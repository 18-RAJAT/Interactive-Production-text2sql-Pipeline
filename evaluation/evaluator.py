"""Evaluation harness: exact match accuracy, execution accuracy, breakdown by SQL type."""

import re
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import torch
from tqdm import tqdm


def normalize_sql(sql: str) -> str:
    sql = sql.strip().lower()
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.rstrip(";").strip()
    sql = re.sub(r"\s*,\s*", ", ", sql)
    sql = re.sub(r"\s*=\s*", " = ", sql)
    sql = re.sub(r"\s*>\s*", " > ", sql)
    sql = re.sub(r"\s*<\s*", " < ", sql)
    return sql


def classify_query(sql: str) -> str:
    sql_lower = sql.lower()
    if "count(" in sql_lower or "sum(" in sql_lower or "avg(" in sql_lower or "max(" in sql_lower or "min(" in sql_lower:
        return "aggregation"
    if " where " in sql_lower:
        return "filter"
    return "simple"


class SQLEvaluator:
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def generate_sql(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        icfg = self.config.inference

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=icfg.max_new_tokens,
                temperature=icfg.temperature,
                top_p=icfg.top_p,
                do_sample=icfg.do_sample,
                repetition_penalty=icfg.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in full_output:
            return full_output.split("[/INST]")[-1].strip()
        return full_output[len(prompt):].strip()

    def exact_match(self, predicted: str, gold: str) -> bool:
        return normalize_sql(predicted) == normalize_sql(gold)

    def execution_accuracy(self, predicted: str, gold: str, schema: str) -> bool:
        try:
            with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
                conn = sqlite3.connect(tmp.name)
                cursor = conn.cursor()
                cursor.execute(
                    "CREATE TABLE IF NOT EXISTS 'table' (id INTEGER PRIMARY KEY)"
                )

                cols = re.findall(r"(\w+)\s*\(", schema.split("(", 1)[-1]) if "(" in schema else []
                for col in cols:
                    if col.lower() != "id":
                        try:
                            cursor.execute(f"ALTER TABLE 'table' ADD COLUMN '{col}' TEXT")
                        except sqlite3.OperationalError:
                            pass

                conn.commit()

                try:
                    pred_result = set(cursor.execute(predicted).fetchall())
                except Exception:
                    conn.close()
                    return False

                try:
                    gold_result = set(cursor.execute(gold).fetchall())
                except Exception:
                    conn.close()
                    return False

                conn.close()
                return pred_result == gold_result
        except Exception:
            return False

    def evaluate_dataset(self, test_dataset, max_samples: int = None) -> Dict:
        if max_samples:
            test_dataset = test_dataset.select(range(min(max_samples, len(test_dataset))))

        results = {
            "exact_match": [],
            "predictions": [],
            "category_results": defaultdict(lambda: {"correct": 0, "total": 0}),
        }

        total_em = 0

        for i, example in enumerate(tqdm(test_dataset, desc="Evaluating")):
            predicted = self.generate_sql(example["prompt"])
            gold = example["sql"]

            em = self.exact_match(predicted, gold)
            category = classify_query(gold)

            results["exact_match"].append(em)
            results["predictions"].append({
                "index": i,
                "prompt": example["prompt"][:200],
                "predicted": predicted,
                "gold": gold,
                "exact_match": em,
                "category": category,
            })

            results["category_results"][category]["total"] += 1
            if em:
                results["category_results"][category]["correct"] += 1
                total_em += 1

        total = len(results["exact_match"])
        summary = {
            "total_samples": total,
            "exact_match_accuracy": total_em / total if total > 0 else 0,
            "category_breakdown": {},
        }

        for cat, vals in results["category_results"].items():
            acc = vals["correct"] / vals["total"] if vals["total"] > 0 else 0
            summary["category_breakdown"][cat] = {
                "accuracy": acc,
                "correct": vals["correct"],
                "total": vals["total"],
            }

        return {"summary": summary, "details": results["predictions"]}

    def save_results(self, results: Dict, output_path: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation results saved to {path}")

    def print_summary(self, results: Dict):
        summary = results["summary"]
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total samples: {summary['total_samples']}")
        print(f"Exact match accuracy: {summary['exact_match_accuracy']:.4f}")
        print("\nCategory breakdown:")
        for cat, vals in summary["category_breakdown"].items():
            print(f"  {cat:15s}: {vals['accuracy']:.4f} ({vals['correct']}/{vals['total']})")
        print("=" * 60)