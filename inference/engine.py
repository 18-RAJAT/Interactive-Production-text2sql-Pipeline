"""Inference engine: load fine-tuned model and generate SQL from natural language + schema."""

import time
import torch
from typing import Dict, Optional

from models.loader import ModelLoader


class InferenceEngine:
    def __init__(self, config, adapter_path: str):
        self.config = config
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        loader = ModelLoader(self.config)
        self.model, self.tokenizer = loader.load_for_inference(self.adapter_path)
        self.device = next(self.model.parameters()).device
        print(f"Model loaded from {self.adapter_path}")

    def build_prompt(self, question: str, schema: str) -> str:
        return (
            f"[INST] Generate SQL for the following question.\n\n"
            f"Schema:\n{schema}\n\n"
            f"Question:\n{question} [/INST]\n"
        )

    def generate(self, question: str, schema: str) -> Dict:
        prompt = self.build_prompt(question, schema)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        icfg = self.config.inference

        start = time.time()
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
        latency = time.time() - start

        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in full_output:
            sql = full_output.split("[/INST]")[-1].strip()
        else:
            sql = full_output[len(prompt):].strip()

        sql = sql.split("\n")[0].strip()
        if sql and not sql.endswith(";"):
            sql += ";"

        return {
            "question": question,
            "schema": schema,
            "generated_sql": sql,
            "latency_ms": round(latency * 1000, 2),
        }

    def batch_generate(self, examples: list) -> list:
        results = []
        for ex in examples:
            result = self.generate(ex["question"], ex["schema"])
            results.append(result)
        return results

    def interactive(self):
        print("\nText-to-SQL Interactive Mode")
        print("Type 'quit' to exit\n")

        while True:
            schema = input("Schema: ").strip()
            if schema.lower() == "quit":
                break
            question = input("Question: ").strip()
            if question.lower() == "quit":
                break

            result = self.generate(question, schema)
            print(f"\nSQL: {result['generated_sql']}")
            print(f"Latency: {result['latency_ms']}ms\n")