"""Training engine with SFTTrainer, logging, checkpointing, and early stopping."""

import re
import json
from pathlib import Path
from datetime import datetime

import torch
from transformers import TrainingArguments, TrainerCallback, EarlyStoppingCallback
from trl import SFTTrainer

from utils.helpers import set_seed


def _normalize_sql(sql: str) -> str:
    sql = sql.strip().lower()
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.rstrip(";").strip()
    sql = re.sub(r"\s*,\s*", ", ", sql)
    sql = re.sub(r"\s*=\s*", " = ", sql)
    sql = re.sub(r"\s*>\s*", " > ", sql)
    sql = re.sub(r"\s*<\s*", " < ", sql)
    return sql


class ExactMatchEvalCallback(TrainerCallback):
    """Computes exact-match accuracy on a validation subset at each eval step.

    Runs model.generate() on a sample of the validation set, extracts the
    predicted SQL, and compares against gold SQL using normalized matching.
    The metric is logged as ``eval_exact_match`` so it can drive best-model
    selection and early stopping.
    """

    def __init__(self, val_dataset, tokenizer, config, max_samples=50):
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.config = config
        self.max_samples = min(max_samples, len(val_dataset))

    def _generate_sql(self, model, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        icfg = self.config.inference

        with torch.no_grad():
            outputs = model.generate(
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

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        model.eval()
        correct = 0
        total = self.max_samples

        for i in range(total):
            example = self.val_dataset[i]
            predicted = self._generate_sql(model, example["prompt"])
            gold = example["sql"]
            if _normalize_sql(predicted) == _normalize_sql(gold):
                correct += 1

        accuracy = correct / total if total > 0 else 0.0

        if state.log_history:
            state.log_history[-1]["eval_exact_match"] = accuracy
        else:
            state.log_history.append({"eval_exact_match": accuracy})

        print(f"  [ExactMatch] {correct}/{total} = {accuracy:.4f}")


class SQLTrainer:
    def __init__(self, config, model, tokenizer, train_dataset, val_dataset):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.output_dir = Path(config.project.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        set_seed(config.project.seed)

    def _build_training_args(self) -> TrainingArguments:
        tcfg = self.config.training
        run_name = f"{self.config.project.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        return TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            run_name=run_name,
            num_train_epochs=tcfg.num_epochs,
            per_device_train_batch_size=tcfg.per_device_train_batch_size,
            per_device_eval_batch_size=tcfg.per_device_eval_batch_size,
            gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
            learning_rate=tcfg.learning_rate,
            weight_decay=tcfg.weight_decay,
            lr_scheduler_type=tcfg.lr_scheduler_type,
            warmup_ratio=tcfg.warmup_ratio,
            max_grad_norm=tcfg.max_grad_norm,
            fp16=tcfg.fp16,
            bf16=tcfg.bf16,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=tcfg.logging_steps,
            eval_strategy="steps",
            eval_steps=tcfg.eval_steps,
            save_strategy="steps",
            save_steps=tcfg.save_steps,
            save_total_limit=tcfg.save_total_limit,
            load_best_model_at_end=tcfg.load_best_model_at_end,
            metric_for_best_model=tcfg.metric_for_best_model,
            greater_is_better=tcfg.greater_is_better,
            report_to=tcfg.report_to,
            seed=self.config.project.seed,
            dataloader_num_workers=self.config.data.num_workers,
            remove_unused_columns=False,
        )

    def _save_training_metadata(self, trainer):
        metadata = {
            "model": self.config.model.name,
            "lora_r": self.config.lora.r,
            "lora_alpha": self.config.lora.lora_alpha,
            "learning_rate": self.config.training.learning_rate,
            "num_epochs": self.config.training.num_epochs,
            "effective_batch_size": (
                self.config.training.per_device_train_batch_size
                * self.config.training.gradient_accumulation_steps
            ),
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset),
            "timestamp": datetime.now().isoformat(),
        }

        meta_path = self.output_dir / "training_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _build_callbacks(self):
        tcfg = self.config.training

        em_callback = ExactMatchEvalCallback(
            val_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            config=self.config,
            max_samples=tcfg.early_stopping_eval_samples,
        )

        es_callback = EarlyStoppingCallback(
            early_stopping_patience=tcfg.early_stopping_patience,
            early_stopping_threshold=tcfg.early_stopping_threshold,
        )

        return [em_callback, es_callback]

    def train(self):
        training_args = self._build_training_args()
        callbacks = self._build_callbacks()

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.model.max_seq_length,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
        )

        tcfg = self.config.training
        print(f"Starting training: {len(self.train_dataset)} train, {len(self.val_dataset)} val")
        print(f"Early stopping: patience={tcfg.early_stopping_patience}, "
              f"threshold={tcfg.early_stopping_threshold}, "
              f"eval_samples={tcfg.early_stopping_eval_samples}")

        result = trainer.train()
        self._save_training_metadata(trainer)

        final_model_path = self.output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))

        metrics = result.metrics
        metrics_path = self.output_dir / "train_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Training complete. Model saved to {final_model_path}")
        print(f"Metrics: loss={metrics.get('train_loss', 'N/A'):.4f}, "
              f"runtime={metrics.get('train_runtime', 0):.1f}s")

        return result, trainer

    def evaluate(self, trainer):
        metrics = trainer.evaluate()
        eval_path = self.output_dir / "eval_metrics.json"
        with open(eval_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Eval loss: {metrics.get('eval_loss', 'N/A'):.4f}")
        return metrics