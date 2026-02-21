"""Training engine with SFTTrainer, logging, and checkpointing."""

import os
import json
from pathlib import Path
from datetime import datetime

from transformers import TrainingArguments
from trl import SFTTrainer

from utils.helpers import set_seed


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

    def train(self):
        training_args = self._build_training_args()

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.model.max_seq_length,
            tokenizer=self.tokenizer,
        )

        print(f"Starting training: {len(self.train_dataset)} train, {len(self.val_dataset)} val")

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