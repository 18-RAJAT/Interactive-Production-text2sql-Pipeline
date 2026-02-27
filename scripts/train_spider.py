"""Train on local Spider CSV dataset using SFT with LoRA on MPS/CPU."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

from configs.config_loader import load_config
from data.pipeline import SpiderCSVPipeline
from utils.helpers import set_seed, Timer, get_device


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on Spider CSV with SFT + LoRA")
    parser.add_argument("--config", type=str, default="configs/config_spider.yaml")
    parser.add_argument("--csv", type=str, default="data/spider_text_sql.csv")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.max_train_samples:
        config.data.max_train_samples = args.max_train_samples
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size

    set_seed(config.project.seed)
    device = get_device()
    print(f"Device: {device}")

    print("\n[1/3] Loading data...")
    with Timer("Data Pipeline"):
        pipeline = SpiderCSVPipeline(config, csv_path=args.csv)
        train_ds, val_ds, test_ds = pipeline.prepare()
        stats = pipeline.get_statistics(train_ds)
        print(f"  Train: {stats['num_examples']} samples, avg length: {stats['avg_text_length']:.0f}")
        print(f"  Val: {len(val_ds)} samples, Test: {len(test_ds)} samples")

    train_ds = train_ds.rename_column("sql", "completion")
    val_ds = val_ds.rename_column("sql", "completion")
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in ("prompt", "completion", "text")])
    val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in ("prompt", "completion", "text")])

    print("\n[2/3] Loading model...")
    with Timer("Model Loading"):
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.name,
            trust_remote_code=config.model.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            torch_dtype=torch.float16,
            device_map={"": device},
            trust_remote_code=config.model.trust_remote_code,
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            task_type=config.lora.task_type,
            target_modules=config.lora.target_modules,
        )
        model = get_peft_model(model, lora_config)
        trainable, total = model.get_nb_trainable_parameters()
        print(f"  {config.model.name}: {trainable:,}/{total:,} trainable ({100*trainable/total:.2f}%)")

    print("\n[3/3] Starting SFT training...")
    output_dir = Path(config.project.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tcfg = config.training

    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        max_length=config.model.max_seq_length,
        num_train_epochs=tcfg.num_epochs,
        per_device_train_batch_size=tcfg.per_device_train_batch_size,
        per_device_eval_batch_size=tcfg.per_device_eval_batch_size,
        gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
        learning_rate=tcfg.learning_rate,
        weight_decay=tcfg.weight_decay,
        lr_scheduler_type=tcfg.lr_scheduler_type,
        warmup_ratio=tcfg.warmup_ratio,
        max_grad_norm=tcfg.max_grad_norm,
        fp16=(device == "cuda"),
        bf16=False,
        logging_dir=str(output_dir / "logs"),
        logging_steps=tcfg.logging_steps,
        eval_strategy="steps",
        eval_steps=tcfg.eval_steps,
        save_strategy="steps",
        save_steps=tcfg.save_steps,
        save_total_limit=tcfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=config.project.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    eff_batch = tcfg.per_device_train_batch_size * tcfg.gradient_accumulation_steps
    print(f"  Epochs: {tcfg.num_epochs}, Batch: {tcfg.per_device_train_batch_size}x{tcfg.gradient_accumulation_steps}={eff_batch}")
    print(f"  LR: {tcfg.learning_rate}, Eval every {tcfg.eval_steps} steps")

    resume_path = None
    if args.resume:
        import re
        ckpt_dir = output_dir / "checkpoints"
        checkpoints = []
        for p in ckpt_dir.glob("checkpoint-*"):
            match = re.match(r"^checkpoint-(\d+)$", p.name)
            if match:
                checkpoints.append((int(match.group(1)), p))
        if checkpoints:
            checkpoints.sort(key=lambda x: x[0])
            resume_path = str(checkpoints[-1][1])
            print(f"  Resuming from: {resume_path}")
        else:
            print("  No checkpoint found, starting fresh")

    with Timer("Training"):
        result = trainer.train(resume_from_checkpoint=resume_path)

    final_path = output_dir / "final_model"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    metrics = result.metrics
    print(f"\nTraining complete!")
    print(f"  Loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Runtime: {metrics.get('train_runtime', 0):.1f}s")
    print(f"  Model saved to: {final_path}")

    eval_metrics = trainer.evaluate()
    print(f"  Eval loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")


if __name__ == "__main__":
    main()