"""Main training script: data loading -> model setup -> LoRA fine-tuning."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_loader import load_config
from data.pipeline import get_pipeline
from models.loader import ModelLoader
from training.trainer import SQLTrainer
from utils.helpers import set_seed, get_device_info, Timer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM on WikiSQL with LoRA")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None, choices=["wikisql", "sql-create-context"])
    args = parser.parse_args()

    config = load_config(args.config)

    if args.max_train_samples:
        config.data.max_train_samples = args.max_train_samples
    if args.max_val_samples:
        config.data.max_val_samples = args.max_val_samples
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.dataset:
        config.data.dataset_name = args.dataset

    set_seed(config.project.seed)

    device_info = get_device_info()
    print(f"Device: {device_info['device']} - {device_info.get('gpu_name', 'N/A')}")

    # Phase 1: Data
    print("\n[1/3] Preparing data...")
    with Timer("Data Pipeline"):
        pipeline = get_pipeline(config)
        train_ds, val_ds, _ = pipeline.prepare()
        stats = pipeline.get_statistics(train_ds)
        print(f"  Train: {stats['num_examples']} samples, avg length: {stats['avg_text_length']:.0f}")
        print(f"  Val: {len(val_ds)} samples")

    # Phase 2: Model
    print("\n[2/3] Loading model...")
    with Timer("Model Loading"):
        loader = ModelLoader(config)
        model, tokenizer = loader.load_for_training()
        info = loader.get_model_info(model)
        print(f"  {info['model_name']}: {info['trainable_pct']} trainable")

    # Phase 3: Training
    print("\n[3/3] Starting training...")
    with Timer("Training"):
        trainer = SQLTrainer(config, model, tokenizer, train_ds, val_ds)
        result, hf_trainer = trainer.train()
        trainer.evaluate(hf_trainer)

    print("\nDone.")


if __name__ == "__main__":
    main()