"""Evaluation script: load fine-tuned model and run metrics on test set."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_loader import load_config
from data.pipeline import get_pipeline
from models.loader import ModelLoader
from evaluation.evaluator import SQLEvaluator
from utils.helpers import set_seed, Timer


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Text-to-SQL model")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--max_samples", type=int, default=500, help="Max test samples")
    parser.add_argument("--output", type=str, default="./outputs/eval_results.json")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.project.seed)

    # Load test data
    print("[1/3] Loading test data...")
    pipeline = get_pipeline(config)
    _, _, test_ds = pipeline.prepare()
    print(f"  Test set: {len(test_ds)} samples")

    # Load model
    print("[2/3] Loading fine-tuned model...")
    with Timer("Model Loading"):
        loader = ModelLoader(config)
        model, tokenizer = loader.load_for_inference(args.adapter_path)

    # Evaluate
    print("[3/3] Running evaluation...")
    evaluator = SQLEvaluator(config, model, tokenizer)
    with Timer("Evaluation"):
        results = evaluator.evaluate_dataset(test_ds, max_samples=args.max_samples)

    evaluator.print_summary(results)
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()