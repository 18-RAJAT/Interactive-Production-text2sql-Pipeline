"""Inference script: generate SQL queries from natural language questions."""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_loader import load_config
from inference.engine import InferenceEngine


def run_examples(engine: InferenceEngine):
    examples = [
        {
            "question": "How many records are in the table?",
            "schema": "TABLE: table ( id (real), name (text), age (real), city (text) )",
        },
        {
            "question": "What is the name where age is greater than 30?",
            "schema": "TABLE: table ( id (real), name (text), age (real), department (text) )",
        },
        {
            "question": "What is the maximum salary?",
            "schema": "TABLE: table ( id (real), name (text), salary (real), role (text) )",
        },
        {
            "question": "Which city has the most entries?",
            "schema": "TABLE: table ( id (real), name (text), city (text), country (text) )",
        },
    ]

    print("\nRunning example queries:")
    print("-" * 60)

    results = engine.batch_generate(examples)
    for r in results:
        print(f"Q: {r['question']}")
        print(f"S: {r['schema'][:80]}...")
        print(f"SQL: {r['generated_sql']}")
        print(f"Latency: {r['latency_ms']}ms")
        print("-" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned model")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--schema", type=str, default=None)
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--examples", action="store_true", help="Run built-in examples")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    engine = InferenceEngine(config, args.adapter_path)

    if args.interactive:
        engine.interactive()
    elif args.examples:
        results = run_examples(engine)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
    elif args.question and args.schema:
        result = engine.generate(args.question, args.schema)
        print(f"\nSQL: {result['generated_sql']}")
        print(f"Latency: {result['latency_ms']}ms")
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
    else:
        parser.print_help()
        print("\nExamples:")
        print('  python scripts/infer.py --adapter_path ./outputs/final_model --examples')
        print('  python scripts/infer.py --adapter_path ./outputs/final_model --interactive')
        print('  python scripts/infer.py --adapter_path ./outputs/final_model \\')
        print('    --question "How many rows?" --schema "TABLE: t (id (real), name (text))"')


if __name__ == "__main__":
    main()