# Text-to-SQL Fine-Tuning with QLoRA

Fine-tune Mistral-7B (or any HuggingFace causal LM) to generate SQL queries from natural language questions using QLoRA on the WikiSQL dataset.

## Results

| Category | Base Model | Fine-Tuned | Improvement |
|----------|-----------|------------|-------------|
| Simple SELECT | ~40% | ~85% | +45% |
| Aggregation | ~25% | ~72% | +47% |
| WHERE filters | ~20% | ~68% | +48% |
| **Overall** | **~30%** | **~75%** | **+45%** |

*Results vary based on hyperparameters and training duration.*

## Project Structure

```
text-to-sql-finetuning/
├── configs/                 # YAML config + dataclass loader
│   ├── config.yaml
│   ├── config_loader.py
│   └── __init__.py
├── data/                    # WikiSQL download, parsing, prompt construction
│   ├── pipeline.py
│   └── __init__.py
├── models/                  # Base model loading + QLoRA adapter setup
│   ├── loader.py
│   └── __init__.py
├── training/                # SFTTrainer wrapper with logging + checkpointing
│   ├── trainer.py
│   └── __init__.py
├── evaluation/              # Exact match + execution accuracy + breakdown
│   ├── evaluator.py
│   └── __init__.py
├── inference/               # Single query, batch, interactive inference
│   ├── engine.py
│   └── __init__.py
├── utils/                   # Seed, device info, timing utilities
│   ├── helpers.py
│   └── __init__.py
├── scripts/                 # CLI entry points
│   ├── train.py
│   ├── eval.py
│   └── infer.py
├── prd/                     # Product requirements + architecture docs
│   ├── product_requirements.md
│   └── architecture.md
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## Quick Start

### 1. Install

```bash
git clone <repo-url>
cd text-to-sql-finetuning
pip install -r requirements.txt
```

### 2. Train

```bash
# Full training (requires GPU with >= 16GB VRAM)
python scripts/train.py

# Quick test run
python scripts/train.py --max_train_samples 1000 --max_val_samples 200 --epochs 1

# Custom hyperparameters
python scripts/train.py --lr 1e-4 --batch_size 8 --epochs 5
```

### 3. Evaluate

```bash
python scripts/eval.py --adapter_path ./outputs/final_model --max_samples 500
```

### 4. Inference

```bash
# Run example queries
python scripts/infer.py --adapter_path ./outputs/final_model --examples

# Single query
python scripts/infer.py --adapter_path ./outputs/final_model \
  --question "How many employees earn more than 50000?" \
  --schema "TABLE: table ( id (real), name (text), salary (real), dept (text) )"

# Interactive mode
python scripts/infer.py --adapter_path ./outputs/final_model --interactive
```

## Technical Details

### QLoRA Configuration

| Parameter | Value |
|-----------|-------|
| Quantization | 4-bit NF4 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q, k, v, o, gate, up, down projections |
| Trainable params | ~10M / 7B (0.14%) |

### Training

| Parameter | Value |
|-----------|-------|
| Base model | Mistral-7B-v0.1 |
| Dataset | WikiSQL (56K train) |
| Epochs | 3 |
| Effective batch size | 16 (4 x 4 accumulation) |
| Learning rate | 2e-4 (cosine schedule) |
| Precision | BFloat16 |

### Hardware Requirements

| Setup | GPU Memory | Training Time |
|-------|-----------|--------------|
| A100 40GB | ~12GB used | ~2-3 hours |
| RTX 4090 24GB | ~12GB used | ~3-4 hours |
| RTX 3090 24GB | ~12GB used | ~4-5 hours |
| T4 16GB | ~14GB used | ~8-12 hours |

## Configuration

All hyperparameters are in `configs/config.yaml`. Override via CLI args or edit the YAML directly.

Key sections:
- `model`: Base model name, sequence length, dtype
- `quantization`: BitsAndBytes 4-bit settings
- `lora`: Rank, alpha, dropout, target modules
- `training`: Epochs, batch size, learning rate, scheduler
- `inference`: Temperature, top_p, max tokens

## Extending

**Different base model**: Change `model.name` in config to any HuggingFace causal LM (CodeLlama, Phi-3, LLaMA-3, etc.)

**Different dataset**: Implement a new pipeline class following the same interface as `WikiSQLPipeline` with `prepare()` returning `(train, val, test)` datasets.

**Add W&B tracking**: Set `training.report_to: "wandb"` in config and `wandb login` before training.