# Text-to-SQL Fine-Tuning with SFT + LoRA

Fine-tune TinyLlama-1.1B (or any HuggingFace causal LM) to generate SQL queries from natural language questions using SFT with LoRA on the Spider dataset.

## Results

| Metric | Value |
|--------|-------|
| Training Loss | 0.40 |
| Eval Loss | 0.43 |
| Token Accuracy | 87.5% |
| Training Time | ~2.5 hours (Apple MPS) |
| Adapter Size | 4.5 MB |

## Project Structure

```
text-to-sql-finetuning/
├── configs/                 # YAML config + dataclass loader
│   ├── config.yaml
│   ├── config_spider.yaml   # Spider dataset training config
│   ├── config_loader.py
│   └── __init__.py
├── data/                    # Dataset loading, parsing, prompt construction
│   ├── pipeline.py          # WikiSQL + SpiderCSV pipelines
│   ├── spider_text_sql.csv  # Spider benchmark (8,035 examples)
│   └── __init__.py
├── models/                  # Base model loading + LoRA adapter setup
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
│   ├── train.py             # Original WikiSQL training
│   ├── train_spider.py      # Spider dataset training (MPS-optimized)
│   ├── eval.py
│   └── infer.py
├── api/                     # FastAPI backend
│   └── serve.py             # REST API with model inference
├── frontend/                # Next.js web UI
├── reports/                 # Generated documentation
│   └── SFT-Workflow-Report.pdf
├── prd/                     # Product requirements + architecture docs
│   ├── product_requirements.md
│   └── architecture.md
├── outputs/                 # Training outputs (not in git)
│   └── spider/
│       ├── final_model/     # Trained LoRA adapter (4.5 MB)
│       └── checkpoints/     # Intermediate checkpoints
├── run.sh                   # One-command project runner
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## Quick Start

### One-Command Setup

```bash
./run.sh
```

This starts both the API server (port 8000) and frontend (port 3000).

### Manual Setup

#### 1. Install

```bash
git clone <repo-url>
cd text-to-sql-finetuning
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

#### 2. Train on Spider Dataset

```bash
# Full training (works on Apple MPS, CUDA, or CPU)
python scripts/train_spider.py --config configs/config_spider.yaml

# Resume from checkpoint
python scripts/train_spider.py --config configs/config_spider.yaml --resume

# Quick test run
python scripts/train_spider.py --max_train_samples 100 --epochs 1
```

#### 3. Run API with Trained Model

```bash
python api/serve.py --adapter_path outputs/spider/final_model
```

#### 4. Evaluate

```bash
python scripts/eval.py --adapter_path ./outputs/spider/final_model --max_samples 500
```

#### 5. Inference

```bash
# Run example queries
python scripts/infer.py --adapter_path ./outputs/spider/final_model --examples

# Single query
python scripts/infer.py --adapter_path ./outputs/spider/final_model \
  --question "How many employees earn more than 50000?" \
  --schema "TABLE: table ( id (real), name (text), salary (real), dept (text) )"

# Interactive mode
python scripts/infer.py --adapter_path ./outputs/spider/final_model --interactive
```

## Technical Details

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, v_proj |
| Trainable params | 1.1M / 1,101M (0.10%) |

### Training

| Parameter | Value |
|-----------|-------|
| Base model | TinyLlama-1.1B-Chat-v1.0 |
| Dataset | Spider (8,035 train examples) |
| Epochs | 3 |
| Effective batch size | 16 (1 x 16 accumulation) |
| Learning rate | 2e-4 (cosine schedule) |
| Max sequence length | 256 tokens |
| Precision | float16 |
| Gradient checkpointing | Enabled |

### Hardware Compatibility

| Device | Status | Notes |
|--------|--------|-------|
| Apple MPS (M-series Mac) | Supported | Primary development target |
| NVIDIA CUDA | Supported | Auto-detected |
| CPU | Supported | Slowest, fallback option |

### HuggingFace Model

The trained LoRA adapter is published at: [Rj18/text-to-sql-tinyllama-lora](https://huggingface.co/Rj18/text-to-sql-tinyllama-lora)

## Configuration

All hyperparameters are in `configs/config_spider.yaml`. Override via CLI args or edit the YAML directly.

Key sections:
- `model`: Base model name, sequence length, dtype
- `lora`: Rank, alpha, dropout, target modules
- `training`: Epochs, batch size, learning rate, scheduler
- `inference`: Temperature, top_p, max tokens

## Extending

**Different base model**: Change `model.name` in config to any HuggingFace causal LM (CodeLlama, Phi-3, LLaMA-3, etc.)

**Different dataset**: Implement a new pipeline class following the same interface as `SpiderCSVPipeline` with `prepare()` returning `(train, val, test)` datasets.

**Add W&B tracking**: Set `training.report_to: "wandb"` in config and `wandb login` before training.
