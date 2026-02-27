# System Architecture Overview

## Text-to-SQL Fine-Tuning Pipeline

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CONFIG SYSTEM                         │
│              configs/config.yaml                         │
│         configs/config_loader.py (dataclasses)           │
└──────────────────────┬──────────────────────────────────┘
                       │ injected into all components
         ┌─────────────┼─────────────────┐
         ▼             ▼                 ▼
┌─────────────┐ ┌─────────────┐ ┌──────────────┐
│    DATA     │ │    MODEL    │ │    UTILS     │
│  pipeline   │ │   loader    │ │   helpers    │
│  Spider     │ │   LoRA      │ │   seed/time  │
│  prompts    │ │   LoRA cfg  │ │   device     │
└──────┬──────┘ └──────┬──────┘ └──────────────┘
       │               │
       ▼               ▼
┌──────────────────────────────────┐
│          TRAINING ENGINE          │
│    SFTTrainer + checkpointing     │
│    logging + metrics              │
└──────────────┬───────────────────┘
               │ saves adapter
               ▼
┌──────────────────────────────────┐
│        SAVED ADAPTER (.pt)        │
│     outputs/final_model/          │
└──────┬───────────────┬───────────┘
       │               │
       ▼               ▼
┌─────────────┐ ┌─────────────────┐
│ EVALUATION  │ │   INFERENCE     │
│ exact match │ │   single query  │
│ exec accur  │ │   batch mode    │
│ breakdown   │ │   interactive   │
└─────────────┘ └─────────────────┘
```

---

## 2. Data Flow Diagram

```
Spider (CSV)
       │
       ▼
┌──────────────────┐
│  Raw Download     │  load_dataset("wikisql")
│  train/val/test   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Schema Parsing   │  table headers + types -> "TABLE: t (col1 (type), ...)"
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  SQL Construction │  sql struct -> "SELECT col FROM table WHERE ..."
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Prompt Formatting│  "[INST] Generate SQL...\nSchema:...\nQuestion:... [/INST]\nSELECT..."
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Tokenization     │  SFTTrainer handles tokenization internally
│  (max_seq=1024)   │
└────────┬─────────┘
         │
         ▼
       Model
```

---

## 3. Model Architecture

```
┌────────────────────────────────────────┐
│         TinyLlama-1.1B (Frozen)        │
│                                        │
│  ┌──────────────────────────────────┐  │
│  │  Self-Attention Layers (x22)     │  │
│  │                                  │  │
│  │  q_proj ──── [LoRA A][LoRA B]   │  │  ◄── Trainable (rank=8)
│  │  v_proj ──── [LoRA A][LoRA B]   │  │  ◄── Trainable
│  │                                  │  │
│  └──────────────────────────────────┘  │
│                                        │
│  Total: ~1.1B params                   │
│  Trainable: ~1.1M params (0.10%)       │
│  Precision: float16                    │
└────────────────────────────────────────┘
```

---

## 4. Training Pipeline Detail

```
                    Epoch Loop
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
    ┌─────────┐   ┌──────────┐   ┌─────────┐
    │ Forward  │   │ Backward │   │ Step    │
    │ Pass     │──▶│ Pass     │──▶│ (every  │
    │ (bf16)   │   │ (grad    │   │  4 acc  │
    │          │   │  accum)  │   │  steps) │
    └─────────┘   └──────────┘   └────┬────┘
                                      │
                        ┌─────────────┼─────────────┐
                        ▼             ▼             ▼
                  ┌──────────┐ ┌──────────┐ ┌──────────┐
                  │ Log loss │ │ Eval on  │ │ Save     │
                  │ (every   │ │ val set  │ │ ckpt     │
                  │  25 steps│ │ (250 st) │ │ (250 st) │
                  └──────────┘ └──────────┘ └──────────┘
                                    │
                                    ▼
                              Best model saved
                              (lowest eval_loss)
```

---

## 5. Downstream Dependencies

```
Training Output (adapter weights)
         │
         ├──▶ Evaluation Pipeline
         │         │
         │         ├── Exact match accuracy
         │         ├── Category breakdown (simple/filter/aggregation)
         │         └── JSON report for CI/tracking
         │
         ├──▶ Inference Engine
         │         │
         │         ├── CLI single query
         │         ├── Batch processing
         │         └── Interactive REPL
         │
         ├──▶ Future: API Server (FastAPI)
         │         │
         │         ├── REST endpoint /generate
         │         ├── Request validation
         │         └── Response caching
         │
         └──▶ Future: Monitoring
                   │
                   ├── Latency tracking
                   ├── Query failure rate
                   └── Drift detection
```

---

## 6. Scaling Considerations

### Compute Scaling
- **Single GPU**: LoRA with gradient accumulation (current)
- **Multi-GPU**: DeepSpeed ZeRO-3 or FSDP via accelerate config
- **Multi-node**: Slurm + accelerate launcher

### Data Scaling
- **Current**: Spider (~7K train examples, multi-table)
- **Next**: Add BIRD-SQL (~12K real-world), WikiSQL (~56K single-table)
- **Advanced**: Synthetic data generation from schema catalog

### Model Scaling
- **7B** (current): Best latency/accuracy for single GPU
- **13B**: Better accuracy, needs 2x A100 or quantization tricks
- **70B**: State-of-the-art accuracy, needs multi-node or API-based

### Serving Scaling
- **Single request**: Current inference engine (~200-500ms)
- **Batch serving**: vLLM with continuous batching
- **Production**: TGI behind load balancer, adapter hot-swapping

### Cost Optimization
- LoRA reduces VRAM from ~4.5GB (full FT) to ~2.5GB
- Adapter is ~4.5MB vs ~2.2GB for full model
- Inference: float16 precision with gradient checkpointing