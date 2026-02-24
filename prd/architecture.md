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
│  WikiSQL    │ │   LoRA      │ │   seed/time  │
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
WikiSQL (HuggingFace)
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
│         Mistral-7B (Frozen)            │
│                                        │
│  ┌──────────────────────────────────┐  │
│  │  Self-Attention Layers (x32)     │  │
│  │                                  │  │
│  │  q_proj ──── [LoRA A][LoRA B]   │  │  ◄── Trainable (rank=16)
│  │  k_proj ──── [LoRA A][LoRA B]   │  │  ◄── Trainable
│  │  v_proj ──── [LoRA A][LoRA B]   │  │  ◄── Trainable
│  │  o_proj ──── [LoRA A][LoRA B]   │  │  ◄── Trainable
│  │                                  │  │
│  └──────────────────────────────────┘  │
│                                        │
│  ┌──────────────────────────────────┐  │
│  │  MLP Layers (x32)               │  │
│  │                                  │  │
│  │  gate_proj ── [LoRA A][LoRA B]  │  │  ◄── Trainable
│  │  up_proj ──── [LoRA A][LoRA B]  │  │  ◄── Trainable
│  │  down_proj ── [LoRA A][LoRA B]  │  │  ◄── Trainable
│  │                                  │  │
│  └──────────────────────────────────┘  │
│                                        │
│  Total: ~7B params                     │
│  Trainable: ~10M params (0.14%)        │
│  Quantized: 4-bit NF4                  │
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
- **Current**: WikiSQL (~56K train, ~8K val, ~15K test)
- **Next**: Add Spider (~10K multi-table), BIRD-SQL (~12K real-world)
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
- LoRA reduces VRAM from ~28GB (full FT) to ~7.6GB
- Adapter is ~40MB vs ~14GB for full model
- Inference: quantized model uses 4x less memory