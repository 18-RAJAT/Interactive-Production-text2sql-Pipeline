# Product Requirements Document (PRD)

## Text-to-SQL Fine-Tuning System

**Version**: 1.0
**Author**: Rajat Joshi
**Status**: Production

---

## 1. Problem Statement

Natural language to SQL translation remains a critical bottleneck in data democratization. Non-technical users cannot query databases without SQL expertise. Existing general-purpose LLMs produce inconsistent SQL that fails execution on real schemas.

**This system** fine-tunes an open-source LLM using SFT with LoRA on the Spider dataset to produce accurate, executable SQL from natural language questions grounded in specific database schemas.

---

## 2. Goals

| Goal | Metric | Target |
|------|--------|--------|
| Exact match accuracy | Normalized SQL comparison | > 60% on WikiSQL test |
| Execution accuracy | Query produces correct results | > 55% on WikiSQL test |
| Training efficiency | GPU hours on single A100 | < 4 hours |
| Inference latency | Time per query | < 500ms |
| Model size | Adapter weights | < 100MB |

---

## 3. Scope

### In Scope

- WikiSQL dataset processing and prompt formatting
- SFT + LoRA fine-tuning of TinyLlama-1.1B (configurable base model)
- Evaluation harness with exact match and execution accuracy
- Single-query and batch inference
- Reproducible training with config-driven hyperparameters
- Checkpoint management and best model selection

### Out of Scope (v1)

- Multi-turn conversational SQL generation
- Cross-database generalization (Spider-level complexity)
- Production API with authentication/rate limiting
- Model distillation or ONNX export
- Web UI (Gradio/Streamlit)

---

## 4. User Personas

### ML Engineer (Primary)
- Runs training experiments with different hyperparameters
- Evaluates model quality across SQL categories
- Deploys fine-tuned adapter for downstream use

### Data Analyst (Secondary)
- Uses inference API to generate SQL from questions
- Provides schema context for their specific database
- Validates generated queries before execution

---

## 5. Functional Requirements

### FR-1: Data Pipeline
- Download WikiSQL from HuggingFace Hub
- Parse table schemas (column names, types)
- Construct human-readable SQL from WikiSQL's structured representation
- Format instruction-tuning prompts with schema + question -> SQL
- Split into train/validation/test

### FR-2: Model Loading
- Load base model with 4-bit quantization (bitsandbytes)
- Apply LoRA adapters to attention and MLP layers
- Support loading pre-trained adapters for inference

### FR-3: Training
- SFTTrainer with configurable hyperparameters
- Gradient accumulation for effective batch size control
- Cosine learning rate schedule with warmup
- Periodic evaluation and checkpointing
- Training metadata and metrics logging

### FR-4: Evaluation
- Exact match after SQL normalization
- Breakdown by query category (simple, filter, aggregation)
- JSON report generation

### FR-5: Inference
- Single query: question + schema -> SQL
- Batch processing
- Interactive REPL mode
- Latency measurement

---

## 6. Non-Functional Requirements

| Requirement | Specification |
|------------|---------------|
| Reproducibility | Seed-controlled, config-driven |
| Modularity | Each component independently testable |
| Memory | LoRA enables training on 16GB unified memory (Apple MPS) |
| Extensibility | Swap base model, dataset, or prompt format via config |

---

## 7. Dependencies

### Training Dependencies
- PyTorch >= 2.0
- transformers >= 4.36
- peft >= 0.7
- trl >= 0.7
- bitsandbytes >= 0.41
- accelerate >= 0.25
- datasets >= 2.14

### Runtime Dependencies
- CUDA >= 11.8 (for GPU inference)
- Python >= 3.10

### Infrastructure
- GPU: NVIDIA A100/A10G (training), T4 (inference)
- Storage: ~20GB for model + data + checkpoints
- RAM: 32GB recommended

---

## 8. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| WikiSQL limited complexity | Model won't handle JOINs/subqueries | Extend with Spider dataset in v2 |
| Catastrophic forgetting | Base model loses general capability | LoRA preserves base weights frozen |
| Overfitting on small val set | Inflated accuracy metrics | Use full test split, manual inspection |
| HuggingFace API rate limits | Data download failures | Cache dataset locally |

---

## 9. Success Criteria

The project is considered successful when:
1. Training completes without OOM on a single 24GB GPU
2. Exact match accuracy exceeds 60% on WikiSQL test set
3. Inference latency is under 500ms per query
4. All components are independently runnable via CLI scripts
5. Results are fully reproducible from the config file