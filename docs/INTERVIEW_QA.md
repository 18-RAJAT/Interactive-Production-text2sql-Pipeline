# Text-to-SQL Fine-Tuning - Interview Q&A Reference

---

## SECTION 1: PROJECT OVERVIEW

### Q1. What is this project about? Give a 30-second elevator pitch.

**A:** This project fine-tunes TinyLlama-1.1B (a small open-source LLM) to convert natural-language questions into SQL queries. We use **Supervised Fine-Tuning (SFT)** with **LoRA** (Low-Rank Adaptation) on the **Spider benchmark** (8,035 examples). The entire adapter is only **4.5 MB** and achieves **0.40 training loss / 0.43 eval loss / 87.5% token accuracy**. The project includes a **FastAPI backend** and a **Next.js frontend** for end-to-end demo. The trained adapter is published on HuggingFace.

### Q2. What is the end-to-end architecture?

**A:**
```
User Question + Schema
       │
       ▼
  Next.js Frontend (React, App Router)
       │  POST /generate_sql
       ▼
  FastAPI Backend (serve.py)
       │
       ├── Model loaded? → LoRA Model Inference
       │                    (TinyLlama + LoRA adapter)
       │
       └── No model?    → Rule-based SQL Fallback
       │
       ▼
  Response: { generated_sql, confidence, latency_ms }
```

### Q3. Walk me through the ML pipeline.

**A:** Five stages:
1. **Data Pipeline** (`data/pipeline.py`): Load Spider CSV → format into `[INST]...[/INST]` prompts → split 90/5/5 (train/val/test)
2. **Model Loading** (`models/loader.py`): Load TinyLlama base → apply LoRA adapters to q_proj and v_proj
3. **Training** (`scripts/train_spider.py`): SFTTrainer from TRL library, cosine LR schedule, gradient checkpointing, early stopping
4. **Evaluation** (`evaluation/evaluator.py`): Exact match accuracy + execution accuracy + category breakdown
5. **Inference** (`inference/engine.py` + `api/serve.py`): Load adapter → generate SQL → compute confidence → return via REST API

---

## SECTION 2: MODEL & FINE-TUNING

### Q4. Why TinyLlama-1.1B? Why not GPT-4 or LLaMA-3-70B?

**A:**
- **Resource constraints**: TinyLlama is 1.1B params — fits on Apple MPS (M-series Mac) without needing a GPU cluster
- **Fast iteration**: Training takes ~2.5 hours on MPS vs days for larger models
- **LoRA efficiency**: Only 0.10% of params are trainable (1.1M out of 1,101M) — 4.5 MB adapter
- **Proof of concept**: Demonstrates that even small models + LoRA can learn structured generation tasks well
- **Production cost**: Inference is cheap and fast compared to API calls to GPT-4

### Q5. What is LoRA? How does it work?

**A:** LoRA (Low-Rank Adaptation) freezes the original model weights and injects small trainable rank-decomposition matrices into selected layers.

For a weight matrix W (d × d), instead of fine-tuning W directly:
```
W' = W + ΔW = W + B × A
```
where A is (r × d) and B is (d × r), with r << d (rank).

**Key idea**: The weight update ΔW has low rank, so we only train 2 × r × d parameters instead of d × d.

**Our config:**
| Parameter | Value | Why |
|-----------|-------|-----|
| r (rank) | 8 | Low rank = fewer params, still expressive for SQL |
| alpha | 16 | Scaling factor = alpha/r = 2, controls update magnitude |
| dropout | 0.05 | Light regularization |
| target_modules | q_proj, v_proj | Attention query + value projections — most impactful for task adaptation |

**Result**: 1.1M trainable / 1,101M total = **0.10%** of the model is trained.

### Q6. Why target only q_proj and v_proj? Why not all attention matrices?

**A:**
- The original LoRA paper (Hu et al., 2021) showed that adapting **query and value** projections gives the best trade-off of performance vs parameter count
- k_proj (key) and o_proj (output) add more params with diminishing returns
- gate_proj/up_proj/down_proj (MLP layers) are available in the WikiSQL config for more complex tasks but weren't needed for Spider
- Fewer target modules = faster training, smaller adapter, less risk of overfitting

### Q7. What is SFT (Supervised Fine-Tuning)? How is it different from RLHF?

**A:**
- **SFT**: Train the model on (input, output) pairs with a standard cross-entropy language modeling loss. The model learns to generate the correct SQL given the prompt.
- **RLHF**: After SFT, use a reward model + PPO to optimize for human preferences — used for alignment, not needed for structured SQL generation.
- We use SFT because SQL has a clear ground truth — no subjective "preference" involved.

### Q8. Explain the training hyperparameters.

**A:**
| Hyperparameter | Value | Reasoning |
|----------------|-------|-----------|
| Epochs | 3 | Standard for SFT; more risks overfitting on 8K examples |
| Batch size | 1 (device) × 16 (accumulation) = **16 effective** | Memory-limited on MPS; accumulation simulates large batch |
| Learning rate | 2e-4 | Standard for LoRA fine-tuning (higher than full fine-tuning) |
| LR scheduler | Cosine | Smooth decay prevents sudden divergence |
| Warmup ratio | 0.05 | 5% of steps for warmup — prevents initial instability |
| Weight decay | 0.01 | Light L2 regularization |
| Max grad norm | 1.0 | Gradient clipping prevents exploding gradients |
| Precision | float16 | Half precision saves memory, no quality loss for inference |
| Gradient checkpointing | Enabled | Trades compute for memory — essential on MPS |
| Early stopping patience | 3 | Stop if eval_loss doesn't improve for 3 consecutive evals |
| Max seq length | 256 | Spider queries are short; saves memory |

### Q9. Why gradient checkpointing? What's the trade-off?

**A:** Gradient checkpointing re-computes activations during backward pass instead of storing them all in memory.
- **Benefit**: ~60% memory reduction — essential for training on Apple MPS with limited unified memory
- **Cost**: ~20-30% slower training due to recomputation
- Enabled via `model.gradient_checkpointing_enable()` in `train_spider.py`

### Q10. What is the prompt template?

**A:**
```
[INST] Generate SQL for the following question.

Question:
{question} [/INST]
{sql}
```
For the API (with schema):
```
[INST] Generate SQL for the following question.

Schema:
{schema}

Question:
{question} [/INST]
{sql}
```
The `[INST]...[/INST]` format is TinyLlama's chat template (inherited from LLaMA-2 chat).

### Q11. Why is the Spider pipeline schema-free while WikiSQL uses schema?

**A:**
- **Spider CSV** only has `text_query` and `sql_command` columns — no schema info in the CSV
- **WikiSQL** has `create_table_statement` — schema is available
- The model still learns SQL syntax and patterns without explicit schema
- At inference time via the API, schema IS provided to help the model generate contextually correct SQL

### Q12. Explain quantization. Why is it disabled for Spider?

**A:**
- **4-bit quantization (NF4)**: Compresses model weights from 16-bit to 4-bit using BitsAndBytes — reduces memory by ~4x
- **Why disabled for Spider**: Apple MPS does not support BitsAndBytes (it's CUDA-only). Training runs on MPS, so `load_in_4bit: false`
- **WikiSQL config**: Uses 4-bit quantization because it targets CUDA GPUs with Mistral-7B (much larger model)
- If running on CUDA, you could enable it for even less memory usage

---

## SECTION 3: DATA PIPELINE

### Q13. Describe the Spider dataset.

**A:**
- **8,035 examples** in `data/spider_text_sql.csv`
- Columns: `text_query` (natural language) and `sql_command` (gold SQL)
- Multi-table, complex queries spanning 200+ databases
- Industry-standard benchmark for text-to-SQL evaluation
- Examples:
  - "How many heads of departments are older than 56?" → `SELECT count(*) FROM head WHERE age > 56`
  - "List the name, born state and age of the heads ordered by age" → `SELECT name, born_state, age FROM head ORDER BY age`

### Q14. How is data split?

**A:**
```python
# 90% train, 10% rest
split = raw.train_test_split(test_size=0.1, seed=42)
# Rest split 50/50 into val and test
val_test = split["test"].train_test_split(test_size=0.5, seed=42)
```
Result: ~7,231 train / ~402 val / ~402 test (approximate)

### Q15. What other datasets does the project support?

**A:** Three pipeline classes:
1. **SpiderCSVPipeline**: Local CSV, used for primary training
2. **WikiSQLPipeline**: HuggingFace `kaxap/pg-wikiSQL-sql-instructions-80k` — 80K examples with schema
3. **SQLContextPipeline**: HuggingFace `b-mc2/sql-create-context` — 78K examples with schema
- Factory function `get_pipeline(config)` selects based on `config.data.dataset_name`

### Q16. How does the prompt formatting work?

**A:** Each pipeline has a `format_example()` method that creates:
```python
{
    "text": prompt + sql,       # Full training text (input + target)
    "prompt": prompt,           # Just the input (for evaluation)
    "sql": sql,                 # Gold SQL (for evaluation)
    "schema": schema            # Schema (empty for Spider CSV)
}
```
SFTTrainer uses the `"text"` field for causal LM training — the model learns to predict `sql` tokens given the `prompt` tokens.

---

## SECTION 4: EVALUATION

### Q17. What evaluation metrics do you use?

**A:** Three metrics:

1. **Exact Match Accuracy**: Normalize both predicted and gold SQL (lowercase, strip whitespace, remove semicolons, standardize operators), then check string equality
2. **Execution Accuracy**: Execute both queries on an in-memory SQLite database, compare result sets
3. **Category Breakdown**: Classify queries by type and report accuracy per category:
   - `aggregation`: Contains COUNT, SUM, AVG, MAX, MIN
   - `filter`: Contains WHERE clause
   - `simple`: Everything else

### Q18. How does SQL normalization work?

**A:**
```python
def normalize_sql(sql):
    sql = sql.strip().lower()           # Lowercase
    sql = re.sub(r"\s+", " ", sql)      # Collapse whitespace
    sql = sql.rstrip(";").strip()       # Remove trailing semicolons
    sql = re.sub(r"\s*,\s*", ", ", sql) # Standardize comma spacing
    sql = re.sub(r"\s*=\s*", " = ", sql)# Standardize operator spacing
    # ... same for > and <
```
This ensures `SELECT  COUNT(*) FROM table ;` matches `select count(*) from table`.

### Q19. How does execution accuracy work?

**A:** Creates a temporary SQLite database, creates a generic table with columns parsed from the schema, runs both predicted and gold SQL, and compares result sets:
```python
pred_result = set(cursor.execute(predicted).fetchall())
gold_result = set(cursor.execute(gold).fetchall())
return pred_result == gold_result
```
- Returns `False` if either query throws an exception
- Uses `set()` comparison so row order doesn't matter

### Q20. What were the actual results?

**A:**
| Metric | Value |
|--------|-------|
| Training Loss | 0.40 |
| Eval Loss | 0.43 |
| Token Accuracy | 87.5% |
| Training Time | ~2.5 hours (Apple MPS) |
| Adapter Size | 4.5 MB |

The gap between train and eval loss (0.03) is small, suggesting the model isn't overfitting badly.

---

## SECTION 5: CONFIDENCE SCORING

### Q21. How does the confidence score work?

**A:** Two modes:

**Model Inference Mode** (real model loaded):
```python
def confidence_from_scores(scores, generated_ids):
    log_probs = []
    for step_idx, logits in enumerate(scores):
        lp = F.log_softmax(logits[0].float(), dim=-1)
        token_id = generated_ids[step_idx]
        log_probs.append(lp[token_id].item())

    avg_conf = exp(mean(log_probs))     # Geometric mean probability
    min_conf = exp(min(log_probs))      # Minimum token probability
    confidence = 0.8 * avg_conf + 0.2 * min_conf  # Blend
```
- 80% weight on geometric mean (overall quality)
- 20% weight on minimum token probability (catches if any token is highly uncertain)
- Clamped to [0, 1]

**Mock/Fallback Mode** (no model loaded):
- Heuristic based on: schema parseability, question pattern matching (how many, average, list, etc.), table/column name overlap with question
- Scores from 0.15 (gibberish input) to ~0.85 (clear pattern + schema match)

### Q22. Why geometric mean instead of arithmetic mean for confidence?

**A:**
- Geometric mean of probabilities = exp(arithmetic mean of log probabilities)
- It's more sensitive to low-probability tokens — a single uncertain token pulls the score down more
- Arithmetic mean could mask a single very uncertain token among many confident ones
- The 20% min-token component further guards against this

---

## SECTION 6: API & SERVING

### Q23. Describe the API endpoints.

**A:**
| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/health` | GET | — | `{"status": "ok", "model_loaded": true/false}` |
| `/generate_sql` | POST | `{"question": "...", "schema": "..."}` | `{"generated_sql": "...", "confidence": 0.83, "latency_ms": 142.1}` |

### Q24. What happens when no model is loaded (mock mode)?

**A:** The API falls back to `_generate_smart_sql()` — a rule-based SQL generator that:
1. Parses the schema to extract table names and columns
2. Classifies the question using regex patterns (how many → COUNT, average → AVG, etc.)
3. Detects JOINs if multiple tables are referenced
4. Generates appropriate SQL with proper column selection
5. Returns a heuristic confidence score

This allows the frontend to work even without the trained model.

### Q25. How does the rule-based SQL generator handle JOINs?

**A:** `_detect_join()` detects multi-table references:
- Checks if the question mentions 2+ table names (including singular/plural variants)
- Looks for join keywords: "with", "along with", "and their", "including", "join"
- Detects foreign key relationships via `_id` column naming conventions
- Supports JOIN types: INNER, LEFT, RIGHT, FULL OUTER, CROSS (detected by regex)
- Falls back to generic JOIN if type is ambiguous

### Q26. What validations does the API perform?

**A:**
1. Question cannot be empty
2. Schema cannot be empty
3. Question must contain at least one word with 2+ letters (prevents gibberish like "a b c ?")
- Each returns 400 status with descriptive error message

### Q27. How does the model loading work at API startup?

**A:** Uses FastAPI's `lifespan` context manager:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    adapter_path = getattr(app.state, "adapter_path", None)
    if adapter_path:
        mdl, tok, dev = load_model(adapter_path)  # Loads base + LoRA adapter
        model = mdl
        tokenizer = tok
    else:
        print("Running in mock mode")
    yield  # App runs here
```
The `load_model()` function:
1. Reads `PeftConfig` from adapter path to find the base model name
2. Loads base model (`AutoModelForCausalLM`) in float16
3. Merges LoRA adapter via `PeftModel.from_pretrained()`
4. Sets model to eval mode

### Q28. How is CORS configured?

**A:**
```python
cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "")
if cors_origins:
    allow_origins = [origin.strip() for origin in cors_origins.split(",")]
elif os.getenv("ENV") == "production":
    allow_origins = []  # No CORS in production
else:
    allow_origins = ["*"]  # Allow all in development
```

---

## SECTION 7: INFERENCE DETAILS

### Q29. What are the inference parameters?

**A:**
| Parameter | Value | Purpose |
|-----------|-------|---------|
| max_new_tokens | 256 | Maximum SQL output length |
| temperature | 0.1 | Low = nearly deterministic output (good for SQL) |
| top_p | 0.95 | Nucleus sampling — consider top 95% probability mass |
| do_sample | True | Enable sampling (with low temp ≈ greedy) |
| repetition_penalty | 1.15 | Penalize repeated tokens (prevents SQL loops) |

### Q30. Why low temperature (0.1) for SQL generation?

**A:**
- SQL is a formal language with precise syntax — you want deterministic, correct output
- High temperature → more randomness → more syntax errors
- Temperature 0.1 ≈ near-greedy decoding while allowing slight flexibility
- Repetition penalty of 1.15 prevents degenerate repetitions without being too aggressive

### Q31. How is the SQL extracted from model output?

**A:**
```python
full_output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
if "[/INST]" in full_output:
    sql = full_output.split("[/INST]")[-1].strip()  # Take everything after [/INST]
else:
    sql = full_output[len(prompt):].strip()
sql = sql.split("\n")[0].strip()  # Take only first line
if not sql.endswith(";"):
    sql += ";"
```

---

## SECTION 8: SYSTEM DESIGN & ENGINEERING

### Q32. How is the codebase organized? Why this structure?

**A:** Modular design following separation of concerns:
```
configs/     → Configuration (YAML + dataclasses)
data/        → Data loading and preprocessing
models/      → Model loading and LoRA setup
training/    → Training loop and callbacks
evaluation/  → Metrics and evaluation
inference/   → Inference engine
utils/       → Shared utilities
scripts/     → CLI entry points
api/         → REST API
frontend/    → Web UI
```
Benefits: Each module is independently testable, swappable (e.g., different data pipeline), and follows single-responsibility principle.

### Q33. How does the config system work?

**A:** Two-layer design:
1. **YAML files** (`config_spider.yaml`): Human-readable configuration
2. **Python dataclasses** (`config_loader.py`): Typed, validated config objects

```python
@dataclass
class Config:
    project: ProjectConfig
    model: ModelConfig
    quantization: QuantizationConfig
    lora: LoraConfig
    data: DataConfig
    training: TrainingConfig
    inference: InferenceConfig
```
The `load_config()` function parses YAML → maps to dataclasses using `_dict_to_dataclass()` which filters unknown keys. This provides type safety and IDE autocomplete.

### Q34. How do you handle different hardware (MPS/CUDA/CPU)?

**A:**
```python
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```
The training script adapts:
- MPS: No BitsAndBytes quantization, fp16 only if CUDA
- CUDA: Full BitsAndBytes support, fp16 enabled
- CPU: Fallback, slowest

### Q35. How does `run.sh` work?

**A:** One-command bootstrap:
1. Creates Python venv if not exists
2. `pip install -r requirements.txt && pip install -e .`
3. `npm install` in frontend/
4. Starts API: `uvicorn api.serve:app --host 0.0.0.0 --port 8000`
5. Starts frontend: `NEXT_PUBLIC_API_URL=http://localhost:8000 npx next dev -p 3000`

### Q36. How do you ensure reproducibility?

**A:**
```python
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
```
Seed = 42 is set everywhere: data splits, model initialization, training.

---

## SECTION 9: DEEP ML THEORY QUESTIONS

### Q37. What is the difference between LoRA, QLoRA, and full fine-tuning?

**A:**
| Method | What's trained | Memory | Quality |
|--------|---------------|--------|---------|
| **Full fine-tune** | All parameters | Very high (full model in fp32) | Best possible |
| **LoRA** | Low-rank adapter matrices (frozen base) | Low (~0.1% params) | Near full fine-tune |
| **QLoRA** | LoRA + 4-bit quantized base | Very low (4-bit base + fp16 adapters) | Slightly lower than LoRA |

Our project uses LoRA (Spider) and supports QLoRA (WikiSQL config with `load_in_4bit: true`).

### Q38. What is NF4 quantization?

**A:** NormalFloat 4-bit (NF4) is an information-theoretically optimal data type for normally distributed weights:
- Assumes weights follow a normal distribution
- Quantization levels are placed to minimize expected quantization error
- Better than uniform 4-bit quantization
- Used with BitsAndBytes library
- `bnb_4bit_use_double_quant`: Quantizes the quantization constants themselves for further compression

### Q39. Explain the cosine learning rate scheduler.

**A:**
```
LR(t) = LR_min + 0.5 × (LR_max - LR_min) × (1 + cos(π × t / T))
```
- Starts at `LR_max` (2e-4), smoothly decays following a cosine curve to near 0
- With warmup_ratio=0.05: first 5% of steps linearly ramp up, then cosine decay
- **Advantage over step decay**: No sudden drops; smoother optimization landscape navigation
- **Advantage over linear decay**: Spends more time at moderate learning rates

### Q40. What is gradient accumulation? Why batch_size=1 with accumulation=16?

**A:**
- **Problem**: MPS has limited memory, can't fit batch_size=16 at once
- **Solution**: Process 1 sample at a time, accumulate gradients for 16 steps, then update weights
- **Effect**: Mathematically equivalent to batch_size=16 (approximately)
- `effective_batch_size = per_device_batch × gradient_accumulation_steps = 1 × 16 = 16`
- Trade-off: Slower per epoch (16x more forward passes) but fits in memory

### Q41. What is the cross-entropy loss for causal language modeling?

**A:**
```
L = -1/T Σ log P(token_t | token_1, ..., token_{t-1})
```
- For each token position, the model predicts a probability distribution over the vocabulary
- Loss is the negative log probability of the correct next token
- This is what SFTTrainer minimizes
- Loss of 0.40 means average token probability ≈ exp(-0.40) ≈ 67%

### Q42. What is the attention mechanism? Why does LoRA target attention?

**A:** Self-attention computes:
```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```
where Q = X × W_q, K = X × W_k, V = X × W_v

LoRA targets W_q and W_v because:
- These projections determine **what to attend to** (Q) and **what information to extract** (V)
- Adapting these changes the model's "focus" most efficiently
- The LoRA paper empirically showed Q+V gives the best rank-efficiency trade-off

### Q43. What is perplexity? How does it relate to your loss?

**A:**
```
Perplexity = exp(cross-entropy loss)
```
- Our eval loss = 0.43 → Perplexity ≈ exp(0.43) ≈ 1.54
- Interpretation: on average, the model is as confused as if choosing between ~1.54 equally likely tokens
- Very low perplexity = model is very confident and mostly correct
- For reference, a random token predictor over 32K vocab would have perplexity ~32,000

### Q44. What are potential failure modes of this approach?

**A:**
1. **Schema mismatch**: Model trained on Spider without explicit schema → may hallucinate column names not in the actual database
2. **Complex queries**: Nested subqueries, CTEs, window functions are rare in training data
3. **Distribution shift**: If user questions are very different from Spider's academic style
4. **Sequence length**: Max 256 tokens truncates long schemas or complex queries
5. **No execution validation**: Model doesn't verify if generated SQL actually runs
6. **Single-turn**: No query refinement or conversation context

### Q45. How would you improve this system for production?

**A:**
1. **Larger model**: CodeLlama-7B or LLaMA-3-8B for better SQL comprehension
2. **Schema-aware training**: Include schema in all training prompts
3. **Execution feedback**: Run SQL on actual DB, use results to verify/retry
4. **RAG**: Retrieve similar queries from a vector database
5. **Beam search**: Use beam search instead of sampling for more reliable outputs
6. **Security**: SQL injection prevention, sandboxed execution, auth on API
7. **Monitoring**: W&B for training metrics, Prometheus for API latency/errors
8. **Docker**: Containerize for reproducible deployment
9. **Rate limiting**: Prevent API abuse
10. **Caching**: Cache common queries to reduce inference cost

---

## SECTION 10: CODING & IMPLEMENTATION QUESTIONS

### Q46. Show me how the model generates SQL (code walkthrough).

**A:**
```python
# 1. Build prompt
prompt = f"[INST] Generate SQL...\nSchema:\n{schema}\nQuestion:\n{question} [/INST]\n"

# 2. Tokenize
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

# 3. Generate with scores for confidence
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.15,
        output_scores=True,           # Needed for confidence
        return_dict_in_generate=True,  # Returns scores + sequences
    )

# 4. Extract generated tokens (skip the prompt tokens)
generated_ids = outputs.sequences[0][inputs["input_ids"].shape[-1]:]

# 5. Compute confidence from per-token logits
confidence = confidence_from_scores(outputs.scores, generated_ids)

# 6. Decode and extract SQL
full_output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
sql = full_output.split("[/INST]")[-1].strip()
sql = sql.split("\n")[0].strip()  # First line only
```

### Q47. How does the data pipeline factory pattern work?

**A:**
```python
def get_pipeline(config):
    source = config.data.dataset_name
    if source == "sql-create-context":
        return SQLContextPipeline(config)
    if source == "spider-csv":
        return SpiderCSVPipeline(config)
    return WikiSQLPipeline(config)  # default
```
All pipelines implement the same interface:
- `prepare() → (train_ds, val_ds, test_ds)`
- `get_statistics(dataset) → dict`
- `format_example(example) → dict`

This is the **Strategy Pattern** — swap datasets without changing training code.

### Q48. How does early stopping work in this project?

**A:** Two callbacks work together:
1. **ExactMatchEvalCallback**: At each eval step, runs model.generate() on 30 validation samples, computes exact match accuracy, logs as `eval_exact_match`
2. **EarlyStoppingCallback** (HuggingFace): Monitors `eval_loss`, stops training if it doesn't improve by `threshold=0.01` for `patience=3` consecutive evaluations

This prevents overfitting — if the model starts memorizing training data without improving on validation, training stops.

### Q49. Explain the `ModelLoader` class design.

**A:** Two usage modes:
```python
loader = ModelLoader(config)

# For training: load base + apply LoRA
model, tokenizer = loader.load_for_training()
# Calls: load_tokenizer() → load_base_model() → apply_lora()
# apply_lora calls prepare_model_for_kbit_training() first

# For inference: load base + merge pre-trained LoRA adapter
model, tokenizer = loader.load_for_inference(adapter_path)
# Calls: load_tokenizer() → load_base_model() → PeftModel.from_pretrained()
```

### Q50. How does checkpoint resumption work?

**A:**
```python
if args.resume:
    ckpt_dir = output_dir / "checkpoints"
    checkpoints = []
    for p in ckpt_dir.glob("checkpoint-*"):
        match = re.match(r"^checkpoint-(\d+)$", p.name)
        if match:
            checkpoints.append((int(match.group(1)), p))
    checkpoints.sort(key=lambda x: x[0])
    resume_path = str(checkpoints[-1][1])  # Latest checkpoint

result = trainer.train(resume_from_checkpoint=resume_path)
```
Finds the highest-numbered checkpoint and resumes from it. HuggingFace Trainer restores optimizer state, scheduler, and RNG state.

---

## SECTION 11: FRONTEND & FULL-STACK QUESTIONS

### Q51. Describe the frontend architecture.

**A:**
- **Framework**: Next.js 14 with App Router
- **Pages**: Landing (`/`), Dashboard (`/dashboard`), Chat (`/chat`)
- **Key components**: SchemaEditor, QueryInput, SQLOutput (with confidence badge), QueryHistory
- **Custom hooks**: `useGenerateSQL` (API call), `useBackendStatus` (health check), `useQueryHistory` (local state)
- **API integration**: `lib/api.ts` — POST `/generate_sql`, GET `/health`
- **Confidence display**: Color-coded badge (green ≥ 80%, amber ≥ 50%, red < 50%)

### Q52. How does the frontend communicate with the backend?

**A:**
```typescript
// lib/api.ts
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export async function generateSQL(question: string, schema: string) {
    const response = await fetch(`${API_URL}/generate_sql`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, schema })
    });
    return response.json();  // { generated_sql, confidence, latency_ms }
}
```

---

## SECTION 12: BEHAVIORAL / DESIGN DECISION QUESTIONS

### Q53. What was the hardest challenge in this project?

**A:** Getting training to work on Apple MPS:
- BitsAndBytes doesn't support MPS → had to disable quantization
- Some PyTorch ops aren't MPS-compatible → needed float16 workarounds
- Memory constraints → gradient checkpointing + batch accumulation
- MPS doesn't fully support all CUDA kernels → careful dtype management

### Q54. If you had more time, what would you add?

**A:**
1. **Schema-grounded training**: Include full CREATE TABLE in all training prompts
2. **Multi-turn conversation**: Track conversation context for query refinement
3. **Execution validation**: Run generated SQL on the actual DB and report errors
4. **Larger model**: Try CodeLlama-7B or SQLCoder for better quality
5. **RAG pipeline**: Retrieve similar past queries to improve generation
6. **Docker + CI/CD**: Containerize and add automated testing
7. **A/B testing**: Compare model versions in production

### Q55. How would you scale this to handle 1000 requests/second?

**A:**
1. **Model optimization**: Use vLLM or TGI for batched inference with continuous batching
2. **Horizontal scaling**: Multiple API instances behind a load balancer
3. **GPU optimization**: TensorRT or ONNX Runtime for faster inference
4. **Caching**: Redis cache for repeated queries
5. **Async processing**: Queue-based architecture for non-real-time requests
6. **Model quantization**: GPTQ or AWQ for faster inference without quality loss
7. **Kubernetes**: Auto-scaling based on request volume

### Q56. How do you monitor model quality in production?

**A:**
1. **Confidence threshold alerting**: If average confidence drops below a threshold
2. **User feedback loop**: Thumbs up/down on generated SQL → retraining data
3. **Execution success rate**: Track what % of generated SQL actually executes
4. **Latency monitoring**: P50, P95, P99 latency tracking
5. **Drift detection**: Compare input distribution against training data
6. **Shadow evaluation**: Run new model versions in parallel, compare outputs

---

## SECTION 13: QUICK-FIRE REVIEW

### Key Numbers to Remember
| What | Value |
|------|-------|
| Base model | TinyLlama-1.1B-Chat-v1.0 |
| Dataset | Spider (8,035 examples) |
| LoRA rank | 8, alpha 16 |
| Target modules | q_proj, v_proj |
| Trainable params | 1.1M / 1,101M = 0.10% |
| Adapter size | 4.5 MB |
| Training loss | 0.40 |
| Eval loss | 0.43 |
| Token accuracy | 87.5% |
| Training time | ~2.5 hours (MPS) |
| LR | 2e-4, cosine schedule |
| Effective batch | 16 (1 × 16 accumulation) |
| Max seq length | 256 tokens |
| Confidence | 80% geo-mean + 20% min-token prob |
| API framework | FastAPI + Uvicorn |
| Frontend | Next.js 14 App Router |
| HuggingFace | Rj18/text-to-sql-tinyllama-lora |

### Key Libraries
| Library | Purpose |
|---------|---------|
| transformers | Model loading, tokenizer |
| peft | LoRA adapters |
| trl | SFTTrainer |
| bitsandbytes | 4-bit quantization (CUDA only) |
| datasets | HuggingFace datasets |
| fastapi | REST API |
| torch | PyTorch backend |

### One-Liner Explanations
- **LoRA**: Freeze base model, train small adapter matrices injected into attention layers
- **SFT**: Standard supervised fine-tuning with (prompt, completion) pairs
- **Gradient checkpointing**: Trade compute for memory by recomputing activations in backward pass
- **Gradient accumulation**: Simulate large batches by accumulating gradients over multiple small batches
- **NF4**: 4-bit quantization optimized for normally-distributed neural network weights
- **Cosine LR**: Learning rate decays following cos curve from max to ~0
- **Exact match**: Normalize + compare predicted SQL string vs gold
- **Execution accuracy**: Run both SQLs on SQLite, compare result sets
- **Confidence**: Per-token log-prob based scoring (geometric mean + min blend)
- **Early stopping**: Stop training when eval metric plateaus for N evaluations

---

*Good luck with your interview! You've built something solid. Focus on explaining WHY you made each decision, not just WHAT you built.*
