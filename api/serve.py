"""FastAPI backend for Text-to-SQL inference."""

import re
import sys
import time
import argparse
import os
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from utils.helpers import get_device

model = None
tokenizer = None


def _parse_schema(schema: str) -> list[tuple[str, list[str]]]:
    tables = []
    for match in re.finditer(
        r'CREATE\s+TABLE\s+(\w+)\s*\((.*?)\)', schema, re.IGNORECASE | re.DOTALL
    ):
        table_name = match.group(1)
        cols_raw = match.group(2)
        cols = []
        for line in cols_raw.split(","):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts and parts[0].upper() not in (
                "PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT", "INDEX"
            ):
                cols.append(parts[0])
        tables.append((table_name, cols))
    return tables


def _mock_confidence(question: str, schema: str, sql: str) -> float:
    """Heuristic confidence for rule-based mock mode (no model loaded).

    Scores higher when the schema has parseable tables and the question
    matches a clear pattern; scores lower for fallback/generic queries.
    """
    score = 0.5
    tables = _parse_schema(schema)
    if tables:
        score += 0.15
    q = question.lower()
    strong_patterns = [
        r'\bhow many\b', r'\baverage\b|\bavg\b', r'\bmax\b|\bmaximum\b',
        r'\bmin\b|\bminimum\b', r'\bsum\b|\btotal\b', r'\blist\b|\bshow\b|\bfind\b',
        r'\bhighest\b|\blowest\b', r'\bgroup\b|\beach\b|\bper\b',
    ]
    for pat in strong_patterns:
        if re.search(pat, q):
            score += 0.15
            break
    if tables and any(col.lower() in q for _, cols in tables for col in cols):
        score += 0.1
    if "SELECT" in sql and "FROM" in sql:
        score += 0.05
    return round(min(1.0, score), 4)


def _generate_smart_sql(question: str, schema: str) -> tuple[str, float, float]:
    start = time.time()
    q = question.lower()
    tables = _parse_schema(schema)

    if not tables:
        table_name = "data"
        columns = ["*"]
    else:
        table_name = tables[0][0]
        columns = tables[0][1]

    numeric_keywords = (
        "salary", "price", "amount", "cost", "age", "count", "num",
        "total", "quantity", "budget", "revenue", "score", "rating", "year",
    )
    numeric_cols = [c for c in columns if any(k in c.lower() for k in numeric_keywords)]
    id_cols = [c for c in columns if c.lower() == "id" or c.lower().endswith("_id")]
    text_cols = [c for c in columns if c not in numeric_cols and c not in id_cols]

    agg_col = numeric_cols[0] if numeric_cols else (columns[-1] if columns else "*")
    filter_col = text_cols[0] if text_cols else (columns[1] if len(columns) > 1 else columns[0])

    for col in columns:
        if col.lower() in q:
            if col in numeric_cols:
                agg_col = col
            elif col in text_cols:
                filter_col = col
            break

    is_ranked = re.search(r'\bnth\b|\brank\b|\btop\s+\d|\b\d+(?:st|nd|rd|th)\s+(?:highest|lowest|largest|smallest)', q)

    if is_ranked or (re.search(r'\bhighest\b|\blowest\b', q) and re.search(r'\bfind\b|\bget\b|\bwhat is\b', q)):
        is_asc = bool(re.search(r'\blowest\b|\bsmallest\b|\bleast\b', q))
        order = "ASC" if is_asc else "DESC"
        target_col = agg_col
        for col in numeric_cols:
            if col.lower() in q:
                target_col = col
                break
        n = 1
        n_match = re.search(r'(\d+)', q)
        if n_match:
            n = int(n_match.group(1))
        select_cols = ", ".join(columns[:4]) if len(columns) > 1 else "*"
        sql = f"SELECT {select_cols} FROM {table_name} ORDER BY {target_col} {order} LIMIT {n}"
    elif re.search(r'\bhow many\b', q):
        sql = f"SELECT COUNT(*) FROM {table_name}"
        if re.search(r'where|with|that|which|whose', q):
            sql += f" WHERE {filter_col} IS NOT NULL"
    elif re.search(r'\baverage\b|\bavg\b|\bmean\b', q):
        sql = f"SELECT AVG({agg_col}) FROM {table_name}"
    elif re.search(r'\bmaximum\b|\bmax\b', q):
        sql = f"SELECT MAX({agg_col}) FROM {table_name}"
    elif re.search(r'\bminimum\b|\bmin\b', q):
        sql = f"SELECT MIN({agg_col}) FROM {table_name}"
    elif re.search(r'\bsum\b|\btotal\b', q):
        sql = f"SELECT SUM({agg_col}) FROM {table_name}"
    elif re.search(r'\bgroup\b|\beach\b|\bper\b|\bby\s+\w+\b', q):
        group_col = filter_col
        sql = f"SELECT {group_col}, COUNT(*) FROM {table_name} GROUP BY {group_col}"
    elif re.search(r'\blist\b|\bshow\b|\bfind\b|\bget\b|\bwhat\b|\bname\b|\bdisplay\b', q):
        select_cols = ", ".join(columns[:4]) if len(columns) > 1 else "*"
        if re.search(r'where|with|that|which|whose|more than|less than|greater|equal|above|below', q):
            sql = f"SELECT {select_cols} FROM {table_name} WHERE {agg_col} IS NOT NULL"
            num_match = re.search(r'(?:more than|greater than|above|over|>)\s*(\d+)', q)
            if num_match:
                sql = f"SELECT {select_cols} FROM {table_name} WHERE {agg_col} > {num_match.group(1)}"
            else:
                num_match = re.search(r'(?:less than|below|under|<)\s*(\d+)', q)
                if num_match:
                    sql = f"SELECT {select_cols} FROM {table_name} WHERE {agg_col} < {num_match.group(1)}"
        else:
            sql = f"SELECT {select_cols} FROM {table_name}"
            if re.search(r'\border\b|\bsort\b', q):
                sql += f" ORDER BY {agg_col} DESC"
    elif len(tables) > 1:
        t1, c1 = tables[0]
        t2, c2 = tables[1]
        join_col = None
        for c in c2:
            if c.lower().endswith("_id") and c.lower().replace("_id", "") == t1.lower().rstrip("s"):
                join_col = c
                break
        if not join_col:
            join_col = c2[0] if c2 else "id"
        select_cols = ", ".join(f"{t1}.{c}" for c in c1[:2])
        sql = f"SELECT {select_cols} FROM {t1} JOIN {t2} ON {t1}.id = {t2}.{join_col}"
    else:
        select_cols = ", ".join(columns[:4]) if len(columns) > 1 else "*"
        sql = f"SELECT {select_cols} FROM {table_name}"

    if not sql.rstrip().endswith(";"):
        sql += ";"

    latency = round((time.time() - start) * 1000, 2)
    confidence = _mock_confidence(question, schema, sql)
    return sql, latency, confidence


class GenerateRequest(BaseModel):
    question: str
    schema: str


class GenerateResponse(BaseModel):
    generated_sql: str
    confidence: Optional[float] = None
    execution_result: Optional[list] = None
    latency_ms: Optional[float] = None


def load_model(adapter_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = get_device()
    print(f"Loading model on {device}...")

    tok = AutoTokenizer.from_pretrained(adapter_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    from peft.config import PeftConfig
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_name = peft_config.base_model_name_or_path

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, adapter_path)
    merged.eval()
    print(f"Model loaded: {base_model_name} + LoRA adapter from {adapter_path}")
    return merged, tok, device


def _confidence_from_scores(
    scores: tuple, generated_ids: torch.Tensor
) -> float:
    """Derive a 0-1 confidence score from per-token logits.

    Uses the geometric mean of token probabilities (exp of mean log-prob)
    as the primary signal, with min token probability as a floor clamp.
    """
    log_probs = []
    for step_idx, logits in enumerate(scores):
        probs = F.log_softmax(logits[0].float(), dim=-1)
        token_id = generated_ids[step_idx]
        log_probs.append(probs[token_id].item())

    if not log_probs:
        return 0.0

    mean_lp = sum(log_probs) / len(log_probs)
    min_lp = min(log_probs)

    avg_conf = math.exp(mean_lp)
    min_conf = math.exp(min_lp)
    confidence = 0.8 * avg_conf + 0.2 * min_conf
    return round(max(0.0, min(1.0, confidence)), 4)



def generate_sql(mdl, tok, device, question: str, schema: str, inference_params: dict | None = None) -> dict:
    prompt = (
        f"[INST] Generate SQL for the following question.\n\n"
        f"Schema:\n{schema}\n\n"
        f"Question:\n{question} [/INST]\n"
    )
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

    params = inference_params or {}
    max_new_tokens = params.get("max_new_tokens", 256)
    temperature = params.get("temperature", 0.1)
    top_p = params.get("top_p", 0.95)
    do_sample = params.get("do_sample", True)
    repetition_penalty = params.get("repetition_penalty", 1.15)

    start = time.time()
    with torch.no_grad():
        outputs = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=tok.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
    latency = time.time() - start

    generated_ids = outputs.sequences[0][inputs["input_ids"].shape[-1]:]
    confidence = _confidence_from_scores(outputs.scores, generated_ids)

    full_output = tok.decode(outputs.sequences[0], skip_special_tokens=True)
    if "[/INST]" in full_output:
        sql = full_output.split("[/INST]")[-1].strip()
    else:
        sql = full_output[len(prompt):].strip()

    sql = sql.split("\n")[0].strip()
    if sql and not sql.endswith(";"):
        sql += ";"

    return {
        "generated_sql": sql,
        "confidence": confidence,
        "latency_ms": round(latency * 1000, 2),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    adapter_path = getattr(app.state, "adapter_path", None)
    if adapter_path:
        mdl, tok, dev = load_model(adapter_path)
        model = mdl
        tokenizer = tok
        app.state.device = dev
        print("Ready for inference.")
    else:
        print("Running in mock mode (no --adapter_path). Use --adapter_path for real inference.")
    yield


app = FastAPI(title="Text-to-SQL API", version="1.0.0", lifespan=lifespan)

cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "")
if cors_origins:
    allow_origins = [origin.strip() for origin in cors_origins.split(",")]
elif os.getenv("ENV", "development").lower() == "production":
    allow_origins = []
else:
    allow_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/generate_sql", response_model=GenerateResponse)
async def handle_generate_sql(req: GenerateRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if not req.schema.strip():
        raise HTTPException(status_code=400, detail="Schema cannot be empty")

    inference_params = getattr(app.state, "inference_params", None)

    if model is not None and tokenizer is not None:
        result = generate_sql(model, tokenizer, app.state.device, req.question, req.schema, inference_params)
        return GenerateResponse(
            generated_sql=result["generated_sql"],
            confidence=result["confidence"],
            latency_ms=result["latency_ms"],
        )

    sql, latency, confidence = _generate_smart_sql(req.question, req.schema)
    return GenerateResponse(
        generated_sql=sql,
        confidence=confidence,
        latency_ms=latency,
    )


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app.state.adapter_path = args.adapter_path
    app.state.config_path = args.config

    if args.config:
        from configs.config_loader import load_config
        config = load_config(args.config)
        app.state.inference_params = config.inference
    else:
        app.state.inference_params = None

    uvicorn.run(app, host=args.host, port=args.port)