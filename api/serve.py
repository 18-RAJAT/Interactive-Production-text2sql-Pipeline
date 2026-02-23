"""FastAPI backend for Text-to-SQL inference."""

import re
import sys
import time
import argparse
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Tuple

engine = None


def _parse_schema(schema: str) -> List[Tuple[str, List[str]]]:
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


def _generate_smart_sql(question: str, schema: str) -> Tuple[str, float]:
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
    return sql, latency


class GenerateRequest(BaseModel):
    question: str
    schema: str


class GenerateResponse(BaseModel):
    generated_sql: str
    confidence: Optional[float] = None
    execution_result: Optional[list] = None
    latency_ms: Optional[float] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # Model loading happens at startup
    # If --adapter_path is provided, load the real model
    # Otherwise run in mock mode for frontend development
    if hasattr(app.state, "adapter_path") and app.state.adapter_path:
        from configs.config_loader import load_config
        from inference.engine import InferenceEngine
        config = load_config(app.state.config_path)
        engine = InferenceEngine(config, app.state.adapter_path)
        print("Model loaded. Ready for inference.")
    else:
        print("Running in mock mode (no adapter_path). Use --adapter_path for real inference.")
    yield


app = FastAPI(title="Text-to-SQL API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": engine is not None}


@app.post("/generate_sql", response_model=GenerateResponse)
async def generate_sql(req: GenerateRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if not req.schema.strip():
        raise HTTPException(status_code=400, detail="Schema cannot be empty")

    if engine is not None:
        result = engine.generate(req.question, req.schema)
        return GenerateResponse(
            generated_sql=result["generated_sql"],
            latency_ms=result.get("latency_ms"),
        )

    sql, latency = _generate_smart_sql(req.question, req.schema)
    return GenerateResponse(
        generated_sql=sql,
        confidence=0.78,
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

    uvicorn.run(app, host=args.host, port=args.port)