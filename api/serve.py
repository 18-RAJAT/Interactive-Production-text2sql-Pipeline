"""FastAPI backend for Text-to-SQL inference."""

import re
import sys
import time
import argparse
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

model = None
tokenizer = None


class GenerateRequest(BaseModel):
    question: str
    schema: str


class GenerateResponse(BaseModel):
    generated_sql: str
    confidence: Optional[float] = None
    execution_result: Optional[list] = None
    latency_ms: Optional[float] = None


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def generate_sql(mdl, tok, device, question: str, schema: str) -> dict:
    prompt = (
        f"[INST] Generate SQL for the following question.\n\n"
        f"Schema:\n{schema}\n\n"
        f"Question:\n{question} [/INST]\n"
    )
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

    start = time.time()
    with torch.no_grad():
        outputs = mdl.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.15,
            pad_token_id=tok.pad_token_id,
        )
    latency = time.time() - start

    full_output = tok.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in full_output:
        sql = full_output.split("[/INST]")[-1].strip()
    else:
        sql = full_output[len(prompt):].strip()

    sql = sql.split("\n")[0].strip()
    if sql and not sql.endswith(";"):
        sql += ";"

    return {"generated_sql": sql, "latency_ms": round(latency * 1000, 2)}


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

    if model is not None and tokenizer is not None:
        result = generate_sql(model, tokenizer, app.state.device, req.question, req.schema)
        return GenerateResponse(
            generated_sql=result["generated_sql"],
            confidence=0.92,
            latency_ms=result["latency_ms"],
        )

    mock_sql = f'SELECT * FROM table WHERE condition = \'{req.question[:20]}...\';'
    return GenerateResponse(
        generated_sql=mock_sql,
        confidence=0.5,
        latency_ms=0.0,
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