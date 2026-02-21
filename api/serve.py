"""FastAPI backend for Text-to-SQL inference."""

import sys
import argparse
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

engine = None


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

    # Mock mode for frontend development
    mock_sql = f'SELECT * FROM table WHERE condition = \'{req.question[:20]}...\';'
    return GenerateResponse(
        generated_sql=mock_sql,
        confidence=0.85,
        latency_ms=42.0,
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