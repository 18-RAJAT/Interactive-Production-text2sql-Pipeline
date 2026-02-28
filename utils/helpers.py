"""Shared utility functions: seeding, device info, formatting, timing, confidence."""

import math
import os
import time
import random

import torch
import torch.nn.functional as F
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_info() -> dict:
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2),
        }
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return {"device": "mps", "gpu_name": "Apple Silicon", "gpu_count": 1}
    return {"device": "cpu", "gpu_name": "N/A", "gpu_count": 0}


def format_number(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


class Timer:
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.name:
            print(f"[{self.name}] {self.elapsed:.2f}s")

    @property
    def elapsed_ms(self):
        return round(self.elapsed * 1000, 2) if self.elapsed else 0


def confidence_from_scores(scores: tuple, generated_ids: torch.Tensor) -> float:
    """Derive a 0-1 confidence from per-token logits via geometric mean probability.

    Blends 80% geometric-mean token probability with 20% minimum token
    probability, clamped to [0, 1].
    """
    log_probs = []
    for step_idx, logits in enumerate(scores):
        lp = F.log_softmax(logits[0].float(), dim=-1)
        token_id = generated_ids[step_idx]
        log_probs.append(lp[token_id].item())

    if not log_probs:
        return 0.0

    mean_lp = sum(log_probs) / len(log_probs)
    min_lp = min(log_probs)

    avg_conf = math.exp(mean_lp)
    min_conf = math.exp(min_lp)
    confidence = 0.8 * avg_conf + 0.2 * min_conf
    return round(max(0.0, min(1.0, confidence)), 4)