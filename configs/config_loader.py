import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ProjectConfig:
    name: str = "text-to-sql-wikisql"
    seed: int = 42
    output_dir: str = "./outputs"
    logging_dir: str = "./outputs/logs"


@dataclass
class ModelConfig:
    name: str = "mistralai/Mistral-7B-v0.1"
    max_seq_length: int = 1024
    dtype: str = "bfloat16"
    trust_remote_code: bool = True


@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class DataConfig:
    dataset_name: str = "wikisql"
    val_split_ratio: float = 0.1
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    num_workers: int = 4


@dataclass
class TrainingConfig:
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 25
    eval_steps: int = 250
    save_steps: int = 250
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_exact_match"
    greater_is_better: bool = True
    report_to: str = "none"
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    early_stopping_eval_samples: int = 50


@dataclass
class InferenceConfig:
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.95
    do_sample: bool = True
    repetition_penalty: float = 1.15


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def _dict_to_dataclass(dc_class, d: dict):
    filtered = {k: v for k, v in d.items() if k in dc_class.__dataclass_fields__}
    return dc_class(**filtered)


def load_config(path: str = None) -> Config:
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    else:
        path = Path(path)

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return Config(
        project=_dict_to_dataclass(ProjectConfig, raw.get("project", {})),
        model=_dict_to_dataclass(ModelConfig, raw.get("model", {})),
        quantization=_dict_to_dataclass(QuantizationConfig, raw.get("quantization", {})),
        lora=_dict_to_dataclass(LoraConfig, raw.get("lora", {})),
        data=_dict_to_dataclass(DataConfig, raw.get("data", {})),
        training=_dict_to_dataclass(TrainingConfig, raw.get("training", {})),
        inference=_dict_to_dataclass(InferenceConfig, raw.get("inference", {})),
    )


_global_config: Optional[Config] = None


def get_config(path: str = None) -> Config:
    global _global_config
    if _global_config is None:
        _global_config = load_config(path)
    return _global_config