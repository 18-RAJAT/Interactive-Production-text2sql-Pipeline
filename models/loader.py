"""Model loading with QLoRA quantization and LoRA adapter configuration."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel


class ModelLoader:
    def __init__(self, config):
        self.config = config

    def _get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(dtype_str, torch.bfloat16)

    def _build_bnb_config(self) -> BitsAndBytesConfig:
        qcfg = self.config.quantization
        return BitsAndBytesConfig(
            load_in_4bit=qcfg.load_in_4bit,
            bnb_4bit_quant_type=qcfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self._get_torch_dtype(qcfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=qcfg.bnb_4bit_use_double_quant,
        )

    def _build_lora_config(self) -> LoraConfig:
        lcfg = self.config.lora
        return LoraConfig(
            r=lcfg.r,
            lora_alpha=lcfg.lora_alpha,
            lora_dropout=lcfg.lora_dropout,
            bias=lcfg.bias,
            task_type=lcfg.task_type,
            target_modules=lcfg.target_modules,
        )

    def load_tokenizer(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def load_base_model(self) -> AutoModelForCausalLM:
        bnb_config = self._build_bnb_config()
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=self.config.model.trust_remote_code,
            torch_dtype=self._get_torch_dtype(self.config.model.dtype),
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        return model

    def apply_lora(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        model = prepare_model_for_kbit_training(model)
        lora_config = self._build_lora_config()
        model = get_peft_model(model, lora_config)
        return model

    def load_for_training(self):
        tokenizer = self.load_tokenizer()
        model = self.load_base_model()
        model = self.apply_lora(model)
        trainable, total = model.get_nb_trainable_parameters()
        print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
        return model, tokenizer

    def load_for_inference(self, adapter_path: str):
        tokenizer = self.load_tokenizer()
        base_model = self.load_base_model()
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        return model, tokenizer

    def get_model_info(self, model) -> dict:
        trainable, total = model.get_nb_trainable_parameters()
        return {
            "model_name": self.config.model.name,
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": f"{100 * trainable / total:.2f}%",
            "lora_rank": self.config.lora.r,
            "lora_alpha": self.config.lora.lora_alpha,
        }