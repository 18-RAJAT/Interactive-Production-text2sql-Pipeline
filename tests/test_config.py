import pytest
from pathlib import Path
from configs.config_loader import (
    load_config, Config, ProjectConfig, ModelConfig,
    QuantizationConfig, LoraConfig, DataConfig, TrainingConfig, InferenceConfig,
)


class TestConfigLoading:
    def test_load_default_config(self, config):
        assert isinstance(config, Config)

    def test_config_has_all_sections(self, config):
        assert isinstance(config.project, ProjectConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.quantization, QuantizationConfig)
        assert isinstance(config.lora, LoraConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.inference, InferenceConfig)

    def test_load_from_explicit_path(self):
        path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
        config = load_config(str(path))
        assert config.project.name == "text-to-sql-wikisql"

    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")


class TestProjectConfig:
    def test_project_name(self, config):
        assert config.project.name == "text-to-sql-wikisql"

    def test_seed_is_int(self, config):
        assert isinstance(config.project.seed, int)
        assert config.project.seed == 42

    def test_output_dir(self, config):
        assert config.project.output_dir == "./outputs"


class TestModelConfig:
    def test_model_name_not_empty(self, config):
        assert len(config.model.name) > 0

    def test_max_seq_length_positive(self, config):
        assert config.model.max_seq_length > 0

    def test_dtype_valid(self, config):
        assert config.model.dtype in ("float16", "bfloat16", "float32")


class TestQuantizationConfig:
    def test_4bit_enabled(self, config):
        assert config.quantization.load_in_4bit is True

    def test_quant_type(self, config):
        assert config.quantization.bnb_4bit_quant_type in ("nf4", "fp4")

    def test_double_quant(self, config):
        assert isinstance(config.quantization.bnb_4bit_use_double_quant, bool)


class TestLoraConfig:
    def test_rank_positive(self, config):
        assert config.lora.r > 0

    def test_alpha_gte_rank(self, config):
        assert config.lora.lora_alpha >= config.lora.r

    def test_dropout_in_range(self, config):
        assert 0.0 <= config.lora.lora_dropout < 1.0

    def test_target_modules_not_empty(self, config):
        assert len(config.lora.target_modules) > 0

    def test_expected_target_modules(self, config):
        expected = {"q_proj", "k_proj", "v_proj", "o_proj"}
        actual = set(config.lora.target_modules)
        assert expected.issubset(actual)

    def test_task_type(self, config):
        assert config.lora.task_type == "CAUSAL_LM"


class TestDataConfig:
    def test_dataset_name(self, config):
        assert config.data.dataset_name in ("wikisql", "sql-create-context")

    def test_val_split_ratio(self, config):
        assert 0.0 < config.data.val_split_ratio < 1.0


class TestTrainingConfig:
    def test_epochs_positive(self, config):
        assert config.training.num_epochs > 0

    def test_batch_size_positive(self, config):
        assert config.training.per_device_train_batch_size > 0

    def test_learning_rate_range(self, config):
        assert 1e-6 < config.training.learning_rate < 1e-2

    def test_warmup_ratio(self, config):
        assert 0.0 <= config.training.warmup_ratio < 1.0

    def test_effective_batch_size(self, config):
        effective = (
            config.training.per_device_train_batch_size
            * config.training.gradient_accumulation_steps
        )
        assert effective >= 4

    def test_scheduler_type(self, config):
        valid = ("linear", "cosine", "cosine_with_restarts", "polynomial", "constant")
        assert config.training.lr_scheduler_type in valid


class TestInferenceConfig:
    def test_temperature_range(self, config):
        assert 0.0 < config.inference.temperature <= 2.0

    def test_top_p_range(self, config):
        assert 0.0 < config.inference.top_p <= 1.0

    def test_max_new_tokens_positive(self, config):
        assert config.inference.max_new_tokens > 0

    def test_repetition_penalty(self, config):
        assert config.inference.repetition_penalty >= 1.0