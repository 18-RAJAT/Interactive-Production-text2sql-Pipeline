import pytest
import torch
from unittest.mock import patch, MagicMock
from inference.engine import InferenceEngine


class TestInferenceEngine:
    @patch.object(InferenceEngine, "_load_model")
    def test_build_prompt_format(self, _mock_load):
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.config = MagicMock()

        schema = "CREATE TABLE t (id INTEGER, name TEXT)"
        question = "How many rows?"
        prompt = engine.build_prompt(question, schema)

        assert "[INST]" in prompt
        assert "[/INST]" in prompt
        assert schema in prompt
        assert question in prompt

    @patch.object(InferenceEngine, "_load_model")
    def test_prompt_contains_schema_and_question(self, _mock_load):
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.config = MagicMock()

        schema = "CREATE TABLE employees (id INTEGER, salary REAL)"
        question = "What is the max salary?"
        prompt = engine.build_prompt(question, schema)

        assert "employees" in prompt
        assert "max salary" in prompt

    @patch("inference.engine.ModelLoader")
    def test_init_calls_model_loader(self, mock_loader_cls):
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_loader.load_for_inference.return_value = (mock_model, mock_tokenizer)
        mock_loader_cls.return_value = mock_loader

        mock_config = MagicMock()
        InferenceEngine(mock_config, "/fake/path")

        mock_loader.load_for_inference.assert_called_once_with("/fake/path")

    @patch("inference.engine.ModelLoader")
    def test_generate_returns_dict(self, mock_loader_cls):
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model.parameters.return_value = iter([mock_param])

        prompt_len = 5
        gen_len = 3
        input_ids = torch.randint(0, 100, (1, prompt_len))
        gen_tokens = torch.randint(0, 100, (1, gen_len))
        full_seq = torch.cat([input_ids, gen_tokens], dim=-1)

        mock_inputs = {"input_ids": input_ids}
        mock_inputs_obj = MagicMock()
        mock_inputs_obj.to.return_value = mock_inputs
        mock_inputs_obj.__getitem__ = lambda _, key: mock_inputs[key]
        mock_tokenizer.return_value = mock_inputs_obj
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.decode.return_value = "blah [/INST]\nSELECT COUNT(*) FROM t;"

        vocab_size = 200
        mock_scores = tuple(torch.randn(1, vocab_size) for _ in range(gen_len))
        mock_output = MagicMock()
        mock_output.sequences = full_seq
        mock_output.scores = mock_scores
        mock_model.generate.return_value = mock_output

        mock_loader.load_for_inference.return_value = (mock_model, mock_tokenizer)
        mock_loader_cls.return_value = mock_loader

        mock_config = MagicMock()
        mock_config.inference.max_new_tokens = 256
        mock_config.inference.temperature = 0.1
        mock_config.inference.top_p = 0.95
        mock_config.inference.do_sample = True
        mock_config.inference.repetition_penalty = 1.15

        engine = InferenceEngine(mock_config, "/fake/path")
        result = engine.generate("How many?", "CREATE TABLE t (id INTEGER)")

        assert "generated_sql" in result
        assert "question" in result
        assert "schema" in result
        assert "latency_ms" in result
        assert "confidence" in result
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0