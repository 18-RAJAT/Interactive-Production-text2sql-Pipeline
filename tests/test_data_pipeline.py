import pytest
from data.pipeline import WikiSQLPipeline, SQLContextPipeline, get_pipeline


class TestWikiSQLPipeline:
    def test_init(self, config):
        pipeline = WikiSQLPipeline(config)
        assert pipeline.config is config

    def test_format_wikisql(self, config, sample_wikisql_row):
        pipeline = WikiSQLPipeline(config)
        result = pipeline.format_wikisql(sample_wikisql_row)

        assert "text" in result
        assert "prompt" in result
        assert "sql" in result
        assert "schema" in result

    def test_prompt_contains_schema(self, config, sample_wikisql_row):
        pipeline = WikiSQLPipeline(config)
        result = pipeline.format_wikisql(sample_wikisql_row)

        assert "CREATE TABLE" in result["prompt"]
        assert "employees" in result["prompt"]

    def test_prompt_contains_question(self, config, sample_wikisql_row):
        pipeline = WikiSQLPipeline(config)
        result = pipeline.format_wikisql(sample_wikisql_row)

        assert sample_wikisql_row["question"] in result["prompt"]

    def test_prompt_has_inst_tags(self, config, sample_wikisql_row):
        pipeline = WikiSQLPipeline(config)
        result = pipeline.format_wikisql(sample_wikisql_row)

        assert "[INST]" in result["prompt"]
        assert "[/INST]" in result["prompt"]

    def test_text_is_prompt_plus_sql(self, config, sample_wikisql_row):
        pipeline = WikiSQLPipeline(config)
        result = pipeline.format_wikisql(sample_wikisql_row)

        assert result["text"] == result["prompt"] + result["sql"]

    def test_sql_extracted_correctly(self, config, sample_wikisql_row):
        pipeline = WikiSQLPipeline(config)
        result = pipeline.format_wikisql(sample_wikisql_row)

        assert result["sql"] == sample_wikisql_row["sql_query"].strip()

    @pytest.mark.slow
    def test_load_raw(self, config):
        pipeline = WikiSQLPipeline(config)
        raw = pipeline.load_raw()

        assert "train" in raw
        assert "validation" in raw
        assert "test" in raw
        assert len(raw["train"]) > 0

    @pytest.mark.slow
    def test_prepare_with_sample_limit(self, config):
        config.data.max_train_samples = 20
        config.data.max_val_samples = 5
        pipeline = WikiSQLPipeline(config)
        train, val, *_ = pipeline.prepare()

        assert len(train) == 20
        assert len(val) == 5
        assert "text" in train.column_names

    @pytest.mark.slow
    def test_get_statistics(self, config):
        config.data.max_train_samples = 10
        pipeline = WikiSQLPipeline(config)
        train, _, _ = pipeline.prepare()
        stats = pipeline.get_statistics(train)

        assert stats["num_examples"] == 10
        assert stats["avg_text_length"] > 0
        assert stats["max_text_length"] >= stats["min_text_length"]


class TestGetPipeline:
    def test_default_returns_wikisql(self, config):
        pipeline = get_pipeline(config)
        assert isinstance(pipeline, WikiSQLPipeline)

    def test_sql_context_returns_correct_type(self, config):
        config.data.dataset_name = "sql-create-context"
        pipeline = get_pipeline(config)
        assert isinstance(pipeline, SQLContextPipeline)