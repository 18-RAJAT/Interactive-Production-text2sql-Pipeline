"""Data pipeline: download, format, prompt construction, and splitting."""

from datasets import load_dataset, Dataset, concatenate_datasets
from typing import Dict, Tuple

AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "OP"]

WIKISQL_HUB_ID = "kaxap/pg-wikiSQL-sql-instructions-80k"
SQL_CONTEXT_HUB_ID = "b-mc2/sql-create-context"


class WikiSQLPipeline:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def load_raw(self) -> Dict[str, Dataset]:
        ds = load_dataset(WIKISQL_HUB_ID)
        return {"train": ds["train"], "validation": ds["validation"], "test": ds["test"]}

    def format_wikisql(self, example: dict) -> dict:
        schema = example["create_table_statement"].strip()
        sql = example["sql_query"].strip()
        question = example["question"].strip()

        prompt = (
            f"[INST] Generate SQL for the following question.\n\n"
            f"Schema:\n{schema}\n\n"
            f"Question:\n{question} [/INST]\n"
        )

        return {"text": f"{prompt}{sql}", "prompt": prompt, "sql": sql, "schema": schema}

    def process_split(self, dataset: Dataset, max_samples: int = None) -> Dataset:
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        cols_to_remove = [c for c in dataset.column_names if c not in {"text", "prompt", "sql", "schema"}]
        return dataset.map(self.format_wikisql, remove_columns=cols_to_remove)

    def prepare(self) -> Tuple[Dataset, Dataset, Dataset]:
        raw = self.load_raw()

        train_ds = self.process_split(raw["train"], self.config.data.max_train_samples)
        val_ds = self.process_split(raw["validation"], self.config.data.max_val_samples)
        test_ds = self.process_split(raw["test"], self.config.data.max_val_samples)

        return train_ds, val_ds, test_ds

    def get_statistics(self, dataset: Dataset) -> dict:
        lengths = [len(t) for t in dataset["text"]]
        return {
            "num_examples": len(dataset),
            "avg_text_length": sum(lengths) / len(lengths),
            "max_text_length": max(lengths),
            "min_text_length": min(lengths),
        }


class SQLContextPipeline:
    """Alternative pipeline using b-mc2/sql-create-context (78K examples, train only)."""

    def __init__(self, config):
        self.config = config

    def load_raw(self) -> Dataset:
        return load_dataset(SQL_CONTEXT_HUB_ID, split="train")

    def format_example(self, example: dict) -> dict:
        schema = example["context"].strip()
        sql = example["answer"].strip()
        question = example["question"].strip()

        prompt = (
            f"[INST] Generate SQL for the following question.\n\n"
            f"Schema:\n{schema}\n\n"
            f"Question:\n{question} [/INST]\n"
        )

        return {"text": f"{prompt}{sql}", "prompt": prompt, "sql": sql, "schema": schema}

    def prepare(self) -> Tuple[Dataset, Dataset, Dataset]:
        raw = self.load_raw()

        if self.config.data.max_train_samples:
            raw = raw.select(range(min(self.config.data.max_train_samples, len(raw))))

        raw = raw.map(self.format_example, remove_columns=raw.column_names)

        split = raw.train_test_split(test_size=self.config.data.val_split_ratio, seed=42)
        train_ds = split["train"]

        val_test = split["test"].train_test_split(test_size=0.5, seed=42)
        val_ds = val_test["train"]
        test_ds = val_test["test"]

        return train_ds, val_ds, test_ds

    def get_statistics(self, dataset: Dataset) -> dict:
        lengths = [len(t) for t in dataset["text"]]
        return {
            "num_examples": len(dataset),
            "avg_text_length": sum(lengths) / len(lengths),
            "max_text_length": max(lengths),
            "min_text_length": min(lengths),
        }


class SpiderCSVPipeline:
    """Pipeline for local Spider CSV with columns: text_query, sql_command."""

    def __init__(self, config, csv_path="data/spider_text_sql.csv"):
        self.config = config
        self.csv_path = csv_path

    def format_example(self, example: dict) -> dict:
        question = example["text_query"].strip()
        sql = example["sql_command"].strip()

        prompt = (
            f"[INST] Generate SQL for the following question.\n\n"
            f"Question:\n{question} [/INST]\n"
        )

        return {"text": f"{prompt}{sql}", "prompt": prompt, "sql": sql, "schema": ""}

    def prepare(self) -> Tuple[Dataset, Dataset, Dataset]:
        import csv
        rows = []
        with open(self.csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("text_query") and row.get("sql_command"):
                    rows.append(row)

        raw = Dataset.from_list(rows)

        if self.config.data.max_train_samples:
            raw = raw.select(range(min(self.config.data.max_train_samples, len(raw))))

        raw = raw.map(self.format_example, remove_columns=raw.column_names)

        split = raw.train_test_split(test_size=self.config.data.val_split_ratio, seed=42)
        train_ds = split["train"]

        val_test = split["test"].train_test_split(test_size=0.5, seed=42)
        val_ds = val_test["train"]
        test_ds = val_test["test"]

        return train_ds, val_ds, test_ds

    def get_statistics(self, dataset: Dataset) -> dict:
        lengths = [len(t) for t in dataset["text"]]
        return {
            "num_examples": len(dataset),
            "avg_text_length": sum(lengths) / len(lengths),
            "max_text_length": max(lengths),
            "min_text_length": min(lengths),
        }


def get_pipeline(config):
    source = getattr(config.data, "dataset_name", "wikisql")
    if source == "sql-create-context":
        return SQLContextPipeline(config)
    if source == "spider-csv":
        return SpiderCSVPipeline(config)
    return WikiSQLPipeline(config)