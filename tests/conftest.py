import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_loader import load_config, Config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def sample_schema():
    return 'CREATE TABLE "employees" (\n  "id" INTEGER,\n  "name" TEXT,\n  "salary" REAL,\n  "dept" TEXT\n);'


@pytest.fixture
def sample_question():
    return "How many employees earn more than 50000?"


@pytest.fixture
def sample_sql():
    return "SELECT COUNT(*) FROM employees WHERE salary > 50000"


@pytest.fixture
def sample_wikisql_row(sample_schema, sample_question):
    return {
        "question": sample_question,
        "create_table_statement": sample_schema,
        "sql_query": "SELECT COUNT(*) FROM employees WHERE salary > 50000;",
        "wiki_sql_table_id": "test-table-1",
    }