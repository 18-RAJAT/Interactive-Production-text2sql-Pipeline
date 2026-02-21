import pytest
from evaluation.evaluator import normalize_sql, classify_query


class TestNormalizeSQL:
    def test_lowercase(self):
        assert "select" in normalize_sql("SELECT * FROM table")

    def test_strip_whitespace(self):
        result = normalize_sql("  SELECT *  FROM  table  ")
        assert result == "select * from table"

    def test_strip_semicolon(self):
        result = normalize_sql("SELECT * FROM table;")
        assert not result.endswith(";")

    def test_collapse_spaces(self):
        result = normalize_sql("SELECT   *   FROM   table")
        assert "   " not in result

    def test_normalize_operators(self):
        result = normalize_sql("WHERE age>30")
        assert "> " in result or " > " in result

    def test_identical_after_normalize(self):
        q1 = "SELECT name FROM users WHERE age > 25"
        q2 = "select  name  from  users  where  age  >  25"
        assert normalize_sql(q1) == normalize_sql(q2)

    def test_empty_string(self):
        result = normalize_sql("")
        assert result == ""

    def test_only_semicolons(self):
        result = normalize_sql(";;;")
        assert ";" not in result


class TestClassifyQuery:
    def test_simple_select(self):
        assert classify_query("SELECT name FROM users") == "simple"

    def test_count_aggregation(self):
        assert classify_query("SELECT COUNT(*) FROM users") == "aggregation"

    def test_sum_aggregation(self):
        assert classify_query("SELECT SUM(salary) FROM employees") == "aggregation"

    def test_avg_aggregation(self):
        assert classify_query("SELECT AVG(age) FROM students") == "aggregation"

    def test_max_aggregation(self):
        assert classify_query("SELECT MAX(price) FROM products") == "aggregation"

    def test_min_aggregation(self):
        assert classify_query("SELECT MIN(score) FROM results") == "aggregation"

    def test_where_filter(self):
        assert classify_query("SELECT name FROM users WHERE age > 25") == "filter"

    def test_aggregation_with_where(self):
        result = classify_query("SELECT COUNT(*) FROM users WHERE active = 1")
        assert result == "aggregation"

    def test_case_insensitive(self):
        assert classify_query("select count(*) from t") == "aggregation"