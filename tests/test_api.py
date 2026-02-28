import pytest
from fastapi.testclient import TestClient
from api.serve import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_model_loaded_field(self, client):
        data = client.get("/health").json()
        assert "model_loaded" in data


class TestGenerateSQLEndpoint:
    def test_valid_request(self, client):
        resp = client.post("/generate_sql", json={
            "question": "How many rows?",
            "schema": "CREATE TABLE t (id INTEGER)",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "generated_sql" in data

    def test_mock_mode_returns_sql(self, client):
        data = client.post("/generate_sql", json={
            "question": "Count all records",
            "schema": "CREATE TABLE t (id INTEGER)",
        }).json()
        assert len(data["generated_sql"]) > 0

    def test_response_has_confidence(self, client):
        data = client.post("/generate_sql", json={
            "question": "test",
            "schema": "CREATE TABLE t (id INTEGER)",
        }).json()
        assert "confidence" in data
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0

    def test_confidence_varies_by_pattern_strength(self, client):
        strong = client.post("/generate_sql", json={
            "question": "How many rows are there?",
            "schema": "CREATE TABLE employees (id INTEGER, name TEXT, salary REAL)",
        }).json()
        weak = client.post("/generate_sql", json={
            "question": "xyz",
            "schema": "CREATE TABLE t (id INTEGER)",
        }).json()
        assert strong["confidence"] > weak["confidence"]

    def test_response_has_latency(self, client):
        data = client.post("/generate_sql", json={
            "question": "test",
            "schema": "CREATE TABLE t (id INTEGER)",
        }).json()
        assert "latency_ms" in data

    def test_empty_question_returns_400(self, client):
        resp = client.post("/generate_sql", json={
            "question": "",
            "schema": "CREATE TABLE t (id INTEGER)",
        })
        assert resp.status_code == 400

    def test_empty_schema_returns_400(self, client):
        resp = client.post("/generate_sql", json={
            "question": "How many?",
            "schema": "",
        })
        assert resp.status_code == 400

    def test_whitespace_question_returns_400(self, client):
        resp = client.post("/generate_sql", json={
            "question": "   ",
            "schema": "CREATE TABLE t (id INTEGER)",
        })
        assert resp.status_code == 400

    def test_punctuation_only_question_returns_400(self, client):
        for q in [".", "-", "!!!", "...", "---", "?", ",,", "@#$"]:
            resp = client.post("/generate_sql", json={
                "question": q,
                "schema": "CREATE TABLE t (id INTEGER)",
            })
            assert resp.status_code == 400, f"Expected 400 for question={q!r}, got {resp.status_code}"

    def test_single_char_question_returns_400(self, client):
        resp = client.post("/generate_sql", json={
            "question": "x",
            "schema": "CREATE TABLE t (id INTEGER)",
        })
        assert resp.status_code == 400

    def test_missing_question_returns_422(self, client):
        resp = client.post("/generate_sql", json={
            "schema": "CREATE TABLE t (id INTEGER)",
        })
        assert resp.status_code == 422

    def test_missing_schema_returns_422(self, client):
        resp = client.post("/generate_sql", json={
            "question": "How many?",
        })
        assert resp.status_code == 422

    def test_empty_body_returns_422(self, client):
        resp = client.post("/generate_sql", json={})
        assert resp.status_code == 422