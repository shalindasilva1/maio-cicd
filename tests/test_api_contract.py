"""API contract tests (requires baked artifacts present in app/model)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.app import app

client = TestClient(app)


def test_health_ok():
    """Health is 200 OK."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_predict_ok():
    """Predict returns a float when payload is valid."""
    payload = {
        "age": 0.02,
        "sex": -0.044,
        "bmi": 0.06,
        "bp": -0.03,
        "s1": -0.02,
        "s2": 0.03,
        "s3": -0.02,
        "s4": 0.02,
        "s5": 0.02,
        "s6": -0.001,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert "prediction" in body and isinstance(body["prediction"], float)


def test_bad_input_returns_json_error():
    """Invalid payload produces JSON 400 with details."""
    res = client.post("/predict", json={"age": "oops"})
    assert res.status_code == 400
    assert "detail" in res.json()
