from pathlib import Path

from fastapi.testclient import TestClient

from sentiment_mlops.api.app import create_app
from sentiment_mlops.config import FRONTEND_DIR, Settings


def make_client(artifact_dir: Path, **overrides: object) -> TestClient:
    return TestClient(create_app(Settings(artifact_dir=artifact_dir, **overrides), FRONTEND_DIR))


def test_predict_positive_and_negative(trained_artifacts: Path) -> None:
    client = make_client(trained_artifacts)
    positive = client.post("/predict", json={"text": "great and wonderful service"}).json()
    negative = client.post("/predict", json={"text": "terrible and awful product"}).json()
    assert positive["predicted_class"] == "positive"
    assert positive["positive_probability"] > 0.5
    assert "great" in {item["term"] for item in positive["positive_terms"]}
    assert negative["predicted_class"] == "negative"
    assert "terrible" in {item["term"] for item in negative["negative_terms"]}


def test_health_ready_model_and_metrics(trained_artifacts: Path) -> None:
    client = make_client(trained_artifacts)
    assert client.get("/health").json()["model_loaded"] is True
    assert client.get("/ready").status_code == 200
    assert client.get("/model").json()["metrics"]["macro_f1"] > 0.5
    client.post("/predict", json={"text": "good"})
    body = client.get("/metrics").text
    assert 'sentiment_predictions_total{predicted_class="positive"}' in body
    assert "sentiment_model_eval_macro_f1" in body


def test_input_validation(trained_artifacts: Path) -> None:
    client = make_client(trained_artifacts, max_text_length=10)
    assert client.post("/predict", json={"text": "   "}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"text": "x" * 11}).status_code == 413


def test_missing_model_is_degraded(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    assert client.get("/health").json()["status"] == "degraded"
    assert client.get("/ready").status_code == 503
    assert client.get("/model").status_code == 404
    assert client.post("/predict", json={"text": "good"}).status_code == 503


def test_frontend_is_served(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    assert "Sentiment" in client.get("/").text
    assert client.get("/script.js").status_code == 200
