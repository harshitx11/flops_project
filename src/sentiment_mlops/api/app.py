"""FastAPI application for review sentiment inference."""

import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field, field_validator

from sentiment_mlops import __version__
from sentiment_mlops.config import FRONTEND_DIR, Settings, settings
from sentiment_mlops.models.predict import ModelService

EVALUATION_METRICS = ("accuracy", "precision_positive", "recall_positive", "f1_positive", "f1_negative", "macro_f1")


class PredictionRequest(BaseModel):
    """Validated review text."""

    text: str = Field(..., min_length=1)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


def create_app(
    runtime_settings: Settings = settings, frontend_dir: Path = FRONTEND_DIR
) -> FastAPI:
    """Create an isolated app and metric registry for production or tests."""

    registry = CollectorRegistry(auto_describe=True)
    requests = Counter(
        "sentiment_http_requests_total", "HTTP requests", ["endpoint", "status"], registry=registry
    )
    prediction_counts = Counter(
        "sentiment_predictions_total", "Predictions by class", ["predicted_class"], registry=registry
    )
    prediction_failures = Counter(
        "sentiment_prediction_failures_total", "Prediction failures", registry=registry
    )
    confidence = Histogram(
        "sentiment_prediction_confidence",
        "Probability of the predicted class",
        buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
        registry=registry,
    )
    latency = Histogram(
        "sentiment_http_request_duration_seconds", "HTTP request latency", ["endpoint"], registry=registry
    )
    model_status = Gauge("sentiment_model_loaded", "Whether the model is loaded", registry=registry)
    model_info = Gauge("sentiment_model_info", "Model metadata", ["version"], registry=registry)
    evaluation_gauges = {
        name: Gauge(f"sentiment_model_eval_{name}", f"Latest offline evaluation {name}", registry=registry)
        for name in EVALUATION_METRICS
    }
    # Artifact metadata changes only when a candidate is accepted, keeping a
    # rejected evaluation from replacing current-model quality metrics.
    metadata: dict[str, Any] = {}
    metadata_path = runtime_settings.artifact_dir / "model_metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("accepted") is True:
                for name, gauge in evaluation_gauges.items():
                    if name in metadata.get("metrics", {}):
                        gauge.set(float(metadata["metrics"][name]))
        except (OSError, ValueError, TypeError):
            metadata = {}
    service = ModelService(runtime_settings.resolved_model_path(), runtime_settings.model_version)
    model_status.set(1 if service.loaded else 0)
    model_info.labels(version=runtime_settings.model_version).set(1)

    app = FastAPI(title="Review Sentiment API", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.cors_origin_list(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.state.model_service = service

    @app.middleware("http")
    async def record_http_metrics(request: Request, call_next: Any) -> Any:
        started = time.perf_counter()
        response = await call_next(request)
        endpoint = (
            request.url.path
            if request.url.path in {"/", "/health", "/ready", "/predict", "/model", "/metrics"}
            else "other"
        )
        requests.labels(endpoint=endpoint, status=str(response.status_code)).inc()
        latency.labels(endpoint=endpoint).observe(time.perf_counter() - started)
        return response

    @app.get("/", include_in_schema=False)
    def root() -> FileResponse:
        return FileResponse(frontend_dir / "index.html")

    @app.get("/api")
    def api_info() -> dict[str, str]:
        return {"service": "review-sentiment", "status": "ok", "docs": "/docs", "health": "/health"}

    @app.get("/health")
    def health() -> dict[str, object]:
        """Report process health and whether an accepted model is available."""
        return {
            "status": "ok" if service.loaded else "degraded",
            "model_loaded": service.loaded,
            "model_version": service.model_version,
        }

    @app.get("/ready")
    def readiness() -> dict[str, object]:
        """Report inference readiness; return 503 until the model is loaded."""
        if not service.loaded:
            raise HTTPException(status_code=503, detail="The model is unavailable")
        return {"ready": True, "model_version": service.model_version}

    @app.get("/model")
    def model_card() -> dict[str, object]:
        """Return the accepted model's offline evaluation summary."""
        if not metadata.get("accepted"):
            raise HTTPException(status_code=404, detail="No accepted model metadata")
        return {
            "model_version": metadata.get("model_version"),
            "trained_at": metadata.get("trained_at"),
            "model_params": metadata.get("model_params"),
            "dataset": metadata.get("dataset"),
            "metrics": {name: metadata["metrics"].get(name) for name in EVALUATION_METRICS},
        }

    @app.post("/predict")
    def predict(payload: PredictionRequest) -> dict[str, object]:
        if len(payload.text) > runtime_settings.max_text_length:
            raise HTTPException(
                status_code=413,
                detail=f"text exceeds {runtime_settings.max_text_length} characters",
            )
        try:
            result = service.predict(payload.text)
        except RuntimeError as exc:
            prediction_failures.inc()
            raise HTTPException(status_code=503, detail="The model is unavailable") from exc
        except Exception as exc:  # noqa: BLE001 - prevent inference details leaking to clients
            prediction_failures.inc()
            raise HTTPException(status_code=500, detail="Prediction failed") from exc
        probability = result["positive_probability"]
        prediction_counts.labels(predicted_class=result["label"]).inc()
        confidence.observe(max(probability, 1 - probability))
        return {
            "predicted_class": result["label"],
            "label": result["label"].capitalize(),
            "positive_probability": probability,
            "positive_terms": result["positive_terms"],
            "negative_terms": result["negative_terms"],
            "model_version": service.model_version,
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    # Keep the static fallback last so API routes take precedence. Root-relative
    # asset paths also work when frontend/ is deployed as a static site.
    app.mount("/", StaticFiles(directory=frontend_dir), name="frontend-assets")
    return app


app = create_app()
