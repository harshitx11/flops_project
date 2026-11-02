"""Load the accepted model artifact and provide inference helpers."""

from pathlib import Path
from typing import Any

import joblib


class ModelService:
    """Single loaded model used by the API process."""

    def __init__(self, model_path: Path, model_version: str) -> None:
        self.model_path = model_path
        self.model_version = model_version
        self.pipeline: Any | None = None
        self.load_error: str | None = None
        try:
            if not model_path.is_file():
                raise FileNotFoundError(f"Model artifact not found: {model_path}")
            self.pipeline = joblib.load(model_path)
        except Exception as exc:  # noqa: BLE001 - expose a stable health state to the API
            self.load_error = str(exc)

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    def predict(self, text: str, top_terms: int = 5) -> dict[str, Any]:
        """Return the label, positive probability, and the terms that drove it."""

        if self.pipeline is None:
            raise RuntimeError(self.load_error or "Model is unavailable")
        label = str(self.pipeline.predict([text])[0])
        classes = list(self.pipeline.classes_)
        positive_probability = float(self.pipeline.predict_proba([text])[0][classes.index("positive")])

        # Per-term contribution = TF-IDF weight x coefficient (positive-class direction).
        vectorizer = self.pipeline.named_steps["tfidf"]
        coefficients = self.pipeline.named_steps["classifier"].coef_[0]
        if classes[1] != "positive":
            coefficients = -coefficients
        row = vectorizer.transform([text])
        names = vectorizer.get_feature_names_out()
        contributions = sorted(
            ((str(names[index]), float(weight * coefficients[index])) for index, weight in zip(row.indices, row.data)),
            key=lambda item: item[1],
        )
        return {
            "label": label,
            "positive_probability": positive_probability,
            "positive_terms": [{"term": t, "weight": w} for t, w in reversed(contributions) if w > 0][:top_terms],
            "negative_terms": [{"term": t, "weight": w} for t, w in contributions if w < 0][:top_terms],
        }
