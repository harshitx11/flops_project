"""Shared fixtures: a tiny, clearly separable dataset and a trained artifact."""

from pathlib import Path

import pandas as pd
import pytest

from sentiment_mlops.config import settings
from sentiment_mlops.models.train import train_model

POSITIVE = ["great", "excellent", "love", "wonderful", "amazing", "fantastic", "good", "perfect"]
NEGATIVE = ["terrible", "awful", "hate", "horrible", "bad", "poor", "boring", "worst"]


@pytest.fixture
def dataset_path(tmp_path: Path) -> Path:
    rows = []
    for index, (good, bad) in enumerate(zip(POSITIVE * 3, NEGATIVE * 3)):
        rows.append({"source": "yelp", "text": f"{good} service number {index}", "label": "1"})
        rows.append({"source": "amazon", "text": f"{bad} product number {index}", "label": "0"})
    path = tmp_path / "sentiment_sentences.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def trained_artifacts(tmp_path: Path, dataset_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(settings, "min_accepted_f1", 0.5)
    artifact_dir = tmp_path / "artifacts"
    train_model(dataset_path, artifact_dir, tmp_path / "reports")
    return artifact_dir
