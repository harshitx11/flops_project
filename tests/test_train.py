import json
from pathlib import Path

import pytest

from sentiment_mlops.config import settings
from sentiment_mlops.models.evaluate import candidate_is_accepted
from sentiment_mlops.models.train import train_model


def test_training_writes_accepted_artifact(trained_artifacts: Path) -> None:
    assert (trained_artifacts / settings.model_filename).is_file()
    metadata = json.loads((trained_artifacts / "model_metadata.json").read_text())
    assert metadata["accepted"] is True
    assert metadata["model_params"]["tfidf__ngram_range"] == [1, 2]
    assert "best_params" not in metadata


def test_rejected_candidate_does_not_replace_artifact(
    tmp_path: Path, dataset_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "min_accepted_f1", 1.01)
    with pytest.raises(RuntimeError, match="rejected"):
        train_model(dataset_path, tmp_path / "artifacts", tmp_path / "reports")
    assert not (tmp_path / "artifacts" / settings.model_filename).exists()
    assert json.loads((tmp_path / "reports" / "latest_evaluation.json").read_text())["accepted"] is False


def test_acceptance_threshold() -> None:
    assert candidate_is_accepted({"macro_f1": 0.8}, 0.75)
    assert not candidate_is_accepted({"macro_f1": 0.7}, 0.75)
