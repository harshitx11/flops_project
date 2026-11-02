import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from sentiment_mlops.data.download_data import combine_archive
from sentiment_mlops.data.prepare_data import load_and_validate, normalize_label


def test_normalize_label() -> None:
    assert normalize_label("1") == "positive"
    assert normalize_label(" 0 ") == "negative"
    assert normalize_label("Positive") == "positive"
    with pytest.raises(ValueError):
        normalize_label("2")


def test_load_removes_blank_and_duplicate_rows(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({
        "source": ["yelp"] * 4,
        "text": ["good food", "good food", "  ", "bad food"],
        "label": ["1", "1", "0", "0"],
    }).to_csv(path, index=False)
    frame, stats = load_and_validate(path)
    assert frame["text"].tolist() == ["good food", "bad food"]
    assert stats["duplicate_records_removed"] == 1
    assert stats["label_distribution"] == {"positive": 1, "negative": 1}


def test_load_rejects_conflicting_labels(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"source": ["imdb"] * 3, "text": ["ok", "ok", "bad"], "label": ["1", "0", "0"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="conflicting"):
        load_and_validate(path)


def test_load_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_and_validate(tmp_path / "missing.csv")


def test_combine_archive_reads_all_sources() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in ("amazon_cells_labelled.txt", "imdb_labelled.txt", "yelp_labelled.txt"):
            archive.writestr(f"sentiment labelled sentences/{name}", 'Said "wow" here.\t1\nNot good.\t0\n')
    frame = combine_archive(buffer.getvalue())
    assert len(frame) == 6
    assert set(frame["source"]) == {"amazon", "imdb", "yelp"}
    assert frame.loc[0, "text"] == 'Said "wow" here.'


def test_combine_archive_requires_every_file() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("yelp_labelled.txt", "Nice.\t1\n")
    with pytest.raises(RuntimeError, match="amazon_cells_labelled.txt"):
        combine_archive(buffer.getvalue())
