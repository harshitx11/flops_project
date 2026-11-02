"""Load, normalize, and validate the combined sentiment dataset."""

from pathlib import Path

import pandas as pd

LABEL_MAP = {"0": "negative", "1": "positive"}
LABELS = ["negative", "positive"]


def normalize_label(value: object) -> str:
    """Map a raw 0/1 label (or an already-normalized name) to a class name."""

    label = str(value).lower()
    if label in LABEL_MAP:
        return LABEL_MAP[label]
    if label in LABELS:
        return label
    raise ValueError(f"Unsupported label {value!r}; expected 0, 1, negative, or positive")


def load_and_validate(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read the combined CSV and return clean records and quality stats."""

    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Dataset file is missing or empty: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"source", "text", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    frame["text"] = frame["text"].str.strip()
    frame = frame[frame["text"] != ""].copy()
    frame["label"] = frame["label"].map(normalize_label)

    duplicate_count = int(frame.duplicated(subset=["text"]).sum())
    conflicts = frame.groupby("text")["label"].nunique()
    conflict_count = int((conflicts > 1).sum())
    if conflict_count:
        raise ValueError(f"Found {conflict_count} sentences with conflicting labels")
    frame = frame.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if set(frame["label"]) != set(LABELS):
        raise ValueError("Dataset must contain both positive and negative classes")
    stats: dict[str, object] = {
        "records": len(frame),
        "duplicate_records_removed": duplicate_count,
        "label_distribution": frame["label"].value_counts().to_dict(),
        "source_distribution": frame["source"].value_counts().to_dict(),
    }
    return frame, stats
