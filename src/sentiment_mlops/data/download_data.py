"""Download the public UCI Sentiment Labelled Sentences without fabricating fallback data."""

import argparse
import csv
import io
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

from sentiment_mlops.config import settings
from sentiment_mlops.data.prepare_data import load_and_validate

SOURCES = {
    "amazon_cells_labelled.txt": "amazon",
    "imdb_labelled.txt": "imdb",
    "yelp_labelled.txt": "yelp",
}


def combine_archive(payload: bytes) -> pd.DataFrame:
    """Read the three tab-separated review files from the archive into one frame."""

    frames = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = {Path(name).name: name for name in archive.namelist() if "__MACOSX" not in name}
        for filename, source in SOURCES.items():
            if filename not in names:
                raise RuntimeError(f"Downloaded archive does not contain {filename}")
            # QUOTE_NONE: sentences contain bare quote characters.
            part = pd.read_csv(
                io.BytesIO(archive.read(names[filename])),
                sep="\t",
                header=None,
                names=["text", "label"],
                dtype=str,
                quoting=csv.QUOTE_NONE,
                encoding="utf-8",
            )
            part.insert(0, "source", source)
            frames.append(part)
    return pd.concat(frames, ignore_index=True)


def download_dataset(data_dir: Path, url: str, force: bool = False) -> Path:
    """Download and validate the dataset, reusing a valid local copy by default."""

    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / settings.dataset_filename
    if destination.exists() and not force:
        load_and_validate(destination)
        return destination
    try:
        with urlopen(url, timeout=30) as response:
            payload = response.read()
    except (OSError, URLError) as exc:
        raise RuntimeError(f"Could not download dataset from {url}: {exc}") from exc
    if not payload:
        raise RuntimeError("Dataset download returned an empty response")
    try:
        frame = combine_archive(payload)
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        raise RuntimeError(f"Downloaded dataset archive is malformed: {exc}") from exc
    temporary = destination.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    load_and_validate(temporary)
    temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--url", default=settings.dataset_url)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(download_dataset(args.data_dir, args.url, args.force))


if __name__ == "__main__":
    main()
