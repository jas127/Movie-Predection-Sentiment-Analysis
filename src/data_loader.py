"""Dataset loading with schema validation.

The loaders return the *raw* data exactly as stored on disk (column names are
whitespace-stripped, as in the original notebook). Cleaning lives in
``preprocessing.py`` so every transformation can be documented.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config


class DataLoadError(Exception):
    """Raised when a dataset is missing or does not match the expected schema."""


def _check_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataLoadError(f"{name} is missing required columns: {missing}")


def load_box_office(path: Path | str = config.BOX_OFFICE_PATH) -> pd.DataFrame:
    """Load the box-office CSV (top-200 worldwide grossers per year)."""
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"Box office dataset not found at {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        raise DataLoadError(f"Could not parse {path.name}: {exc}") from exc
    df.columns = df.columns.str.strip()
    _check_columns(df, config.BOX_OFFICE_REQUIRED_COLUMNS, path.name)
    if df.empty:
        raise DataLoadError(f"{path.name} contains no rows")
    return df


def load_reviews(path: Path | str = config.REVIEWS_PATH) -> pd.DataFrame:
    """Load the labelled movie-review TSV (label: 'pos'/'neg', review: text)."""
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"Review dataset not found at {path}")
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception as exc:  # pragma: no cover - defensive
        raise DataLoadError(f"Could not parse {path.name}: {exc}") from exc
    df.columns = df.columns.str.strip()
    _check_columns(df, config.REVIEWS_REQUIRED_COLUMNS, path.name)
    if df.empty:
        raise DataLoadError(f"{path.name} contains no rows")
    return df
