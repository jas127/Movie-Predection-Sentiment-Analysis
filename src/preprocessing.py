"""Documented, non-destructive preprocessing.

Design rule: analysis tables keep *every* record. Derived columns are added and
problems are flagged; rows are only removed where the original notebooks removed
them (model training set, VADER evaluation set) and each removal is logged in a
``PreprocessingLog`` that the Data Quality page displays.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config

RATING_BANDS = ["< 5.0", "5.0 – 5.9", "6.0 – 6.9", "7.0 – 7.9", "8.0 +", "Not available"]


@dataclass
class PreprocessingLog:
    """Ordered record of preprocessing steps and their effect on row counts."""

    steps: list[dict] = field(default_factory=list)

    def add(self, dataset: str, step: str, rows_before: int, rows_after: int, note: str = "") -> None:
        self.steps.append({
            "Dataset": dataset,
            "Step": step,
            "Rows before": rows_before,
            "Rows after": rows_after,
            "Rows removed": rows_before - rows_after,
            "Note": note,
        })

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


def split_list(value) -> list[str]:
    """Split a comma-separated string ('Action, Drama') into a clean list."""
    if not isinstance(value, str) or not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_rating(value) -> float:
    """Parse the dataset's audience score format ('6.126/10') into a float.

    Returns NaN for missing or malformed values instead of raising.
    """
    if not isinstance(value, str):
        return np.nan
    text = value.strip()
    if text.endswith("/10"):
        text = text[:-3]
    try:
        score = float(text)
    except ValueError:
        return np.nan
    return score if 0 <= score <= 10 else np.nan


def rating_band(score: float, votes: float) -> str:
    """Bucket an audience score. A 0.0 score with 0 votes means 'no score yet'."""
    if pd.isna(score) or (score == 0 and (pd.isna(votes) or votes == 0)):
        return "Not available"
    if score < 5:
        return RATING_BANDS[0]
    if score < 6:
        return RATING_BANDS[1]
    if score < 7:
        return RATING_BANDS[2]
    if score < 8:
        return RATING_BANDS[3]
    return RATING_BANDS[4]


def prepare_box_office(raw: pd.DataFrame, log: PreprocessingLog | None = None) -> pd.DataFrame:
    """Add analysis-friendly derived columns. No rows are removed."""
    df = raw.copy()
    n = len(df)
    df["Title"] = df["Release Group"].astype(str)
    df["Rating_Score"] = df["Rating"].map(parse_rating)
    df["Rating_Band"] = [rating_band(s, v) for s, v in zip(df["Rating_Score"], df["Vote_Count"])]
    df["Genre_List"] = df["Genres"].map(split_list)
    df["Country_List"] = df["Production_Countries"].map(split_list)
    df["Primary_Genre"] = df["Genre_List"].map(lambda g: g[0] if g else "Unknown")
    df["Original_Language"] = df["Original_Language"].astype("object")
    df["Has_Full_Metadata"] = df[config.MODEL_FEATURES].notna().all(axis=1)
    df["Worldwide_M"] = df["$Worldwide"] / 1e6
    if log is not None:
        log.add("Box office", "Load CSV + strip column names", n, n)
        log.add("Box office", "Derive Rating_Score (parse 'x/10'), Rating_Band, genre/country lists", n, len(df),
                "Derived columns only; original columns untouched")
    return df


def model_training_frame(raw: pd.DataFrame, log: PreprocessingLog | None = None) -> pd.DataFrame:
    """Reproduce the original notebook's training-set filtering exactly.

    Original code: ``df.dropna(subset=['$Worldwide'])`` then ``df.dropna()``.
    """
    n0 = len(raw)
    df = raw.dropna(subset=[config.TARGET])
    n1 = len(df)
    df = df.dropna()
    if log is not None:
        log.add("Box office (model)", "Drop rows with missing target ($Worldwide)", n0, n1)
        log.add("Box office (model)", "Drop rows with any missing value (original df.dropna())", n1, len(df),
                "Mostly re-releases / director's cuts without metadata")
    return df


def prepare_reviews(raw: pd.DataFrame, log: PreprocessingLog | None = None) -> pd.DataFrame:
    """Reproduce the original VADER notebook cleaning: drop NaN and blank reviews.

    Duplicates are *flagged* (``Is_Duplicate``) but kept so the original
    evaluation can be reproduced; the Data Quality page reports their impact.
    """
    n0 = len(raw)
    df = raw.dropna().copy()
    n1 = len(df)
    df = df[~df["review"].astype(str).str.isspace()].copy()
    n2 = len(df)
    df["review"] = df["review"].astype(str)
    df["label"] = df["label"].astype(str).str.strip()
    df["Is_Duplicate"] = df.duplicated(subset=["review"], keep="first")
    df = df.reset_index(drop=True)
    if log is not None:
        log.add("Reviews", "Drop rows with missing review text (original dropna)", n0, n1)
        log.add("Reviews", "Drop whitespace-only reviews (original 'blanks' loop)", n1, n2)
        log.add("Reviews", "Flag duplicate review texts (kept, not removed)", n2, n2,
                f"{int(df['Is_Duplicate'].sum())} duplicates flagged")
    return df


def explode_column(df: pd.DataFrame, list_col: str, new_col: str) -> pd.DataFrame:
    """One row per (film, list item). Films with an empty list are dropped from
    the exploded view only. NOTE: multi-genre films are counted once per genre."""
    out = df.explode(list_col).rename(columns={list_col: new_col})
    out = out[out[new_col].notna()]
    return out
