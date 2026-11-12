"""Streamlit-cached access to datasets, scores and models.

Analytics live in Streamlit-free modules (tested with pytest); this file only
adds caching so heavy work (VADER scoring of ~2k long reviews, RF training)
runs once per server process.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src import config, data_loader, prediction, preprocessing, sentiment_analysis


@st.cache_data(show_spinner="Loading box office data…")
def box_office_raw() -> pd.DataFrame:
    return data_loader.load_box_office()


@st.cache_data(show_spinner=False)
def box_office() -> pd.DataFrame:
    return preprocessing.prepare_box_office(box_office_raw())


@st.cache_data(show_spinner=False)
def model_frame() -> pd.DataFrame:
    return preprocessing.model_training_frame(box_office_raw())


@st.cache_data(show_spinner="Loading reviews…")
def reviews_raw() -> pd.DataFrame:
    return data_loader.load_reviews()


@st.cache_resource(show_spinner="Loading VADER lexicon…")
def vader():
    return sentiment_analysis.get_vader()


@st.cache_data(show_spinner="Scoring reviews with VADER…")
def reviews_scored() -> pd.DataFrame:
    clean = preprocessing.prepare_reviews(reviews_raw())
    return sentiment_analysis.score_reviews(clean, vader())


@st.cache_data(show_spinner="Scoring titles…")
def titles_scored() -> pd.DataFrame:
    return sentiment_analysis.add_title_sentiment(box_office(), vader())


@st.cache_data(show_spinner=False)
def titles_exploded() -> pd.DataFrame:
    return preprocessing.explode_column(titles_scored(), "Genre_List", "Genre")


@st.cache_resource(show_spinner="Training Random Forest model…")
def model(variant: str = "original") -> prediction.ModelResult:
    return prediction.train_and_evaluate(model_frame(), variant)


@st.cache_data(show_spinner="Computing permutation importance…")
def permutation_importance(variant: str = "original") -> pd.DataFrame:
    return prediction.permutation_importances(model(variant))


@st.cache_data(show_spinner=False)
def preprocessing_log() -> pd.DataFrame:
    log = preprocessing.PreprocessingLog()
    preprocessing.prepare_box_office(box_office_raw(), log)
    preprocessing.model_training_frame(box_office_raw(), log)
    preprocessing.prepare_reviews(reviews_raw(), log)
    return log.to_frame()


def saved_model_report() -> dict | None:
    """Cross-validation results written by scripts/evaluate_models.py (if run)."""
    path = config.REPORTS_DIR / "model_metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
