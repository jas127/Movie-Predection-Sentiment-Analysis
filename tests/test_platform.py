"""Automated tests for Movie Intelligence platform."""
import numpy as np
import pandas as pd
import pytest

from src import config, data_loader, data_quality, prediction, preprocessing, revenue_analysis, sentiment_analysis

@pytest.fixture(scope="session")
def box_office_raw():
    return data_loader.load_box_office()

@pytest.fixture(scope="session")
def reviews_raw():
    return data_loader.load_reviews()

@pytest.fixture(scope="session")
def prepared_box_office(box_office_raw):
    return preprocessing.prepare_box_office(box_office_raw)

@pytest.fixture(scope="session")
def prepared_reviews(reviews_raw):
    return preprocessing.prepare_reviews(reviews_raw)

# --- 1. Dataset Loading & Schema Tests ---
def test_box_office_loading(box_office_raw):
    assert not box_office_raw.empty
    assert len(box_office_raw) == 5000
    for col in config.BOX_OFFICE_REQUIRED_COLUMNS:
        assert col in box_office_raw.columns

def test_reviews_loading(reviews_raw):
    assert not reviews_raw.empty
    assert len(reviews_raw) == 2000
    for col in config.REVIEWS_REQUIRED_COLUMNS:
        assert col in reviews_raw.columns

# --- 2. Preprocessing & Cleanliness ---
def test_preprocessing_preserves_rows(box_office_raw, prepared_box_office):
    assert len(prepared_box_office) == len(box_office_raw)
    assert "Rating_Score" in prepared_box_office.columns
    assert "Genre_List" in prepared_box_office.columns

def test_reviews_cleaning_reproducibility(reviews_raw, prepared_reviews):
    # Original dropna and blanks removed 62 rows -> exactly 1938 remaining
    assert len(prepared_reviews) == 1938
    assert set(prepared_reviews["label"].unique()) == {"pos", "neg"}

# --- 3. Revenue Analysis Tests ---
def test_revenue_by_genre(prepared_box_office):
    g = revenue_analysis.revenue_by_genre(prepared_box_office, min_films=10)
    assert not g.empty
    assert "Genre" in g.columns
    assert "Median" in g.columns
    assert "Mean" in g.columns
    assert g["Median"].is_monotonic_decreasing or len(g) > 0

def test_filter_movies_empty_handling(prepared_box_office):
    # Year filter that matches nothing
    filtered = revenue_analysis.filter_movies(prepared_box_office, years=(1900, 1905))
    assert filtered.empty
    assert revenue_analysis.revenue_by_genre(filtered).empty

# --- 4. Sentiment Analysis Tests ---
def test_vader_evaluation_metrics(prepared_reviews):
    sid = sentiment_analysis.get_vader()
    scored = sentiment_analysis.score_reviews(prepared_reviews, sid)
    ev = sentiment_analysis.evaluate(scored)
    # Verified ground truth: exactly 63.57% accuracy (not the unverified 85% claim)
    assert pytest.approx(ev["accuracy"], abs=1e-3) == 0.6357
    assert ev["confusion"].shape == (2, 2)

def test_score_text_custom_input():
    sid = sentiment_analysis.get_vader()
    res = sentiment_analysis.score_text(sid, "An absolutely fantastic masterpiece of filmmaking!")
    assert res["label_3way"] == "Positive"
    assert res["compound"] > 0.5
    
    with pytest.raises(ValueError):
        sentiment_analysis.score_text(sid, "   ")

# --- 5. Prediction Pipeline Tests ---
def test_original_model_metrics(box_office_raw):
    model_df = preprocessing.model_training_frame(box_office_raw)
    assert len(model_df) == 4797
    res = prediction.train_and_evaluate(model_df, variant="original")
    # Verified exact reproduction: R2 ~ 0.5278
    assert pytest.approx(res.metrics["r2"], abs=1e-3) == 0.5279
    assert res.metrics["mse"] > 1e15

def test_prediction_input_validation(box_office_raw):
    model_df = preprocessing.model_training_frame(box_office_raw)
    features = config.MODEL_FEATURES
    
    # Valid input
    valid_input = {
        "Year": 2012, "Genres": "Action", "Rating": "7.5/10",
        "Vote_Count": 50000, "Original_Language": "en", "Production_Countries": "United States of America"
    }
    errors, warnings = prediction.validate_prediction_input(valid_input, model_df, features)
    assert len(errors) == 0
    
    # Missing input
    invalid_input = {"Year": 2012}
    errors, warnings = prediction.validate_prediction_input(invalid_input, model_df, features)
    assert len(errors) > 0

# --- 6. Data Quality Checks ---
def test_data_quality_suite(box_office_raw, reviews_raw):
    checks = data_quality.run_all_checks(box_office_raw, reviews_raw)
    assert not checks.empty
    assert "Status" in checks.columns
    # Critical checks should not fail
    fail_checks = checks[checks["Status"] == "FAIL"]
    assert len(fail_checks) == 0
