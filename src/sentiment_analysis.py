"""Sentiment analysis built on the project's original methods.

* Review sentiment: NLTK VADER ``SentimentIntensityAnalyzer`` with the original
  notebook's rule ``compound >= 0 -> 'pos' else 'neg'``.
* Title sentiment: TextBlob polarity, as used in the original title notebook.
  VADER compound on titles is provided as a secondary comparison.

Both are lexicon/rule-based methods - not machine-learned transformers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src import config


class SentimentModelError(RuntimeError):
    """Raised when the VADER lexicon cannot be loaded or downloaded."""


def get_vader():
    """Return a VADER analyser, using a project-local NLTK data directory.

    If the lexicon is absent it is downloaded once into ``./nltk_data``
    (works on Streamlit Community Cloud, which has outbound internet).
    """
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    local = str(config.NLTK_DATA_DIR)
    if local not in nltk.data.path:
        nltk.data.path.insert(0, local)
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        try:
            config.NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
            nltk.download("vader_lexicon", download_dir=local, quiet=True)
            return SentimentIntensityAnalyzer()
        except Exception as exc:
            raise SentimentModelError(
                "VADER lexicon unavailable. Run: python -m nltk.downloader -d ./nltk_data vader_lexicon"
            ) from exc


def three_way_label(compound: float) -> str:
    """Conventional VADER 3-class bucketing (display only; dataset labels are binary)."""
    if compound >= config.VADER_POS_THRESHOLD:
        return "Positive"
    if compound <= config.VADER_NEG_THRESHOLD:
        return "Negative"
    return "Neutral"


def score_text(analyzer, text: str) -> dict:
    """Score one piece of free text. Raises ValueError for empty/non-text input."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter some review text to analyse.")
    if len(text) > 20_000:
        raise ValueError("Text is too long (limit 20,000 characters).")
    scores = analyzer.polarity_scores(text)
    scores["label_3way"] = three_way_label(scores["compound"])
    scores["label_binary"] = "pos" if scores["compound"] >= config.ORIGINAL_CLASSIFICATION_THRESHOLD else "neg"
    return scores


def score_reviews(df: pd.DataFrame, analyzer) -> pd.DataFrame:
    """Add VADER neg/neu/pos/compound + predicted labels (original notebook logic)."""
    out = df.copy()
    scores = out["review"].map(analyzer.polarity_scores)
    for key in ("neg", "neu", "pos", "compound"):
        out[key] = scores.map(lambda d, k=key: d[k])
    out["predicted"] = np.where(out["compound"] >= config.ORIGINAL_CLASSIFICATION_THRESHOLD, "pos", "neg")
    out["sentiment_3way"] = out["compound"].map(three_way_label)
    out["correct"] = out["predicted"] == out["label"]
    out["word_count"] = out["review"].str.split().str.len()
    return out


def evaluate(df: pd.DataFrame, threshold: float = config.ORIGINAL_CLASSIFICATION_THRESHOLD) -> dict:
    """Accuracy, per-class report and confusion matrix (rows=true, cols=pred; order neg,pos)."""
    if df.empty:
        raise ValueError("No reviews to evaluate.")
    pred = np.where(df["compound"] >= threshold, "pos", "neg")
    labels = ["neg", "pos"]
    return {
        "n": int(len(df)),
        "accuracy": float(accuracy_score(df["label"], pred)),
        "report": classification_report(df["label"], pred, labels=labels, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(df["label"], pred, labels=labels),
        "threshold": threshold,
    }


def threshold_sweep(df: pd.DataFrame, thresholds=None) -> pd.DataFrame:
    """Accuracy at alternative compound thresholds (in-sample, diagnostic only)."""
    if thresholds is None:
        thresholds = np.round(np.arange(-0.5, 0.95, 0.05), 2)
    rows = []
    for t in thresholds:
        pred = np.where(df["compound"] >= t, "pos", "neg")
        rows.append({"Threshold": float(t), "Accuracy": float((pred == df["label"]).mean())})
    return pd.DataFrame(rows)


# --- Title sentiment -----------------------------------------------------------

def textblob_polarity(text) -> float:
    """Original title-notebook method: TextBlob polarity in [-1, 1]."""
    from textblob import TextBlob
    return float(TextBlob(str(text)).sentiment.polarity)


def add_title_sentiment(df: pd.DataFrame, analyzer) -> pd.DataFrame:
    out = df.copy()
    out["Title_Sentiment_TextBlob"] = out["Title"].map(textblob_polarity)
    out["Title_Sentiment_VADER"] = out["Title"].map(lambda t: analyzer.polarity_scores(str(t))["compound"])
    return out


def title_sentiment_by_genre(exploded: pd.DataFrame, column: str, min_films: int = 1) -> pd.DataFrame:
    """Per-genre title sentiment. Includes the share of titles with a non-zero
    score, because most titles contain no sentiment-bearing words."""
    if exploded.empty:
        return pd.DataFrame(columns=["Genre", "Films", "Mean", "Median", "Non_Zero_Share"])
    g = exploded.groupby("Genre")[column]
    out = pd.DataFrame({
        "Films": g.count(),
        "Mean": g.mean(),
        "Median": g.median(),
        "Non_Zero_Share": g.apply(lambda s: float((s != 0).mean())),
    }).reset_index()
    out = out[out["Films"] >= min_films]
    return out.sort_values("Mean", ascending=False).reset_index(drop=True)
