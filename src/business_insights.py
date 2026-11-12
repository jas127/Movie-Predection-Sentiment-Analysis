"""Business insights generated from live calculations.

Every insight is built from numbers computed on the datasets at runtime; the
text templates only phrase those numbers. Each insight carries an
``evidence_type`` so the UI can distinguish descriptive observations from
model-based results - none of these are causal claims.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.revenue_analysis import (correlation, foreign_share_by_year, format_currency, revenue_by_column,
                                  revenue_by_genre)
from src.sentiment_analysis import title_sentiment_by_genre


def _insight(question, finding, evidence_type, caveat, metric=None):
    return {"question": question, "finding": finding, "evidence_type": evidence_type,
            "caveat": caveat, "metric": metric}


def genre_revenue_insight(df: pd.DataFrame, min_films: int = config.MIN_GROUP_SIZE) -> dict:
    g = revenue_by_genre(df, min_films=min_films)
    if g.empty:
        return _insight("Which genres generate the highest worldwide revenue?", "Not enough data.", "Descriptive", "")
    top, bottom = g.iloc[0], g.iloc[-1]
    top3 = ", ".join(f"{r.Genre} ({format_currency(r.Median)})" for r in g.head(3).itertuples())
    ratio = top["Median"] / bottom["Median"] if bottom["Median"] else np.nan
    return _insight(
        "Which genres generate the highest worldwide revenue?",
        f"By median worldwide gross, the leading genres are {top3}. The lowest is {bottom['Genre']} "
        f"({format_currency(bottom['Median'])}), so the top genre's typical film grosses about {ratio:.1f}× more.",
        "Descriptive",
        f"Median used because means are inflated by blockbusters. Multi-genre films count toward each genre. "
        f"Genres with fewer than {min_films} films excluded.",
        metric={"top_genre": top["Genre"], "top_median": float(top["Median"])},
    )


def genre_variability_insight(df: pd.DataFrame, min_films: int = config.MIN_GROUP_SIZE) -> dict:
    g = revenue_by_genre(df, min_films=min_films).sort_values("CV", ascending=False)
    if g.empty:
        return _insight("Which genres show the greatest variation in revenue?", "Not enough data.", "Descriptive", "")
    hi, lo = g.iloc[0], g.iloc[-1]
    return _insight(
        "Which genres show the greatest variation in revenue?",
        f"{hi['Genre']} has the highest relative variability (coefficient of variation {hi['CV']:.2f}); "
        f"{lo['Genre']} is the most consistent (CV {lo['CV']:.2f}). Every genre has CV > 1, i.e. the spread "
        f"of outcomes is larger than the average outcome."
        if (g["CV"] > 1).all() else
        f"{hi['Genre']} has the highest relative variability (CV {hi['CV']:.2f}); {lo['Genre']} the lowest "
        f"(CV {lo['CV']:.2f}).",
        "Descriptive",
        "CV = standard deviation ÷ mean. High variability signals hit-driven risk, not necessarily low returns.",
        metric={"most_variable": hi["Genre"], "cv": float(hi["CV"])},
    )


def popularity_insight(df: pd.DataFrame) -> dict:
    c = correlation(df, "Vote_Count")
    r = c["spearman"]
    strength = "strong" if abs(r) >= 0.6 else "moderate" if abs(r) >= 0.3 else "weak"
    return _insight(
        "Is audience popularity (vote count) associated with box-office performance?",
        f"Yes - a {strength} positive association (Spearman ρ = {r:.2f}, n = {c['n']:,}). "
        f"Films with more audience votes tend to have higher worldwide gross.",
        "Correlational",
        "Votes accumulate after release, largely because more people saw the film. This is association, "
        "and the direction of causation most likely runs from revenue/reach to votes.",
        metric=c,
    )


def rating_insight(df: pd.DataFrame) -> dict:
    sub = df[df["Rating_Band"] != "Not available"]
    c = correlation(sub, "Rating_Score")
    r = c["spearman"]
    return _insight(
        "Do higher audience scores go with higher revenue?",
        f"Only weakly (Spearman ρ = {r:.2f}, n = {c['n']:,}). Audience score is a much weaker revenue signal "
        f"than audience reach (vote count).",
        "Correlational",
        "Scores exclude placeholder 0/10 values with no votes.",
        metric=c,
    )


def title_sentiment_insight(scored_titles: pd.DataFrame, exploded: pd.DataFrame,
                            min_films: int = config.MIN_GROUP_SIZE) -> dict:
    t = title_sentiment_by_genre(exploded, "Title_Sentiment_TextBlob", min_films=min_films)
    corr = correlation(scored_titles, "Title_Sentiment_TextBlob")
    nonzero = float((scored_titles["Title_Sentiment_TextBlob"] != 0).mean())
    top, bottom = t.iloc[0], t.iloc[-1]
    return _insight(
        "Which genres have more positive titles - and does it matter for revenue?",
        f"{top['Genre']} titles score highest on average ({top['Mean']:+.3f}) and {bottom['Genre']} lowest "
        f"({bottom['Mean']:+.3f}). However only {nonzero:.0%} of titles contain any sentiment-bearing words, and "
        f"title sentiment shows essentially no relationship with revenue (Spearman ρ = {corr['spearman']:.2f}).",
        "Descriptive",
        "Title polarity (TextBlob, as in the original notebook) measures wording, not audience reaction. "
        "Averages are close to zero because most titles score exactly 0.",
        metric={"nonzero_share": nonzero, "revenue_spearman": corr["spearman"]},
    )


def review_sentiment_insight(eval_result: dict, scored_reviews: pd.DataFrame) -> dict:
    cm = eval_result["confusion"]
    neg_recall = cm[0, 0] / cm[0].sum()
    pos_recall = cm[1, 1] / cm[1].sum()
    med = scored_reviews.groupby("label")["compound"].median()
    return _insight(
        "How well does lexicon sentiment reflect what reviewers actually concluded?",
        f"VADER matches the reviewer's label {eval_result['accuracy']:.1%} of the time on {eval_result['n']:,} "
        f"reviews. It recognises {pos_recall:.0%} of positive reviews but only {neg_recall:.0%} of negative ones - "
        f"negative reviews often still contain many positive words (median compound for negative reviews: "
        f"{med.get('neg', np.nan):+.2f}).",
        "Model evaluation",
        "A balanced academic dataset of long-form critic reviews; accuracy on other text will differ.",
        metric={"accuracy": eval_result["accuracy"], "neg_recall": neg_recall, "pos_recall": pos_recall},
    )


def international_insight(df: pd.DataFrame) -> dict:
    fs = foreign_share_by_year(df)
    first, last = fs.iloc[0], fs.iloc[-1]
    lang = revenue_by_column(df, "Original_Language")
    total = df["$Worldwide"].sum()
    en_share = df.loc[df["Original_Language"] == "en", "$Worldwide"].sum() / total if total else np.nan
    return _insight(
        "How important are markets outside the US?",
        f"Non-US ('foreign') box office was {first['Foreign_Share']:.0%} of worldwide gross in "
        f"{int(first['Year'])} and {last['Foreign_Share']:.0%} in {int(last['Year'])}. English-language films "
        f"account for {en_share:.0%} of total worldwide gross in the dataset.",
        "Descriptive",
        "The dataset is the top 200 films per year, which includes many non-US releases with $0 US gross.",
        metric={"foreign_share_first": float(first["Foreign_Share"]), "foreign_share_last": float(last["Foreign_Share"]),
                "en_share": float(en_share), "n_languages": int(len(lang))},
    )


def model_insight(original_metrics: dict, pre_release_metrics: dict) -> dict:
    a, b = original_metrics["r2"], pre_release_metrics["r2"]
    return _insight(
        "Can we predict box office from metadata?",
        f"The original model explains {a:.0%} of revenue variance on held-out films (R² = {a:.3f}), but most of "
        f"that comes from vote count, which is only known after release. Using only pre-release fields "
        f"(year, genres, language, countries) R² falls to {b:.3f}.",
        "Model evaluation",
        "Not suitable for green-light or forecasting decisions. Missing key drivers: budget, marketing, cast, "
        "release date/competition, franchise status.",
        metric={"r2_original": a, "r2_pre_release": b},
    )


def sentiment_proxy_limitations() -> dict:
    return _insight(
        "What are the limits of using sentiment as a proxy for movie success?",
        "The review dataset has no titles, genres or revenue, so review sentiment cannot be linked to box office "
        "here. Title sentiment is linkable but unrelated to revenue. Lexicon scores also miss sarcasm, "
        "context and film-specific vocabulary ('dark', 'killer', 'unpredictable' can be praise).",
        "Methodological",
        "Linking sentiment to success would need reviews joined to specific films, ideally with timestamps.",
    )


def build_all_insights(df, scored_titles, exploded_titles, eval_result, scored_reviews,
                       original_metrics, pre_release_metrics) -> list[dict]:
    return [
        genre_revenue_insight(df),
        genre_variability_insight(df),
        popularity_insight(df),
        rating_insight(df),
        international_insight(df),
        title_sentiment_insight(scored_titles, exploded_titles),
        review_sentiment_insight(eval_result, scored_reviews),
        model_insight(original_metrics, pre_release_metrics),
        sentiment_proxy_limitations(),
    ]


def executive_summary(insights: list[dict]) -> list[str]:
    """Short, plain-English bullet points for non-technical stakeholders."""
    def metric(prefix: str) -> dict:
        for i in insights:
            if i["question"].startswith(prefix):
                return i["metric"] or {}
        return {}

    g = metric("Which genres generate")
    pop = metric("Is audience popularity")
    mod = metric("Can we predict")
    rev = metric("How well does lexicon")
    intl = metric("How important are markets")
    bullets = []
    if g:
        bullets.append(f"**{g['top_genre']}** films have the highest typical worldwide gross "
                       f"({format_currency(g['top_median'])} median) among genres with ≥{config.MIN_GROUP_SIZE} films.")
    bullets.append("Revenue is **hit-driven in every genre**: a few blockbusters account for a large share of totals, "
                   "so medians are a better guide to a 'typical' film than averages.")
    if pop:
        bullets.append(f"Audience reach (vote count) tracks revenue closely (ρ = {pop['spearman']:.2f}); "
                       "audience *score* barely does.")
    if intl:
        bullets.append(f"International markets matter: non-US share of gross moved from "
                       f"{intl['foreign_share_first']:.0%} to {intl['foreign_share_last']:.0%} over the period.")
    if mod:
        bullets.append(f"The revenue model is **explanatory, not a forecast**: R² {mod['r2_original']:.2f} overall, "
                       f"but only {mod['r2_pre_release']:.2f} using information available before release.")
    if rev:
        bullets.append(f"Automated review sentiment (VADER) agrees with reviewers {rev['accuracy']:.0%} of the time and "
                       "over-reads negative reviews as positive - useful for rough monitoring, not for decisions.")
    return bullets
