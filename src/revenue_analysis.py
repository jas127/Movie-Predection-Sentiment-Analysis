"""Box-office revenue aggregations and filters (pure pandas, UI-independent)."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.preprocessing import explode_column

REV = "$Worldwide"


def format_currency(value: float, decimals: int = 1) -> str:
    """Human-readable USD: 2_799_439_100 -> '$2.8B'. NaN -> 'n/a'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    sign = "-" if value < 0 else ""
    v = abs(float(value))
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= divisor:
            return f"{sign}${v / divisor:,.{decimals}f}{suffix}"
    return f"{sign}${v:,.0f}"


def filter_movies(
    df: pd.DataFrame,
    years: tuple[int, int] | None = None,
    genres: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
    countries: Iterable[str] | None = None,
    rating_bands: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Apply dashboard filters. Empty/None selections mean 'no filter'.

    Genre and country filters match a film if *any* of its listed values is
    selected (multi-genre / co-production aware).
    """
    mask = pd.Series(True, index=df.index)
    if years is not None:
        lo, hi = sorted(years)
        mask &= df["Year"].between(lo, hi)
    genres = set(genres or [])
    if genres:
        mask &= df["Genre_List"].map(lambda gl: bool(genres.intersection(gl)))
    languages = set(languages or [])
    if languages:
        mask &= df["Original_Language"].isin(languages)
    countries = set(countries or [])
    if countries:
        mask &= df["Country_List"].map(lambda cl: bool(countries.intersection(cl)))
    bands = set(rating_bands or [])
    if bands:
        mask &= df["Rating_Band"].isin(bands)
    return df[mask]


def _summary(grouped) -> pd.DataFrame:
    out = grouped[REV].agg(
        Films="count", Total="sum", Mean="mean", Median="median", Std="std",
        Q1=lambda s: s.quantile(0.25), Q3=lambda s: s.quantile(0.75),
    )
    out["IQR"] = out["Q3"] - out["Q1"]
    # Coefficient of variation: dispersion relative to the mean (unitless).
    out["CV"] = out["Std"] / out["Mean"]
    return out


def revenue_by_genre(df: pd.DataFrame, min_films: int = 1) -> pd.DataFrame:
    """Revenue statistics per genre. Multi-genre films count toward each genre."""
    if df.empty:
        return pd.DataFrame(columns=["Genre", "Films", "Total", "Mean", "Median", "Std", "Q1", "Q3", "IQR", "CV"])
    exploded = explode_column(df, "Genre_List", "Genre")
    out = _summary(exploded.groupby("Genre"))
    out = out[out["Films"] >= min_films].reset_index()
    return out.sort_values("Median", ascending=False).reset_index(drop=True)


def revenue_by_year(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Year", "Films", "Total", "Mean", "Median"])
    out = df.groupby("Year")[REV].agg(Films="count", Total="sum", Mean="mean", Median="median")
    return out.reset_index().sort_values("Year")


def revenue_by_column(df: pd.DataFrame, column: str, min_films: int = 1) -> pd.DataFrame:
    """Revenue stats grouped by a single-valued column (e.g. Rating_Band, language)."""
    if df.empty:
        return pd.DataFrame(columns=[column, "Films", "Total", "Mean", "Median"])
    out = _summary(df.groupby(column, dropna=False)).reset_index()
    out = out[out["Films"] >= min_films]
    return out.sort_values("Median", ascending=False).reset_index(drop=True)


def revenue_by_country(df: pd.DataFrame, top_n: int = 15, min_films: int = 1) -> pd.DataFrame:
    """Revenue stats per production country (co-productions count for each country)."""
    if df.empty:
        return pd.DataFrame(columns=["Country", "Films", "Total", "Mean", "Median"])
    exploded = explode_column(df, "Country_List", "Country")
    out = _summary(exploded.groupby("Country")).reset_index()
    out = out[out["Films"] >= min_films]
    return out.sort_values("Total", ascending=False).head(top_n).reset_index(drop=True)


def top_movies(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    cols = ["Title", "Year", "Genres", REV, "$Domestic", "$Foreign", "Rating_Score", "Vote_Count",
            "Original_Language", "Production_Countries"]
    return df.nlargest(n, REV)[cols].reset_index(drop=True)


def correlation(df: pd.DataFrame, x: str, y: str = REV) -> dict:
    """Spearman (rank) and Pearson-on-log correlations, robust to heavy tails.

    Rows with missing values, and non-positive values for the log version,
    are excluded; the number of rows used is returned for transparency.
    """
    sub = df[[x, y]].dropna()
    result = {"n": int(len(sub)), "spearman": np.nan, "pearson_log": np.nan}
    if len(sub) < 3:
        return result
    result["spearman"] = float(sub[x].corr(sub[y], method="spearman"))
    pos = sub[(sub[x] > 0) & (sub[y] > 0)]
    if len(pos) >= 3:
        result["pearson_log"] = float(np.log10(pos[x]).corr(np.log10(pos[y])))
    return result


def iqr_outliers(series: pd.Series, k: float = 1.5, log_scale: bool = False) -> tuple[pd.Series, float, float]:
    """Tukey IQR outlier flags. Returns (mask, lower_bound, upper_bound).

    With ``log_scale=True`` the fences are computed on log10 values (more
    appropriate for right-skewed revenue) and returned in original units.
    """
    s = series.astype(float)
    work = np.log10(s.where(s > 0)) if log_scale else s
    q1, q3 = work.quantile(0.25), work.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    mask = (work < lo) | (work > hi)
    if log_scale:
        lo, hi = 10 ** lo, 10 ** hi
    return mask.fillna(False), float(lo), float(hi)


def foreign_share_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate foreign share of worldwide gross per year (sum-based, not mean of %)."""
    if df.empty:
        return pd.DataFrame(columns=["Year", "Foreign_Share"])
    g = df.groupby("Year")[["$Foreign", REV]].sum()
    g["Foreign_Share"] = g["$Foreign"] / g[REV]
    return g.reset_index()[["Year", "Foreign_Share"]]
