"""Automated data quality checks.

Every check returns a row with a status (PASS / WARN / FAIL), the number of
affected records and a plain-English explanation. Nothing is modified or
removed here - this module only reports.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src import config
from src.preprocessing import parse_rating
from src.revenue_analysis import iqr_outliers

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

EXPECTED_BOX_OFFICE_TYPES = {
    "Rank": "integer", "Release Group": "text", "$Worldwide": "numeric", "$Domestic": "numeric",
    "Domestic %": "numeric", "$Foreign": "numeric", "Foreign %": "numeric", "Year": "integer",
    "Genres": "text", "Rating": "text ('x/10')", "Vote_Count": "numeric", "Original_Language": "text",
    "Production_Countries": "text",
}

EDITION_PATTERN = re.compile(r"re-?release|director'?s cut|anniversary|remaster|\b3d\b|special edition|imax",
                             re.IGNORECASE)


def _check(dataset, name, status, affected, detail):
    return {"Dataset": dataset, "Check": name, "Status": status, "Affected rows": int(affected), "Detail": detail}


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    out = pd.DataFrame({
        "Column": df.columns,
        "Missing": df.isna().sum().values,
    })
    out["Missing %"] = (out["Missing"] / n * 100).round(2) if n else 0.0
    out["Completeness %"] = 100 - out["Missing %"]
    return out


def dtype_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Compare actual pandas dtypes with the expected logical type."""
    rows = []
    for col, expected in EXPECTED_BOX_OFFICE_TYPES.items():
        if col not in df.columns:
            rows.append({"Column": col, "Expected": expected, "Actual dtype": "MISSING", "Valid": False})
            continue
        s = df[col]
        if expected == "integer":
            ok = pd.api.types.is_integer_dtype(s)
        elif expected == "numeric":
            ok = pd.api.types.is_numeric_dtype(s)
        else:
            ok = not pd.api.types.is_numeric_dtype(s)
        rows.append({"Column": col, "Expected": expected, "Actual dtype": str(s.dtype), "Valid": bool(ok)})
    return pd.DataFrame(rows)


def box_office_checks(raw: pd.DataFrame) -> list[dict]:
    d = "Box office"
    checks = []
    n = len(raw)

    missing_cols = [c for c in config.BOX_OFFICE_REQUIRED_COLUMNS if c not in raw.columns]
    checks.append(_check(d, "Required columns present", FAIL if missing_cols else PASS, 0,
                         f"Missing: {missing_cols}" if missing_cols else "All 13 expected columns found"))

    rows_with_missing = int(raw.isna().any(axis=1).sum())
    checks.append(_check(d, "Rows with any missing value", WARN if rows_with_missing else PASS, rows_with_missing,
                         "Missing values are confined to metadata columns (genre, rating, votes, language, "
                         "country); revenue fields are complete."
                         if raw[["$Worldwide", "$Domestic", "$Foreign", "Year"]].notna().all().all()
                         else "Some revenue/year fields are missing"))

    exact_dups = int(raw.duplicated().sum())
    checks.append(_check(d, "Exact duplicate rows", WARN if exact_dups else PASS, exact_dups,
                         "Identical records across all columns"))

    ty = raw.duplicated(subset=["Release Group", "Year"], keep=False)
    checks.append(_check(d, "Same title + year appears more than once", WARN if ty.any() else PASS, int(ty.sum()),
                         "Kept: rows differ in revenue, so they are likely distinct films sharing a title"))

    year_ok = raw["Year"].between(1900, 2100)
    checks.append(_check(d, "Year within plausible range", PASS if year_ok.all() else FAIL, int((~year_ok).sum()),
                         f"Observed range {raw['Year'].min()}–{raw['Year'].max()}"))

    rank_ok = raw.groupby("Year")["Rank"].apply(lambda s: s.is_unique and s.min() >= 1)
    checks.append(_check(d, "Rank unique within each year", PASS if rank_ok.all() else WARN,
                         int((~rank_ok).sum()), "Dataset is the top-ranked films per year by worldwide gross"))

    nonpos = raw["$Worldwide"] <= 0
    checks.append(_check(d, "Worldwide revenue > 0", FAIL if nonpos.any() else PASS, int(nonpos.sum()),
                         f"Min ${raw['$Worldwide'].min():,.0f}"))

    neg = (raw["$Domestic"] < 0) | (raw["$Foreign"] < 0)
    checks.append(_check(d, "Domestic / foreign revenue non-negative", FAIL if neg.any() else PASS, int(neg.sum()), ""))

    zero_dom = raw["$Domestic"] == 0
    checks.append(_check(d, "Films with $0 domestic (US) gross", WARN if zero_dom.any() else PASS, int(zero_dom.sum()),
                         "Plausible for non-US releases (no US theatrical run); kept, but means 'Domestic' "
                         "should not be read as 'home market'"))

    diff = (raw["$Domestic"] + raw["$Foreign"] - raw["$Worldwide"]).abs()
    tol = np.maximum(1.0, raw["$Worldwide"] * 0.001)
    bad_sum = diff > tol
    checks.append(_check(d, "Domestic + Foreign ≈ Worldwide (±0.1%)", WARN if bad_sum.any() else PASS,
                         int(bad_sum.sum()), f"Largest gap ${diff.max():,.0f}; source rounding / reporting differences"))

    pct_gap = (raw["Domestic %"] + raw["Foreign %"] - 100).abs()
    bad_pct = pct_gap > 0.2
    checks.append(_check(d, "Domestic % + Foreign % ≈ 100", WARN if bad_pct.any() else PASS, int(bad_pct.sum()),
                         f"Largest deviation {pct_gap.max():.2f} pp"))

    present = raw["Rating"].notna()
    parsed = raw.loc[present, "Rating"].map(parse_rating)
    bad_rating = int(parsed.isna().sum())
    checks.append(_check(d, "Rating parses as 'x/10' within 0–10", FAIL if bad_rating else PASS, bad_rating,
                         "Rating is an audience score stored as text (e.g. '6.126/10') - NOT an MPAA "
                         "certificate. The original model one-hot encodes it as a category."))

    placeholder = (raw["Rating"].map(parse_rating) == 0) & (raw["Vote_Count"].fillna(0) == 0)
    checks.append(_check(d, "Placeholder score (0/10 with 0 votes)", WARN if placeholder.any() else PASS,
                         int(placeholder.sum()), "Treated as 'Not available' in analytics; left unchanged for the "
                         "original model to preserve reproducibility"))

    vc = raw["Vote_Count"].dropna()
    bad_vc = (vc < 0) | (vc % 1 != 0)
    checks.append(_check(d, "Vote_Count non-negative whole number", FAIL if bad_vc.any() else PASS, int(bad_vc.sum()),
                         "Stored as float only because of missing values"))

    lang = raw["Original_Language"].dropna()
    bad_lang = ~lang.str.fullmatch(r"[a-z]{2}")
    checks.append(_check(d, "Language is a 2-letter lowercase code", WARN if bad_lang.any() else PASS,
                         int(bad_lang.sum()), ""))
    cn = int((lang == "cn").sum())
    checks.append(_check(d, "Non-ISO language code 'cn'", WARN if cn else PASS, cn,
                         "'cn' is not ISO 639-1 (Chinese is 'zh'); likely a source-specific code for "
                         "Cantonese. Kept as a separate category."))

    editions = raw["Release Group"].astype(str).str.contains(EDITION_PATTERN)
    checks.append(_check(d, "Re-releases / special editions", WARN if editions.any() else PASS, int(editions.sum()),
                         "Titles such as 'Re-release' or 'Director's Cut' - often lack metadata"))

    mask, _, hi = iqr_outliers(raw["$Worldwide"])
    checks.append(_check(d, "Revenue outliers (IQR, raw scale)", WARN if mask.any() else PASS, int(mask.sum()),
                         f"Above ${hi:,.0f}. Expected for box office (heavy right tail) - kept, not removed"))
    return checks


def review_checks(raw: pd.DataFrame) -> list[dict]:
    d = "Reviews"
    checks = []
    missing = int(raw["review"].isna().sum())
    checks.append(_check(d, "Missing review text", WARN if missing else PASS, missing, "Removed before scoring (original notebook)"))
    blank = int(raw["review"].fillna("x").astype(str).str.isspace().sum())
    checks.append(_check(d, "Whitespace-only reviews", WARN if blank else PASS, blank, "Removed before scoring (original notebook)"))
    valid = raw.dropna(subset=["review"])
    valid = valid[~valid["review"].astype(str).str.isspace()]
    dups = int(valid["review"].duplicated().sum())
    checks.append(_check(d, "Duplicate review texts (after cleaning)", WARN if dups else PASS, dups,
                         "Kept to reproduce the original evaluation; impact on accuracy shown below"))
    bad_label = int((~raw["label"].isin(["pos", "neg"])).sum())
    checks.append(_check(d, "Labels are 'pos' or 'neg'", FAIL if bad_label else PASS, bad_label, ""))
    counts = valid["label"].value_counts()
    balance = counts.min() / counts.max() if len(counts) == 2 else 0
    checks.append(_check(d, "Class balance (min/max ratio)", PASS if balance >= 0.8 else WARN, 0,
                         f"{dict(counts)} → ratio {balance:.2f}"))
    short = int((valid["review"].astype(str).str.len() < 50).sum())
    checks.append(_check(d, "Very short reviews (< 50 chars)", WARN if short else PASS, short,
                         "Too short for reliable lexicon scoring"))
    return checks


def run_all_checks(box_raw: pd.DataFrame, reviews_raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(box_office_checks(box_raw) + review_checks(reviews_raw))


def quality_score(checks: pd.DataFrame) -> dict:
    """Share of checks passing - a simple headline, not a statistical measure."""
    counts = checks["Status"].value_counts()
    total = len(checks)
    return {
        "total": total,
        "pass": int(counts.get(PASS, 0)),
        "warn": int(counts.get(WARN, 0)),
        "fail": int(counts.get(FAIL, 0)),
        "pass_rate": float(counts.get(PASS, 0) / total) if total else 0.0,
    }
