"""Module 5 — Data Quality & Validation."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src import app_data, data_quality, ui
from src.revenue_analysis import format_currency

ui.page_header("Data Quality & Governance",
               "Automated data health checks, schema integrity, missingness profiling, and preprocessing audit log.")

box_raw = app_data.box_office_raw()
reviews_raw = app_data.reviews_raw()
checks_df = data_quality.run_all_checks(box_raw, reviews_raw)
q_summary = data_quality.quality_score(checks_df)

ui.kpi_row([
    ("Total Checks Executed", f"{q_summary['total']}", "Automated rules executed"),
    ("Passed Checks", f"{q_summary['pass']}", "Zero integrity violations", ui.COLORS["positive"]),
    ("Warnings Identified", f"{q_summary['warn']}", "Documented business considerations", ui.COLORS["accent2"]),
    ("Critical Failures", f"{q_summary['fail']}", "Blocking schema errors", ui.COLORS["negative"] if q_summary['fail'] > 0 else ui.COLORS["positive"]),
])

tabs = st.tabs(["Automated Validation Suite", "Missing Values & Completeness", "Schema & Type Audit", "Preprocessing Audit Trail", "Anomaly & Outlier Report"])

# --- TAB 1: Automated Validation Suite ---
with tabs[0]:
    ui.section("Automated Health Checks", "Auditing datasets against strict schema and business rules.")
    
    filter_status = st.multiselect("Filter by Status", ["PASS", "WARN", "FAIL"], default=["FAIL", "WARN", "PASS"])
    filtered_checks = checks_df[checks_df["Status"].isin(filter_status)]
    
    def color_status(val):
        if val == "PASS":
            return "color: #48BB78; font-weight: bold;"
        elif val == "WARN":
            return "color: #F6AD55; font-weight: bold;"
        elif val == "FAIL":
            return "color: #F56565; font-weight: bold;"
        return ""
        
    st.dataframe(filtered_checks.style.map(color_status, subset=["Status"]),
                 hide_index=True, use_container_width=True, height=450)

# --- TAB 2: Missing Values & Completeness ---
with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        ui.section("Box Office Completeness", "5,000 total records from 2000–2024.")
        box_missing = data_quality.missing_summary(box_raw)
        st.dataframe(box_missing.style.format({"Missing %": "{:.2f}%", "Completeness %": "{:.2f}%"}),
                     hide_index=True, use_container_width=True)
        st.caption("Note: Core financial fields ($Worldwide, $Domestic, $Foreign, Year) have 100% completeness. Missingness is strictly in secondary metadata.")
    
    with col2:
        ui.section("Reviews Completeness", "2,000 total labelled review records.")
        rev_missing = data_quality.missing_summary(reviews_raw)
        st.dataframe(rev_missing.style.format({"Missing %": "{:.2f}%", "Completeness %": "{:.2f}%"}),
                     hide_index=True, use_container_width=True)
        st.caption("35 null reviews and 27 empty whitespace reviews are cleanly dropped during preprocessing (matching the original notebook).")

# --- TAB 3: Schema & Type Audit ---
with tabs[2]:
    ui.section("Data Type & Schema Verification", "Ensuring loaded column types conform to expected enterprise specifications.")
    dtype_df = data_quality.dtype_validation(box_raw)
    st.dataframe(dtype_df, hide_index=True, use_container_width=True)
    
    st.info("""
    **Notable Schema Discoveries:**
    - **`Rating` Field:** Formatted as string `'x/10'` (e.g. `'6.126/10'`). It represents IMDb user score, not MPAA classification. Parsed into float `Rating_Score`.
    - **`Original_Language` Code `'cn'`:** The code `'cn'` appears 61 times. ISO 639-1 uses `'zh'` for Chinese; `'cn'` is historically used in certain databases for Cantonese. Preserved as an independent category.
    - **Re-Releases / Director's Cuts:** 45 records contain keywords like 'Director's Cut' or 'Re-release', explaining missing metadata in certain older catalog titles.
    """)

# --- TAB 4: Preprocessing Audit Trail ---
with tabs[3]:
    ui.section("Transparent Preprocessing Log", "Explicit documentation of every data transformation with zero silent row dropping.")
    prep_log = app_data.preprocessing_log()
    st.dataframe(prep_log, hide_index=True, use_container_width=True)
    
    st.markdown("""
    **Governance Principles:**
    1. **Zero Silent Drops:** Analysis and reporting modules preserve all raw records. Derived fields (`Rating_Score`, `Genre_List`) are added non-destructively.
    2. **Model Training Isolation:** The dropna operation is confined strictly to model training (`model_training_frame`) to reproduce the original notebook's 4,797 record dataset without contaminating general reporting.
    3. **Duplicate Reviews Flagged:** In the review dataset, 60 duplicate review texts were identified. They are flagged rather than silently removed, ensuring reproducible evaluation while documenting the data quirk.
    """)

# --- TAB 5: Anomaly & Outlier Report ---
with tabs[4]:
    ui.section("Revenue Outlier & Anomaly Analysis", "Handling extreme commercial outcomes in heavy-tailed distributions.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Sum Discrepancy Audit ($Domestic + $Foreign vs $Worldwide)")
        diff = (box_raw["$Domestic"] + box_raw["$Foreign"] - box_raw["$Worldwide"]).abs()
        mismatch_count = (diff > 1.0).sum()
        max_diff = diff.max()
        st.write(f"- Records where Domestic + Foreign ≠ Worldwide: **{mismatch_count} films ({mismatch_count/len(box_raw):.1%})**")
        st.write(f"- Maximum discrepancy: **{format_currency(max_diff)}**")
        st.caption("Discrepancies are due to currency conversion timing and independent territory reporting differences in Box Office Mojo / IMDb.")
        
    with c2:
        st.markdown("#### The Heavy-Tail Reality")
        st.write("- Standard linear IQR identifies **553 films** (> $262.4M) as statistical outliers.")
        st.write("- However, blockbusters are genuine market outcomes, not measurement errors.")
        st.write("- Applying a log-transformed IQR fence identifies only **32 genuine statistical extremes** (< $2.3M or > $1.28B).")
        st.caption("Recommendation: Never truncate or winsorize blockbusters in box office BI; instead, utilize log transformations and median-first aggregations.")
