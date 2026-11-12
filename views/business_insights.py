"""Module 6 — Business Insights & Executive Reporting."""
import streamlit as st

from src import app_data, business_insights, sentiment_analysis, ui

ui.page_header("Strategic Business Insights & Recommendations",
               "Translating empirical analytical findings and machine learning results into actionable executive guidance.")

# Load core datasets and models
df = app_data.box_office()
reviews = app_data.reviews_scored()
titles_scored = app_data.titles_scored()
titles_exploded = app_data.titles_exploded()
orig_model = app_data.model("original")
pre_model = app_data.model("pre_release")
ev = sentiment_analysis.evaluate(reviews)

all_insights = business_insights.build_all_insights(
    df, titles_scored, titles_exploded, ev, reviews, orig_model.metrics, pre_model.metrics
)

# Executive Summary Section
ui.section("Executive Briefing for Senior Stakeholders",
           "Key empirical findings synthesized for product, commercial, and executive decision-makers.")

exec_bullets = business_insights.executive_summary(all_insights)
for bullet in exec_bullets:
    st.markdown(f"- {bullet}")

st.divider()

# Filter by Evidence Type
c_filter1, c_filter2 = st.columns([2, 3])
with c_filter1:
    selected_type = st.multiselect(
        "Filter by Evidence Rigor",
        ["Descriptive", "Correlational", "Model evaluation", "Methodological"],
        default=["Descriptive", "Correlational", "Model evaluation", "Methodological"]
    )

st.markdown("### Deep-Dive Analytical Investigations")

filtered_insights = [ins for ins in all_insights if any(ins["evidence_type"].startswith(t) for t in selected_type)]

for ins in filtered_insights:
    ui.insight_card(ins)

st.divider()

# Commercial Recommendations Section
ui.section("Strategic & Commercial Recommendations",
           "Pragmatic guidance for film financing, marketing analytics, and technology strategy.")

rec_col1, rec_col2 = st.columns(2)

with rec_col1:
    st.markdown("""
    #### 1. Film Slate Portfolio Strategy
    - **Balance High-Ceiling & High-Variance Genres:** Science Fiction, Adventure, and Fantasy offer the highest median returns but exhibit extreme coefficient of variation ($CV > 1.35$). Slates must be cushioned with consistent mid-tier performers.
    - **Beware the Averages:** Box office data follows a Pareto/power-law distribution. Never base green-light ROI hurdles on genre mean revenues; median benchmarks provide the only honest baseline.
    - **Global-First Financing:** International theatrical gross represents 58–65% of worldwide box office. Projects without strong international localization and appeal face structural ceiling caps.
    """)

with rec_col2:
    st.markdown("""
    #### 2. Analytics & Machine Learning Governance
    - **Reject Leaked Metrics in Pre-Release Forecasting:** IMDb vote count is a post-release consequence of ticket sales. Commercial forecasting algorithms must strictly restrict feature sets to pre-release observables (budget, IP strength, talent track record, trailer engagement velocity).
    - **Contextualize Automated Sentiment (VADER):** Dictionary-based lexicon analyzers misclassify 56% of negative critic reviews because sophisticated critics use nuanced, ironical, or qualified prose. 
    - **Don't Conflate Title Wording with Reception:** Movie title sentiment has zero correlation with financial success. Creative title decisions should be guided by brand recognition and clarity rather than sentiment polarity.
    """)
