"""Movie Intelligence — Business Analytics & Insights Platform.

Run locally:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Movie Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src import ui  # noqa: E402  (import after set_page_config)
from src.data_loader import DataLoadError  # noqa: E402

ui.inject_css()

pages = {
    "Dashboard": [
        st.Page("views/overview.py", title="Executive Overview", icon=":material/dashboard:", default=True),
        st.Page("views/business_insights.py", title="Business Insights", icon=":material/lightbulb:"),
    ],
    "Analytics": [
        st.Page("views/revenue_analytics.py", title="Box Office Revenue", icon=":material/payments:"),
        st.Page("views/sentiment_analytics.py", title="Audience Sentiment", icon=":material/sentiment_satisfied:"),
        st.Page("views/prediction.py", title="Revenue Prediction", icon=":material/model_training:"),
    ],
    "Governance": [
        st.Page("views/data_quality.py", title="Data Quality", icon=":material/fact_check:"),
    ],
}

with st.sidebar:
    st.markdown("### 🎬 Movie Intelligence")
    st.caption("Business Analytics & Insights Platform")

nav = st.navigation(pages)

with st.sidebar:
    st.divider()
    st.caption(
        "Data: top-200 worldwide grossers per year, 2000–2024 (5,000 films) and 2,000 labelled critic reviews. "
        "Revenue figures are nominal USD (not inflation-adjusted)."
    )

try:
    nav.run()
except DataLoadError as exc:
    st.error(f"Data could not be loaded: {exc}")
    st.stop()
