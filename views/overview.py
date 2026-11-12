"""Module 1 — Executive Overview."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import app_data, business_insights, config, sentiment_analysis, ui
from src.revenue_analysis import format_currency, revenue_by_genre

ui.page_header("Executive Overview",
               "Headline metrics computed live from the datasets and the reproduced models.")

df = app_data.box_office()
raw_reviews = app_data.reviews_raw()
reviews = app_data.reviews_scored()
original = app_data.model("original")
pre = app_data.model("pre_release")
ev = sentiment_analysis.evaluate(reviews)

n_genres = df["Genre_List"].explode().dropna().nunique()
coverage = df["Has_Full_Metadata"].mean()

ui.kpi_row([
    ("Movie records", f"{len(df):,}", f"{df['Year'].min()}–{df['Year'].max()}, top 200 per year"),
    ("Review records", f"{len(raw_reviews):,}", f"{len(reviews):,} usable after cleaning"),
    ("Genres represented", f"{n_genres}", "multi-genre films counted per genre"),
    ("Avg worldwide revenue", format_currency(df["$Worldwide"].mean()),
     f"median {format_currency(df['$Worldwide'].median())}"),
])
st.write("")
ui.kpi_row([
    ("Revenue model R²", f"{original.metrics['r2']:.3f}", f"held-out test set, n={original.metrics['n_test']}",
     ui.COLORS["accent2"]),
    ("Pre-release model R²", f"{pre.metrics['r2']:.3f}", "without vote count & rating", ui.COLORS["accent2"]),
    ("Sentiment accuracy", f"{ev['accuracy']:.1%}", f"VADER vs reviewer labels, n={ev['n']:,}", ui.COLORS["accent3"]),
    ("Metadata coverage", f"{coverage:.1%}", "films with all model fields present", ui.COLORS["accent3"]),
])

st.write("")
c1, c2 = st.columns(2)
with c1:
    ui.section("Revenue distribution", "Log scale: box office is heavily right-skewed; a linear axis hides most films.")
    fig = px.histogram(df, x=np.log10(df["$Worldwide"]), nbins=50, color_discrete_sequence=[ui.COLORS["accent"]])
    ticks = [1e6, 1e7, 1e8, 1e9]
    fig.update_xaxes(title="Worldwide gross (log scale)", tickvals=np.log10(ticks),
                     ticktext=[format_currency(t, 0) for t in ticks])
    fig.update_yaxes(title="Films")
    fig.add_vline(np.log10(df["$Worldwide"].median()), line_dash="dash", line_color=ui.COLORS["accent2"],
                  annotation_text="median", annotation_font_color=ui.COLORS["accent2"])
    st.plotly_chart(ui.style_fig(fig, 340), use_container_width=True)
with c2:
    ui.section("Genre performance", f"Median worldwide gross; genres with ≥{config.MIN_GROUP_SIZE} films.")
    g = revenue_by_genre(df, config.MIN_GROUP_SIZE).sort_values("Median")
    fig = px.bar(g, x="Median", y="Genre", orientation="h", color_discrete_sequence=[ui.COLORS["accent"]],
                 hover_data={"Films": True, "Mean": ":,.0f", "Median": ":,.0f"})
    fig.update_xaxes(title="Median worldwide gross (USD)", tickformat="$.2s")
    fig.update_yaxes(title=None)
    st.plotly_chart(ui.style_fig(fig, 340), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    ui.section("Review sentiment distribution", "VADER 3-way buckets (compound ≥ 0.05 positive, ≤ −0.05 negative).")
    counts = reviews["sentiment_3way"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
    fig = px.pie(values=counts.values, names=counts.index, hole=0.6, color=counts.index,
                 color_discrete_map=ui.SENTIMENT_COLORS)
    fig.update_traces(textinfo="label+percent")
    st.plotly_chart(ui.style_fig(fig, 320), use_container_width=True)
with c4:
    ui.section("Actual reviewer labels", "Ground truth is balanced; VADER skews positive.")
    comp = (reviews.assign(Source="Reviewer label", Label=reviews["label"].map({"pos": "Positive", "neg": "Negative"}))
            [["Source", "Label"]])
    comp2 = reviews.assign(Source="VADER prediction",
                           Label=reviews["predicted"].map({"pos": "Positive", "neg": "Negative"}))[["Source", "Label"]]
    both = pd.concat([comp, comp2]).groupby(["Source", "Label"]).size()
    both = both.reset_index(name="Reviews")
    fig = px.bar(both, x="Source", y="Reviews", color="Label", barmode="group",
                 color_discrete_map=ui.SENTIMENT_COLORS)
    fig.update_xaxes(title=None)
    st.plotly_chart(ui.style_fig(fig, 320), use_container_width=True)

ui.section("Key findings", "Generated from the calculations above — see Business Insights for detail and caveats.")
insights = business_insights.build_all_insights(
    df, app_data.titles_scored(), app_data.titles_exploded(), ev, reviews, original.metrics, pre.metrics)
for bullet in business_insights.executive_summary(insights):
    st.markdown(f"- {bullet}")
