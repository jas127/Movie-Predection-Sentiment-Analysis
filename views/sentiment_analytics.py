"""Module 3 — Audience Sentiment Analytics."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import app_data, config, ui
from src import sentiment_analysis as sa
from src.revenue_analysis import correlation

ui.page_header("Audience Sentiment Analytics",
               "Lexicon-based sentiment for 2,000 labelled critic reviews and for movie titles by genre.")

with st.expander("About the method — read before interpreting", expanded=False):
    st.markdown(
        "- **Reviews** are scored with **NLTK VADER**, a *lexicon- and rule-based* model: each word has a "
        "hand-assigned valence and rules adjust for negation, intensifiers ('very'), punctuation and capitals. "
        "It is **not** a transformer or deep-learning model and was not trained on this data.\n"
        "- Classification follows the original notebook: `compound ≥ 0 → positive`, otherwise negative.\n"
        "- **Titles** are scored with **TextBlob polarity** (the original title notebook's method); "
        "VADER on titles is offered for comparison.\n"
        "- **Limitations:** VADER does not understand sarcasm, irony, long-range context (e.g. a review that "
        "praises the cast but concludes the film fails), or film-specific vocabulary ('dark', 'killer', "
        "'gripping'). Long reviews tend to accumulate positive words, which inflates scores.\n"
        "- The review dataset has **no title, genre or revenue fields**, so review sentiment cannot be "
        "broken down by genre or linked to box office. Genre-level sentiment below is title-based only."
    )

reviews = app_data.reviews_scored()
ev = sa.evaluate(reviews)
rep = ev["report"]

ui.kpi_row([
    ("Reviews scored", f"{ev['n']:,}", "after removing missing/blank text"),
    ("Accuracy", f"{ev['accuracy']:.1%}", "vs reviewer labels (balanced classes)", ui.COLORS["accent3"]),
    ("Positive recall", f"{rep['pos']['recall']:.0%}", f"precision {rep['pos']['precision']:.0%}", ui.COLORS["positive"]),
    ("Negative recall", f"{rep['neg']['recall']:.0%}", f"precision {rep['neg']['precision']:.0%}", ui.COLORS["negative"]),
])
st.caption("Note: the original README stated ~85% accuracy; the original notebook output and this app both "
           f"give {ev['accuracy']:.1%} on the same data. The verified figure is used here.")

tabs = st.tabs(["Review sentiment", "Title sentiment by genre", "Analyse your own review", "Sample reviews"])

# ---- Review sentiment ----------------------------------------------------------------
with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        ui.section("Positive / neutral / negative split", "Conventional VADER buckets (±0.05).")
        counts = reviews["sentiment_3way"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
        fig = px.bar(x=counts.index, y=counts.values, color=counts.index, color_discrete_map=ui.SENTIMENT_COLORS,
                     text=counts.values)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title=None)
        fig.update_yaxes(title="Reviews")
        st.plotly_chart(ui.style_fig(fig, 340), use_container_width=True)
    with c2:
        ui.section("Compound score by true label", "Overlap = where the lexicon cannot separate the classes.")
        fig = px.histogram(reviews.assign(Label=reviews["label"].map({"pos": "Positive", "neg": "Negative"})),
                           x="compound", color="Label", nbins=40, barmode="overlay", opacity=0.65,
                           color_discrete_map=ui.SENTIMENT_COLORS)
        fig.add_vline(config.ORIGINAL_CLASSIFICATION_THRESHOLD, line_dash="dash", line_color=ui.COLORS["accent2"],
                      annotation_text="threshold 0", annotation_font_color=ui.COLORS["accent2"])
        fig.update_xaxes(title="VADER compound score")
        fig.update_yaxes(title="Reviews")
        st.plotly_chart(ui.style_fig(fig, 340), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        ui.section("Confusion matrix", "Rows = reviewer label, columns = VADER prediction.")
        cm = ev["confusion"]
        fig = go.Figure(go.Heatmap(z=cm, x=["Pred: neg", "Pred: pos"], y=["True: neg", "True: pos"],
                                   text=cm, texttemplate="%{text}", colorscale=[[0, ui.COLORS["panel"]],
                                                                                [1, ui.COLORS["accent"]]],
                                   showscale=False))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(ui.style_fig(fig, 300), use_container_width=True)
        rows = [{"Class": k, "Precision": rep[k]["precision"], "Recall": rep[k]["recall"],
                 "F1": rep[k]["f1-score"], "Support": int(rep[k]["support"])} for k in ("neg", "pos")]
        st.dataframe(pd.DataFrame(rows).style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}"}),
                     hide_index=True, use_container_width=True)
    with c4:
        ui.section("Threshold explorer", "What if the cut-off were different? (diagnostic, in-sample)")
        t = st.slider("Compound threshold for 'positive'", -0.5, 0.95, 0.0, 0.05, key="thr")
        ev_t = sa.evaluate(reviews, threshold=t)
        st.metric("Accuracy at this threshold", f"{ev_t['accuracy']:.1%}",
                  delta=f"{(ev_t['accuracy'] - ev['accuracy']) * 100:+.1f} pp vs original")
        sweep = sa.threshold_sweep(reviews)
        fig = px.line(sweep, x="Threshold", y="Accuracy", markers=True, color_discrete_sequence=[ui.COLORS["accent3"]])
        fig.add_vline(t, line_dash="dot", line_color=ui.COLORS["accent2"])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(ui.style_fig(fig, 240), use_container_width=True)
        st.caption("Choosing a threshold on the same data used for evaluation overstates performance; a fair "
                   "estimate would need a separate validation split.")

    ui.section("Score components by label", "Mean share of positive / neutral / negative words per review.")
    comp = reviews.groupby("label")[["pos", "neu", "neg", "compound"]].mean().rename(
        index={"pos": "Positive reviews", "neg": "Negative reviews"})
    st.dataframe(comp.style.format("{:.3f}"), use_container_width=True)

# ---- Title sentiment ----------------------------------------------------------------
with tabs[1]:
    titles = app_data.titles_scored()
    ex = app_data.titles_exploded()
    c1, c2 = st.columns([1, 3])
    method = c1.radio("Method", ["TextBlob (original)", "VADER"], key="title_method")
    col = "Title_Sentiment_TextBlob" if method.startswith("TextBlob") else "Title_Sentiment_VADER"
    min_n = c1.number_input("Min films per genre", 1, 500, config.MIN_GROUP_SIZE, key="title_min")
    tg = sa.title_sentiment_by_genre(ex, col, min_films=int(min_n))
    nonzero = (titles[col] != 0).mean()
    rc = correlation(titles, col)
    c1.metric("Titles with any sentiment", f"{nonzero:.0%}")
    c1.metric("Spearman ρ with revenue", f"{rc['spearman']:.2f}")
    with c2:
        if tg.empty:
            ui.empty_state("No genre meets the minimum film count.")
        else:
            d = tg.sort_values("Mean")
            fig = px.bar(d, x="Mean", y="Genre", orientation="h",
                         color=np.where(d["Mean"] >= 0, "Positive", "Negative"),
                         color_discrete_map=ui.SENTIMENT_COLORS,
                         hover_data={"Films": True, "Non_Zero_Share": ":.0%"},
                         title=f"Average title sentiment by genre — {method}")
            fig.update_layout(showlegend=False)
            fig.update_xaxes(title="Mean polarity (−1 to +1)")
            fig.update_yaxes(title=None)
            st.plotly_chart(ui.style_fig(fig, max(340, 26 * len(d))), use_container_width=True)
    st.caption("Most titles are names or nouns ('Gladiator', 'Avatar') and score exactly 0, so genre averages "
               "are small and driven by a minority of titles. Title wording is not audience sentiment.")
    cA, cB = st.columns(2)
    show_cols = ["Title", "Year", "Genres", col]
    cA.markdown("**Most positive titles**")
    cA.dataframe(titles.nlargest(8, col)[show_cols], hide_index=True, use_container_width=True)
    cB.markdown("**Most negative titles**")
    cB.dataframe(titles.nsmallest(8, col)[show_cols], hide_index=True, use_container_width=True)

# ---- Live analysis ----------------------------------------------------------------
with tabs[2]:
    st.markdown("Enter any review text. It is scored with the same VADER analyser used above.")
    examples = {
        "— choose an example —": "",
        "Clear praise": "A stunning, heartfelt film with brilliant performances. I loved every minute.",
        "Clear criticism": "Boring, predictable and far too long. The worst film I have seen this year.",
        "Sarcasm (VADER limitation)": "Oh great, another two hours of my life I'll never get back. Just wonderful.",
        "Mixed / conclusion at end": "The cast is talented and the visuals are beautiful, but the story is a "
                                     "complete mess and ultimately the film fails.",
    }
    pick = st.selectbox("Examples", list(examples), key="example_pick")
    text = st.text_area("Review text", value=examples[pick], height=140, max_chars=20000, key="review_text")
    if st.button("Analyse sentiment", type="primary"):
        try:
            s = sa.score_text(app_data.vader(), text)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            label = s["label_3way"]
            colour = ui.SENTIMENT_COLORS[label]
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                ui.kpi("Overall", label, "VADER 3-way bucket", colour)
            with c2:
                ui.kpi("Compound", f"{s['compound']:+.3f}", "−1 (neg) to +1 (pos)", colour)
            with c3:
                ui.kpi("Positive words", f"{s['pos']:.0%}", "share of text")
            with c4:
                ui.kpi("Neutral words", f"{s['neu']:.0%}", "share of text")
            with c5:
                ui.kpi("Negative words", f"{s['neg']:.0%}", "share of text")
            st.caption(f"Original binary rule would classify this as **{s['label_binary']}**. "
                       "Treat the result as an indicative lexicon score, not an understanding of meaning.")

# ---- Sample reviews ----------------------------------------------------------------
with tabs[3]:
    outcome = st.radio("Show", ["Correctly classified", "False positives (negative reviews scored positive)",
                                "False negatives (positive reviews scored negative)"], horizontal=True,
                       key="sample_outcome")
    if outcome.startswith("Correct"):
        pool = reviews[reviews["correct"]]
    elif outcome.startswith("False positives"):
        pool = reviews[(reviews["label"] == "neg") & (reviews["predicted"] == "pos")]
    else:
        pool = reviews[(reviews["label"] == "pos") & (reviews["predicted"] == "neg")]
    st.caption(f"{len(pool):,} reviews in this group.")
    if pool.empty:
        ui.empty_state("No reviews in this group.")
    else:
        seed = st.number_input("Sample seed", 0, 10_000, 7, key="sample_seed")
        sample = pool.sample(min(5, len(pool)), random_state=int(seed))
        for r in sample.itertuples():
            with st.expander(f"Label: {r.label} · Predicted: {r.predicted} · compound {r.compound:+.3f} · "
                             f"{r.word_count} words"):
                st.write(r.review[:2500] + ("…" if len(r.review) > 2500 else ""))
