"""Module 2 — Box Office Revenue Analytics."""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import app_data, ui
from src import revenue_analysis as ra
from src.preprocessing import RATING_BANDS

LANG_NAMES = {"en": "English", "ja": "Japanese", "zh": "Chinese (zh)", "fr": "French", "ko": "Korean",
              "hi": "Hindi", "de": "German", "es": "Spanish", "cn": "cn (source code, likely Cantonese)",
              "it": "Italian", "ru": "Russian", "tr": "Turkish", "te": "Telugu", "sv": "Swedish", "pt": "Portuguese",
              "ta": "Tamil", "da": "Danish", "th": "Thai", "nl": "Dutch", "pl": "Polish"}

ui.page_header("Box Office Revenue Analytics",
               "Explore worldwide gross by genre, year, rating, country and language. All values nominal USD.")

df = app_data.box_office()

# ---- Filters -------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### Filters")
    y0, y1 = int(df["Year"].min()), int(df["Year"].max())
    years = st.slider("Release year", y0, y1, (y0, y1), key="rev_years")
    all_genres = sorted(df["Genre_List"].explode().dropna().unique())
    genres = st.multiselect("Genre (any match)", all_genres, key="rev_genres")
    langs = df["Original_Language"].value_counts().index.tolist()
    languages = st.multiselect("Original language", langs, key="rev_langs",
                               format_func=lambda c: f"{LANG_NAMES.get(c, c)} [{c}]")
    countries_all = df["Country_List"].explode().value_counts().index.tolist()
    countries = st.multiselect("Production country (any match)", countries_all, key="rev_countries")
    bands = st.multiselect("Audience rating band", RATING_BANDS, key="rev_bands",
                           help="Audience score out of 10 from the dataset's 'Rating' column. "
                                "The dataset does not contain MPAA certificates.")

f = ra.filter_movies(df, years, genres, languages, countries, bands)

if f.empty:
    ui.empty_state()
    st.stop()

top = f.loc[f["$Worldwide"].idxmax()]
ui.kpi_row([
    ("Films in selection", f"{len(f):,}", f"of {len(df):,}"),
    ("Total worldwide gross", ra.format_currency(f["$Worldwide"].sum()), "sum across selection"),
    ("Median per film", ra.format_currency(f["$Worldwide"].median()),
     f"mean {ra.format_currency(f['$Worldwide'].mean())}"),
    ("Top grosser", ra.format_currency(top["$Worldwide"]), f"{top['Title']} ({top['Year']})"),
])
st.caption("Selection bias: the dataset only contains the top 200 grossers of each year, so it describes "
           "successful releases, not all films.")

tabs = st.tabs(["Genre", "Release year", "Top films", "Distribution & outliers", "Popularity",
                "Audience rating", "Country & language"])

# ---- Genre ---------------------------------------------------------------------
with tabs[0]:
    c1, c2 = st.columns([1, 3])
    stat = c1.radio("Statistic", ["Median", "Mean", "Total"], key="genre_stat",
                    help="Median is robust to blockbusters; mean is pulled up by a few hits.")
    min_n = c1.number_input("Min films per genre", 1, 500, 10, key="genre_min")
    g = ra.revenue_by_genre(f, min_films=int(min_n))
    with c2:
        if g.empty:
            ui.empty_state("No genre has enough films for this threshold.")
        else:
            g = g.sort_values(stat)
            fig = px.bar(g, x=stat, y="Genre", orientation="h", color_discrete_sequence=[ui.COLORS["accent"]],
                         hover_data={"Films": True, "Median": ":$,.0f", "Mean": ":$,.0f", "Total": ":$,.0f"},
                         title=f"{stat} worldwide gross by genre")
            fig.update_xaxes(title=f"{stat} worldwide gross (USD)", tickformat="$.2s")
            fig.update_yaxes(title=None)
            st.plotly_chart(ui.style_fig(fig, max(320, 26 * len(g))), use_container_width=True)
    if not g.empty:
        ui.section("Spread within genres", "Box = middle 50% of films; log scale so outliers don't flatten the chart.")
        ex = f.explode("Genre_List").rename(columns={"Genre_List": "Genre"}).dropna(subset=["Genre"])
        ex = ex[ex["Genre"].isin(g["Genre"])]
        order = g.sort_values("Median", ascending=False)["Genre"].tolist()
        fig = px.box(ex, x="Genre", y="$Worldwide", category_orders={"Genre": order}, log_y=True,
                     points=False, color_discrete_sequence=[ui.COLORS["accent3"]])
        fig.update_yaxes(title="Worldwide gross (log)", tickformat="$.2s")
        fig.update_xaxes(title=None)
        st.plotly_chart(ui.style_fig(fig, 400), use_container_width=True)
        st.caption("Multi-genre films are counted in every genre they list, so genre totals overlap.")

# ---- Year ----------------------------------------------------------------------
with tabs[1]:
    y = ra.revenue_by_year(f)
    fig = go.Figure()
    fig.add_bar(x=y["Year"], y=y["Total"], name="Total gross", marker_color=ui.COLORS["accent"], opacity=0.85)
    fig.add_scatter(x=y["Year"], y=y["Median"], name="Median per film", yaxis="y2", mode="lines+markers",
                    line=dict(color=ui.COLORS["accent2"], width=2))
    fig.update_layout(title="Worldwide gross by release year",
                      yaxis=dict(title="Total (USD)", tickformat="$.2s"),
                      yaxis2=dict(title="Median (USD)", overlaying="y", side="right", tickformat="$.2s",
                                  showgrid=False))
    st.plotly_chart(ui.style_fig(fig, 420), use_container_width=True)
    st.caption("Note 2020–2021 (COVID-19 cinema closures). Values are nominal - not adjusted for inflation.")
    fs = ra.foreign_share_by_year(f)
    fig = px.line(fs, x="Year", y="Foreign_Share", markers=True, title="Share of gross earned outside the US",
                  color_discrete_sequence=[ui.COLORS["accent3"]])
    fig.update_yaxes(tickformat=".0%", title="Foreign share")
    st.plotly_chart(ui.style_fig(fig, 300), use_container_width=True)

# ---- Top films -------------------------------------------------------------------
with tabs[2]:
    n = st.slider("Number of films", 5, 50, 15, key="top_n")
    t = ra.top_movies(f, n).copy()
    for col in ["$Worldwide", "$Domestic", "$Foreign"]:
        t[col] = t[col].map(lambda v: ra.format_currency(v, 2))
    t["Vote_Count"] = t["Vote_Count"].map(lambda v: "" if np.isnan(v) else f"{v:,.0f}")
    t["Rating_Score"] = t["Rating_Score"].map(lambda v: "" if np.isnan(v) else f"{v:.1f}")
    t.index = t.index + 1
    st.dataframe(t.rename(columns={"Rating_Score": "Score /10", "Vote_Count": "Votes",
                                   "Original_Language": "Lang", "Production_Countries": "Countries"}),
                 use_container_width=True, height=min(600, 38 + 35 * len(t)))

# ---- Distribution ----------------------------------------------------------------
with tabs[3]:
    log = st.toggle("Log scale", value=True, key="dist_log")
    c1, c2 = st.columns([3, 2])
    with c1:
        x = np.log10(f["$Worldwide"]) if log else f["$Worldwide"] / 1e6
        fig = px.histogram(x=x, nbins=50, color_discrete_sequence=[ui.COLORS["accent"]],
                           title="Distribution of worldwide gross")
        if log:
            ticks = [1e6, 1e7, 1e8, 1e9]
            fig.update_xaxes(tickvals=np.log10(ticks), ticktext=[ra.format_currency(v, 0) for v in ticks],
                             title="Worldwide gross (log scale)")
        else:
            fig.update_xaxes(title="Worldwide gross (USD millions)")
        fig.update_yaxes(title="Films")
        st.plotly_chart(ui.style_fig(fig, 380), use_container_width=True)
    with c2:
        mask_raw, _, hi_raw = ra.iqr_outliers(f["$Worldwide"])
        mask_log, lo_log, hi_log = ra.iqr_outliers(f["$Worldwide"], log_scale=True)
        st.markdown("**Outlier detection (Tukey IQR, k = 1.5)**")
        st.markdown(f"- Raw scale: **{int(mask_raw.sum()):,}** films above {ra.format_currency(hi_raw)} "
                    f"({mask_raw.mean():.1%})")
        st.markdown(f"- Log scale: **{int(mask_log.sum()):,}** films outside "
                    f"{ra.format_currency(lo_log)} – {ra.format_currency(hi_log)}")
        top_share = f.nlargest(max(1, len(f) // 10), "$Worldwide")["$Worldwide"].sum() / f["$Worldwide"].sum()
        st.markdown(f"- Top 10% of films earn **{top_share:.0%}** of total gross")
        st.caption("Box office is naturally heavy-tailed, so raw-scale IQR flags many legitimate hits. "
                   "Outliers are reported, never removed.")
        if mask_log.any():
            st.dataframe(f.loc[mask_log, ["Title", "Year", "$Worldwide"]].sort_values("$Worldwide", ascending=False)
                         .assign(**{"$Worldwide": lambda d: d["$Worldwide"].map(ra.format_currency)}),
                         hide_index=True, use_container_width=True, height=220)

# ---- Popularity --------------------------------------------------------------------
with tabs[4]:
    corr = ra.correlation(f, "Vote_Count")
    plot = f[(f["Vote_Count"] > 0)].dropna(subset=["Vote_Count"])
    excluded = len(f) - len(plot)
    c1, c2, c3 = st.columns(3)
    c1.metric("Spearman ρ (votes vs gross)", "n/a" if np.isnan(corr["spearman"]) else f"{corr['spearman']:.2f}")
    c2.metric("Pearson r (log–log)", "n/a" if np.isnan(corr["pearson_log"]) else f"{corr['pearson_log']:.2f}")
    c3.metric("Films used", f"{corr['n']:,}")
    if plot.empty:
        ui.empty_state("No films with vote counts in this selection.")
    else:
        fig = px.scatter(plot, x="Vote_Count", y="$Worldwide", log_x=True, log_y=True, opacity=0.55,
                         color="Rating_Band", category_orders={"Rating_Band": RATING_BANDS},
                         hover_name="Title", hover_data={"Year": True, "$Worldwide": ":$,.0f"},
                         title="Audience vote count vs worldwide gross")
        fig.update_xaxes(title="Vote count (log)")
        fig.update_yaxes(title="Worldwide gross (log)", tickformat="$.2s")
        st.plotly_chart(ui.style_fig(fig, 480), use_container_width=True)
    st.caption(f"{excluded:,} films without votes (missing or 0) are hidden from the log–log plot. "
               "Correlation ≠ causation: votes accumulate after release, largely as a consequence of reach.")

# ---- Rating ----------------------------------------------------------------------------
with tabs[5]:
    st.info("The original README describes this column as an MPAA rating, but the data actually holds an "
            "**audience score out of 10** (e.g. '6.126/10'). MPAA (G/PG/PG-13/R) comparisons are therefore not "
            "possible with this dataset; score bands are shown instead.", icon="ℹ️")
    rb = ra.revenue_by_column(f, "Rating_Band")
    order = [b for b in RATING_BANDS if b in set(f["Rating_Band"])]
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.box(f, x="Rating_Band", y="$Worldwide", log_y=True, category_orders={"Rating_Band": order},
                     points=False, color_discrete_sequence=[ui.COLORS["accent"]],
                     title="Worldwide gross by audience score band")
        fig.update_yaxes(title="Worldwide gross (log)", tickformat="$.2s")
        fig.update_xaxes(title="Audience score")
        st.plotly_chart(ui.style_fig(fig, 400), use_container_width=True)
    with c2:
        show = rb.set_index("Rating_Band").reindex(order)[["Films", "Median", "Mean"]].reset_index()
        show["Median"] = show["Median"].map(ra.format_currency)
        show["Mean"] = show["Mean"].map(ra.format_currency)
        st.dataframe(show, hide_index=True, use_container_width=True)
        rc = ra.correlation(f[f["Rating_Band"] != "Not available"], "Rating_Score")
        if not np.isnan(rc["spearman"]):
            st.metric("Spearman ρ (score vs gross)", f"{rc['spearman']:.2f}", help=f"n = {rc['n']:,}")

# ---- Country & language ----------------------------------------------------------------
with tabs[6]:
    c1, c2 = st.columns(2)
    with c1:
        ct = ra.revenue_by_country(f, top_n=15)
        if ct.empty:
            ui.empty_state("No country data in this selection.")
        else:
            fig = px.bar(ct.sort_values("Total"), x="Total", y="Country", orientation="h",
                         hover_data={"Films": True, "Median": ":$,.0f"},
                         color_discrete_sequence=[ui.COLORS["accent"]],
                         title="Top production countries by total gross")
            fig.update_xaxes(title="Total gross (USD)", tickformat="$.2s")
            fig.update_yaxes(title=None)
            st.plotly_chart(ui.style_fig(fig, 480), use_container_width=True)
            st.caption("Co-productions count fully toward each listed country.")
    with c2:
        lg = ra.revenue_by_column(f, "Original_Language").sort_values("Total", ascending=False).head(12)
        lg = lg.dropna(subset=["Original_Language"])
        if lg.empty:
            ui.empty_state("No language data in this selection.")
        else:
            lg["Language"] = lg["Original_Language"].map(lambda c: LANG_NAMES.get(c, c))
            fig = px.bar(lg.sort_values("Total"), x="Total", y="Language", orientation="h",
                         hover_data={"Films": True, "Median": ":$,.0f"},
                         color_discrete_sequence=[ui.COLORS["accent3"]],
                         title="Top original languages by total gross")
            fig.update_xaxes(title="Total gross (USD)", tickformat="$.2s")
            fig.update_yaxes(title=None)
            st.plotly_chart(ui.style_fig(fig, 480), use_container_width=True)

