"""Shared UI: colour palette, CSS, Plotly template and small components."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

COLORS = {
    "bg": "#0B1220",
    "panel": "#111A2E",
    "border": "#1F2A44",
    "text": "#E5E9F0",
    "muted": "#8B95A7",
    "accent": "#4FD1C5",   # teal - primary
    "accent2": "#F6AD55",  # amber - highlight
    "accent3": "#7F9CF5",  # indigo - secondary series
    "positive": "#48BB78",
    "negative": "#F56565",
    "neutral": "#A0AEC0",
}
SERIES = [COLORS["accent"], COLORS["accent3"], COLORS["accent2"], "#B794F4", "#68D391", "#FC8181", "#63B3ED"]
SENTIMENT_COLORS = {"Positive": COLORS["positive"], "Negative": COLORS["negative"], "Neutral": COLORS["neutral"],
                    "pos": COLORS["positive"], "neg": COLORS["negative"]}

_template = go.layout.Template()
_template.layout = go.Layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color=COLORS["text"], size=13),
    colorway=SERIES,
    xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"], linecolor=COLORS["border"]),
    yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"], linecolor=COLORS["border"]),
    margin=dict(l=10, r=10, t=48, b=10),
    title=dict(font=dict(size=15, color=COLORS["text"]), x=0, xanchor="left"),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.02, yanchor="bottom", x=0),
    hoverlabel=dict(bgcolor=COLORS["panel"], bordercolor=COLORS["border"], font=dict(color=COLORS["text"])),
)
pio.templates["movie_intel"] = _template
pio.templates.default = "plotly_dark+movie_intel"

CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }}
  .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }}
  h1, h2, h3 {{ letter-spacing: -0.01em; }}
  .page-header h1 {{ font-size: 1.75rem; font-weight: 700; margin-bottom: 0.15rem; }}
  .page-header p {{ color: {COLORS['muted']}; margin-top: 0; font-size: 0.95rem; }}
  .kpi {{ background: {COLORS['panel']}; border: 1px solid {COLORS['border']}; border-radius: 10px;
          padding: 16px 18px; height: 100%; border-top: 3px solid {COLORS['accent']}; }}
  .kpi .label {{ color: {COLORS['muted']}; font-size: 0.75rem; text-transform: uppercase;
                 letter-spacing: 0.06em; font-weight: 600; }}
  .kpi .value {{ color: {COLORS['text']}; font-size: 1.6rem; font-weight: 700; margin-top: 4px; }}
  .kpi .sub {{ color: {COLORS['muted']}; font-size: 0.78rem; margin-top: 2px; }}
  .insight {{ background: {COLORS['panel']}; border: 1px solid {COLORS['border']}; border-radius: 10px;
              padding: 16px 18px; margin-bottom: 12px; }}
  .insight .q {{ font-weight: 600; font-size: 0.98rem; margin-bottom: 6px; }}
  .insight .f {{ color: {COLORS['text']}; font-size: 0.92rem; line-height: 1.5; }}
  .insight .c {{ color: {COLORS['muted']}; font-size: 0.8rem; margin-top: 8px; }}
  .tag {{ display: inline-block; font-size: 0.68rem; font-weight: 600; padding: 2px 8px; border-radius: 999px;
          margin-left: 8px; vertical-align: middle; text-transform: uppercase; letter-spacing: 0.05em; }}
  .tag-Descriptive {{ background: rgba(79,209,197,0.15); color: {COLORS['accent']}; }}
  .tag-Correlational {{ background: rgba(127,156,245,0.15); color: {COLORS['accent3']}; }}
  .tag-Model {{ background: rgba(246,173,85,0.15); color: {COLORS['accent2']}; }}
  .tag-Methodological {{ background: rgba(160,174,192,0.15); color: {COLORS['neutral']}; }}
  .section-note {{ color: {COLORS['muted']}; font-size: 0.85rem; margin: -6px 0 10px 0; }}
  [data-testid="stSidebar"] {{ border-right: 1px solid {COLORS['border']}; }}
  div[data-testid="stExpander"] details {{ border-color: {COLORS['border']}; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='page-header'><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>",
                unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", accent: str | None = None) -> None:
    style = f" style='border-top-color:{accent}'" if accent else ""
    st.markdown(
        f"<div class='kpi'{style}><div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{html.escape(value)}</div><div class='sub'>{html.escape(sub)}</div></div>",
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            kpi(*item)


def section(title: str, note: str = "") -> None:
    st.markdown(f"#### {title}")
    if note:
        st.markdown(f"<div class='section-note'>{html.escape(note)}</div>", unsafe_allow_html=True)


def insight_card(ins: dict) -> None:
    tag = ins["evidence_type"].split()[0]
    md = html.escape(ins["finding"])
    st.markdown(
        f"<div class='insight'><div class='q'>{html.escape(ins['question'])}"
        f"<span class='tag tag-{tag}'>{html.escape(ins['evidence_type'])}</span></div>"
        f"<div class='f'>{md}</div><div class='c'>Caveat: {html.escape(ins['caveat'])}</div></div>",
        unsafe_allow_html=True,
    )


def empty_state(message: str = "No records match the current filters. Widen the filters to see results.") -> None:
    st.info(message, icon="🔎")


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height)
    return fig
