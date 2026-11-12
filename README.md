<div align="center">

# 🎬 Movie Intelligence
### Enterprise Business Analytics & Predictive Insights Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)
[![Tests Passing](https://img.shields.io/badge/pytest-11%20passed-success.svg?logo=pytest&logoColor=white)](tests/test_platform.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*An end-to-end transformation of an academic machine learning study into a production-grade business intelligence and predictive auditing platform for the entertainment industry.*

[Key Features](#-key-features) • [Platform Architecture](#-technical-architecture) • [Empirical Audit & Findings](#-empirical-audit--analytical-findings) • [Quickstart](#-local-installation--quickstart) • [Streamlit Cloud Deployment](#-free-deployment-streamlit-community-cloud)

</div>

---

## 📌 Executive Summary & Business Context

The theatrical film industry is a hit-driven, capital-intensive market defined by extreme revenue skew, long feedback loops, and high financial risk. Studio executives, content acquisitions teams, and financial sponsors frequently struggle with four fundamental questions:

1. **Portfolio Slate Risk:** Which movie genres offer consistent commercial floors versus extreme hit-driven variance?
2. **International Revenue Exposure:** How critical are non-US theatrical revenues relative to domestic gross over time?
3. **Sentiment as a Signal:** Does natural language processing of viewer sentiment reliably correlate with box office gross?
4. **Predictive Model Rigor:** Can machine learning algorithms reliably forecast worldwide box office prior to release using metadata?

**Movie Intelligence** answers these questions by pairing rich data visualization with **methodological auditing**. Rather than blindly trusting machine learning metrics, this platform evaluates data quality, audits predictive pipelines for post-release information leakage, and translates findings into clear executive guidance.

---

## 🎯 Key Features

```
┌────────────────────────────────────────────────────────────────────────┐
│                      🎬 MOVIE INTELLIGENCE PLATFORM                    │
├─────────────────────┬───────────────────────┬──────────────────────────┤
│ 📊 Executive Dash   │ 💰 Revenue Analytics  │ 💬 Audience Sentiment    │
│ • Live KPI Suite    │ • Slicing & Dicing    │ • VADER Evaluation       │
│ • Skew Distribution │ • Medians & IQRs      │ • TextBlob Genre Polarity│
│ • Model Highlights  │ • 25-Year Trends      │ • Interactive Scorer     │
├─────────────────────┼───────────────────────┼──────────────────────────┤
│ 🤖 Machine Learning │ 🛡️ Data Quality Suite │ 💡 Business Insights     │
│ • RF Pipeline Fit   │ • 17 Automated Rules  │ • Evidence Rigor Tags    │
│ • Leakage Proof     │ • Zero Silent Drops   │ • Slate Risk Strategies  │
│ • Tree Uncertainty  │ • Reconciliation Audit│ • Executive Briefing     │
└─────────────────────┴───────────────────────┴──────────────────────────┘
```

### 1. Executive Overview
- **Real-Time KPIs:** Live calculations for 5,000 top-200 box office titles (2000–2024), 2,000 labelled reviews, 19 distinct genres, average ($119.2M) and median ($52.4M) worldwide revenues, and metadata coverage.
- **Heavy-Tail Distributions:** Dual-axis log-scale distribution visualizing commercial clustering and blockbuster skew.
- **Sentiment Alignment:** Compares ground-truth critic polarity against dictionary-based classification predictions.

### 2. Box Office Revenue Analytics
- **Multi-Dimensional Filters:** Dynamic slicing across release year range, genres, original languages, production countries, and audience score tiers.
- **Median-First Metric Framework:** Reports median gross alongside interquartile ranges (IQR) and coefficients of variation ($CV$) to prevent megahits from skewing benchmarks.
- **Dual Outlier Detection:** Tukey outlier detection on raw linear scales (553 statistical outliers) versus log-transformed scales (32 genuine extremes).
- **International Market Trends:** Tracks international ("Foreign") theatrical share over 25 years (rising from 45% to over 65% in peak international periods).

### 3. Audience Sentiment Analytics
- **VADER Benchmark:** Evaluates VADER across 1,938 clean critic reviews, documenting verified $63.57\%$ accuracy against ground truth.
- **Threshold Diagnostic Explorer:** Interactive compound score threshold slider analyzing trade-offs between precision and recall.
- **Genre-Wise Title Sentiment:** Explodes multi-genre films and scores titles via TextBlob polarity and VADER compound scores.
- **Live Sentiment Tester:** Test custom reviews with transparency regarding lexicon limitations (irony, sarcasm, and domain-specific vocabulary).

### 4. Predictive Modeling & Information Leakage Audit
- **Original Random Forest Pipeline:** Exact reproduction of the original pipeline ($R^2 = 0.5279$, $\text{RMSE} = \$137.3\text{M}$).
- **The Information Leakage Proof:** Trains an identical pre-release model excluding post-release signals (`Vote_Count` and `Rating`). Demonstrates that removing post-release votes drops $R^2$ from **0.528 to 0.227**, proving the original model's predictive power is an artifact of post-release popularity.
- **Feature Importance:** Permutation importance on the held-out test set demonstrates `Vote_Count` accounts for almost all predictive utility.
- **Interactive Prediction Simulator:** Input validation guarding against unseen categories and negative values, accompanied by tree ensemble percentile spreads.

### 5. Data Quality & Governance
- **Automated Validation Suite:** Automated checks covering schema conformity, range validity, balance, and missingness.
- **Transparent Preprocessing Audit Trail:** Explicitly logs every transformation, ensuring zero silent row deletion.
- **Financial Reconciliation:** Identifies rounding discrepancies between domestic/foreign sums and worldwide totals.

### 6. Business Insights & Strategic Recommendations
- **Categorized Evidence Rigor:** Every insight is tagged as *Descriptive*, *Correlational*, *Model Evaluation*, or *Methodological* to prevent confusing association with causation.
- **Commercial Slate Recommendations:** Pragmatic guidance for studio slate risk management and ML production deployment.

---

## 🔬 Empirical Audit & Analytical Findings

| Analytical Dimension | Academic Notebook Claim | Verified Empirical Finding | Business & Technical Implications |
| :--- | :--- | :--- | :--- |
| **VADER Review Accuracy** | Claimed "~85%" | **63.57%** (1,232 / 1,938 correct) | VADER exhibits severe positive bias; negative recall is only 44.1%, frequently misclassifying nuanced critic reviews. |
| **Revenue Model Fit ($R^2$)** | Claimed "~0.528" | **0.5279** (Test Split), **0.5327** (5-Fold CV Mean) | Verified exact reproduction on identical 80/20 train/test split. |
| **Pre-Release Fit ($R^2$)** | Not tested | **0.2274** (Test Split), **0.1041** (5-Fold CV Mean) | Removing `Vote_Count` drops explanatory power by over 30 pp, proving the model cannot forecast before release. |
| **Audience Votes vs Gross** | Core feature | Spearman $\rho = 0.679$, Pearson log-log $r = 0.626$ | Strong correlation driven by post-release exposure, not pre-release causality. |
| **Title Sentiment vs Gross** | Implied predictive value | Spearman $\rho = -0.042$ ($p > 0.05$) | Movie title wording has zero correlation with theatrical box office revenue. |

---

## 🏗 Technical Architecture

```
Movie-Predection-Sentiment-Analysis/
├── app.py                      # Application entrypoint with st.navigation
├── requirements.txt            # Streamlit, scikit-learn, plotly, nltk, textblob, pytest
├── README.md                   # Enterprise technical documentation
├── data/                       # Preserved datasets
│   ├── box_office_data.csv     # 5,000 top-200 worldwide grossers (2000–2024)
│   └── moviereviews.tsv        # 2,000 labelled critic reviews (pos/neg)
├── docs/original_reports/       # Preserved original PDF research reports
├── nltk_data/                  # Self-contained NLTK resources (vader_lexicon)
├── notebooks/                  # Path-corrected original notebooks
├── reports/model_metrics.json   # 5-fold cross-validation output report
├── scripts/evaluate_models.py   # Cross-validation & benchmark evaluation script
├── src/                        # UI-independent business logic
│   ├── config.py               # Schemas, features, and relative paths
│   ├── data_loader.py          # Data ingestion with schema validation
│   ├── preprocessing.py        # Non-destructive cleaning & PreprocessingLog
│   ├── revenue_analysis.py     # Aggregations, medians, IQR fences, correlations
│   ├── sentiment_analysis.py   # VADER, TextBlob, and confusion matrix evaluators
│   ├── prediction.py           # RandomForest pipelines & leakage benchmarks
│   ├── data_quality.py         # Automated validation suite (PASS / WARN / FAIL)
│   ├── business_insights.py    # Live calculated insights with evidence tags
│   ├── app_data.py             # Streamlit caching layer
│   └── ui.py                   # Theming, CSS, and chart styling
├── tests/test_platform.py       # Automated pytest suite (11 unit & integration tests)
└── views/                      # Streamlit page views
    ├── overview.py             # Executive dashboard & real-time KPIs
    ├── revenue_analytics.py    # Multi-dimensional box office analytics & filters
    ├── sentiment_analytics.py  # VADER review & title sentiment analytics
    ├── prediction.py           # Prediction simulator & leakage analysis
    ├── data_quality.py         # Data governance & validation audit logs
    └── business_insights.py    # Strategic recommendations & evidence tagging
```

---

## ⚡ Local Installation & Quickstart

### Prerequisites
- Python 3.10 to 3.13
- Git

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/jas127/Movie-Predection-Sentiment-Analysis.git
cd Movie-Predection-Sentiment-Analysis

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
```bash
PYTHONPATH=. pytest -v tests/test_platform.py
```
*Expected output: `11 passed in 10.73s`.*

### 3. Launch the Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Free Deployment (Streamlit Community Cloud)

This repository is optimized for one-click deployment on **Streamlit Community Cloud**:

1. Fork or push this repository to your GitHub account.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub.
3. Select your repository and branch (`main`).
4. Set the **Main file path** to `app.py`.
5. Click **Deploy**. Streamlit Cloud will automatically install dependencies and launch the platform.

---

## 💡 Interview Discussion Guide

When presenting this project for **Data Analyst**, **Business Analyst**, **Product Analyst**, or **Technology Consultant** interviews:

- **Data Integrity & Governance:** Discuss how you implemented non-destructive preprocessing and automated data validation rules rather than silently dropping imperfect rows.
- **Critical Model Auditing:** Explain how identifying post-release information leakage in `Vote_Count` showed that a model with $R^2 = 0.528$ was actually explaining popularity rather than forecasting success.
- **Empirical Rigor in NLP:** Share how verifying VADER's real-world accuracy on long-form reviews ($63.6\%$ vs. the claimed $85\%$) highlighted the difference between dictionary matching and semantic comprehension.
- **Stakeholder Communication:** Highlight the use of evidence-tagging (*Descriptive*, *Correlational*, *Model Evaluation*) to help non-technical stakeholders distinguish observed patterns from causal claims.

---

## 📄 License & Attribution

- **License:** MIT License
- **Author:** Jas
- **References:**
  - Hutto, C.J. & Gilbert, E.E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.*
  - Datasets sourced from Box Office Mojo & IMDb.
