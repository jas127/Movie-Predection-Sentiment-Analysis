"""Module 4 — Revenue Prediction."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import app_data, config, prediction, ui
from src.revenue_analysis import format_currency

ui.page_header("Box Office Revenue Prediction",
               "Exploratory machine learning using the project's original RandomForestRegressor pipeline.")

with st.expander("Methodology & Critical Limitations — Please Read", expanded=True):
    st.markdown("""
    **Core Methodology & Information Leakage Warning:**
    - **Original Pipeline:** Uses `Year`, `Genres`, `Rating` (audience score), `Vote_Count`, `Original_Language`, and `Production_Countries`.
    - **CRITICAL LIMITATION (Information Leakage):** `Vote_Count` and `Rating` accumulate **after** a movie's theatrical release. In fact, `Vote_Count` is a proxy for audience reach/popularity after release. Using post-release metrics to predict box office creates significant information leakage and **cannot serve as a commercial pre-release forecasting model**.
    - **Pre-Release Baseline Model:** To quantify this leakage, we train a comparative model using only metadata available prior to release (`Year`, `Genres`, `Original_Language`, `Production_Countries`). Notice the drastic reduction in explanatory power ($R^2$), proving that the original model's performance relies predominantly on post-release signals.
    - **Exploratory Nature:** These models omit crucial commercial drivers such as production budget, marketing spend, screen count, star power, and franchise IP. Predictions should be interpreted as academic exercises, not commercial guidance.
    """)

model_df = app_data.model_frame()
orig_model = app_data.model("original")
pre_model = app_data.model("pre_release")

ui.kpi_row([
    ("Original Model R²", f"{orig_model.metrics['r2']:.3f}", f"MSE: {orig_model.metrics['mse']:.2e}", ui.COLORS["accent"]),
    ("Original Test RMSE", format_currency(orig_model.metrics['rmse']), f"MAE: {format_currency(orig_model.metrics['mae'])}"),
    ("Pre-Release Model R²", f"{pre_model.metrics['r2']:.3f}", "Excludes Vote_Count & Rating", ui.COLORS["accent2"]),
    ("Pre-Release Test RMSE", format_currency(pre_model.metrics['rmse']), f"MAE: {format_currency(pre_model.metrics['mae'])}"),
])

tabs = st.tabs(["Interactive Predictor", "Model Evaluation & Diagnostics", "Feature Importance", "Pre-Release Comparison"])

# --- TAB 1: Interactive Predictor ---
with tabs[0]:
    ui.section("Exploratory Prediction Simulator", "Test single-movie inputs against the trained Random Forest model.")
    
    st.info("💡 **Pre-fill with Sample Data:** The original notebook provided three reference test cases.", icon="ℹ️")
    sample_options = {
        "Custom Input": None,
        "Sample 1: Romance (1998, R, 12 votes, South Africa)": {
            "Year": 1998, "Genres": "Romance", "Rating": "6.0/10", "Vote_Count": 12, "Original_Language": "en", "Production_Countries": "South Africa"
        },
        "Sample 2: Action Blockbuster (2012, PG-13, 50,000 votes, US)": {
            "Year": 2012, "Genres": "Action", "Rating": "7.5/10", "Vote_Count": 50000, "Original_Language": "en", "Production_Countries": "United States of America"
        },
        "Sample 3: Comedy (2005, G, 230 votes, India)": {
            "Year": 2005, "Genres": "Comedy", "Rating": "6.5/10", "Vote_Count": 230, "Original_Language": "en", "Production_Countries": "India"
        }
    }
    selected_sample = st.selectbox("Select Preset or Custom", list(sample_options.keys()))
    preset = sample_options[selected_sample]

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            in_year = st.number_input("Release Year", min_value=1950, max_value=2030, 
                                      value=preset["Year"] if preset else 2015, step=1)
            unique_genres = sorted(model_df["Genres"].dropna().unique().tolist())
            def_genre_idx = unique_genres.index(preset["Genres"]) if preset and preset["Genres"] in unique_genres else 0
            in_genre = st.selectbox("Genre Combination", unique_genres, index=def_genre_idx)
        
        with col2:
            in_votes = st.number_input("IMDb Vote Count (Post-Release)", min_value=0, max_value=2000000, 
                                       value=preset["Vote_Count"] if preset else 25000, step=500,
                                       help="⚠️ Note: High vote counts represent post-release popularity.")
            unique_ratings = sorted(model_df["Rating"].dropna().unique().tolist())
            def_rat_idx = unique_ratings.index(preset["Rating"]) if preset and preset["Rating"] in unique_ratings else 0
            in_rating = st.selectbox("Audience Rating (x/10 format)", unique_ratings, index=def_rat_idx)
            
        with col3:
            unique_langs = sorted(model_df["Original_Language"].dropna().unique().tolist())
            def_lang_idx = unique_langs.index(preset["Original_Language"]) if preset and preset["Original_Language"] in unique_langs else 0
            in_lang = st.selectbox("Original Language", unique_langs, index=def_lang_idx)
            
            unique_countries = sorted(model_df["Production_Countries"].dropna().unique().tolist())
            def_country_idx = unique_countries.index(preset["Production_Countries"]) if preset and preset["Production_Countries"] in unique_countries else 0
            in_country = st.selectbox("Production Country", unique_countries, index=def_country_idx)
            
        submit_btn = st.form_submit_button("Generate Prediction", type="primary")

    if submit_btn or preset is not None:
        user_inputs = {
            "Year": in_year,
            "Genres": in_genre,
            "Rating": in_rating,
            "Vote_Count": in_votes,
            "Original_Language": in_lang,
            "Production_Countries": in_country
        }
        
        errors, warnings = prediction.validate_prediction_input(user_inputs, model_df, orig_model.features)
        
        for w in warnings:
            st.warning(f"⚠️ {w}")
            
        if errors:
            for e in errors:
                st.error(f"❌ {e}")
        else:
            pred_orig = prediction.predict(orig_model, user_inputs)
            pred_pre = prediction.predict(pre_model, user_inputs)
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                ui.kpi("Predicted Worldwide Gross (Original Model)", 
                       format_currency(pred_orig["prediction"]), 
                       f"10th–90th Tree Spread: {format_currency(pred_orig['p10'])} – {format_currency(pred_orig['p90'])}",
                       ui.COLORS["accent"])
            with res_col2:
                ui.kpi("Predicted Gross (Pre-Release Model Only)", 
                       format_currency(pred_pre["prediction"]), 
                       f"10th–90th Tree Spread: {format_currency(pred_pre['p10'])} – {format_currency(pred_pre['p90'])}",
                       ui.COLORS["accent2"])
                
            st.caption("Note: The tree percentile spread reflects ensemble disagreement across random trees, not a calibrated statistical confidence interval.")

# --- TAB 2: Model Evaluation & Diagnostics ---
with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        ui.section("Actual vs. Predicted Revenue (Test Set)", "Evaluated on 20% held-out test data (960 movies).")
        eval_df = pd.DataFrame({
            "Actual": orig_model.y_test,
            "Predicted": orig_model.y_pred
        })
        fig_scatter = px.scatter(eval_df, x="Actual", y="Predicted", opacity=0.5,
                                 color_discrete_sequence=[ui.COLORS["accent"]])
        min_val = min(eval_df["Actual"].min(), eval_df["Predicted"].min())
        max_val = max(eval_df["Actual"].max(), eval_df["Predicted"].max())
        fig_scatter.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                              line=dict(color=ui.COLORS["negative"], dash="dash"))
        fig_scatter.update_xaxes(title="Actual Worldwide Revenue (USD)", tickformat="$.2s")
        fig_scatter.update_yaxes(title="Predicted Worldwide Revenue (USD)", tickformat="$.2s")
        st.plotly_chart(ui.style_fig(fig_scatter, 380), use_container_width=True)
        
    with c2:
        ui.section("Residual Distribution", "Residuals = Actual - Predicted.")
        eval_df["Residual"] = eval_df["Actual"] - eval_df["Predicted"]
        fig_resid = px.histogram(eval_df, x="Residual", nbins=50,
                                 color_discrete_sequence=[ui.COLORS["accent3"]])
        fig_resid.add_vline(x=0, line_dash="dash", line_color=ui.COLORS["accent2"])
        fig_resid.update_xaxes(title="Residual Error (USD)", tickformat="$.2s")
        fig_resid.update_yaxes(title="Count")
        st.plotly_chart(ui.style_fig(fig_resid, 380), use_container_width=True)

    st.markdown("#### Detailed Performance Metrics")
    metrics_table = pd.DataFrame([
        {"Metric": "R² Score (Coefficient of Determination)", "Value": f"{orig_model.metrics['r2']:.4f}", "Interpretation": "Explains ~52.8% of variance in worldwide gross"},
        {"Metric": "Mean Squared Error (MSE)", "Value": f"{orig_model.metrics['mse']:.4e}", "Interpretation": "Reflects high variance across wide revenue distributions"},
        {"Metric": "Root Mean Squared Error (RMSE)", "Value": format_currency(orig_model.metrics['rmse']), "Interpretation": "Penalizes large errors on multi-billion dollar blockbusters"},
        {"Metric": "Mean Absolute Error (MAE)", "Value": format_currency(orig_model.metrics['mae']), "Interpretation": "Average deviation across all test predictions"},
        {"Metric": "Median Absolute Error", "Value": format_currency(orig_model.metrics['median_abs_error']), "Interpretation": "Typical error for middle-tier releases"}
    ])
    st.dataframe(metrics_table, hide_index=True, use_container_width=True)

# --- TAB 3: Feature Importance ---
with tabs[2]:
    c_imp1, c_imp2 = st.columns(2)
    with c_imp1:
        ui.section("Grouped Impurity Importance (Gini)", "Aggregates one-hot dummy features back to raw categorical columns.")
        gini_df = prediction.grouped_impurity_importance(orig_model)
        fig_gini = px.bar(gini_df, x="Importance", y="Feature", orientation="h",
                          color_discrete_sequence=[ui.COLORS["accent"]],
                          hover_data={"Encoded columns": True})
        fig_gini.update_layout(yaxis=dict(autorange="reversed"))
        fig_gini.update_xaxes(title="Feature Importance")
        st.plotly_chart(ui.style_fig(fig_gini, 340), use_container_width=True)
        st.caption("⚠️ Note: Gini impurity importance can be biased toward high-cardinality categorical variables.")
        
    with c_imp2:
        ui.section("Permutation Importance on Test Set", "Measures drop in test R² when a feature column is randomly shuffled.")
        perm_df = app_data.permutation_importance("original")
        fig_perm = px.bar(perm_df, x="R2 drop (mean)", y="Feature", orientation="h",
                          error_x="R2 drop (std)", color_discrete_sequence=[ui.COLORS["accent3"]])
        fig_perm.update_layout(yaxis=dict(autorange="reversed"))
        fig_perm.update_xaxes(title="Decrease in Test R²")
        st.plotly_chart(ui.style_fig(fig_perm, 340), use_container_width=True)
        st.caption("Demonstrates conclusively that shuffling `Vote_Count` degrades test R² by nearly 1.0.")

# --- TAB 4: Pre-Release Comparison ---
with tabs[3]:
    ui.section("The Information Leakage Proof: Pre-Release vs Full Model", 
               "Direct comparison demonstrating why the model cannot be deployed before movie theatrical debut.")
    
    comp_df = pd.DataFrame([
        {"Feature Set": "Original Pipeline (Full)", "Included Features": "Year, Genres, Rating, Vote_Count, Language, Countries", 
         "Test R²": orig_model.metrics['r2'], "RMSE": format_currency(orig_model.metrics['rmse']), "Leakage Risk": "HIGH (Post-release data)"},
        {"Feature Set": "Pre-Release Model (Legitimate)", "Included Features": "Year, Genres, Language, Countries", 
         "Test R²": pre_model.metrics['r2'], "RMSE": format_currency(pre_model.metrics['rmse']), "Leakage Risk": "NONE (Available pre-debut)"}
    ])
    st.dataframe(comp_df, hide_index=True, use_container_width=True)
    
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(name="Original Model (with Post-Release Leakage)", x=["R² Score"], y=[orig_model.metrics['r2']], marker_color=ui.COLORS["accent"]))
    fig_comp.add_trace(go.Bar(name="Pre-Release Model (True Forecasting Baseline)", x=["R² Score"], y=[pre_model.metrics['r2']], marker_color=ui.COLORS["accent2"]))
    fig_comp.update_layout(barmode='group', title="R² Drop When Removing Post-Release Leaked Features", yaxis_title="R² Score", yaxis_range=[0, 0.7])
    st.plotly_chart(ui.style_fig(fig_comp, 320), use_container_width=True)
    
    st.markdown("""
    ### Takeaways for Technical Interview / Business Evaluation:
    1. **52.8% vs 22.7% R²:** When stripping out post-release audience engagement indicators (`Vote_Count`), the predictive power drops by over 30 percentage points.
    2. **Causal Direction:** Box office success drives high IMDb vote counts, not the other way around. Treating vote counts as an independent predictive variable in forecasting is conceptually flawed.
    3. **Production Recommendation:** A production-grade pre-release forecasting tool would require pre-release data assets: production budget, marketing/P&A spend, trailer views & engagement velocity, theater booking counts, franchise intellectual property tier, and competitive release calendar density.
    """)
