"""Revenue prediction - reuses the original RandomForestRegressor pipeline.

``variant='original'`` reproduces Movie-Revenue-Prediction.ipynb exactly
(same features, preprocessing, hyper-parameters, 80/20 split, random_state=42).

``variant='pre_release'`` is a comparison model trained on the *same split*
but without Vote_Count and Rating, which are only known after release. The gap
between the two quantifies how much of the original R² depends on
post-release (leakage-prone) information.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

VARIANTS = {
    "original": (config.NUMERICAL_FEATURES, config.CATEGORICAL_FEATURES),
    "pre_release": (config.PRE_RELEASE_NUMERICAL, config.PRE_RELEASE_CATEGORICAL),
}


def build_pipeline(numerical: list[str], categorical: list[str]) -> Pipeline:
    """Same structure as the original notebook."""
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE)),
    ])


@dataclass
class ModelResult:
    variant: str
    pipeline: Pipeline
    numerical: list[str]
    categorical: list[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray
    metrics: dict = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        return self.numerical + self.categorical


def split_data(model_df: pd.DataFrame):
    """The original split: X has MODEL_FEATURES columns in the original order."""
    X = model_df[config.MODEL_FEATURES]
    y = model_df[config.TARGET]
    return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)


def train_and_evaluate(model_df: pd.DataFrame, variant: str = "original") -> ModelResult:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(VARIANTS)}")
    if len(model_df) < 50:
        raise ValueError("Not enough rows to train a model.")
    numerical, categorical = VARIANTS[variant]
    X_train, X_test, y_train, y_test = split_data(model_df)
    cols = numerical + categorical
    pipe = build_pipeline(numerical, categorical)
    pipe.fit(X_train[cols], y_train)
    y_pred = pipe.predict(X_test[cols])
    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "median_abs_error": float(np.median(np.abs(y_test - y_pred))),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return ModelResult(variant, pipe, numerical, categorical,
                       X_train[cols], X_test[cols], y_train, y_test, y_pred, metrics)


def cross_validated_r2(model_df: pd.DataFrame, variant: str = "original", folds: int = 5) -> np.ndarray:
    """5-fold CV R² on the full model dataset - checks the single split is not a fluke."""
    numerical, categorical = VARIANTS[variant]
    X = model_df[numerical + categorical]
    y = model_df[config.TARGET]
    cv = KFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)
    return cross_val_score(build_pipeline(numerical, categorical), X, y, cv=cv, scoring="r2", n_jobs=1)


def grouped_impurity_importance(result: ModelResult) -> pd.DataFrame:
    """Sum one-hot dummy importances back to their source feature.

    Note: impurity importance is biased toward high-cardinality features;
    see ``permutation_importances`` for a less biased view.
    """
    pre = result.pipeline.named_steps["preprocessor"]
    importances = result.pipeline.named_steps["regressor"].feature_importances_
    ohe = pre.named_transformers_["cat"]
    sizes = [1] * len(result.numerical) + [len(c) for c in ohe.categories_]
    names = result.numerical + result.categorical
    rows, start = [], 0
    for name, size in zip(names, sizes):
        rows.append({"Feature": name, "Importance": float(importances[start:start + size].sum()),
                     "Encoded columns": size})
        start += size
    return pd.DataFrame(rows).sort_values("Importance", ascending=False).reset_index(drop=True)


def permutation_importances(result: ModelResult, n_repeats: int = 5) -> pd.DataFrame:
    """Drop in test-set R² when each raw input column is shuffled."""
    pi = permutation_importance(result.pipeline, result.X_test, result.y_test, n_repeats=n_repeats,
                                random_state=config.RANDOM_STATE, scoring="r2", n_jobs=1)
    return pd.DataFrame({
        "Feature": result.X_test.columns,
        "R2 drop (mean)": pi.importances_mean,
        "R2 drop (std)": pi.importances_std,
    }).sort_values("R2 drop (mean)", ascending=False).reset_index(drop=True)


def validate_prediction_input(inputs: dict, model_df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for a single prediction request.

    Errors block prediction (missing / non-numeric / negative values).
    Warnings flag extrapolation and categories unseen in training - the
    original notebook's sample input ('PG-13', 'United States') hit this
    silently, because OneHotEncoder(handle_unknown='ignore') zeroes them out.
    """
    errors, warnings = [], []
    for f in features:
        if f not in inputs or inputs[f] is None or (isinstance(inputs[f], str) and not inputs[f].strip()):
            errors.append(f"Missing value for {f}.")
    if errors:
        return errors, warnings
    for f in ("Year", "Vote_Count"):
        if f in features:
            try:
                v = float(inputs[f])
            except (TypeError, ValueError):
                errors.append(f"{f} must be numeric.")
                continue
            if v < 0:
                errors.append(f"{f} cannot be negative.")
            lo, hi = model_df[f].min(), model_df[f].max()
            if v < lo or v > hi:
                warnings.append(f"{f}={v:,.0f} is outside the training range ({lo:,.0f}–{hi:,.0f}); "
                                "random forests cannot extrapolate beyond observed values.")
    for f in features:
        if f in ("Year", "Vote_Count"):
            continue
        if inputs[f] not in set(model_df[f].dropna().unique()):
            warnings.append(f"'{inputs[f]}' was never seen for {f} in training; the model will ignore it.")
    return errors, warnings


def predict(result: ModelResult, inputs: dict) -> dict:
    """Point prediction plus the 10th–90th percentile spread of individual trees.

    The tree spread is a rough indication of model disagreement, NOT a
    calibrated prediction interval.
    """
    row = pd.DataFrame([{f: inputs[f] for f in result.features}])
    row[result.numerical] = row[result.numerical].astype(float)
    point = float(result.pipeline.predict(row)[0])
    Xt = result.pipeline.named_steps["preprocessor"].transform(row)
    trees = np.array([t.predict(Xt)[0] for t in result.pipeline.named_steps["regressor"].estimators_])
    return {"prediction": point, "p10": float(np.percentile(trees, 10)), "p90": float(np.percentile(trees, 90))}
