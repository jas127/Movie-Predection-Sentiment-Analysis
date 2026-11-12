"""Offline evaluation and cross-validation script.

Computes 5-fold cross-validated R2 scores for both the original pipeline and
the pre-release pipeline, saving the output to reports/model_metrics.json.
"""
import json
from pathlib import Path
import numpy as np

from src import config, data_loader, prediction, preprocessing

def run():
    print("Loading box office data...")
    raw = data_loader.load_box_office()
    model_df = preprocessing.model_training_frame(raw)
    
    print(f"Training dataset size: {len(model_df)} records")
    
    print("\n--- Evaluating Original Pipeline ---")
    orig_res = prediction.train_and_evaluate(model_df, variant="original")
    orig_cv = prediction.cross_validated_r2(model_df, variant="original", folds=5)
    print(f"Test R2: {orig_res.metrics['r2']:.4f}")
    print(f"Test RMSE: ${orig_res.metrics['rmse']:,.2f}")
    print(f"5-Fold CV R2: {orig_cv} -> Mean: {orig_cv.mean():.4f} (+/- {orig_cv.std():.4f})")
    
    print("\n--- Evaluating Pre-Release Pipeline ---")
    pre_res = prediction.train_and_evaluate(model_df, variant="pre_release")
    pre_cv = prediction.cross_validated_r2(model_df, variant="pre_release", folds=5)
    print(f"Test R2: {pre_res.metrics['r2']:.4f}")
    print(f"Test RMSE: ${pre_res.metrics['rmse']:,.2f}")
    print(f"5-Fold CV R2: {pre_cv} -> Mean: {pre_cv.mean():.4f} (+/- {pre_cv.std():.4f})")
    
    output = {
        "original": {
            "metrics": orig_res.metrics,
            "cv_scores": [float(x) for x in orig_cv],
            "cv_mean": float(orig_cv.mean()),
            "cv_std": float(orig_cv.std())
        },
        "pre_release": {
            "metrics": pre_res.metrics,
            "cv_scores": [float(x) for x in pre_cv],
            "cv_mean": float(pre_cv.mean()),
            "cv_std": float(pre_cv.std())
        }
    }
    
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = config.REPORTS_DIR / "model_metrics.json"
    with open(report_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved metrics to {report_file}")

if __name__ == "__main__":
    run()
