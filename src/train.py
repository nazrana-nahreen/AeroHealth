"""
AeroHealth AI - Model Training & Evaluation Pipeline
Trains baseline and advanced regression models, evaluates using NASA PHM metrics,
and saves the production-ready model pipeline.
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import MODELS_DIR, DEFAULT_RUL_CAP
from src.data_loader import prepare_dataset
from src.feature_engineering import prepare_features
from src.evaluate import compute_metrics


def train_and_evaluate(subset: str = "FD001", rul_cap: int = DEFAULT_RUL_CAP) -> Dict[str, Any]:
    """
    Train candidate models, evaluate against test ground truth,
    and persist the top-performing model artifact.
    """
    print(f"=== [1/4] Ingesting NASA C-MAPSS dataset ({subset}) ===")
    train_df, test_df, true_rul = prepare_dataset(subset=subset, rul_cap=rul_cap)
    print(f"Loaded {train_df['unit_number'].nunique()} train units and {len(true_rul)} test units.")

    print("\n=== [2/4] Engineering Temporal Features (Rolling Stats & Scaling) ===")
    X_train, y_train, X_test_last, scaler, feature_cols = prepare_features(
        train_df, test_df, window=5
    )
    print(f"Feature matrix shape: {X_train.shape} with {len(feature_cols)} engineered features.")

    # Candidate models
    models = {
        "Baseline (Ridge Regression)": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_split=5, n_jobs=-1, random_state=42
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=120, max_depth=5, learning_rate=0.08, random_state=42
        )
    }

    results = {}
    fitted_models = {}

    print("\n=== [3/4] Training & Benchmarking Candidate Models ===")
    print(f"{'Model Name':<30} | {'RMSE':<8} | {'MAE':<8} | {'R2':<8} | {'NASA Score':<12}")
    print("-" * 75)

    best_score = float("inf")
    best_model_name = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test_last)
        
        # Clip test predictions to positive RUL
        y_pred = np.clip(y_pred, 0, None)
        
        metrics = compute_metrics(true_rul.values, y_pred)
        results[name] = metrics
        fitted_models[name] = model

        print(
            f"{name:<30} | {metrics['rmse']:<8.3f} | {metrics['mae']:<8.3f} | "
            f"{metrics['r2']:<8.4f} | {metrics['nasa_score']:<12.2f}"
        )

        # Select by lowest NASA PHM Score (or RMSE if tied)
        if metrics["nasa_score"] < best_score:
            best_score = metrics["nasa_score"]
            best_model_name = name

    print("-" * 75)
    print(f"[BEST MODEL] Top Performing Model: {best_model_name} (NASA Score: {best_score:.2f})")

    print("\n=== [4/4] Serializing Model Pipeline Artifact ===")
    pipeline_artifact = {
        "model_name": best_model_name,
        "model": fitted_models[best_model_name],
        "scaler": scaler,
        "feature_cols": feature_cols,
        "window": 5,
        "rul_cap": rul_cap,
        "metrics": results[best_model_name],
        "all_results": results,
    }

    output_path = MODELS_DIR / "aerohealth_model.joblib"
    joblib.dump(pipeline_artifact, output_path)
    print(f"Artifact successfully saved to {output_path}")

    return pipeline_artifact


if __name__ == "__main__":
    train_and_evaluate()
