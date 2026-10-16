"""
AeroHealth AI - Unit tests for data loading and feature engineering
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_raw_dataset, add_train_rul_labels, load_ground_truth_rul
from src.feature_engineering import extract_rolling_features, prepare_features
from src.evaluate import nasa_scoring_function, compute_metrics


def test_load_raw_dataset():
    df = load_raw_dataset("FD001", "train")
    assert not df.empty
    assert "unit_number" in df.columns
    assert "time_in_cycles" in df.columns
    assert "sensor_1" in df.columns
    assert df["unit_number"].nunique() == 100


def test_rul_label_calculation():
    df = load_raw_dataset("FD001", "train")
    labeled_df = add_train_rul_labels(df, rul_cap=125)
    
    assert "RUL" in labeled_df.columns
    assert labeled_df["RUL"].max() == 125
    assert labeled_df["RUL"].min() == 0
    
    # Engine 1 final cycle must have RUL == 0
    unit_1 = labeled_df[labeled_df["unit_number"] == 1]
    last_cycle = unit_1["time_in_cycles"].max()
    final_record = unit_1[unit_1["time_in_cycles"] == last_cycle]
    assert final_record["RUL"].values[0] == 0


def test_nasa_scoring_function():
    y_true = np.array([50.0, 50.0])
    # Early prediction (-5 cycles) vs Late prediction (+5 cycles)
    early_pred = np.array([45.0, 50.0])
    late_pred = np.array([55.0, 50.0])
    
    score_early = nasa_scoring_function(y_true, early_pred)
    score_late = nasa_scoring_function(y_true, late_pred)
    
    # NASA score strictly penalizes late prediction more than early prediction
    assert score_late > score_early


def test_feature_engineering_pipeline():
    train_df = load_raw_dataset("FD001", "train").iloc[:1000]
    train_df = add_train_rul_labels(train_df, rul_cap=125)
    test_df = load_raw_dataset("FD001", "test").iloc[:500]
    
    X_train, y_train, X_test_last, scaler, feature_cols = prepare_features(
        train_df, test_df, window=5
    )
    
    assert len(X_train) == len(train_df)
    assert len(feature_cols) > 0
    assert not np.isnan(X_train.values).any()
