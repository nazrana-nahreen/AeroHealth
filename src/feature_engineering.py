"""
AeroHealth AI - Feature Engineering Module
Computes time-series degradation features: rolling means, rolling standard deviations,
and normalized sensor trends per engine unit.
"""
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    SETTING_COLUMNS,
    SENSOR_COLUMNS,
    FD001_DROP_SENSORS,
)


def get_feature_columns() -> List[str]:
    """Get the list of raw active sensors and operational settings."""
    active_sensors = [s for s in SENSOR_COLUMNS if s not in FD001_DROP_SENSORS]
    return SETTING_COLUMNS + active_sensors


def extract_rolling_features(
    df: pd.DataFrame, 
    window: int = 5,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract rolling mean and standard deviation per engine unit.
    
    Args:
        df: DataFrame with unit_number and time_in_cycles
        window: Rolling window size (cycles)
        columns: Columns to compute rolling statistics on
        
    Returns:
        DataFrame with added rolling mean and rolling std columns.
    """
    df = df.copy()
    if columns is None:
        columns = [s for s in SENSOR_COLUMNS if s not in FD001_DROP_SENSORS]
        
    grouped = df.groupby("unit_number")
    
    # Rolling mean
    rolling_mean = (
        grouped[columns]
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    rolling_mean.columns = [f"{col}_roll_mean_{window}" for col in columns]
    
    # Rolling standard deviation
    rolling_std = (
        grouped[columns]
        .rolling(window=window, min_periods=1)
        .std()
        .fillna(0.0)
        .reset_index(level=0, drop=True)
    )
    rolling_std.columns = [f"{col}_roll_std_{window}" for col in columns]
    
    # Concatenate features
    result = pd.concat([df, rolling_mean, rolling_std], axis=1)
    return result


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    window: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler, List[str]]:
    """
    Apply full feature engineering pipeline:
    1. Compute rolling window stats on train and test per engine.
    2. Extract last cycle record of each engine in test set for final RUL evaluation.
    3. Fit StandardScaler on train feature columns.
    
    Returns:
        X_train, y_train, X_test_last, scaler, feature_names
    """
    # 1. Rolling features
    train_feat = extract_rolling_features(train_df, window=window)
    test_feat = extract_rolling_features(test_df, window=window)
    
    # Identify non-feature columns
    exclude_cols = {"unit_number", "time_in_cycles", "RUL"} | set(FD001_DROP_SENSORS)
    feature_cols = [c for c in train_feat.columns if c not in exclude_cols]
    
    # 2. Extract last cycle record for each test engine
    test_last = test_feat.groupby("unit_number").last().reset_index()
    
    # 3. Fit scaler
    scaler = StandardScaler()
    scaler.fit(train_feat[feature_cols])
    
    # Transform
    X_train = pd.DataFrame(
        scaler.transform(train_feat[feature_cols]),
        columns=feature_cols,
        index=train_feat.index
    )
    y_train = train_feat["RUL"]
    
    X_test_last = pd.DataFrame(
        scaler.transform(test_last[feature_cols]),
        columns=feature_cols,
        index=test_last.index
    )
    
    return X_train, y_train, X_test_last, scaler, feature_cols
