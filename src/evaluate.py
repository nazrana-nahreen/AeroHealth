"""
AeroHealth AI - Evaluation Metrics Module
Includes standard regression metrics and the official NASA PHM 2008 Asymmetric Scoring Function.
"""
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def nasa_scoring_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the official NASA PHM 2008 Data Challenge Asymmetric Score.
    
    Penalizes late predictions (optimistic RUL, which causes flight risks) 
    more heavily than early predictions (conservative RUL, which schedules maintenance early).
    
    Formula:
        d = y_pred - y_true
        if d < 0 (early):  score = exp(-d / 13) - 1
        if d >= 0 (late):  score = exp(d / 10) - 1
        
    Args:
        y_true: Ground truth RUL values
        y_pred: Predicted RUL values
        
    Returns:
        Total NASA score (lower is better).
    """
    diff = y_pred - y_true
    score = np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
    return float(np.sum(score))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute a full dictionary of evaluation metrics.
    
    Returns:
        dict with RMSE, MAE, R2, and NASA asymmetric score.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    nasa_score = float(nasa_scoring_function(y_true, y_pred))
    
    return {
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "r2": round(r2, 4),
        "nasa_score": round(nasa_score, 2),
    }
