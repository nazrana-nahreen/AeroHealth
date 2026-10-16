"""
AeroHealth AI - Data Loader & Ingestion Module
Handles loading NASA C-MAPSS data, parsing fixed column structure,
and computing Remaining Useful Life (RUL) labels.
"""
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    RAW_DATA_DIR,
    ALL_COLUMNS,
    INDEX_COLUMNS,
    DEFAULT_RUL_CAP,
)


def load_raw_dataset(subset: str = "FD001", split: str = "train") -> pd.DataFrame:
    """
    Load raw space-delimited C-MAPSS dataset.
    
    Args:
        subset: 'FD001', 'FD002', 'FD003', or 'FD004'
        split: 'train' or 'test'
        
    Returns:
        pd.DataFrame with labeled columns.
    """
    filename = f"{split}_{subset}.txt"
    filepath = RAW_DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found at {filepath}")
        
    # File is space-delimited, may have trailing whitespaces
    df = pd.read_csv(filepath, sep=r"\s+", header=None)
    
    # Drop any extra columns resulting from trailing whitespace
    if df.shape[1] > len(ALL_COLUMNS):
        df = df.iloc[:, :len(ALL_COLUMNS)]
        
    df.columns = ALL_COLUMNS
    return df


def load_ground_truth_rul(subset: str = "FD001") -> pd.Series:
    """
    Load ground truth RUL values for the test set.
    
    Args:
        subset: 'FD001', 'FD002', 'FD003', or 'FD004'
        
    Returns:
        pd.Series containing the true RUL for each test unit at its last recorded cycle.
    """
    filename = f"RUL_{subset}.txt"
    filepath = RAW_DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"RUL ground truth file not found at {filepath}")
        
    rul_series = pd.read_csv(filepath, sep=r"\s+", header=None).squeeze()
    rul_series.index = np.arange(1, len(rul_series) + 1)
    rul_series.index.name = "unit_number"
    rul_series.name = "true_rul"
    return rul_series


def add_train_rul_labels(
    df: pd.DataFrame, 
    rul_cap: Optional[int] = DEFAULT_RUL_CAP
) -> pd.DataFrame:
    """
    Calculate RUL for training data where engines run until failure.
    RUL = max_cycle_for_unit - current_cycle.
    
    Args:
        df: Raw training DataFrame
        rul_cap: Maximum RUL threshold (piecewise linear target). Defaults to 125.
        
    Returns:
        DataFrame with added 'RUL' column.
    """
    df = df.copy()
    max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df["RUL"] = max_cycles - df["time_in_cycles"]
    
    if rul_cap is not None:
        df["RUL"] = df["RUL"].clip(upper=rul_cap)
        
    return df


def prepare_dataset(
    subset: str = "FD001", 
    rul_cap: Optional[int] = DEFAULT_RUL_CAP
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load and prepare train data, test data, and ground truth RUL.
    
    Returns:
        train_df (with RUL labels), test_df, ground_truth_rul
    """
    train_df = load_raw_dataset(subset, "train")
    train_df = add_train_rul_labels(train_df, rul_cap=rul_cap)
    
    test_df = load_raw_dataset(subset, "test")
    ground_truth_rul = load_ground_truth_rul(subset)
    
    return train_df, test_df, ground_truth_rul


if __name__ == "__main__":
    print("Testing data loader on FD001...")
    train, test, true_rul = prepare_dataset("FD001")
    print(f"Train shape: {train.shape}, Unique engines: {train['unit_number'].nunique()}")
    print(f"Test shape:  {test.shape}, Unique engines:  {test['unit_number'].nunique()}")
    print(f"True RUL count: {len(true_rul)}")
    print(f"Train RUL summary:\n{train['RUL'].describe()}")
