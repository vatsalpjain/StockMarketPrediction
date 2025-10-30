"""Utilities for building sequence datasets for LSTM models.

Non-invasive helpers: no torch dependency here. These functions operate on
NumPy/Pandas and return NumPy arrays ready to be wrapped by DL frameworks.
"""
from __future__ import annotations

from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_supervised_sequences(
    features_df: pd.DataFrame,
    target_series: pd.Series,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Build sliding-window sequences X_seq, y for sequence models.

    - features_df: feature matrix with time index (no NaNs)
    - target_series: target aligned to features_df index
    - lookback: number of past timesteps per sample

    Returns:
        X_seq: shape (num_samples, lookback, num_features)
        y: shape (num_samples,)
        meta: dict with 'last_window' for future inference
    """
    assert features_df.index.equals(target_series.index), "Features and target indices must align"
    X = features_df.values.astype(np.float32)
    y = target_series.values.astype(np.float32)

    sequences = []
    targets = []
    for t in range(lookback, len(X)):
        sequences.append(X[t - lookback : t])
        targets.append(y[t])

    X_seq = np.asarray(sequences, dtype=np.float32)
    y_seq = np.asarray(targets, dtype=np.float32)

    last_window = X[-lookback:]
    meta = {"last_window": last_window}
    return X_seq, y_seq, meta


def train_val_test_split_sequences(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Time-ordered split for sequences: train, val, test.

    Ratios must sum to <= 1. Remaining goes to test.
    """
    assert 0 < train_ratio < 1 and 0 < val_ratio < 1
    n = len(X_seq)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1

    X_train, y_train = X_seq[:n_train], y_seq[:n_train]
    X_val, y_val = X_seq[n_train : n_train + n_val], y_seq[n_train : n_train + n_val]
    X_test, y_test = X_seq[n_train + n_val :], y_seq[n_train + n_val :]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def fit_feature_scaler(
    features_df: pd.DataFrame,
    train_end_index: int,
    scaler: Optional[StandardScaler] = None,
) -> StandardScaler:
    """Fit a StandardScaler on the feature columns up to train_end_index (exclusive)."""
    scaler = scaler or StandardScaler()
    scaler.fit(features_df.iloc[:train_end_index].values)
    return scaler


def apply_feature_scaler(features_df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply a fitted scaler to features_df and return a new DataFrame with same index/columns."""
    scaled = scaler.transform(features_df.values)
    return pd.DataFrame(scaled, index=features_df.index, columns=features_df.columns)


