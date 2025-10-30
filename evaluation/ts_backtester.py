"""Sequence-aware backtester for LSTM models with rolling-origin splits."""
from __future__ import annotations

from typing import Dict
import numpy as np
import torch

from features.sequence_builder import build_supervised_sequences, apply_feature_scaler, fit_feature_scaler
from evaluation.metrics import MetricsCalculator


def rolling_origin_backtest_lstm(
    X_df,
    y_series,
    lookback: int,
    trainer_factory,
    n_splits: int = 3,
) -> Dict[str, float]:
    """Run rolling-origin backtests and return averaged metrics.

    trainer_factory: () -> (trainer, cfg) that yields a fresh LSTMTrainer per split.
    """
    # Fit scaler on entire history of each split's train portion
    N = len(X_df)
    split_sizes = np.linspace(int(0.6 * N), int(0.9 * N), num=n_splits, dtype=int)
    results = []

    for i, train_end in enumerate(split_sizes):
        scaler = fit_feature_scaler(X_df, train_end_index=train_end)
        X_scaled = apply_feature_scaler(X_df, scaler)
        X_seq, y_seq, _ = build_supervised_sequences(X_scaled, y_series, lookback)

        adj_train_end = max(1, train_end - lookback)
        X_train, y_train = X_seq[:adj_train_end], y_seq[:adj_train_end]
        X_test, y_test = X_seq[adj_train_end:], y_seq[adj_train_end:]
        if len(X_test) == 0:
            continue

        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.float32)
        Xtt = torch.tensor(X_test, dtype=torch.float32)

        trainer, _ = trainer_factory()
        trainer.fit(Xt, yt, X_val=None, y_val=None)
        yp, _, _ = trainer.predict(Xtt)
        y_pred = yp.numpy()
        metrics = MetricsCalculator.calculate_all(y_test, y_pred)
        results.append(metrics)

    # Average
    if not results:
        return {}
    keys = results[0].keys()
    avg = {k: float(np.mean([r[k] for r in results])) for k in keys}
    return {f"avg_{k}": v for k, v in avg.items()}


