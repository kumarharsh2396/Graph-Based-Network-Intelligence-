"""ETA model metrics and comparison helpers."""

from __future__ import annotations

import numpy as np


def eta_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.clip(np.asarray(predicted, dtype=float), 0, None)
    absolute_error = np.abs(actual - predicted)
    relative_error = np.divide(
        absolute_error,
        actual,
        out=np.full_like(absolute_error, np.nan),
        where=actual > 0,
    )
    return {
        "count": int(len(actual)),
        "mae_minutes": float(np.nanmean(absolute_error)),
        "median_absolute_error_minutes": float(np.nanmedian(absolute_error)),
        "median_absolute_percentage_error": float(np.nanmedian(relative_error) * 100),
        "within_15_percent": float(np.nanmean(relative_error <= 0.15) * 100),
    }


def from_log_prediction(predicted_log: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(np.asarray(predicted_log, dtype=float)), 0, None)

