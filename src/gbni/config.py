"""Configuration loading and deterministic defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "raw_path": "data/delivery_data.csv",
        "split_column": "data",
        "train_label": "training",
        "test_label": "test",
    },
    "graph": {
        "late_ratio_threshold": 1.20,
        "minimum_corridor_observations": 5,
    },
    "model": {
        "random_seed": 42,
        "validation_fraction": 0.20,
        "ridge_penalty": 2.0,
        "graphsage_hidden_dim": 24,
        "graphsage_epochs": 80,
        "learning_rate": 0.01,
        "weight_decay": 0.0001,
        "patience": 12,
    },
    "simulation": {
        "hub_improvement_fraction": 0.30,
        "revenue_per_prevented_breach": 1000.0,
        "ftl_incremental_cost": 500.0,
        "value_per_minute_saved": 10.0,
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration on top of safe defaults."""
    if path is None:
        return _merge({}, DEFAULT_CONFIG)
    with Path(path).open("r", encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}
    return _merge(DEFAULT_CONFIG, supplied)

