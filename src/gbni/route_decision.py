"""Predictive FTL-versus-Carting time/cost decision framework."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .benchmark import ETABenchmarkResult


def route_mode_recommendations(
    profiles: pd.DataFrame,
    benchmark: ETABenchmarkResult,
    incremental_ftl_cost: float = 500.0,
    value_per_minute_saved: float = 10.0,
) -> pd.DataFrame:
    """Score both route modes and expose the break-even premium.

    These are predictive scenarios, not causal effects. Confidence is high only
    where both modes were historically observed on the same corridor.
    """
    carting = profiles.copy()
    carting["route_type"] = "Carting"
    ftl = profiles.copy()
    ftl["route_type"] = "FTL"
    carting_eta = benchmark.predict_scenarios(carting)
    ftl_eta = benchmark.predict_scenarios(ftl)

    output = profiles.copy()
    output["predicted_carting_eta_min"] = carting_eta
    output["predicted_ftl_eta_min"] = ftl_eta
    output["ftl_time_saving_min"] = carting_eta - ftl_eta
    output["break_even_ftl_premium"] = np.clip(
        output["ftl_time_saving_min"] * value_per_minute_saved, 0, None
    )
    output["ftl_net_value"] = output["break_even_ftl_premium"] - incremental_ftl_cost
    output["recommended_route_type"] = np.where(
        output["ftl_net_value"].gt(0), "FTL", "Carting"
    )

    edge_modes = benchmark.graph_artifacts.edges.set_index(
        ["source_center", "destination_center"]
    )["route_type_count"]
    keys = pd.MultiIndex.from_frame(output[["source_center", "destination_center"]])
    mode_count = edge_modes.reindex(keys).fillna(0).to_numpy()
    output["evidence_level"] = np.select(
        [mode_count >= 2, mode_count == 1],
        ["supported: both modes observed", "low: one mode observed"],
        default="unsupported: unseen corridor",
    )
    output["decision_note"] = np.where(
        mode_count >= 2,
        "Predictive recommendation within observed corridor support",
        "Treat as a scenario; validate through a controlled operational pilot",
    )
    return output


def corridor_mode_summary(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Aggregate leg scenarios into actionable corridor/time profiles."""
    keys = ["source_center", "destination_center", "time_bucket"]
    return (
        recommendations.groupby(keys, observed=True)
        .agg(
            evaluated_legs=("trip_uuid", "size"),
            median_carting_eta_min=("predicted_carting_eta_min", "median"),
            median_ftl_eta_min=("predicted_ftl_eta_min", "median"),
            median_ftl_time_saving_min=("ftl_time_saving_min", "median"),
            median_break_even_ftl_premium=("break_even_ftl_premium", "median"),
            median_ftl_net_value=("ftl_net_value", "median"),
            recommended_route_type=(
                "recommended_route_type",
                lambda values: values.value_counts().index[0],
            ),
            evidence_level=("evidence_level", "first"),
        )
        .reset_index()
        .sort_values("median_ftl_net_value", ascending=False)
    )

