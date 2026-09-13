"""Transparent hub-upgrade impact scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_hub_upgrades(
    legs: pd.DataFrame,
    ranked_hubs: pd.DataFrame,
    top_n: int = 3,
    improvement_fraction: float = 0.30,
    late_ratio_threshold: float = 1.20,
    revenue_per_prevented_breach: float = 1000.0,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Reduce attributable movement excess at selected hubs and recompute risk."""
    selected = ranked_hubs.head(top_n)["facility_id"].tolist()
    frame = legs.copy()
    original_movement = frame["actual_movement_min"].to_numpy(dtype=float)
    osrm = frame["osrm_time_min"].to_numpy(dtype=float)
    adjusted = original_movement.copy()
    original_breach = original_movement > late_ratio_threshold * osrm
    remaining_breach_probability = original_breach.astype(float)
    detail_rows = []

    for rank, hub in enumerate(selected, start=1):
        affected = frame["source_center"].eq(hub) | frame["destination_center"].eq(hub)
        # Split corridor excess across endpoints so a two-hub upgrade cannot
        # remove more than the observed excess.
        attributable = 0.5 * np.clip(adjusted - osrm, 0, None)
        savings = np.where(affected, attributable * improvement_fraction, 0.0)
        adjusted = np.clip(adjusted - savings, osrm, None)
        incremental_expected_prevented = float(
            (remaining_breach_probability[affected] * 0.5 * improvement_fraction).sum()
        )
        remaining_breach_probability[affected] *= 1.0 - 0.5 * improvement_fraction
        detail_rows.append(
            {
                "upgrade_rank": rank,
                "facility_id": hub,
                "affected_legs": int(affected.sum()),
                "movement_minutes_saved": float(savings.sum()),
                "incremental_expected_breaches_prevented": incremental_expected_prevented,
            }
        )

    expected_remaining = float(remaining_breach_probability.sum())
    expected_prevented = float(original_breach.sum() - expected_remaining)
    summary = {
        "selected_hubs": len(selected),
        "original_proxy_breaches": int(original_breach.sum()),
        "expected_remaining_proxy_breaches": expected_remaining,
        "expected_prevented_proxy_breaches": expected_prevented,
        "late_delivery_reduction_pct": float(
            100.0 * expected_prevented / original_breach.sum()
            if original_breach.sum()
            else 0.0
        ),
        "movement_minutes_saved": float((original_movement - adjusted).sum()),
        "scenario_revenue_recovered": float(
            expected_prevented * revenue_per_prevented_breach
        ),
        "improvement_fraction": float(improvement_fraction),
        "revenue_per_prevented_breach": float(revenue_per_prevented_breach),
    }
    return pd.DataFrame(detail_rows), summary
