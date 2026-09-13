"""Business-oriented hub and corridor prioritization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .graph_analysis import GraphArtifacts


def _percentile(series: pd.Series) -> pd.Series:
    if len(series) <= 1:
        return pd.Series(np.ones(len(series)), index=series.index)
    return series.rank(method="average", pct=True)


def rank_bottleneck_hubs(
    artifacts: GraphArtifacts, legs: pd.DataFrame
) -> pd.DataFrame:
    """Rank facilities by structural criticality and observed delay exposure."""
    hubs = artifacts.nodes.copy()
    names = pd.concat(
        [
            legs[["source_center", "source_name"]].rename(
                columns={"source_center": "facility_id", "source_name": "facility_name"}
            ),
            legs[["destination_center", "destination_name"]].rename(
                columns={
                    "destination_center": "facility_id",
                    "destination_name": "facility_name",
                }
            ),
        ],
        ignore_index=True,
    ).dropna()
    name_map = (
        names.groupby("facility_id")["facility_name"]
        .agg(lambda values: values.value_counts().index[0])
        .to_dict()
    )
    hubs["facility_name"] = hubs["facility_id"].map(name_map).fillna(hubs["facility_id"])

    hubs["structural_percentile"] = _percentile(hubs["betweenness_centrality"])
    hubs["weighted_structural_percentile"] = _percentile(
        hubs["weighted_betweenness_centrality"]
    )
    hubs["throughput_percentile"] = _percentile(hubs["throughput_legs"])
    hubs["excess_time_percentile"] = _percentile(hubs["attributed_excess_min"])
    hubs["low_redundancy_percentile"] = _percentile(1.0 - hubs["clustering_coefficient"])
    total_breaches = hubs["attributed_breaches"].sum()
    total_excess = hubs["attributed_excess_min"].sum()
    hubs["sla_breach_contribution_pct"] = np.where(
        total_breaches > 0, 100.0 * hubs["attributed_breaches"] / total_breaches, 0.0
    )
    hubs["excess_time_contribution_pct"] = np.where(
        total_excess > 0, 100.0 * hubs["attributed_excess_min"] / total_excess, 0.0
    )
    max_breach_share = hubs["sla_breach_contribution_pct"].max() or 1.0
    max_excess_share = hubs["excess_time_contribution_pct"].max() or 1.0
    hubs["observed_impact_index"] = 0.5 * (
        hubs["sla_breach_contribution_pct"] / max_breach_share
    ) + 0.5 * (hubs["excess_time_contribution_pct"] / max_excess_share)
    # Half of the ranking is realized operational impact. This prevents a highly
    # central but lightly used facility from outranking a proven delay hotspot.
    hubs["bottleneck_score"] = 100.0 * (
        0.15 * hubs["structural_percentile"]
        + 0.10 * hubs["weighted_structural_percentile"]
        + 0.15 * hubs["throughput_percentile"]
        + 0.10 * hubs["low_redundancy_percentile"]
        + 0.50 * hubs["observed_impact_index"]
    )
    return hubs.sort_values(
        ["bottleneck_score", "attributed_excess_min"], ascending=False
    ).reset_index(drop=True)


def rank_delayed_corridors(artifacts: GraphArtifacts) -> pd.DataFrame:
    corridors = artifacts.edges.copy()
    breach_count = corridors["observation_count"] * corridors["breach_rate"]
    denominator = breach_count.sum()
    corridors["sla_breach_contribution_pct"] = np.where(
        denominator > 0, 100.0 * breach_count / denominator, 0.0
    )
    corridors["corridor_priority_score"] = (
        _percentile(corridors["total_excess_min"]) * 0.45
        + _percentile(corridors["observation_count"]) * 0.30
        + _percentile(corridors["median_movement_delay_ratio"].clip(upper=10)) * 0.25
    ) * 100.0
    corridors["recommended_intervention"] = np.select(
        [
            corridors["route_type_count"].gt(1) & corridors["chronic_delay"],
            corridors["chronic_delay"] & corridors["observation_count"].ge(15),
            corridors["chronic_delay"],
        ],
        [
            "Evaluate FTL/Carting shift on supported corridor",
            "Assess parallel route and capacity relief",
            "Audit departure window and facility handoff",
        ],
        default="Monitor; current evidence does not justify capital action",
    )
    return corridors.sort_values(
        ["corridor_priority_score", "total_excess_min"], ascending=False
    ).reset_index(drop=True)
