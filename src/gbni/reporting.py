"""Generate an operations-facing strategy memo from analytical outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _overall_metric(metrics: pd.DataFrame, model: str, column: str) -> float:
    row = metrics[(metrics["segment"] == "overall") & (metrics["model"] == model)]
    return float(row.iloc[0][column])


def write_strategy_memo(
    output_path: str | Path,
    metrics: pd.DataFrame,
    hubs: pd.DataFrame,
    corridors: pd.DataFrame,
    simulation_summary: dict,
    route_summary: pd.DataFrame,
) -> Path:
    baseline_mae = _overall_metric(metrics, "tabular_baseline", "mae_minutes")
    graph_mae = _overall_metric(metrics, "graphsage", "mae_minutes")
    baseline_accuracy = _overall_metric(metrics, "tabular_baseline", "within_15_percent")
    graph_accuracy = _overall_metric(metrics, "graphsage", "within_15_percent")
    mae_improvement = 100.0 * (baseline_mae - graph_mae) / baseline_mae

    lines = [
        "# Network Operations Strategy Memo",
        "",
        "**Decision requested:** prioritize a targeted hub-and-corridor improvement program and use graph-corrected ETAs for operational planning.",
        "",
        "## Executive recommendation",
        "",
        f"The graph-enhanced ETA reduced holdout MAE from {baseline_mae:.1f} to {graph_mae:.1f} minutes ({mae_improvement:.1f}% improvement) and raised trips predicted within 15% of actual from {baseline_accuracy:.1f}% to {graph_accuracy:.1f}%. This supports a controlled production pilot, with cold-start routes monitored separately.",
        "",
        "Concentrate the first improvement wave on the three highest-ranked hubs below. The ranking combines network dependency, traffic exposure, observed excess minutes, and lack of alternate connectivity; it is not a centrality-only list.",
        "",
        "## Five priority hubs",
        "",
    ]
    for rank, row in hubs.head(5).iterrows():
        lines.append(
            f"{rank + 1}. **{row['facility_name']} ({row['facility_id']})** — bottleneck score {row['bottleneck_score']:.1f}; approximately {row['sla_breach_contribution_pct']:.2f}% of observed proxy breaches and {row['excess_time_contribution_pct']:.2f}% of network excess movement time are attributed to movements touching this hub."
        )

    lines.extend(["", "## Corridor actions", ""])
    for _, row in corridors.head(5).iterrows():
        lines.append(
            f"- **{row['source_center']} → {row['destination_center']}**: median movement is {row['median_movement_delay_ratio']:.2f}× OSRM across {int(row['observation_count'])} legs. {row['recommended_intervention']}."
        )

    lines.extend(
        [
            "",
            "## Expected impact of the first three hub upgrades",
            "",
            f"Under the explicit scenario that each selected hub removes {100 * simulation_summary['improvement_fraction']:.0f}% of its attributable corridor excess and proportionally reduces its attributable breach risk, the historical test window projects {simulation_summary['expected_prevented_proxy_breaches']:,.1f} prevented proxy breaches, a {simulation_summary['late_delivery_reduction_pct']:.2f}% reduction, and {simulation_summary['movement_minutes_saved']:,.0f} movement minutes recovered.",
            "",
            f"At the configured value of ₹{simulation_summary['revenue_per_prevented_breach']:,.0f} per prevented breach, the scenario value is ₹{simulation_summary['scenario_revenue_recovered']:,.0f}. This is a planning scenario—not booked revenue—until Finance supplies actual SLA penalties and shipment economics.",
            "",
            "## FTL versus Carting policy",
            "",
            "Use FTL when its predicted time saving, multiplied by the value of a minute saved, exceeds the incremental FTL premium. Recommendations outside corridors where both modes were historically observed should be treated as pilots rather than automatic switches.",
        ]
    )
    supported = route_summary[route_summary["evidence_level"].str.startswith("supported")]
    if not supported.empty:
        top = supported.iloc[0]
        lines.extend(
            [
                "",
                f"The strongest supported profile in this sample is **{top['source_center']} → {top['destination_center']} ({top['time_bucket']})**, with median modeled FTL savings of {top['median_ftl_time_saving_min']:.1f} minutes and a break-even premium of ₹{top['median_break_even_ftl_premium']:,.0f}.",
            ]
        )

    lines.extend(
        [
            "",
            "## Controls before rollout",
            "",
            "- Pilot ETA correction on high-volume, previously seen corridors and report unseen-node performance separately.",
            "- Validate the top hub recommendations with scan-event and capacity data before approving capital expenditure.",
            "- Replace the 20%-over-OSRM breach proxy with contractual promised-delivery timestamps when available.",
            "- Replace scenario revenue and route premiums with Finance and Procurement values.",
            "- Run controlled FTL/Carting trials; the historical dataset has limited same-corridor mode overlap and is not sufficient for broad causal claims.",
        ]
    )
    destination = Path(output_path)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
