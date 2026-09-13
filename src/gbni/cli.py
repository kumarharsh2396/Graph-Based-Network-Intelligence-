"""Command-line orchestration for the complete analytical workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .benchmark import run_eta_benchmark
from .bottlenecks import rank_bottleneck_hubs, rank_delayed_corridors
from .config import load_config
from .data_pipeline import build_leg_table, data_quality_report, load_raw_data
from .graph_analysis import graph_summary
from .reporting import write_strategy_memo
from .route_decision import corridor_mode_summary, route_mode_recommendations
from .simulation import simulate_hub_upgrades
from .visualization import write_bottleneck_svg


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print("[1/7] Loading and aggregating checkpoint data", flush=True)
    raw = load_raw_data(args.data or config["data"]["raw_path"])
    legs = build_leg_table(raw)
    legs.to_csv(output / "canonical_legs.csv", index=False)
    quality = data_quality_report(raw, legs)
    _write_json(output / "data_quality.json", quality)

    model_config = config["model"]
    graph_config = config["graph"]
    epochs = args.epochs if args.epochs is not None else model_config["graphsage_epochs"]
    print("[2/7] Training and benchmarking chronological ETA models", flush=True)
    benchmark = run_eta_benchmark(
        legs,
        train_label=config["data"]["train_label"],
        test_label=config["data"]["test_label"],
        validation_fraction=model_config["validation_fraction"],
        ridge_penalty=model_config["ridge_penalty"],
        hidden_dim=model_config["graphsage_hidden_dim"],
        epochs=epochs,
        learning_rate=model_config["learning_rate"],
        weight_decay=model_config["weight_decay"],
        patience=model_config["patience"],
        seed=model_config["random_seed"],
        late_ratio_threshold=graph_config["late_ratio_threshold"],
        minimum_observations=graph_config["minimum_corridor_observations"],
    )
    benchmark.metrics.to_csv(output / "model_metrics.csv", index=False)
    benchmark.predictions.to_csv(output / "test_predictions.csv", index=False)
    np.savez(
        output / "baseline_model.npz",
        coefficients=benchmark.baseline_model.coefficients,
        mean=benchmark.baseline_model.mean,
        scale=benchmark.baseline_model.scale,
    )
    torch.save(
        {
            "state_dict": benchmark.graph_model.state_dict(),
            "node_to_index": benchmark.node_store.node_to_index,
            "node_features": benchmark.node_store.matrix,
            "adjacency": benchmark.node_store.adjacency,
            "tabular_mean": benchmark.graph_scaler.mean,
            "tabular_scale": benchmark.graph_scaler.scale,
            "target_mean": benchmark.graph_target_mean,
            "target_scale": benchmark.graph_target_scale,
            "selected_epochs": benchmark.selected_epochs,
        },
        output / "graphsage_model.pt",
    )

    print("[3/7] Ranking bottleneck hubs and corridors", flush=True)
    hubs = rank_bottleneck_hubs(benchmark.graph_artifacts, legs[legs.data.eq("training")])
    corridors = rank_delayed_corridors(benchmark.graph_artifacts)
    hubs.to_csv(output / "bottleneck_hubs.csv", index=False)
    corridors.to_csv(output / "delayed_corridors.csv", index=False)
    benchmark.graph_artifacts.stratified_edges.to_csv(
        output / "corridors_by_mode_time.csv", index=False
    )
    _write_json(output / "graph_summary.json", graph_summary(benchmark.graph_artifacts))

    print("[4/7] Evaluating FTL versus Carting scenarios", flush=True)
    test_legs = legs[legs.data.eq(config["data"]["test_label"])].copy()
    mode_scenarios = route_mode_recommendations(
        test_legs,
        benchmark,
        incremental_ftl_cost=config["simulation"]["ftl_incremental_cost"],
        value_per_minute_saved=config["simulation"]["value_per_minute_saved"],
    )
    mode_summary = corridor_mode_summary(mode_scenarios)
    mode_summary.to_csv(output / "route_mode_recommendations.csv", index=False)

    print("[5/7] Simulating the top-three hub upgrades", flush=True)
    simulation_detail, simulation_summary = simulate_hub_upgrades(
        test_legs,
        hubs,
        top_n=3,
        improvement_fraction=config["simulation"]["hub_improvement_fraction"],
        late_ratio_threshold=graph_config["late_ratio_threshold"],
        revenue_per_prevented_breach=config["simulation"]["revenue_per_prevented_breach"],
    )
    simulation_detail.to_csv(output / "hub_upgrade_scenarios.csv", index=False)
    _write_json(output / "hub_upgrade_summary.json", simulation_summary)

    print("[6/7] Rendering network visualization", flush=True)
    write_bottleneck_svg(
        benchmark.graph_artifacts.graph,
        hubs,
        corridors,
        output / "network_bottlenecks.svg",
    )

    print("[7/7] Writing the operations strategy memo", flush=True)
    write_strategy_memo(
        output / "network_operations_strategy_memo.md",
        benchmark.metrics,
        hubs,
        corridors,
        simulation_summary,
        mode_summary,
    )
    _write_json(output / "run_config.json", config)
    overall = benchmark.metrics[benchmark.metrics.segment.eq("overall")]
    print("\nCompleted. Overall holdout metrics:")
    print(overall[["model", "mae_minutes", "within_15_percent"]].to_string(index=False))
    print(f"Artifacts written to: {output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run the complete pipeline")
    run_parser.add_argument("--data", help="Raw delivery CSV path")
    run_parser.add_argument("--config", default="config.yaml", help="YAML configuration path")
    run_parser.add_argument("--output", default="artifacts", help="Output directory")
    run_parser.add_argument("--epochs", type=int, help="Maximum GraphSAGE validation epochs")
    run_parser.set_defaults(function=run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

