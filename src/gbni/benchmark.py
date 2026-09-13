"""Chronological baseline-versus-GraphSAGE ETA benchmarking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluation import eta_metrics, from_log_prediction
from .features import (
    HistoricalGraphFeatures,
    NodeFeatureStore,
    baseline_matrix,
    target_log_elapsed,
)
from .graph_analysis import GraphArtifacts, build_graph_artifacts
from .models import (
    GraphSAGEETA,
    RidgeRegressor,
    fit_graphsage,
    predict_graphsage,
)


@dataclass
class ArrayScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "ArrayScaler":
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return ((matrix - self.mean) / self.scale).astype(np.float32)


@dataclass
class ETABenchmarkResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    baseline_model: RidgeRegressor
    graph_model: GraphSAGEETA
    graph_artifacts: GraphArtifacts
    node_store: NodeFeatureStore
    history: HistoricalGraphFeatures
    graph_scaler: ArrayScaler
    graph_target_mean: float
    graph_target_scale: float
    selected_epochs: int

    def predict_scenarios(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict elapsed minutes for counterfactual rows with known schema."""
        tabular = np.column_stack(
            [baseline_matrix(frame), self.history.transform(frame, leave_one_out=False)]
        ).astype(np.float32)
        tabular = self.graph_scaler.transform(tabular)
        source, destination = self.node_store.endpoint_indices(frame)
        log_prediction = predict_graphsage(
            self.graph_model,
            self.node_store.matrix,
            self.node_store.adjacency,
            tabular,
            source,
            destination,
        )
        log_prediction = (
            log_prediction * self.graph_target_scale + self.graph_target_mean
        )
        return from_log_prediction(log_prediction)


def temporal_train_validation_split(
    training: pd.DataFrame, validation_fraction: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by trip creation time while keeping every trip in one partition."""
    trip_times = training.groupby("trip_uuid")["trip_creation_time"].min().sort_values()
    split_position = max(1, int(len(trip_times) * (1.0 - validation_fraction)))
    fit_trip_ids = set(trip_times.iloc[:split_position].index)
    fit = training[training["trip_uuid"].isin(fit_trip_ids)].copy()
    validation = training[~training["trip_uuid"].isin(fit_trip_ids)].copy()
    if fit.empty or validation.empty:
        raise ValueError("Temporal split produced an empty training or validation partition")
    return fit, validation


def _graph_inputs(
    fit: pd.DataFrame,
    evaluation: pd.DataFrame,
    artifacts: GraphArtifacts,
) -> tuple[
    HistoricalGraphFeatures,
    NodeFeatureStore,
    ArrayScaler,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    history = HistoricalGraphFeatures(smoothing=10.0)
    fit_history = history.fit_transform(fit)
    evaluation_history = history.transform(evaluation)
    fit_tabular = np.column_stack([baseline_matrix(fit), fit_history]).astype(np.float32)
    evaluation_tabular = np.column_stack(
        [baseline_matrix(evaluation), evaluation_history]
    ).astype(np.float32)
    scaler = ArrayScaler.fit(fit_tabular)
    fit_tabular = scaler.transform(fit_tabular)
    evaluation_tabular = scaler.transform(evaluation_tabular)
    node_store = NodeFeatureStore.from_graph(artifacts)
    fit_source, fit_destination = node_store.endpoint_indices(fit)
    evaluation_source, evaluation_destination = node_store.endpoint_indices(evaluation)
    return (
        history,
        node_store,
        scaler,
        fit_tabular,
        evaluation_tabular,
        fit_source,
        fit_destination,
        evaluation_source,
        evaluation_destination,
    )


def _metric_rows(
    actual: np.ndarray,
    osrm: np.ndarray,
    baseline: np.ndarray,
    graph: np.ndarray,
    test: pd.DataFrame,
    unseen_corridor: np.ndarray,
    unseen_node: np.ndarray,
) -> pd.DataFrame:
    rows = []
    models = {"OSRM": osrm, "tabular_baseline": baseline, "graphsage": graph}
    segments: dict[str, np.ndarray] = {
        "overall": np.ones(len(test), dtype=bool),
        "FTL": test["route_type"].eq("FTL").to_numpy(),
        "Carting": test["route_type"].eq("Carting").to_numpy(),
        "seen_corridor": ~unseen_corridor,
        "unseen_corridor": unseen_corridor,
        "seen_nodes": ~unseen_node,
        "unseen_node": unseen_node,
    }
    for segment, mask in segments.items():
        if not mask.any():
            continue
        for model_name, prediction in models.items():
            row = {"segment": segment, "model": model_name}
            row.update(eta_metrics(actual[mask], prediction[mask]))
            rows.append(row)
    return pd.DataFrame(rows)


def run_eta_benchmark(
    legs: pd.DataFrame,
    train_label: str = "training",
    test_label: str = "test",
    validation_fraction: float = 0.20,
    ridge_penalty: float = 2.0,
    hidden_dim: int = 24,
    epochs: int = 80,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    patience: int = 12,
    seed: int = 42,
    late_ratio_threshold: float = 1.20,
    minimum_observations: int = 5,
) -> ETABenchmarkResult:
    """Fit both models without using test topology or outcomes."""
    eligible = legs[legs["is_model_eligible"]].copy()
    training = eligible[eligible["data"].eq(train_label)].copy()
    test = eligible[eligible["data"].eq(test_label)].copy()
    if training.empty or test.empty:
        raise ValueError("Both training and test legs are required")

    fit, validation = temporal_train_validation_split(training, validation_fraction)
    validation_artifacts = build_graph_artifacts(
        fit, late_ratio_threshold, minimum_observations
    )
    (
        _,
        validation_node_store,
        _,
        fit_tabular,
        validation_tabular,
        fit_source,
        fit_destination,
        validation_source,
        validation_destination,
    ) = _graph_inputs(fit, validation, validation_artifacts)
    fit_target = target_log_elapsed(fit)
    validation_target = target_log_elapsed(validation)
    fit_target_mean = float(fit_target.mean())
    fit_target_scale = float(fit_target.std()) or 1.0
    validation_training = fit_graphsage(
        validation_node_store.matrix,
        validation_node_store.adjacency,
        fit_tabular,
        fit_source,
        fit_destination,
        (fit_target - fit_target_mean) / fit_target_scale,
        validation_tabular,
        validation_source,
        validation_destination,
        (validation_target - fit_target_mean) / fit_target_scale,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        seed=seed,
    )
    selected_epochs = max(1, validation_training.best_epoch)

    baseline = RidgeRegressor(penalty=ridge_penalty).fit(
        baseline_matrix(training), target_log_elapsed(training)
    )
    baseline_prediction = from_log_prediction(baseline.predict(baseline_matrix(test)))

    final_artifacts = build_graph_artifacts(
        training, late_ratio_threshold, minimum_observations
    )
    (
        history,
        node_store,
        graph_scaler,
        training_tabular,
        test_tabular,
        training_source,
        training_destination,
        test_source,
        test_destination,
    ) = _graph_inputs(training, test, final_artifacts)

    training_target = target_log_elapsed(training)
    graph_target_mean = float(training_target.mean())
    graph_target_scale = float(training_target.std()) or 1.0
    normalized_training_target = (
        training_target - graph_target_mean
    ) / graph_target_scale
    final_training = fit_graphsage(
        node_store.matrix,
        node_store.adjacency,
        training_tabular,
        training_source,
        training_destination,
        normalized_training_target,
        training_tabular,
        training_source,
        training_destination,
        normalized_training_target,
        hidden_dim=hidden_dim,
        epochs=selected_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=selected_epochs + 1,
        seed=seed,
    )
    graph_log_prediction = predict_graphsage(
            final_training.model,
            node_store.matrix,
            node_store.adjacency,
            test_tabular,
            test_source,
            test_destination,
        )
    graph_prediction = from_log_prediction(
        graph_log_prediction * graph_target_scale + graph_target_mean
    )

    actual = test["actual_elapsed_min"].to_numpy(dtype=float)
    osrm = test["osrm_time_min"].to_numpy(dtype=float)
    training_edges = set(
        map(
            tuple,
            training[["source_center", "destination_center"]].itertuples(
                index=False, name=None
            ),
        )
    )
    unseen_corridor = np.array(
        [
            pair not in training_edges
            for pair in test[["source_center", "destination_center"]].itertuples(
                index=False, name=None
            )
        ]
    )
    unseen_node = (test_source < 0) | (test_destination < 0)
    metrics = _metric_rows(
        actual,
        osrm,
        baseline_prediction,
        graph_prediction,
        test,
        unseen_corridor,
        unseen_node,
    )

    predictions = test[
        [
            "data",
            "trip_uuid",
            "source_center",
            "destination_center",
            "route_type",
            "od_start_time",
            "actual_elapsed_min",
            "actual_movement_min",
            "osrm_time_min",
            "osrm_distance_km",
        ]
    ].copy()
    predictions["osrm_prediction_min"] = osrm
    predictions["baseline_prediction_min"] = baseline_prediction
    predictions["graph_prediction_min"] = graph_prediction
    predictions["unseen_corridor"] = unseen_corridor
    predictions["unseen_node"] = unseen_node
    predictions["graph_within_15_percent"] = (
        np.abs(graph_prediction - actual) / actual <= 0.15
    )

    return ETABenchmarkResult(
        metrics=metrics,
        predictions=predictions,
        baseline_model=baseline,
        graph_model=final_training.model,
        graph_artifacts=final_artifacts,
        node_store=node_store,
        history=history,
        graph_scaler=graph_scaler,
        graph_target_mean=graph_target_mean,
        graph_target_scale=graph_target_scale,
        selected_epochs=selected_epochs,
    )
