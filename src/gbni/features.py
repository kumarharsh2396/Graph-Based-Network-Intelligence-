"""Departure-time and leakage-safe historical graph features."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .graph_analysis import GraphArtifacts


BASELINE_FEATURE_NAMES = [
    "log_osrm_time",
    "log_osrm_distance",
    "osrm_speed_kmph",
    "is_ftl",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
]


def baseline_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Features that are known when an OD leg departs."""
    hour = frame["departure_hour"].to_numpy(dtype=float)
    weekday = frame["departure_weekday"].to_numpy(dtype=float)
    osrm_time = frame["osrm_time_min"].to_numpy(dtype=float)
    distance = frame["osrm_distance_km"].to_numpy(dtype=float)
    speed = np.divide(distance * 60.0, osrm_time, out=np.zeros_like(distance), where=osrm_time > 0)
    return np.column_stack(
        [
            np.log1p(np.clip(osrm_time, 0, None)),
            np.log1p(np.clip(distance, 0, None)),
            np.clip(speed, 0, 150) / 100.0,
            frame["route_type"].eq("FTL").to_numpy(dtype=float),
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * weekday / 7.0),
            np.cos(2 * np.pi * weekday / 7.0),
            frame["is_weekend"].to_numpy(dtype=float),
        ]
    ).astype(np.float32)


def target_log_elapsed(frame: pd.DataFrame) -> np.ndarray:
    return np.log1p(frame["actual_elapsed_min"].to_numpy(dtype=float)).astype(np.float32)


def _history_target(frame: pd.DataFrame) -> np.ndarray:
    ratio = frame["actual_elapsed_min"].to_numpy(dtype=float) / frame[
        "osrm_time_min"
    ].to_numpy(dtype=float)
    return np.log(np.clip(ratio, 0.10, 20.0))


@dataclass
class HistoricalGraphFeatures:
    """Smoothed corridor/node priors fitted strictly on historical labels."""

    smoothing: float = 10.0
    global_mean: float = 0.0
    summaries: dict[str, dict[tuple, tuple[float, int]]] = field(default_factory=dict)
    definitions: dict[str, list[str]] = field(
        default_factory=lambda: {
            "corridor": ["source_center", "destination_center"],
            "source": ["source_center"],
            "destination": ["destination_center"],
            "route_time": ["route_type", "time_bucket"],
        }
    )

    @staticmethod
    def _key(values) -> tuple:
        if isinstance(values, tuple):
            return values
        return (values,)

    def fit(self, frame: pd.DataFrame) -> "HistoricalGraphFeatures":
        target = _history_target(frame)
        self.global_mean = float(np.mean(target))
        work = frame.copy()
        work["_history_target"] = target
        self.summaries = {}
        for name, columns in self.definitions.items():
            grouped = work.groupby(columns, observed=True)["_history_target"].agg(["sum", "count"])
            self.summaries[name] = {
                self._key(index): (float(row["sum"]), int(row["count"]))
                for index, row in grouped.iterrows()
            }
        return self

    def transform(self, frame: pd.DataFrame, leave_one_out: bool = False) -> np.ndarray:
        if not self.summaries:
            raise RuntimeError("HistoricalGraphFeatures must be fitted before transform")
        current = _history_target(frame) if leave_one_out else np.zeros(len(frame))
        output: list[np.ndarray] = []
        for name, columns in self.definitions.items():
            summary = self.summaries[name]
            values = np.empty(len(frame), dtype=np.float32)
            counts = np.empty(len(frame), dtype=np.float32)
            for position, row_values in enumerate(frame[columns].itertuples(index=False, name=None)):
                total, count = summary.get(tuple(row_values), (0.0, 0))
                if leave_one_out and count:
                    total -= float(current[position])
                    count -= 1
                values[position] = (total + self.smoothing * self.global_mean) / (
                    count + self.smoothing
                )
                counts[position] = np.log1p(count)
            output.extend([values, counts])
        return np.column_stack(output).astype(np.float32)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        self.fit(frame)
        return self.transform(frame, leave_one_out=True)


NODE_FEATURE_COLUMNS = [
    "in_degree",
    "out_degree",
    "betweenness_centrality",
    "weighted_betweenness_centrality",
    "clustering_coefficient",
    "weak_component_size",
    "throughput_legs",
]


@dataclass
class NodeFeatureStore:
    node_to_index: dict[str, int]
    matrix: np.ndarray
    adjacency: np.ndarray

    @classmethod
    def from_graph(cls, artifacts: GraphArtifacts) -> "NodeFeatureStore":
        node_frame = artifacts.nodes.sort_values("facility_id").reset_index(drop=True)
        node_to_index = {node: index for index, node in enumerate(node_frame["facility_id"])}
        matrix = node_frame[NODE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        matrix[:, [0, 1, 5, 6]] = np.log1p(matrix[:, [0, 1, 5, 6]])
        mean = matrix.mean(axis=0, keepdims=True)
        std = matrix.std(axis=0, keepdims=True)
        matrix = (matrix - mean) / np.where(std > 1e-8, std, 1.0)
        # Betweenness and throughput are heavy-tailed. Clipping prevents a small
        # number of national hubs from destabilizing message-passing gradients.
        matrix = np.clip(matrix, -5.0, 5.0).astype(np.float32)

        adjacency = np.zeros((len(node_frame), len(node_frame)), dtype=np.float32)
        for source, destination in artifacts.graph.edges:
            i = node_to_index[source]
            j = node_to_index[destination]
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0
        adjacency += np.eye(len(node_frame), dtype=np.float32)
        degrees = adjacency.sum(axis=1, keepdims=True)
        adjacency /= np.where(degrees > 0, degrees, 1.0)
        return cls(node_to_index=node_to_index, matrix=matrix, adjacency=adjacency)

    def endpoint_indices(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        source = frame["source_center"].map(self.node_to_index).fillna(-1).to_numpy(dtype=np.int64)
        destination = (
            frame["destination_center"].map(self.node_to_index).fillna(-1).to_numpy(dtype=np.int64)
        )
        return source, destination
