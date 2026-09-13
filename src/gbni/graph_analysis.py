"""Directed logistics graph construction and structural metrics."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class GraphArtifacts:
    graph: nx.DiGraph
    edges: pd.DataFrame
    stratified_edges: pd.DataFrame
    nodes: pd.DataFrame


def aggregate_edges(
    legs: pd.DataFrame,
    late_ratio_threshold: float = 1.20,
    minimum_observations: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build overall and route/time-stratified corridor summaries."""
    frame = legs.copy()
    frame["is_late_proxy"] = frame["movement_delay_ratio"].gt(late_ratio_threshold)

    overall = (
        frame.groupby(["source_center", "destination_center"], observed=True)
        .agg(
            observation_count=("trip_uuid", "size"),
            trip_count=("trip_uuid", "nunique"),
            route_type_count=("route_type", "nunique"),
            median_osrm_time_min=("osrm_time_min", "median"),
            median_osrm_distance_km=("osrm_distance_km", "median"),
            median_actual_movement_min=("actual_movement_min", "median"),
            median_actual_elapsed_min=("actual_elapsed_min", "median"),
            median_movement_delay_ratio=("movement_delay_ratio", "median"),
            mean_movement_delay_ratio=("movement_delay_ratio", "mean"),
            median_excess_min=("movement_excess_min", "median"),
            total_excess_min=("movement_excess_min", "sum"),
            breach_rate=("is_late_proxy", "mean"),
        )
        .reset_index()
    )
    overall["chronic_delay"] = (
        overall["observation_count"].ge(minimum_observations)
        & overall["median_movement_delay_ratio"].gt(late_ratio_threshold)
    )

    stratified = (
        frame.groupby(
            ["source_center", "destination_center", "route_type", "time_bucket"],
            observed=True,
        )
        .agg(
            observation_count=("trip_uuid", "size"),
            median_osrm_time_min=("osrm_time_min", "median"),
            median_actual_movement_min=("actual_movement_min", "median"),
            median_actual_elapsed_min=("actual_elapsed_min", "median"),
            median_movement_delay_ratio=("movement_delay_ratio", "median"),
            median_excess_min=("movement_excess_min", "median"),
            breach_rate=("is_late_proxy", "mean"),
        )
        .reset_index()
    )
    stratified["chronic_delay"] = (
        stratified["observation_count"].ge(minimum_observations)
        & stratified["median_movement_delay_ratio"].gt(late_ratio_threshold)
    )
    return overall, stratified


def build_graph(edge_table: pd.DataFrame) -> nx.DiGraph:
    """Create the directed topology with positive travel-time path weights."""
    graph = nx.DiGraph()
    for row in edge_table.itertuples(index=False):
        attributes = row._asdict()
        source = attributes.pop("source_center")
        destination = attributes.pop("destination_center")
        attributes["weight"] = max(float(attributes["median_osrm_time_min"]), 1e-6)
        graph.add_edge(source, destination, **attributes)
    return graph


def compute_node_metrics(graph: nx.DiGraph, legs: pd.DataFrame) -> pd.DataFrame:
    """Compute graph structure and observed operational exposure per facility."""
    nodes = list(graph.nodes)
    if not nodes:
        return pd.DataFrame(columns=["facility_id"])

    unweighted_betweenness = nx.betweenness_centrality(graph, normalized=True)
    weighted_betweenness = nx.betweenness_centrality(graph, weight="weight", normalized=True)
    clustering = nx.clustering(graph)

    component_size: dict[str, int] = {}
    for component in nx.weakly_connected_components(graph):
        size = len(component)
        component_size.update({node: size for node in component})

    frame = legs.copy()
    if "is_late_proxy" not in frame:
        frame["is_late_proxy"] = frame["movement_delay_ratio"].gt(1.20)
    frame["positive_excess_min"] = frame["movement_excess_min"].clip(lower=0)

    outgoing = frame.groupby("source_center").agg(
        outgoing_legs=("trip_uuid", "size"),
        outgoing_breaches=("is_late_proxy", "sum"),
        outgoing_excess_min=("positive_excess_min", "sum"),
        source_dwell_min=("dwell_time_min", "sum"),
    )
    incoming = frame.groupby("destination_center").agg(
        incoming_legs=("trip_uuid", "size"),
        incoming_breaches=("is_late_proxy", "sum"),
        incoming_excess_min=("positive_excess_min", "sum"),
    )

    records = []
    for node in nodes:
        records.append(
            {
                "facility_id": node,
                "in_degree": graph.in_degree(node),
                "out_degree": graph.out_degree(node),
                "betweenness_centrality": unweighted_betweenness[node],
                "weighted_betweenness_centrality": weighted_betweenness[node],
                "clustering_coefficient": clustering[node],
                "weak_component_size": component_size[node],
            }
        )
    metrics = pd.DataFrame(records).set_index("facility_id")
    metrics = metrics.join(outgoing, how="left").join(incoming, how="left").fillna(0)
    metrics["throughput_legs"] = metrics["outgoing_legs"] + metrics["incoming_legs"]
    metrics["attributed_breaches"] = (
        metrics["outgoing_breaches"] + metrics["incoming_breaches"]
    ) / 2.0
    metrics["attributed_excess_min"] = (
        metrics["outgoing_excess_min"] + metrics["incoming_excess_min"]
    ) / 2.0
    return metrics.reset_index()


def build_graph_artifacts(
    legs: pd.DataFrame,
    late_ratio_threshold: float = 1.20,
    minimum_observations: int = 5,
) -> GraphArtifacts:
    """Build all graph tables from a supplied historical window."""
    edges, stratified = aggregate_edges(
        legs,
        late_ratio_threshold=late_ratio_threshold,
        minimum_observations=minimum_observations,
    )
    graph = build_graph(edges)
    scored_legs = legs.copy()
    scored_legs["is_late_proxy"] = scored_legs["movement_delay_ratio"].gt(
        late_ratio_threshold
    )
    nodes = compute_node_metrics(graph, scored_legs)
    return GraphArtifacts(graph=graph, edges=edges, stratified_edges=stratified, nodes=nodes)


def graph_summary(artifacts: GraphArtifacts) -> dict[str, int | float]:
    graph = artifacts.graph
    largest_weak = max((len(c) for c in nx.weakly_connected_components(graph)), default=0)
    largest_strong = max((len(c) for c in nx.strongly_connected_components(graph)), default=0)
    return {
        "nodes": graph.number_of_nodes(),
        "directed_edges": graph.number_of_edges(),
        "weak_components": nx.number_weakly_connected_components(graph),
        "strong_components": nx.number_strongly_connected_components(graph),
        "largest_weak_component": largest_weak,
        "largest_strong_component": largest_strong,
        "chronic_delay_corridors": int(artifacts.edges["chronic_delay"].sum()),
    }

