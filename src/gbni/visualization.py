"""Dependency-free SVG rendering for a prioritized network subgraph."""

from __future__ import annotations

from html import escape
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def write_bottleneck_svg(
    graph: nx.DiGraph,
    ranked_hubs: pd.DataFrame,
    ranked_corridors: pd.DataFrame,
    output_path: str | Path,
    max_nodes: int = 60,
) -> Path:
    """Render the most important hubs and corridors to a standalone SVG."""
    top_hubs = ranked_hubs.head(max_nodes // 2)["facility_id"].tolist()
    top_edges = ranked_corridors.head(max_nodes // 2)[
        ["source_center", "destination_center"]
    ]
    selected = list(dict.fromkeys(top_hubs + top_edges.to_numpy().ravel().tolist()))[:max_nodes]
    subgraph = graph.subgraph(selected).copy()
    if not subgraph.nodes:
        raise ValueError("Cannot visualize an empty graph")
    positions = nx.spring_layout(subgraph, seed=42, weight="weight", iterations=80)
    width, height, margin = 1200, 800, 55
    xs = np.array([positions[n][0] for n in subgraph.nodes])
    ys = np.array([positions[n][1] for n in subgraph.nodes])

    def scale(value: float, values: np.ndarray, extent: int) -> float:
        span = float(values.max() - values.min()) or 1.0
        return margin + (value - float(values.min())) / span * (extent - 2 * margin)

    xy = {
        node: (scale(positions[node][0], xs, width), scale(positions[node][1], ys, height))
        for node in subgraph.nodes
    }
    top_five = set(ranked_hubs.head(5)["facility_id"])
    chronic = set(
        map(
            tuple,
            ranked_corridors.loc[
                ranked_corridors["chronic_delay"], ["source_center", "destination_center"]
            ].itertuples(index=False, name=None),
        )
    )
    score_map = ranked_hubs.set_index("facility_id")["bottleneck_score"].to_dict()

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs><marker id=\"arrow\" markerWidth=\"8\" markerHeight=\"8\" refX=\"7\" refY=\"3\" orient=\"auto\"><path d=\"M0,0 L0,6 L8,3 z\" fill=\"#64748b\"/></marker></defs>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="35" y="35" font-family="Arial" font-size="23" font-weight="700" fill="#0f172a">Priority Logistics Network</text>',
        '<text x="35" y="58" font-family="Arial" font-size="13" fill="#475569">Red = top five bottleneck hubs; orange = chronic-delay corridor</text>',
    ]
    for source, destination in subgraph.edges:
        x1, y1 = xy[source]
        x2, y2 = xy[destination]
        color = "#f97316" if (source, destination) in chronic else "#94a3b8"
        stroke = 2.4 if (source, destination) in chronic else 1.0
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{stroke}" opacity="0.72" marker-end="url(#arrow)"/>'
        )
    for node in subgraph.nodes:
        x, y = xy[node]
        score = float(score_map.get(node, 0.0))
        radius = 5.0 + 8.0 * score / 100.0
        fill = "#dc2626" if node in top_five else "#2563eb"
        label = escape(str(node))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" opacity="0.88"><title>{label}: score {score:.1f}</title></circle>')
        if node in top_five:
            parts.append(f'<text x="{x + radius + 3:.1f}" y="{y + 4:.1f}" font-family="Arial" font-size="11" fill="#0f172a">{label}</text>')
    parts.append("</svg>")
    destination = Path(output_path)
    destination.write_text("\n".join(parts), encoding="utf-8")
    return destination

