import unittest

import networkx as nx
import pandas as pd

from gbni.bottlenecks import rank_bottleneck_hubs
from gbni.graph_analysis import GraphArtifacts
from gbni.simulation import simulate_hub_upgrades


class BusinessAnalysisTests(unittest.TestCase):
    def test_bottleneck_contributions_sum_to_one_hundred(self):
        nodes = pd.DataFrame(
            {
                "facility_id": ["A", "B"],
                "betweenness_centrality": [0.0, 1.0],
                "weighted_betweenness_centrality": [0.0, 1.0],
                "throughput_legs": [1, 3],
                "attributed_excess_min": [2, 8],
                "attributed_breaches": [1, 4],
                "clustering_coefficient": [0.5, 0.0],
            }
        )
        legs = pd.DataFrame(
            {
                "source_center": ["A"],
                "source_name": ["Alpha"],
                "destination_center": ["B"],
                "destination_name": ["Beta"],
            }
        )
        artifacts = GraphArtifacts(nx.DiGraph(), pd.DataFrame(), pd.DataFrame(), nodes)
        ranked = rank_bottleneck_hubs(artifacts, legs)
        self.assertAlmostEqual(ranked.sla_breach_contribution_pct.sum(), 100.0)
        self.assertEqual(ranked.iloc[0].facility_id, "B")

    def test_upgrade_scenario_never_increases_breaches(self):
        legs = pd.DataFrame(
            {
                "source_center": ["A", "B"],
                "destination_center": ["B", "C"],
                "actual_movement_min": [14.0, 20.0],
                "osrm_time_min": [10.0, 10.0],
            }
        )
        hubs = pd.DataFrame({"facility_id": ["B"]})
        _, summary = simulate_hub_upgrades(legs, hubs, improvement_fraction=1.0)
        self.assertLessEqual(
            summary["expected_remaining_proxy_breaches"],
            summary["original_proxy_breaches"],
        )


if __name__ == "__main__":
    unittest.main()
