import unittest

import pandas as pd

from gbni.graph_analysis import build_graph_artifacts, graph_summary


class GraphAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.legs = pd.DataFrame(
            {
                "trip_uuid": ["t1", "t2", "t3", "t4"],
                "source_center": ["A", "A", "B", "B"],
                "destination_center": ["B", "B", "C", "C"],
                "route_type": ["FTL", "FTL", "Carting", "Carting"],
                "time_bucket": ["morning", "morning", "evening", "evening"],
                "osrm_time_min": [10, 10, 10, 10],
                "osrm_distance_km": [5, 5, 6, 6],
                "actual_movement_min": [14, 16, 8, 9],
                "actual_elapsed_min": [20, 22, 12, 13],
                "movement_delay_ratio": [1.4, 1.6, 0.8, 0.9],
                "movement_excess_min": [4, 6, 0, 0],
                "dwell_time_min": [6, 6, 4, 4],
            }
        )

    def test_directed_edges_and_chronic_delay(self):
        artifacts = build_graph_artifacts(self.legs, minimum_observations=2)
        summary = graph_summary(artifacts)
        self.assertEqual(summary["nodes"], 3)
        self.assertEqual(summary["directed_edges"], 2)
        self.assertEqual(summary["chronic_delay_corridors"], 1)

    def test_middle_node_is_structural_chokepoint(self):
        artifacts = build_graph_artifacts(self.legs, minimum_observations=2)
        nodes = artifacts.nodes.set_index("facility_id")
        self.assertGreater(
            nodes.loc["B", "betweenness_centrality"],
            nodes.loc["A", "betweenness_centrality"],
        )


if __name__ == "__main__":
    unittest.main()
