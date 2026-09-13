import unittest
from pathlib import Path

import pandas as pd

from gbni.reporting import write_strategy_memo


class ReportingTests(unittest.TestCase):
    def test_memo_contains_decisions_and_caveat(self):
        metrics = pd.DataFrame(
            [
                {"segment": "overall", "model": "tabular_baseline", "mae_minutes": 100, "within_15_percent": 20},
                {"segment": "overall", "model": "graphsage", "mae_minutes": 80, "within_15_percent": 30},
            ]
        )
        hubs = pd.DataFrame(
            [
                {
                    "facility_name": f"Hub {i}",
                    "facility_id": f"H{i}",
                    "bottleneck_score": 90 - i,
                    "sla_breach_contribution_pct": 2.0,
                    "excess_time_contribution_pct": 3.0,
                }
                for i in range(5)
            ]
        )
        corridors = pd.DataFrame(
            [
                {
                    "source_center": "A",
                    "destination_center": "B",
                    "median_movement_delay_ratio": 1.5,
                    "observation_count": 20,
                    "recommended_intervention": "Assess parallel route",
                }
            ]
        )
        simulation = {
            "improvement_fraction": 0.3,
            "expected_prevented_proxy_breaches": 10.0,
            "late_delivery_reduction_pct": 5.0,
            "movement_minutes_saved": 100,
            "revenue_per_prevented_breach": 1000,
            "scenario_revenue_recovered": 10000,
        }
        routes = pd.DataFrame(columns=["evidence_level"])
        directory = Path("tests") / "_test_outputs"
        directory.mkdir(exist_ok=True)
        path = directory / "memo.md"
        write_strategy_memo(path, metrics, hubs, corridors, simulation, routes)
        text = path.read_text(encoding="utf-8")
        path.unlink()
        self.assertIn("Executive recommendation", text)
        self.assertIn("not booked revenue", text)


if __name__ == "__main__":
    unittest.main()
