import unittest

import pandas as pd

from gbni.data_pipeline import build_leg_table, data_quality_report


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame(
            {
                "data": ["training", "training", "test"],
                "trip_creation_time": ["2024-01-01"] * 2 + ["2024-01-02"],
                "route_schedule_uuid": ["r1", "r1", "r2"],
                "route_type": ["FTL", "FTL", "Carting"],
                "trip_uuid": ["t1", "t1", "t2"],
                "source_center": ["A", "A", "B"],
                "source_name": ["Alpha", "Alpha", "Beta"],
                "destination_center": ["B", "B", "C"],
                "destination_name": ["Beta", "Beta", "Gamma"],
                "od_start_time": ["2024-01-01 07:00"] * 2 + ["2024-01-02 21:00"],
                "od_end_time": ["2024-01-01 09:00"] * 2 + ["2024-01-02 22:00"],
                "start_scan_to_end_scan": [120, 120, 60],
                "actual_distance_to_destination": [10, 20, 15],
                "actual_time": [30, 70, 40],
                "osrm_time": [25, 50, 30],
                "osrm_distance": [11, 22, 16],
                "segment_actual_time": [30, 40, 40],
                "segment_osrm_time": [25, 25, 30],
                "segment_osrm_distance": [11, 11, 16],
            }
        )

    def test_checkpoints_collapse_to_leg(self):
        legs = build_leg_table(self.raw)
        self.assertEqual(len(legs), 2)
        first = legs.loc[legs.trip_uuid.eq("t1")].iloc[0]
        self.assertEqual(first.checkpoint_count, 2)
        self.assertEqual(first.actual_movement_min, 70)
        self.assertEqual(first.osrm_time_min, 50)
        self.assertEqual(first.dwell_time_min, 50)
        self.assertAlmostEqual(first.movement_delay_ratio, 1.4)
        self.assertEqual(first.time_bucket, "morning")

    def test_quality_report_is_serializable(self):
        legs = build_leg_table(self.raw)
        report = data_quality_report(self.raw, legs)
        self.assertEqual(report["raw_rows"], 3)
        self.assertEqual(report["leg_rows"], 2)
        self.assertEqual(report["unique_facilities"], 3)


if __name__ == "__main__":
    unittest.main()
