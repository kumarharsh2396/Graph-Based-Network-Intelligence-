import unittest

import pandas as pd

from gbni.benchmark import temporal_train_validation_split


class BenchmarkTests(unittest.TestCase):
    def test_temporal_split_keeps_trips_separate(self):
        frame = pd.DataFrame(
            {
                "trip_uuid": ["a", "a", "b", "c", "d"],
                "trip_creation_time": pd.to_datetime(
                    ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
                ),
            }
        )
        fit, validation = temporal_train_validation_split(frame, 0.25)
        self.assertFalse(set(fit.trip_uuid) & set(validation.trip_uuid))
        self.assertLess(fit.trip_creation_time.max(), validation.trip_creation_time.min())


if __name__ == "__main__":
    unittest.main()
