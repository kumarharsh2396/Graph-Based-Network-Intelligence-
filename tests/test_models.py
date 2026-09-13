import unittest

import numpy as np

from gbni.evaluation import eta_metrics
from gbni.models import RidgeRegressor, fit_graphsage, predict_graphsage


class ModelTests(unittest.TestCase):
    def test_ridge_fits_linear_signal(self):
        x = np.arange(20, dtype=float).reshape(-1, 1)
        y = 3.0 + 2.0 * x[:, 0]
        model = RidgeRegressor(penalty=1e-8).fit(x, y)
        self.assertLess(np.mean(np.abs(model.predict(x) - y)), 1e-5)

    def test_eta_business_metric(self):
        metrics = eta_metrics(np.array([100, 100]), np.array([110, 130]))
        self.assertEqual(metrics["within_15_percent"], 50.0)
        self.assertEqual(metrics["mae_minutes"], 20.0)

    def test_graphsage_trains_and_predicts(self):
        nodes = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        adjacency = np.eye(3, dtype=np.float32)
        tabular = np.ones((4, 2), dtype=np.float32)
        source = np.array([0, 0, 1, 1])
        destination = np.array([1, 2, 2, 0])
        target = np.array([3.0, 3.1, 3.2, 3.0], dtype=np.float32)
        result = fit_graphsage(
            nodes,
            adjacency,
            tabular,
            source,
            destination,
            target,
            tabular,
            source,
            destination,
            target,
            hidden_dim=4,
            epochs=3,
            patience=3,
        )
        prediction = predict_graphsage(
            result.model, nodes, adjacency, tabular, source, destination
        )
        self.assertEqual(prediction.shape, (4,))
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
