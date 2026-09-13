"""Self-contained baseline and graph-enhanced ETA estimators."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class RidgeRegressor:
    penalty: float = 2.0
    coefficients: np.ndarray | None = None
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, features: np.ndarray, target: np.ndarray) -> "RidgeRegressor":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-10] = 1.0
        normalized = (x - self.mean) / self.scale
        design = np.column_stack([np.ones(len(normalized)), normalized])
        penalty_matrix = np.eye(design.shape[1]) * self.penalty
        penalty_matrix[0, 0] = 0.0
        self.coefficients = np.linalg.solve(
            design.T @ design + penalty_matrix,
            design.T @ y,
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.coefficients is None or self.mean is None or self.scale is None:
            raise RuntimeError("RidgeRegressor must be fitted before predict")
        normalized = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        design = np.column_stack([np.ones(len(normalized)), normalized])
        return design @ self.coefficients


class GraphSAGEETA(nn.Module):
    """Two-layer mean-aggregation GraphSAGE with an edge ETA head."""

    def __init__(self, node_feature_dim: int, tabular_dim: int, hidden_dim: int = 24):
        super().__init__()
        self.self_1 = nn.Linear(node_feature_dim, hidden_dim)
        self.neighbor_1 = nn.Linear(node_feature_dim, hidden_dim)
        self.self_2 = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(tabular_dim + hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim * 2, 1),
        )

    def encode(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbors = adjacency @ node_features
        hidden = torch.relu(self.self_1(node_features) + self.neighbor_1(neighbors))
        hidden_neighbors = adjacency @ hidden
        return torch.relu(self.self_2(hidden) + self.neighbor_2(hidden_neighbors))

    @staticmethod
    def _lookup(embeddings: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        safe = indices.clamp(min=0)
        selected = embeddings[safe]
        return selected * indices.ge(0).unsqueeze(1)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        tabular: torch.Tensor,
        source_index: torch.Tensor,
        destination_index: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.encode(node_features, adjacency)
        source = self._lookup(embedding, source_index)
        destination = self._lookup(embedding, destination_index)
        edge = torch.cat(
            [tabular, source, destination, torch.abs(source - destination), source * destination],
            dim=1,
        )
        return self.head(edge).squeeze(1)


@dataclass
class GraphSAGETrainingResult:
    model: GraphSAGEETA
    best_epoch: int
    validation_mae_log: float


def fit_graphsage(
    node_features: np.ndarray,
    adjacency: np.ndarray,
    train_tabular: np.ndarray,
    train_source: np.ndarray,
    train_destination: np.ndarray,
    train_target: np.ndarray,
    validation_tabular: np.ndarray,
    validation_source: np.ndarray,
    validation_destination: np.ndarray,
    validation_target: np.ndarray,
    hidden_dim: int = 24,
    epochs: int = 80,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    patience: int = 12,
    seed: int = 42,
) -> GraphSAGETrainingResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GraphSAGEETA(node_features.shape[1], train_tabular.shape[1], hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.SmoothL1Loss()

    nf = torch.tensor(node_features, dtype=torch.float32)
    adj = torch.tensor(adjacency, dtype=torch.float32)
    train_x = torch.tensor(train_tabular, dtype=torch.float32)
    train_s = torch.tensor(train_source, dtype=torch.long)
    train_d = torch.tensor(train_destination, dtype=torch.long)
    train_y = torch.tensor(train_target, dtype=torch.float32)
    val_x = torch.tensor(validation_tabular, dtype=torch.float32)
    val_s = torch.tensor(validation_source, dtype=torch.long)
    val_d = torch.tensor(validation_destination, dtype=torch.long)
    val_y = torch.tensor(validation_target, dtype=torch.float32)

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        prediction = model(nf, adj, train_x, train_s, train_d)
        loss = loss_function(prediction, train_y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_prediction = model(nf, adj, val_x, val_s, val_d)
            validation_loss = torch.mean(torch.abs(validation_prediction - val_y)).item()
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return GraphSAGETrainingResult(model, best_epoch, best_loss)


def predict_graphsage(
    model: GraphSAGEETA,
    node_features: np.ndarray,
    adjacency: np.ndarray,
    tabular: np.ndarray,
    source: np.ndarray,
    destination: np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        result = model(
            torch.tensor(node_features, dtype=torch.float32),
            torch.tensor(adjacency, dtype=torch.float32),
            torch.tensor(tabular, dtype=torch.float32),
            torch.tensor(source, dtype=torch.long),
            torch.tensor(destination, dtype=torch.long),
        )
    return result.numpy()

