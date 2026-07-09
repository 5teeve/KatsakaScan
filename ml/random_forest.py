from __future__ import annotations
import numpy as np
from ml.decision_tree import DecisionTreeMaxMinority


class RandomForestMaxMinority:
    def __init__(self, n_estimators: int = 100,
                 max_depth: int = 8,
                 random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.random_state = random_state
        self.trees_: list[DecisionTreeMaxMinority] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestMaxMinority":
        np.random.seed(self.random_state)
        n = len(y)
        y = y.astype(int)
        self.trees_ = []

        for _ in range(self.n_estimators):
            # Bagging : échantillonnage avec remplacement
            indices = np.random.choice(n, size=n, replace=True)
            tree = DecisionTreeMaxMinority(max_depth=self.max_depth)
            tree.fit(X[indices], y[indices])
            self.trees_.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vote majoritaire des arbres."""
        votes = np.stack([t.predict(X) for t in self.trees_], axis=1)
        return (votes.mean(axis=1) >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Moyenne des probabilités feuille de chaque arbre — [[p(0), p(1)], ...]."""
        p1 = np.stack([t.predict_proba(X)[:, 1] for t in self.trees_], axis=1).mean(axis=1)
        return np.stack([1 - p1, p1], axis=1)
