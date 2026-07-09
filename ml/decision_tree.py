from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional

from ml.max_minority import trouver_meilleur_split, purete


@dataclass
class Node:
    feature_idx: Optional[int]   = None
    threshold:   Optional[float] = None
    left:        Optional["Node"] = None
    right:       Optional["Node"] = None
    value:       Optional[int]   = None
    proba:       float            = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


def _majority_class(y: np.ndarray) -> int:
    return int(np.bincount(y, minlength=2).argmax())


def build_tree(X: np.ndarray, y: np.ndarray,
               depth: int, max_depth: int) -> Node:
    """Construction récursive de l'arbre selon la métrique Max-Minority."""
    # Conditions d'arrêt : nœud pur ou profondeur max atteinte
    if purete(y) == 1.0 or depth >= max_depth or len(y) < 2:
        return Node(value=_majority_class(y), proba=float(y.mean()))

    n_features = X.shape[1]
    best_feat, best_seuil, best_purete = None, None, purete(y)

    for feat in range(n_features):
        seuil, p = trouver_meilleur_split(X[:, feat], y)
        if seuil is not None and p > best_purete:
            best_purete, best_feat, best_seuil = p, feat, seuil

    if best_feat is None:
        return Node(value=_majority_class(y), proba=float(y.mean()))

    mask = X[:, best_feat] <= best_seuil
    return Node(
        feature_idx=best_feat,
        threshold=best_seuil,
        left=build_tree(X[mask],  y[mask],  depth + 1, max_depth),
        right=build_tree(X[~mask], y[~mask], depth + 1, max_depth),
    )


class DecisionTreeMaxMinority:
    def __init__(self, max_depth: int = 8) -> None:
        self.max_depth = max_depth
        self.root: Optional[Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeMaxMinority":
        self.root = build_tree(X, y.astype(int), depth=0, max_depth=self.max_depth)
        return self

    def _predict_one(self, x: np.ndarray) -> int:
        node = self.root
        while not node.is_leaf:
            node = node.left if x[node.feature_idx] <= node.threshold else node.right
        return node.value

    def _proba_one(self, x: np.ndarray) -> float:
        node = self.root
        while not node.is_leaf:
            node = node.left if x[node.feature_idx] <= node.threshold else node.right
        return node.proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retourne [[p(0), p(1)], ...] — p(1) = proportion classe 1 dans la feuille."""
        p1 = np.array([self._proba_one(x) for x in X])
        return np.stack([1 - p1, p1], axis=1)
