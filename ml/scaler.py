# ml/scaler.py
import numpy as np
import pickle
from pathlib import Path


class MinMaxScaler:
    def __init__(self) -> None:
        self._min: np.ndarray | None = None
        self._range: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        self._min = X.min(axis=0)
        self._range = X.max(axis=0) - self._min
        self._range[self._range == 0] = 1  # évite division par zéro
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self._min) / self._range

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"min": self._min, "range": self._range}, f)

    @classmethod
    def load(cls, path: Path) -> "MinMaxScaler":
        with open(path, "rb") as f:
            data = pickle.load(f)
        scaler = cls()
        scaler._min = data["min"]
        scaler._range = data["range"]
        return scaler
