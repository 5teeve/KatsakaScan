from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent


@dataclass(frozen=True)
class Paths:
    raw_healthy:   Path = ROOT / "data/raw/saines"
    raw_diseased:  Path = ROOT / "data/raw/malades"
    features_csv:  Path = ROOT / "data/processed/features.csv"
    train_csv:     Path = ROOT / "data/splits/train.csv"
    test_csv:      Path = ROOT / "data/splits/test.csv"
    model_fs_pkl:   Path = ROOT / "models/forest_from_scratch.pkl"
    model_sk_tree:  Path = ROOT / "models/tree_sklearn.pkl"
    model_sk_rf:    Path = ROOT / "models/forest_sklearn.pkl"
    scaler_pkl:    Path = ROOT / "models/scaler.pkl"
    uploads_dir:   Path = ROOT / "uploads"


@dataclass(frozen=True)
class HSVThresholds:
    """Plage HSV pour isoler les teintes 'rouille' (marron/jaune/orange)."""
    rust_h_min: int = 5
    rust_h_max: int = 30
    rust_s_min: int = 50
    rust_v_min: int = 50


@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int = 100
    max_depth:    int = 8
    random_state: int = 42


@dataclass(frozen=True)
class TrainConfig:
    test_size:    float = 0.20   # split 80/20
    random_state: int   = 42


PATHS = Paths()
HSV   = HSVThresholds()
MODEL = ModelConfig()
TRAIN = TrainConfig()
