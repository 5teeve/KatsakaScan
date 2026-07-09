# scripts/train.py
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import PATHS, MODEL
from ml.decision_tree import DecisionTreeMaxMinority
from ml.random_forest import RandomForestMaxMinority
from ml.scaler import MinMaxScaler

FEATURE_COLS = ["pct_rouille", "rugosite", "pct_jaunissement"]
LABEL_COL = "label_malade"


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return df[FEATURE_COLS].values, df[LABEL_COL].values.astype(int)


def train() -> None:
    X_train, y_train = load_split(PATHS.train_csv)

    scaler = MinMaxScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    # 1. Arbre From Scratch (Max-Minority)
    tree_fs = DecisionTreeMaxMinority(max_depth=MODEL.max_depth)
    tree_fs.fit(X_train_scaled, y_train)

    # 2. Random Forest From Scratch (Max-Minority)
    forest_fs = RandomForestMaxMinority(n_estimators=MODEL.n_estimators,
                                         max_depth=MODEL.max_depth,
                                         random_state=MODEL.random_state)
    forest_fs.fit(X_train_scaled, y_train)

    # 3. Arbre sklearn (Gini)
    tree_sk = DecisionTreeClassifier(criterion="gini",
                                      max_depth=MODEL.max_depth,
                                      random_state=MODEL.random_state)
    tree_sk.fit(X_train_scaled, y_train)

    # 4. Random Forest sklearn (Gini)
    forest_sk = RandomForestClassifier(n_estimators=MODEL.n_estimators,
                                        max_depth=MODEL.max_depth,
                                        random_state=MODEL.random_state)
    forest_sk.fit(X_train_scaled, y_train)

    PATHS.model_fs_pkl.parent.mkdir(parents=True, exist_ok=True)

    with open(PATHS.model_fs_pkl.with_name("tree_from_scratch.pkl"), "wb") as f:
        pickle.dump(tree_fs, f)
    with open(PATHS.model_fs_pkl, "wb") as f:
        pickle.dump(forest_fs, f)
    with open(PATHS.model_sk_tree, "wb") as f:
        pickle.dump(tree_sk, f)
    with open(PATHS.model_sk_rf, "wb") as f:
        pickle.dump(forest_sk, f)

    scaler.save(PATHS.scaler_pkl)

    print("✅ 4 modèles entraînés et sauvegardés dans models/.")


if __name__ == "__main__":
    train()
