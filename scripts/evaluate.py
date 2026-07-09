# scripts/evaluate.py
import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import PATHS
from ml.scaler import MinMaxScaler
from ml.metrics import accuracy, confusion_matrix, precision_recall_f1, roc_auc
from scripts.train import load_split


def evaluate() -> None:
    X_test, y_test = load_split(PATHS.test_csv)

    scaler = MinMaxScaler.load(PATHS.scaler_pkl)
    X_test_scaled = scaler.transform(X_test)

    with open(PATHS.model_fs_pkl.with_name("tree_from_scratch.pkl"), "rb") as f:
        tree_fs = pickle.load(f)
    with open(PATHS.model_fs_pkl, "rb") as f:
        forest_fs = pickle.load(f)
    with open(PATHS.model_sk_tree, "rb") as f:
        tree_sk = pickle.load(f)
    with open(PATHS.model_sk_rf, "rb") as f:
        forest_sk = pickle.load(f)

    models = {
        "Arbre From Scratch (Max-Minority)":  tree_fs,
        "Forest From Scratch (Max-Minority)": forest_fs,
        "Arbre sklearn (Gini)":               tree_sk,
        "Forest sklearn (Gini)":              forest_sk,
    }

    print(f"{'Modèle':<38} {'Accuracy':>9} {'Précision':>10} {'Rappel':>8} {'F1':>6} {'AUC':>7}")
    print("-" * 82)

    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        acc  = accuracy(y_test, y_pred)
        prf  = precision_recall_f1(y_test, y_pred)
        cm   = confusion_matrix(y_test, y_pred)

        if hasattr(model, "predict_proba"):
            scores  = model.predict_proba(X_test_scaled)[:, 1]
            auc_str = f"{roc_auc(y_test, scores):>7.3f}"
        else:
            auc_str = "    N/A"

        print(f"{name:<38} {acc:>9.3f} {prf['precision']:>10.3f}"
              f" {prf['recall']:>8.3f} {prf['f1']:>6.3f} {auc_str}")
        print(f"  Matrice de confusion (TN FP / FN TP):\n{cm}\n")


if __name__ == "__main__":
    evaluate()
