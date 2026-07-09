from __future__ import annotations
import numpy as np


def purete(y: np.ndarray) -> float:
    """P(t) = proportion de la classe majoritaire dans le nœud."""
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y, minlength=2)
    return float(counts.max()) / len(y)


def trouver_meilleur_split(X_column: np.ndarray,
                            y: np.ndarray) -> tuple[float | None, float]:
    """
    Teste tous les seuils candidats pour une variable et retourne
    (meilleur_seuil, purete_max_ponderee).

    Retourne (None, purete(y)) si aucun split n'améliore la pureté
    du nœud parent.
    """
    n = len(y)

    # Étape 1 : trier par X_column
    order = np.argsort(X_column)
    x_sorted, y_sorted = X_column[order], y[order]

    # Étape 2 : initialisation
    best_seuil: float | None = None
    best_purete: float = purete(y)

    unique_vals = np.unique(x_sorted)
    if len(unique_vals) < 2:
        return None, best_purete

    thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

    # Étape 3 : parcourir les seuils candidats
    for s in thresholds:
        split = np.searchsorted(x_sorted, s, side="right")
        if split == 0 or split == n:
            continue

        y_gauche, y_droite = y_sorted[:split], y_sorted[split:]
        p_gauche, p_droite = purete(y_gauche), purete(y_droite)

        p_split = (len(y_gauche) / n) * p_gauche + (len(y_droite) / n) * p_droite

        if p_split > best_purete:
            best_purete = p_split
            best_seuil = s

    # Étape 4 : retour
    return best_seuil, best_purete
