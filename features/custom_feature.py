import cv2
import numpy as np


def pct_jaunissement(hsv: np.ndarray) -> float:
    """
    X3 : ratio de pixels jaune pâle (chlorose) sur le total.

    Justification agronomique : avant l'apparition des pustules de
    rouille, la maladie provoque souvent un jaunissement (chlorose)
    de la feuille autour des futures lésions, par dégradation locale
    de la chlorophylle. Ce signal est plus précoce que pct_rouille
    (qui détecte les pustules déjà formées) et apporte donc une
    information complémentaire pour la classification.
    """
    lower = np.array([20, 30, 100], dtype=np.uint8)
    upper = np.array([35, 255, 255], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)
    return float(np.count_nonzero(mask)) / mask.size
