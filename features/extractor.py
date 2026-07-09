from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import NamedTuple

from features.color_features import pct_rouille
from features.texture_features import rugosite_sobel
from features.custom_feature import pct_jaunissement


class FeatureVector(NamedTuple):
    pct_rouille:     float
    rugosite:        float
    pct_jaunissement: float


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Image illisible : {path}")
    return img


def extract_features(path: Path) -> FeatureVector:
    """Orchestre l'extraction des 3 caractéristiques d'une image."""
    img  = _load_image(path)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return FeatureVector(
        pct_rouille    = pct_rouille(hsv),
        rugosite       = rugosite_sobel(gray),
        pct_jaunissement = pct_jaunissement(hsv),
    )
