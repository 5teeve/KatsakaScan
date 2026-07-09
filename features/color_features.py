import cv2
import numpy as np
from config import HSV


def pct_rouille(hsv: np.ndarray) -> float:
    """X1 : ratio de pixels 'rouille' sur le total."""
    lower = np.array([HSV.rust_h_min, HSV.rust_s_min, HSV.rust_v_min], dtype=np.uint8)
    upper = np.array([HSV.rust_h_max, 255, 255], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)
    return float(np.count_nonzero(mask)) / mask.size
