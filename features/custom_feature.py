import cv2
import numpy as np


def pct_jaunissement(hsv: np.ndarray) -> float:
    lower = np.array([20, 30, 100], dtype=np.uint8)
    upper = np.array([35, 255, 255], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)
    return float(np.count_nonzero(mask)) / mask.size
