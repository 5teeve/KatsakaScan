import cv2
import numpy as np


def rugosite_sobel(gray: np.ndarray) -> float:
    """X2 : variance des gradients de Sobel (rugosité de surface)."""
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return float(magnitude.var())
