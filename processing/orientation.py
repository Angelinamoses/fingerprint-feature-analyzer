import cv2
import numpy as np


def estimate_orientation(image, block_size=16):
    """
    Estimate local fingerprint ridge orientation
    and orientation coherence.

    Returns:
        orientation: orientation map in radians
        coherence: confidence/coherence map from 0 to 1
    """

    gx = cv2.Sobel(
        image,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    kernel_size = block_size | 1

    gxx = cv2.GaussianBlur(
        gx * gx,
        (kernel_size, kernel_size),
        0
    )

    gyy = cv2.GaussianBlur(
        gy * gy,
        (kernel_size, kernel_size),
        0
    )

    gxy = cv2.GaussianBlur(
        gx * gy,
        (kernel_size, kernel_size),
        0
    )

    orientation = 0.5 * np.arctan2(
        2 * gxy,
        gxx - gyy
    )

    numerator = np.sqrt(
        (gxx - gyy) ** 2 +
        (2 * gxy) ** 2
    )

    denominator = gxx + gyy + 1e-8

    coherence = numerator / denominator
    coherence = np.clip(coherence, 0, 1)

    return orientation, coherence