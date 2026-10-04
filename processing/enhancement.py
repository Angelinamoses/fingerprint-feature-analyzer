"""
Oriented Gabor ridge enhancement (Hong, Wan & Jain style).

This file was empty in the original repository, so ridges were
binarised straight from a CLAHE image. Without ridge-oriented
filtering, adaptive thresholding yields broken ridges, bridges and
spurs, and the skeleton produces mostly false minutiae.
"""

import cv2
import numpy as np


def _gabor_kernel(phi, period, sigma_n, sigma_a, size):
    """
    Even-symmetric Gabor kernel.

    phi:     direction of the ridge NORMAL (the modulation direction)
    sigma_n: Gaussian sigma across the ridge
    sigma_a: Gaussian sigma along the ridge
    """
    half = size // 2
    ys, xs = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)

    across = xs * np.cos(phi) + ys * np.sin(phi)
    along = -xs * np.sin(phi) + ys * np.cos(phi)

    kernel = np.exp(
        -(across ** 2) / (2.0 * sigma_n ** 2)
        - (along ** 2) / (2.0 * sigma_a ** 2)
    ) * np.cos(2.0 * np.pi * across / period)

    return kernel - kernel.mean()


def gabor_enhance(
    image,
    orientation,
    frequency=None,
    mask=None,
    n_orientations=16,
    sigma_n_factor=0.45,
    sigma_a_factor=0.9,
    ridge_polarity="dark",
):
    """
    Enhance ridges with an oriented Gabor filter bank.

    Args:
        image:       Grayscale image (uint8 / float).
        orientation: Ridge orientation map (radians, period pi).
        frequency:   Ridge frequency in cycles/pixel (None -> 0.1).
        mask:        Foreground mask.
        n_orientations: number of discrete filter orientations.
        sigma_n_factor / sigma_a_factor:
            Gaussian envelope sigmas as a fraction of the ridge period.
        ridge_polarity:
            "dark" (ink ridges on light paper, as in NIST SD4) or
            "bright".

    Returns:
        enhanced: uint8 image, dark ridges on a mid-grey (128)
                  background, zero outside the mask -> 255.
        response: float32 filter response (positive on ridges).
    """

    img = image.astype(np.float32)

    if frequency is None or not np.isfinite(frequency) or frequency <= 0:
        frequency = 0.1

    period = float(np.clip(1.0 / frequency, 4.0, 25.0))

    inside = (mask > 0) if mask is not None else np.ones(img.shape, bool)

    # --- normalise: z-score inside the print, then remove slow
    #     illumination variation
    if np.any(inside):
        mu = float(img[inside].mean())
        sd = float(img[inside].std()) + 1e-6
    else:
        mu, sd = float(img.mean()), float(img.std()) + 1e-6

    z = (img - mu) / sd
    z[~inside] = 0.0

    weights = inside.astype(np.float32)
    sigma_bg = 2.0 * period
    background = cv2.GaussianBlur(z * weights, (0, 0), sigma_bg)
    background /= cv2.GaussianBlur(weights, (0, 0), sigma_bg) + 1e-6
    z = (z - background) * weights

    if ridge_polarity == "dark":
        z = -z                      # make ridges positive

    # --- filter bank
    size = int(round(period * 4.5)) | 1
    sigma_n = sigma_n_factor * period
    sigma_a = sigma_a_factor * period

    kernels = [
        _gabor_kernel(
            b * np.pi / n_orientations + np.pi / 2.0,
            period,
            sigma_n,
            sigma_a,
            size,
        )
        for b in range(n_orientations)
    ]

    bank = np.stack([cv2.filter2D(z, -1, k) for k in kernels], axis=0)

    theta = np.mod(np.nan_to_num(orientation, nan=0.0), np.pi)
    position = theta / np.pi * n_orientations
    lower = np.floor(position).astype(np.int64) % n_orientations
    upper = (lower + 1) % n_orientations
    weight_upper = (position - np.floor(position)).astype(np.float32)

    r_lower = np.take_along_axis(bank, lower[None], axis=0)[0]
    r_upper = np.take_along_axis(bank, upper[None], axis=0)[0]

    response = ((1.0 - weight_upper) * r_lower + weight_upper * r_upper)
    response = response.astype(np.float32)
    response[~inside] = 0.0

    scale = float(np.std(response[inside])) + 1e-6 if np.any(inside) else 1.0

    enhanced = np.clip(128.0 - 55.0 * response / scale, 0, 255)
    enhanced[~inside] = 255.0

    return enhanced.astype(np.uint8), response
