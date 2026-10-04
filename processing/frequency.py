import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from processing.orientation import estimate_orientation


def _normal_signature(image, cx, cy, theta, length, width):
    """
    Average the image along the ridge direction and return the 1-D
    profile ACROSS the ridges ("x-signature").

    theta is the ridge orientation, so the sampling axis is the ridge
    normal.
    """
    tx, ty = np.cos(theta), np.sin(theta)       # along the ridge
    nx, ny = -np.sin(theta), np.cos(theta)      # across the ridge

    u = np.arange(length, dtype=np.float32) - (length - 1) / 2.0
    v = np.arange(width, dtype=np.float32) - (width - 1) / 2.0

    U, V = np.meshgrid(u, v)

    xs = cx + U * nx + V * tx
    ys = cy + U * ny + V * ty

    patch = map_coordinates(
        image,
        [ys, xs],
        order=1,
        mode="nearest",
    )

    return patch.mean(axis=0)


def _dominant_frequency(signature, min_freq, max_freq, min_peak_ratio):
    """
    Dominant spatial frequency (cycles / pixel) of a 1-D ridge
    profile via a windowed, zero-padded FFT.

    Returns None when no clear periodic component exists.
    """
    s = signature - signature.mean()

    if np.std(s) < 1e-3:
        return None

    s = s * np.hanning(len(s))

    n_fft = 512
    spectrum = np.abs(np.fft.rfft(s, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0)

    band = (freqs >= min_freq) & (freqs <= max_freq)

    if not np.any(band):
        return None

    band_spectrum = spectrum[band]
    band_freqs = freqs[band]

    peak_index = int(np.argmax(band_spectrum))
    peak = band_spectrum[peak_index]

    if peak < min_peak_ratio * (np.median(band_spectrum) + 1e-9):
        return None

    return float(band_freqs[peak_index])


def estimate_ridge_frequency(
    image,
    orientation=None,
    mask=None,
    coherence=None,
    window_length=48,
    window_width=16,
    step=24,
    min_period=4.0,
    max_period=25.0,
    min_coherence=0.40,
    min_peak_ratio=3.0,
):
    """
    Estimate ridge frequency from many oriented, local ridge profiles.

    The original estimator used 8 *horizontal* intensity rows. A
    horizontal line crosses oblique ridges at a stretched distance
    (period / sin(angle)), so spacing was overestimated and depended
    on the ridge direction (e.g. 15.5 px reported where the true
    period was ~10-12 px). Here the profile is taken along the ridge
    NORMAL at many locations, so the result is independent of the
    local ridge direction.

    Args:
        image:       Grayscale image (denoised, not masked).
        orientation: Ridge orientation map (radians). Computed if None.
        mask:        Foreground mask.
        coherence:   Orientation coherence (computed if orientation is None).

    Returns:
        frequency: median frequency in cycles/pixel (or None)
        spacing:   median ridge period in pixels (or None)
        estimates: per-location frequency estimates (np.ndarray)
    """

    img = image.astype(np.float32)

    if orientation is None:
        orientation, coherence = estimate_orientation(
            img,
            mask=mask,
        )

    height, width = img.shape[:2]

    if mask is not None:
        inside = (mask > 0).astype(np.uint8)
        distance = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
    else:
        distance = np.full((height, width), 1e6, dtype=np.float32)

    margin = window_length / 2.0 + 2.0

    estimates = []

    for cy in range(int(margin), int(height - margin), step):

        for cx in range(int(margin), int(width - margin), step):

            if distance[cy, cx] < margin:
                continue

            if coherence is not None and coherence[cy, cx] < min_coherence:
                continue

            theta = float(orientation[cy, cx])

            if not np.isfinite(theta):
                continue

            signature = _normal_signature(
                img,
                cx,
                cy,
                theta,
                window_length,
                window_width,
            )

            freq = _dominant_frequency(
                signature,
                1.0 / max_period,
                1.0 / min_period,
                min_peak_ratio,
            )

            if freq is not None:
                estimates.append(freq)

    if len(estimates) < 3:
        return None, None, np.array([])

    estimates = np.asarray(estimates, dtype=np.float64)

    frequency = float(np.median(estimates))
    spacing = 1.0 / frequency

    return frequency, spacing, estimates
