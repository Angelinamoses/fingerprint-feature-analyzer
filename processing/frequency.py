import numpy as np
from scipy.signal import find_peaks


def estimate_ridge_frequency(
    image,
    rows=None,
    min_peak_distance=5,
    prominence=20
):
    """
    Estimate fingerprint ridge frequency from multiple
    horizontal intensity profiles.

    Returns:
        frequency: robust median frequency in cycles/pixel
        spacing: robust median ridge spacing in pixels
        frequencies: individual valid row estimates
    """

    if rows is None:
        height = image.shape[0]
        rows = np.linspace(
            height * 0.25,
            height * 0.75,
            8
        ).astype(int)

    frequencies = []

    for y in rows:
        profile = image[y, :].astype(np.float32)

        peaks, _ = find_peaks(
            profile,
            distance=min_peak_distance,
            prominence=prominence
        )

        if len(peaks) < 5:
            continue

        spacing = np.diff(peaks)
        median_spacing = np.median(spacing)

        if median_spacing > 0:
            frequencies.append(1.0 / median_spacing)

    if not frequencies:
        return None, None, []

    frequencies = np.array(frequencies)

    frequency = np.median(frequencies)
    spacing = 1.0 / frequency

    return frequency, spacing, frequencies