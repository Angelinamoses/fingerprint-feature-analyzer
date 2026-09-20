import numpy as np
from scipy.signal import find_peaks


def estimate_ridge_density(
    image,
    rows=None,
    min_peak_distance=5,
    prominence=20
):
    """
    Estimate ridge density from multiple horizontal
    fingerprint profiles.

    Returns:
        density: robust median ridge density
        densities: individual row density estimates
    """

    if rows is None:
        height = image.shape[0]
        rows = np.linspace(
            height * 0.25,
            height * 0.75,
            8
        ).astype(int)

    densities = []

    for y in rows:
        profile = image[y, :].astype(np.float32)

        peaks, _ = find_peaks(
            profile,
            distance=min_peak_distance,
            prominence=prominence
        )

        if len(peaks) < 5:
            continue

        density = len(peaks) / image.shape[1]
        densities.append(density)

    if not densities:
        return None, []

    density = np.median(densities)

    return density, np.array(densities)