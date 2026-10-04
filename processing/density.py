import numpy as np

from processing.frequency import estimate_ridge_frequency


def estimate_ridge_density(
    image,
    orientation=None,
    mask=None,
    coherence=None,
    **kwargs,
):
    """
    Ridge density = number of ridges per pixel measured ACROSS the ridges.

    The original implementation counted intensity peaks along a few
    horizontal rows and divided by the image width. That mixes in the
    local ridge direction (oblique ridges are crossed more rarely per
    pixel along the row than perpendicular ones) and counts peaks over
    the whole row width, including background.

    The estimate now comes from the same oriented-profile estimator
    used for ridge frequency, so density (ridges/pixel) is consistent
    with frequency and spacing. Multiply by pixels-per-mm
    (19.69 for 500 dpi) to get ridges/mm.

    Returns:
        density:   median ridges/pixel (or None)
        densities: per-location estimates (np.ndarray)
    """

    frequency, _, estimates = estimate_ridge_frequency(
        image,
        orientation=orientation,
        mask=mask,
        coherence=coherence,
        **kwargs,
    )

    if frequency is None:
        return None, np.array([])

    return float(np.median(estimates)), np.asarray(estimates)
