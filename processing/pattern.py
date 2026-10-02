import numpy as np


def compute_poincare_index(orientation):
    """
    Compute the Poincare index for a fingerprint
    orientation field.

    Args:
        orientation: 2D orientation map in radians.

    Returns:
        2D array containing Poincare index values.
    """

    height, width = orientation.shape

    poincare = np.full(
        (height - 2, width - 2),
        np.nan,
        dtype=np.float32
    )

    for y in range(1, height - 1):
        for x in range(1, width - 1):

            center = orientation[y, x]

            neighbors = [
                orientation[y - 1, x - 1],
                orientation[y - 1, x],
                orientation[y - 1, x + 1],
                orientation[y, x + 1],
                orientation[y + 1, x + 1],
                orientation[y + 1, x],
                orientation[y + 1, x - 1],
                orientation[y, x - 1],
            ]

            angles = [
                2 * (angle - center)
                for angle in neighbors
            ]

            wrapped = np.arctan2(
                np.sin(angles),
                np.cos(angles)
            )

            total_rotation = np.sum(wrapped)

            poincare[y - 1, x - 1] = (
                total_rotation / (2 * np.pi)
            )

    return poincare


def find_poincare_candidates(
    orientation,
    threshold=0.25,
    margin=3
):
    """
    Find possible singularity candidates using
    the Poincare index.

    Candidates near the image boundary are excluded.

    Returns:
        List of candidate dictionaries.
    """

    poincare = compute_poincare_index(
        orientation
    )

    candidates = []

    height, width = poincare.shape

    for y in range(margin, height - margin):
        for x in range(margin, width - margin):

            value = poincare[y, x]

            if np.isnan(value):
                continue

            if abs(value) >= threshold:

                candidates.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "poincare_index": float(value),
                    }
                )

    return candidates