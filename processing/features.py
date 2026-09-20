import numpy as np


def extract_features(results):
    """
    Convert pipeline results into a clean fingerprint
    feature structure.

    Features that are not reliably detected are returned
    as None rather than being guessed.
    """

    summary = results["summary"]
    quality = results["quality"]

    orientation_map = results["orientation"]["orientation"]
    coherence_map = results["orientation"]["coherence"]
    mask = results["segmentation"]["mask"]

    # --------------------------------------------------
    # Orientation measurement
    # --------------------------------------------------
    valid_orientation = orientation_map[mask > 0]
    valid_coherence = coherence_map[mask > 0]

    if valid_orientation.size > 0:
        ridge_orientation = float(
            np.median(valid_orientation)
        )

        orientation_coherence = float(
            np.median(valid_coherence)
        )
    else:
        ridge_orientation = None
        orientation_coherence = None

    # --------------------------------------------------
    # Feature structure
    # --------------------------------------------------
    features = {
        "pattern_type": None,

        "image_quality": {
            "mean_intensity": quality["mean_intensity"],
            "intensity_std": quality["intensity_std"],
            "local_variance": quality["local_variance"],
            "foreground_pixels": quality["foreground_pixels"],
        },

        "core": {
            "x": None,
            "y": None,
            "confidence": None,
        },

        "delta": {
            "x": None,
            "y": None,
            "confidence": None,
        },

        "ridge": {
            "count": None,
            "density": summary["ridge_density"],
            "orientation": ridge_orientation,
            "orientation_coherence": orientation_coherence,
            "frequency": summary["ridge_frequency"],
            "spacing": summary["ridge_spacing"],
        },

        "minutiae": {
            "total": None,
            "ridge_endings": None,
            "bifurcations": None,
            "points": [],
        },
    }

    return features