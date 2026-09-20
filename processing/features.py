def extract_features(results):
    """
    Convert pipeline results into a clean fingerprint
    feature structure.

    Features that are not reliably detected are returned
    as None rather than being guessed.
    """

    summary = results["summary"]

    features = {
        "pattern_type": None,
        "image_quality": None,

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
            "orientation": None,
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