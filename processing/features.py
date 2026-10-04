import numpy as np


def _safe_float(value):
    """
    Convert NumPy/Python numeric values into a JSON-safe float.

    Returns None for invalid or non-finite values.
    """
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None

    return value


def _safe_int(value):
    """
    Convert NumPy/Python numeric values into a JSON-safe integer.

    Returns None for invalid values.
    """
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value):
    """
    Convert NumPy/Python boolean values into native Python bool.
    """
    if value is None:
        return False

    return bool(value)


def _extract_point(point):
    """
    Convert a minutiae point into a clean JSON-safe structure.
    """

    if not isinstance(point, dict):
        return None

    x = _safe_int(point.get("x"))
    y = _safe_int(point.get("y"))

    if x is None or y is None:
        return None

    orientation = _safe_float(
        point.get("orientation")
    )

    confidence = _safe_float(
        point.get("confidence")
    )

    point_type = point.get("type")

    if point_type is not None:
        point_type = str(point_type)

    return {
        "x": x,
        "y": y,
        "type": point_type,
        "orientation": orientation,
        "angle": _safe_float(point.get("angle")),
        "confidence": confidence,
    }


def _extract_minutiae(results):
    """
    Read minutiae results from the pipeline.

    Supports the detector output:

        results["minutiae"]

    If no detector has been connected yet, the function
    safely returns an unavailable result.
    """

    detector = results.get("minutiae")

    if not isinstance(detector, dict):
        return {
            "total": None,
            "ridge_endings": None,
            "bifurcations": None,
            "points": [],
            "reliable": False,
        }

    raw_points = detector.get(
        "points",
        []
    )

    points = []

    if isinstance(raw_points, list):
        for point in raw_points:

            clean_point = _extract_point(point)

            if clean_point is not None:
                points.append(clean_point)

    reliable = _safe_bool(
        detector.get("reliable", False)
    )

    total = _safe_int(
        detector.get("total")
    )

    ridge_endings = _safe_int(
        detector.get("ridge_endings")
    )

    bifurcations = _safe_int(
        detector.get("bifurcations")
    )

    # Points are reported even when the global sanity checks flag
    # the result; ``reliable`` / ``reasons`` tell the consumer how far
    # to trust it (the old code blanked everything, hiding real data).
    reasons = detector.get("reasons", [])

    # Recalculate total from the actual returned points
    # when possible. This prevents inconsistent counts.
    if points:
        total = len(points)

        ridge_endings = sum(
            1
            for point in points
            if point["type"] == "ridge_ending"
        )

        bifurcations = sum(
            1
            for point in points
            if point["type"] == "bifurcation"
        )

    return {
        "total": total,
        "ridge_endings": ridge_endings,
        "bifurcations": bifurcations,
        "points": points,
        "reliable": reliable,
        "reasons": list(reasons) if isinstance(reasons, list) else [],
    }


def _extract_singular_point(results, name):
    """
    Extract a core or delta result from the pipeline.

    Supported forms:

        {
            "x": ...,
            "y": ...,
            "confidence": ...
        }

    or:

        {
            "detected": True,
            "x": ...,
            "y": ...,
            "confidence": ...
        }
    """

    singular_results = results.get(
        "singular_points"
    )

    if not isinstance(
        singular_results,
        dict
    ):
        return {
            "x": None,
            "y": None,
            "confidence": None,
        }

    point = singular_results.get(name)

    if not isinstance(point, dict):
        return {
            "x": None,
            "y": None,
            "confidence": None,
        }

    detected = point.get(
        "detected",
        True
    )

    if not bool(detected):
        return {
            "x": None,
            "y": None,
            "confidence": None,
        }

    x = _safe_int(
        point.get("x")
    )

    y = _safe_int(
        point.get("y")
    )

    confidence = _safe_float(
        point.get("confidence")
    )

    if x is None or y is None:
        return {
            "x": None,
            "y": None,
            "confidence": None,
        }

    return {
        "x": x,
        "y": y,
        "confidence": confidence,
    }


def _extract_pattern_details(results):
    pattern_result = results.get("pattern")

    if not isinstance(pattern_result, dict):
        return {"confidence": None, "method": None, "subtype": None}

    return {
        "confidence": _safe_float(pattern_result.get("confidence")),
        "method": pattern_result.get("method"),
        "subtype": pattern_result.get("subtype"),
    }


def _extract_pattern(results):
    """
    Extract pattern classification from the pipeline.

    Supported pipeline format:

        results["pattern"]

    Example:

        {
            "label": "Loop",
            "confidence": 0.91
        }

    The classifier must provide the result.
    This function never infers the class from the filename.
    """

    pattern_result = results.get(
        "pattern"
    )

    if not isinstance(
        pattern_result,
        dict
    ):
        return None

    label = pattern_result.get(
        "label"
    )

    if label is None:
        label = pattern_result.get(
            "pattern_type"
        )

    if label is None:
        return None

    label = str(label).strip()

    if not label:
        return None

    allowed_patterns = {
        "arch",
        "loop",
        "whorl",
        "unknown",
    }

    normalized = label.lower()

    if normalized not in allowed_patterns:
        return None

    if normalized == "unknown":
        return None

    # Keep the UI-friendly capitalization.
    return normalized.capitalize()


def extract_features(results):
    """
    Convert complete fingerprint pipeline results into
    a clean, JSON-safe feature structure.

    This function DOES NOT perform detection itself.

    It collects outputs from:
        - pattern detector
        - singular-point detector
        - minutiae detector
        - ridge analysis
        - image-quality analysis

    Missing or unreliable detector results remain None.
    """

    if not isinstance(results, dict):
        raise TypeError(
            "results must be a dictionary."
        )

    summary = results.get(
        "summary",
        {}
    )

    quality = results.get(
        "quality",
        {}
    )

    orientation_results = results.get(
        "orientation",
        {}
    )

    segmentation_results = results.get(
        "segmentation",
        {}
    )

    orientation_map = orientation_results.get(
        "orientation"
    )

    coherence_map = orientation_results.get(
        "coherence"
    )

    mask = segmentation_results.get(
        "mask"
    )

    # --------------------------------------------------
    # Orientation measurement
    # --------------------------------------------------

    ridge_orientation = None
    orientation_coherence = None

    if (
        orientation_map is not None
        and coherence_map is not None
        and mask is not None
    ):

        try:

            valid_mask = mask > 0

            valid_orientation = (
                orientation_map[valid_mask]
            )

            valid_coherence = (
                coherence_map[valid_mask]
            )

            valid_orientation = (
                valid_orientation[
                    np.isfinite(valid_orientation)
                ]
            )

            valid_coherence = (
                valid_coherence[
                    np.isfinite(valid_coherence)
                ]
            )

            if valid_orientation.size > 0:
                # orientation has period pi -> circular mean,
                # not an arithmetic median
                ridge_orientation = float(
                    0.5 * np.angle(
                        np.mean(np.exp(2j * valid_orientation))
                    )
                )

            if valid_coherence.size > 0:
                orientation_coherence = float(
                    np.clip(
                        np.median(
                            valid_coherence
                        ),
                        0.0,
                        1.0,
                    )
                )

        except (
            TypeError,
            ValueError,
            IndexError,
        ):
            ridge_orientation = None
            orientation_coherence = None

    # --------------------------------------------------
    # Pattern
    # --------------------------------------------------

    pattern_type = _extract_pattern(
        results
    )

    # --------------------------------------------------
    # Core
    # --------------------------------------------------

    core = _extract_singular_point(
        results,
        "core"
    )

    # --------------------------------------------------
    # Delta
    # --------------------------------------------------

    delta = _extract_singular_point(
        results,
        "delta"
    )

    # --------------------------------------------------
    # Minutiae
    # --------------------------------------------------

    minutiae = _extract_minutiae(
        results
    )

    # --------------------------------------------------
    # Image quality
    # --------------------------------------------------

    image_quality = {
        "mean_intensity": _safe_float(
            quality.get(
                "mean_intensity"
            )
        ),

        "intensity_std": _safe_float(
            quality.get(
                "intensity_std"
            )
        ),

        "local_variance": _safe_float(
            quality.get(
                "local_variance"
            )
        ),

        "foreground_pixels": _safe_int(
            quality.get(
                "foreground_pixels"
            )
        ),
    }

    # --------------------------------------------------
    # Ridge features
    # --------------------------------------------------

    ridge = {
        "count": _safe_int(
            summary.get(
                "ridge_count"
            )
        ),

        "density": _safe_float(
            summary.get(
                "ridge_density"
            )
        ),

        "orientation": ridge_orientation,

        "orientation_coherence":
            orientation_coherence,

        "frequency": _safe_float(
            summary.get(
                "ridge_frequency"
            )
        ),

        "spacing": _safe_float(
            summary.get(
                "ridge_spacing"
            )
        ),
    }

    # --------------------------------------------------
    # Final structure
    # --------------------------------------------------

    features = {
        "pattern_type": pattern_type,

        "pattern_details": _extract_pattern_details(results),

        "image_quality": image_quality,

        "core": core,

        "delta": delta,

        "ridge": ridge,

        "minutiae": minutiae,
    }

    return features