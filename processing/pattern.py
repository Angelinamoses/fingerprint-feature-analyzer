import numpy as np


# ============================================================
# Utility
# ============================================================

def _safe_float(value):
    """
    Convert a value to a finite Python float.

    Returns None for invalid values.
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


# ============================================================
# Block orientation
# ============================================================

def block_orientation(
    orientation,
    coherence,
    mask,
    block_size=16,
    min_foreground_ratio=0.50,
    min_coherence=0.25,
):
    """
    Convert the pixel-level orientation field into a
    block-level orientation field.

    Fingerprint orientation has 180-degree periodicity,
    therefore circular averaging is performed using:

        exp(i * 2 * theta)

    Args:
        orientation:
            Pixel-level orientation map in radians.

        coherence:
            Pixel-level orientation coherence map.

        mask:
            Fingerprint foreground mask.

        block_size:
            Size of orientation blocks in pixels.

        min_foreground_ratio:
            Minimum percentage of a block that must belong
            to the fingerprint foreground.

        min_coherence:
            Minimum median coherence required for a block.

    Returns:
        orientation_grid:
            Block-level orientation map.

        coherence_grid:
            Block-level coherence map.

        valid_grid:
            Boolean map identifying usable blocks.
    """

    if orientation is None:
        raise ValueError("Orientation map cannot be None.")

    if coherence is None:
        raise ValueError("Coherence map cannot be None.")

    if mask is None:
        raise ValueError("Segmentation mask cannot be None.")

    if orientation.shape != coherence.shape:
        raise ValueError(
            "Orientation and coherence maps must have the same shape."
        )

    if orientation.shape != mask.shape:
        raise ValueError(
            "Orientation and mask must have the same shape."
        )

    height, width = orientation.shape

    grid_height = height // block_size
    grid_width = width // block_size

    orientation_grid = np.full(
        (grid_height, grid_width),
        np.nan,
        dtype=np.float32,
    )

    coherence_grid = np.zeros(
        (grid_height, grid_width),
        dtype=np.float32,
    )

    valid_grid = np.zeros(
        (grid_height, grid_width),
        dtype=bool,
    )

    for by in range(grid_height):

        for bx in range(grid_width):

            y1 = by * block_size
            y2 = (by + 1) * block_size

            x1 = bx * block_size
            x2 = (bx + 1) * block_size

            block_mask = mask[y1:y2, x1:x2] > 0

            foreground_ratio = float(
                np.mean(block_mask)
            )

            if foreground_ratio < min_foreground_ratio:
                continue

            block_coherence = coherence[
                y1:y2,
                x1:x2
            ][block_mask]

            if block_coherence.size == 0:
                continue

            block_coherence = block_coherence[
                np.isfinite(block_coherence)
            ]

            if block_coherence.size == 0:
                continue

            median_coherence = float(
                np.median(block_coherence)
            )

            if median_coherence < min_coherence:
                continue

            block_orientation = orientation[
                y1:y2,
                x1:x2
            ][block_mask]

            block_orientation = block_orientation[
                np.isfinite(block_orientation)
            ]

            if block_orientation.size == 0:
                continue

            # Circular mean for an orientation field
            # with pi-periodicity.
            doubled = 2.0 * block_orientation

            vector = np.mean(
                np.exp(1j * doubled)
            )

            angle = 0.5 * np.angle(vector)

            orientation_grid[
                by,
                bx
            ] = float(angle)

            coherence_grid[
                by,
                bx
            ] = median_coherence

            valid_grid[
                by,
                bx
            ] = True

    return (
        orientation_grid,
        coherence_grid,
        valid_grid,
    )


# ============================================================
# Poincare index
# ============================================================

def compute_poincare_index(
    orientation_grid,
    valid_grid,
):
    """
    Compute the Poincare index on a block orientation field.

    Because fingerprint orientation has 180-degree ambiguity,
    orientation is doubled before angular differences are
    calculated.

    A value close to:

        +0.5 -> core-like singularity
        -0.5 -> delta-like singularity

    Returns:
        2D Poincare-index map.
    """

    if orientation_grid.ndim != 2:
        raise ValueError(
            "orientation_grid must be a 2D array."
        )

    if orientation_grid.shape != valid_grid.shape:
        raise ValueError(
            "orientation_grid and valid_grid must have "
            "the same shape."
        )

    height, width = orientation_grid.shape

    if height < 3 or width < 3:
        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    poincare = np.full(
        (height - 2, width - 2),
        np.nan,
        dtype=np.float32,
    )

    for y in range(1, height - 1):

        for x in range(1, width - 1):

            local_valid = valid_grid[
                y - 1:y + 2,
                x - 1:x + 2
            ]

            if not np.all(local_valid):
                continue

            ring = np.array(
                [
                    orientation_grid[y - 1, x - 1],
                    orientation_grid[y - 1, x],
                    orientation_grid[y - 1, x + 1],
                    orientation_grid[y, x + 1],
                    orientation_grid[y + 1, x + 1],
                    orientation_grid[y + 1, x],
                    orientation_grid[y + 1, x - 1],
                    orientation_grid[y, x - 1],
                    orientation_grid[y - 1, x - 1],
                ],
                dtype=np.float64,
            )

            if not np.all(np.isfinite(ring)):
                continue

            doubled = ring * 2.0

            differences = []

            for index in range(
                len(doubled) - 1
            ):
                delta = (
                    doubled[index + 1]
                    - doubled[index]
                )

                wrapped = np.arctan2(
                    np.sin(delta),
                    np.cos(delta),
                )

                differences.append(
                    wrapped
                )

            # Because the orientation was doubled,
            # divide by 2*pi to obtain the standard
            # fingerprint Poincare index.
            value = (
                np.sum(differences)
                / (2.0 * np.pi)
            )

            poincare[
                y - 1,
                x - 1
            ] = float(value)

    return poincare


# ============================================================
# Candidate detection
# ============================================================

def find_poincare_candidates(
    orientation,
    coherence,
    mask,
    block_size=16,
    poincare_tolerance=0.18,
    min_coherence=0.35,
    min_foreground_ratio=0.50,
):
    """
    Find possible core and delta singularities.

    Candidate detection is intentionally conservative.

    A candidate must:
        1. Exist inside the fingerprint foreground.
        2. Have adequate local orientation coherence.
        3. Have a Poincare value close to +0.5 or -0.5.
        4. Not be located directly at the orientation-grid boundary.

    Returns:
        Dictionary containing:
            candidates
            orientation_grid
            coherence_grid
            valid_grid
            poincare
    """

    (
        orientation_grid,
        coherence_grid,
        valid_grid,
    ) = block_orientation(
        orientation=orientation,
        coherence=coherence,
        mask=mask,
        block_size=block_size,
        min_foreground_ratio=min_foreground_ratio,
        min_coherence=min_coherence,
    )

    poincare = compute_poincare_index(
        orientation_grid,
        valid_grid,
    )

    candidates = []

    if poincare.size == 0:
        return {
            "candidates": [],
            "orientation_grid": orientation_grid,
            "coherence_grid": coherence_grid,
            "valid_grid": valid_grid,
            "poincare": poincare,
        }

    height, width = poincare.shape

    for y in range(height):

        for x in range(width):

            value = poincare[y, x]

            if not np.isfinite(value):
                continue

            # Core ≈ +0.5
            # Delta ≈ -0.5
            distance_to_singularity = min(
                abs(value - 0.5),
                abs(value + 0.5),
            )

            if (
                distance_to_singularity
                > poincare_tolerance
            ):
                continue

            grid_y = y + 1
            grid_x = x + 1

            local_coherence = coherence_grid[
                max(0, grid_y - 1):
                min(
                    coherence_grid.shape[0],
                    grid_y + 2,
                ),
                max(0, grid_x - 1):
                min(
                    coherence_grid.shape[1],
                    grid_x + 2,
                ),
            ]

            local_coherence = local_coherence[
                np.isfinite(local_coherence)
            ]

            if local_coherence.size == 0:
                continue

            candidate_coherence = float(
                np.median(local_coherence)
            )

            if candidate_coherence < min_coherence:
                continue

            # Convert block coordinates into image coordinates.
            pixel_x = int(
                grid_x * block_size
                + block_size // 2
            )

            pixel_y = int(
                grid_y * block_size
                + block_size // 2
            )

            # Determine candidate type.
            if value > 0:
                candidate_type = "core_candidate"
            else:
                candidate_type = "delta_candidate"

            # Confidence combines:
            # - Poincare closeness to ±0.5
            # - orientation coherence
            poincare_score = float(
                np.clip(
                    1.0
                    - (
                        distance_to_singularity
                        / max(
                            poincare_tolerance,
                            1e-8,
                        )
                    ),
                    0.0,
                    1.0,
                )
            )

            confidence = float(
                np.clip(
                    0.5 * poincare_score
                    + 0.5 * candidate_coherence,
                    0.0,
                    1.0,
                )
            )

            candidates.append(
                {
                    "type": candidate_type,
                    "x": pixel_x,
                    "y": pixel_y,
                    "poincare_index": float(value),
                    "coherence": candidate_coherence,
                    "confidence": confidence,
                }
            )

    # --------------------------------------------------------
    # Non-maximum suppression
    # --------------------------------------------------------

    selected = []

    min_distance = max(
        block_size * 2,
        int(
            min(mask.shape)
            * 0.04
        ),
    )

    for candidate in sorted(
        candidates,
        key=lambda item: item["confidence"],
        reverse=True,
    ):

        too_close = False

        for existing in selected:

            distance_squared = (
                (
                    candidate["x"]
                    - existing["x"]
                ) ** 2
                +
                (
                    candidate["y"]
                    - existing["y"]
                ) ** 2
            )

            if (
                distance_squared
                < min_distance ** 2
            ):
                too_close = True
                break

        if not too_close:
            selected.append(candidate)

    return {
        "candidates": selected,
        "orientation_grid": orientation_grid,
        "coherence_grid": coherence_grid,
        "valid_grid": valid_grid,
        "poincare": poincare,
    }


# ============================================================
# Core / Delta selection
# ============================================================

def select_core_delta(
    candidates,
    image_shape,
    min_confidence=0.55,
):
    """
    Select at most one core and one delta candidate.

    The detector does not force a result.

    If evidence is insufficient, the corresponding point
    remains unavailable.
    """

    height, width = image_shape[:2]

    margin = max(
        16,
        int(
            min(height, width)
            * 0.06
        ),
    )

    usable = []

    for candidate in candidates:

        x = candidate["x"]
        y = candidate["y"]

        if (
            x < margin
            or y < margin
            or x >= width - margin
            or y >= height - margin
        ):
            continue

        if (
            candidate["confidence"]
            < min_confidence
        ):
            continue

        usable.append(candidate)

    cores = [
        candidate
        for candidate in usable
        if candidate["type"]
        == "core_candidate"
    ]

    deltas = [
        candidate
        for candidate in usable
        if candidate["type"]
        == "delta_candidate"
    ]

    core = (
        max(
            cores,
            key=lambda item: item["confidence"],
        )
        if cores
        else None
    )

    # A delta that is extremely close to the core is
    # suspicious and is therefore rejected.
    valid_deltas = deltas

    if core is not None:

        minimum_separation = (
            min(height, width)
            * 0.10
        )

        valid_deltas = [
            delta
            for delta in deltas
            if (
                (
                    delta["x"]
                    - core["x"]
                ) ** 2
                +
                (
                    delta["y"]
                    - core["y"]
                ) ** 2
            )
            >= minimum_separation ** 2
        ]

    delta = (
        max(
            valid_deltas,
            key=lambda item: item["confidence"],
        )
        if valid_deltas
        else None
    )

    return core, delta


# ============================================================
# Pattern evidence
# ============================================================

def infer_pattern(
    candidates,
    mean_coherence,
):
    """
    Infer fingerprint pattern from singularity evidence.

    This is deliberately conservative.

    Evidence rules:

        Core + >=2 deltas -> Whorl evidence
        Core + >=1 delta  -> Loop evidence
        Otherwise         -> Unknown

    An absence of detected singularities is NOT automatically
    treated as Arch because a failed singularity detector
    cannot prove an arch.

    Returns:
        {
            "label": ...,
            "confidence": ...,
            "method": ...
        }
    """

    if not candidates:
        return {
            "label": None,
            "confidence": None,
            "method": "poincare_singularity_analysis",
        }

    cores = [
        candidate
        for candidate in candidates
        if candidate["type"]
        == "core_candidate"
    ]

    deltas = [
        candidate
        for candidate in candidates
        if candidate["type"]
        == "delta_candidate"
    ]

    mean_coherence = _safe_float(
        mean_coherence
    )

    if mean_coherence is None:
        mean_coherence = 0.0

    if (
        len(cores) >= 1
        and len(deltas) >= 2
    ):

        confidence = min(
            0.95,
            0.55
            + 0.08 * len(cores)
            + 0.06 * len(deltas),
        )

        if (
            mean_coherence < 0.45
        ):
            confidence *= 0.85

        if confidence >= 0.60:

            return {
                "label": "Whorl",
                "confidence": float(
                    np.clip(
                        confidence,
                        0.0,
                        1.0,
                    )
                ),
                "method":
                    "poincare_singularity_analysis",
            }

    if (
        len(cores) >= 1
        and len(deltas) >= 1
    ):

        confidence = min(
            0.90,
            0.52
            + 0.10 * min(len(cores), 1)
            + 0.08 * min(len(deltas), 1),
        )

        if (
            mean_coherence < 0.45
        ):
            confidence *= 0.85

        if confidence >= 0.60:

            return {
                "label": "Loop",
                "confidence": float(
                    np.clip(
                        confidence,
                        0.0,
                        1.0,
                    )
                ),
                "method":
                    "poincare_singularity_analysis",
            }

    return {
        "label": None,
        "confidence": None,
        "method": "poincare_singularity_analysis",
    }


# ============================================================
# Complete pattern analysis
# ============================================================

def analyze_pattern(
    orientation,
    coherence,
    mask,
    image_shape,
    block_size=16,
):
    """
    Complete singularity and pattern analysis.

    Returns:
        {
            "pattern": {...},
            "singular_points": {
                "core": {...},
                "delta": {...},
            },
            "candidates": [...],
            "poincare": ...
        }
    """

    detection = find_poincare_candidates(
        orientation=orientation,
        coherence=coherence,
        mask=mask,
        block_size=block_size,
    )

    candidates = detection[
        "candidates"
    ]

    core, delta = select_core_delta(
        candidates=candidates,
        image_shape=image_shape,
    )

    foreground = mask > 0

    if np.any(foreground):

        valid_coherence = coherence[
            foreground
        ]

        valid_coherence = valid_coherence[
            np.isfinite(valid_coherence)
        ]

        mean_coherence = (
            float(
                np.median(
                    valid_coherence
                )
            )
            if valid_coherence.size
            else 0.0
        )

    else:
        mean_coherence = 0.0

    pattern = infer_pattern(
        candidates=candidates,
        mean_coherence=mean_coherence,
    )

    def point_output(point):

        if point is None:
            return {
                "detected": False,
                "x": None,
                "y": None,
                "confidence": None,
            }

        return {
            "detected": True,
            "x": int(point["x"]),
            "y": int(point["y"]),
            "confidence": _safe_float(
                point["confidence"]
            ),
        }

    return {
        "pattern": pattern,

        "singular_points": {
            "core": point_output(core),
            "delta": point_output(delta),
        },

        "candidates": candidates,

        "poincare": detection[
            "poincare"
        ],

        "orientation_grid": detection[
            "orientation_grid"
        ],

        "coherence_grid": detection[
            "coherence_grid"
        ],

        "valid_grid": detection[
            "valid_grid"
        ],
    }