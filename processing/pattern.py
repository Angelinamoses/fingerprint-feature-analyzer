"""
Fingerprint pattern / singular-point analysis.

Pipeline
--------
    ridge orientation map
        -> block orientation grid        (block_orientation)
        -> Poincare index map            (compute_poincare_index)
        -> core / delta candidates       (find_poincare_candidates)
        -> selected core + delta         (select_core_delta)
        -> Arch / Loop / Whorl evidence  (infer_pattern)

Conventions
-----------
* Orientation is the ridge-line direction (radians, period pi).
* Poincare index: +0.5 = core, -0.5 = delta, +1.0 = double core
  (concentric / spiral whorl centre).
"""

import cv2
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


def _wrap_half_pi(delta):
    """Wrap an orientation difference (period pi) into (-pi/2, pi/2]."""
    return (delta + np.pi / 2.0) % np.pi - np.pi / 2.0


# ============================================================
# Block orientation
# ============================================================

def block_orientation(
    orientation,
    coherence,
    mask,
    block_size=16,
    min_foreground_ratio=0.35,
    min_coherence=0.25,
):
    """
    Convert the pixel-level orientation field into a block-level field.

    Fingerprint orientation has 180-degree periodicity, so circular
    averaging is done on exp(i * 2 * theta).

    Args:
        orientation: Pixel-level ridge orientation (radians).
        coherence:   Pixel-level coherence map.
        mask:        Foreground mask.
        block_size:  Block edge in pixels.
        min_foreground_ratio:
            Minimum fraction of a block inside the foreground.
            (Was 0.50; relaxed because deltas are usually close to
            the print border.)
        min_coherence:
            Minimum median coherence for a block to be usable.

    Returns:
        orientation_grid, coherence_grid, valid_grid
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
        (grid_height, grid_width), np.nan, dtype=np.float32
    )
    coherence_grid = np.zeros(
        (grid_height, grid_width), dtype=np.float32
    )
    valid_grid = np.zeros(
        (grid_height, grid_width), dtype=bool
    )

    for by in range(grid_height):

        for bx in range(grid_width):

            y1 = by * block_size
            y2 = y1 + block_size
            x1 = bx * block_size
            x2 = x1 + block_size

            block_mask = mask[y1:y2, x1:x2] > 0

            if float(np.mean(block_mask)) < min_foreground_ratio:
                continue

            block_coherence = coherence[y1:y2, x1:x2][block_mask]
            block_coherence = block_coherence[np.isfinite(block_coherence)]

            if block_coherence.size == 0:
                continue

            median_coherence = float(np.median(block_coherence))

            if median_coherence < min_coherence:
                continue

            block_theta = orientation[y1:y2, x1:x2][block_mask]
            block_theta = block_theta[np.isfinite(block_theta)]

            if block_theta.size == 0:
                continue

            vector = np.mean(np.exp(2j * block_theta))

            orientation_grid[by, bx] = float(0.5 * np.angle(vector))
            coherence_grid[by, bx] = median_coherence
            valid_grid[by, bx] = True

    return orientation_grid, coherence_grid, valid_grid


# ============================================================
# Poincare index
# ============================================================

def _ring_offsets(radius):
    """Clockwise (on screen) ring of integer offsets at Chebyshev radius."""
    r = int(radius)
    ring = []
    for x in range(-r, r + 1):          # top edge, left -> right
        ring.append((-r, x))
    for y in range(-r + 1, r + 1):      # right edge, top -> bottom
        ring.append((y, r))
    for x in range(r - 1, -r - 1, -1):  # bottom edge, right -> left
        ring.append((r, x))
    for y in range(r - 1, -r, -1):      # left edge, bottom -> top
        ring.append((y, -r))
    return ring


def compute_poincare_index(
    orientation_grid,
    valid_grid,
    radius=1,
    require_center_valid=False,
):
    """
    Poincare index of a block orientation field.

    For every cell the orientation change is summed along the closed
    ring of cells at Chebyshev distance ``radius`` and divided by 2*pi:

        +0.5 -> core        -0.5 -> delta
        +1.0 -> double core  0.0 -> regular flow

    Orientation has period pi, so each step is wrapped to
    (-pi/2, pi/2]. For a closed ring the sum is an exact multiple of
    pi, so valid outputs are quantised to multiples of 0.5.

    BUGS FIXED relative to the original implementation
    --------------------------------------------------
    1. The original doubled the angles, wrapped to (-pi, pi] and divided
       by 2*pi, which is TWICE the Poincare index. A true core gave
       +1.0 and a true delta -1.0. Because the ring is closed the result
       could only be a whole number, so the +/-0.5 acceptance window in
       the candidate detector was unreachable: no core or delta could
       ever be reported, so pattern/core/delta were always None.
    2. The original required all 9 blocks (including the centre block)
       to be valid. The centre block of a singular point has low
       coherence by definition, so true singularities were discarded.
       Only the ring cells need to be valid now.

    Returns:
        2D array, shape (H - 2*radius, W - 2*radius). Entry [y, x]
        belongs to grid cell (y + radius, x + radius).
    """

    if orientation_grid.ndim != 2:
        raise ValueError(
            "orientation_grid must be a 2D array."
        )

    if orientation_grid.shape != valid_grid.shape:
        raise ValueError(
            "orientation_grid and valid_grid must have the same shape."
        )

    radius = int(radius)
    height, width = orientation_grid.shape

    if height < 2 * radius + 1 or width < 2 * radius + 1:
        return np.empty((0, 0), dtype=np.float32)

    poincare = np.full(
        (height - 2 * radius, width - 2 * radius),
        np.nan,
        dtype=np.float32,
    )

    offsets = _ring_offsets(radius)
    theta = np.asarray(orientation_grid, dtype=np.float64)

    for y in range(radius, height - radius):

        for x in range(radius, width - radius):

            if require_center_valid and not valid_grid[y, x]:
                continue

            ring_valid = all(
                valid_grid[y + dy, x + dx] for dy, dx in offsets
            )

            if not ring_valid:
                continue

            ring = [theta[y + dy, x + dx] for dy, dx in offsets]
            ring.append(ring[0])
            ring = np.asarray(ring, dtype=np.float64)

            if not np.all(np.isfinite(ring)):
                continue

            steps = _wrap_half_pi(np.diff(ring))

            poincare[y - radius, x - radius] = float(
                np.sum(steps) / (2.0 * np.pi)
            )

    return poincare


# ============================================================
# Candidate detection
# ============================================================

def _cluster_cells(cells, max_gap):
    """
    Group neighbouring flagged cells (a real singular point fires on
    2-4 adjacent cells) and return a list of clusters (lists of cells).
    """
    remaining = list(cells)
    clusters = []

    while remaining:
        seed = remaining.pop()
        cluster = [seed]
        queue = [seed]

        while queue:
            cy, cx = queue.pop()
            near = [
                c for c in remaining
                if max(abs(c[0] - cy), abs(c[1] - cx)) <= max_gap
            ]
            for c in near:
                remaining.remove(c)
                cluster.append(c)
                queue.append(c)

        clusters.append(cluster)

    return clusters


def find_poincare_candidates(
    orientation,
    coherence,
    mask,
    block_size=16,
    ring_radius=1,
    poincare_tolerance=0.2,
    min_coherence=0.30,
    min_foreground_ratio=0.35,
):
    """
    Find possible core and delta singularities.

    A candidate must:
        1. be inside the foreground (all ring blocks valid),
        2. have a Poincare index near +0.5 (core), -0.5 (delta)
           or +1.0 (double core),
        3. have adequate ring coherence.

    Adjacent flagged cells are merged and the centroid is reported,
    which makes the location stable and gives a natural confidence
    measure (number of agreeing cells).

    Returns:
        dict with candidates, orientation_grid, coherence_grid,
        valid_grid, poincare
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
        radius=ring_radius,
    )

    result = {
        "candidates": [],
        "orientation_grid": orientation_grid,
        "coherence_grid": coherence_grid,
        "valid_grid": valid_grid,
        "poincare": poincare,
    }

    if poincare.size == 0:
        return result

    tol = float(poincare_tolerance)
    off = int(ring_radius)

    flagged = {"core": [], "delta": [], "double_core": []}

    for y, x in np.argwhere(np.isfinite(poincare)):

        value = float(poincare[y, x])

        if abs(value - 0.5) <= tol:
            flagged["core"].append((int(y), int(x)))
        elif abs(value + 0.5) <= tol:
            flagged["delta"].append((int(y), int(x)))
        elif abs(value - 1.0) <= tol:
            flagged["double_core"].append((int(y), int(x)))

    candidates = []

    for kind, cells in flagged.items():

        for cluster in _cluster_cells(cells, max_gap=1):

            ys = np.array([c[0] + off for c in cluster], dtype=np.float64)
            xs = np.array([c[1] + off for c in cluster], dtype=np.float64)

            gy = float(np.mean(ys))
            gx = float(np.mean(xs))

            pixel_x = int(round(gx * block_size + block_size / 2.0))
            pixel_y = int(round(gy * block_size + block_size / 2.0))

            iy = int(round(gy))
            ix = int(round(gx))

            window = coherence_grid[
                max(0, iy - off - 1):iy + off + 2,
                max(0, ix - off - 1):ix + off + 2,
            ]
            window = window[window > 0]

            ring_coherence = float(np.median(window)) if window.size else 0.0

            support = min(1.0, len(cluster) / 3.0)

            confidence = float(
                np.clip(
                    0.55 * ring_coherence + 0.45 * support,
                    0.0,
                    1.0,
                )
            )

            values = [
                float(poincare[c[0], c[1]]) for c in cluster
            ]

            candidates.append(
                {
                    "type": (
                        "delta_candidate"
                        if kind == "delta"
                        else "core_candidate"
                    ),
                    "multiplicity": 2 if kind == "double_core" else 1,
                    "x": pixel_x,
                    "y": pixel_y,
                    "poincare_index": float(np.mean(values)),
                    "coherence": ring_coherence,
                    "support_cells": len(cluster),
                    "confidence": confidence,
                }
            )

    # --------------------------------------------------------
    # Non-maximum suppression (same type only)
    # --------------------------------------------------------

    selected = []
    min_distance = max(block_size * 1.5, min(mask.shape) * 0.04)

    for candidate in sorted(
        candidates,
        key=lambda item: item["confidence"],
        reverse=True,
    ):
        too_close = any(
            existing["type"] == candidate["type"]
            and (
                (candidate["x"] - existing["x"]) ** 2
                + (candidate["y"] - existing["y"]) ** 2
            ) < min_distance ** 2
            for existing in selected
        )

        if not too_close:
            selected.append(candidate)

    result["candidates"] = selected

    return result


# ============================================================
# Core / Delta selection
# ============================================================

def usable_candidates(
    candidates,
    image_shape,
    min_confidence=0.45,
):
    """
    Candidates that are far enough from the image border and have
    adequate confidence.
    """

    height, width = image_shape[:2]

    margin = max(16, int(min(height, width) * 0.05))

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

        if candidate["confidence"] < min_confidence:
            continue

        usable.append(candidate)

    return usable


def select_core_delta(
    candidates,
    image_shape,
    min_confidence=0.45,
):
    """
    Select at most one core and one delta candidate.

    The detector does not force a result; if evidence is
    insufficient the corresponding point stays unavailable.
    """

    height, width = image_shape[:2]

    usable = usable_candidates(
        candidates,
        image_shape,
        min_confidence=min_confidence,
    )

    cores = [c for c in usable if c["type"] == "core_candidate"]
    deltas = [c for c in usable if c["type"] == "delta_candidate"]

    core = (
        max(cores, key=lambda item: item["confidence"])
        if cores
        else None
    )

    valid_deltas = deltas

    # A delta extremely close to the core is suspicious.
    if core is not None:

        minimum_separation = min(height, width) * 0.08

        valid_deltas = [
            delta
            for delta in deltas
            if (
                (delta["x"] - core["x"]) ** 2
                + (delta["y"] - core["y"]) ** 2
            ) >= minimum_separation ** 2
            or core.get("multiplicity", 1) > 1
        ]

    delta = (
        max(valid_deltas, key=lambda item: item["confidence"])
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
    image_shape=(512, 512),
    valid_fraction=None,
):
    """
    Infer Arch / Loop / Whorl from singular-point evidence.

    Rules (cores counted with multiplicity, so a +1 index counts as
    two cores):

        >=2 cores and >=1 delta     -> Whorl
        >=1 core  and >=2 deltas    -> Whorl
        1 core, 1 delta, close and
        vertically stacked          -> Arch (tented arch)
        1 core, 1 delta             -> Loop (+ left/right subtype)
        2+ cores, no delta          -> Whorl (delta outside the print)
        1 core, no delta            -> Loop (partial print, low conf.)
        no singular point, and the
        flow field is coherent and
        well covered                -> Arch
        otherwise                   -> unknown (None)

    The previous version could not output Arch at all (any print
    without singularities returned None, "because a failed detector
    cannot prove an arch"). A plain arch genuinely has no
    singularities, so absence is accepted as evidence only when the
    orientation field was reliably measured over most of the print
    (``valid_fraction``) and coherence is high.

    Returns:
        {"label", "confidence", "method", "subtype", "evidence"}
    """

    height, width = image_shape[:2]
    min_dim = float(min(height, width))

    mean_coherence = _safe_float(mean_coherence) or 0.0

    cores = [c for c in candidates if c["type"] == "core_candidate"]
    deltas = [c for c in candidates if c["type"] == "delta_candidate"]

    n_cores = int(sum(c.get("multiplicity", 1) for c in cores))
    n_deltas = len(deltas)

    evidence = {
        "cores": n_cores,
        "deltas": n_deltas,
        "mean_coherence": mean_coherence,
        "valid_fraction": _safe_float(valid_fraction),
    }

    def result(label, confidence, subtype=None, why=""):
        if label is None:
            return {
                "label": None,
                "confidence": None,
                "method": "poincare_singularity_analysis",
                "subtype": None,
                "evidence": evidence,
            }

        confidence = float(np.clip(confidence, 0.0, 1.0))

        if mean_coherence < 0.40:
            confidence *= 0.85

        evidence["rule"] = why

        return {
            "label": label,
            "confidence": confidence,
            "method": "poincare_singularity_analysis",
            "subtype": subtype,
            "evidence": evidence,
        }

    # ---- Whorl ----
    if n_cores >= 2 and n_deltas >= 1:
        return result(
            "Whorl",
            min(0.92, 0.70 + 0.06 * min(n_deltas, 2) + 0.04 * min(n_cores, 2)),
            why=">=2 cores and >=1 delta",
        )

    if n_cores >= 1 and n_deltas >= 2:
        return result(
            "Whorl",
            min(0.90, 0.68 + 0.05 * min(n_deltas, 3)),
            why=">=1 core and >=2 deltas",
        )

    if n_cores >= 2 and n_deltas == 0:
        return result(
            "Whorl",
            0.62,
            why=">=2 cores, deltas outside the print",
        )

    # ---- Loop / tented arch ----
    if n_cores == 1 and n_deltas == 1:

        core = cores[0]
        delta = deltas[0]

        dx = delta["x"] - core["x"]
        dy = delta["y"] - core["y"]
        separation = float(np.hypot(dx, dy))

        if separation < 0.14 * min_dim and abs(dx) < 0.07 * min_dim:
            return result(
                "Arch",
                0.64,
                subtype="tented_arch",
                why="core and delta close together and vertically stacked",
            )

        subtype = "left_loop" if dx > 0 else "right_loop"

        return result(
            "Loop",
            0.78,
            subtype=subtype,
            why="1 core and 1 delta",
        )

    if n_cores == 1 and n_deltas == 0:
        return result(
            "Loop",
            0.58,
            subtype=None,
            why="single core, delta not visible (partial print?)",
        )

    # ---- Plain arch ----
    if n_cores == 0 and n_deltas == 0:

        if (
            valid_fraction is not None
            and valid_fraction >= 0.60
            and mean_coherence >= 0.45
        ):
            return result(
                "Arch",
                0.66,
                subtype="plain_arch",
                why="no singular point in a well-measured, coherent field",
            )

    return result(None, None)


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
            "singular_points": {"core": {...}, "delta": {...}},
            "candidates": [...],
            "poincare": ...,
            ...
        }
    """

    detection = find_poincare_candidates(
        orientation=orientation,
        coherence=coherence,
        mask=mask,
        block_size=block_size,
    )

    candidates = detection["candidates"]

    core, delta = select_core_delta(
        candidates=candidates,
        image_shape=image_shape,
    )

    usable = usable_candidates(
        candidates,
        image_shape,
    )

    foreground = mask > 0

    if np.any(foreground):

        valid_coherence = coherence[foreground]
        valid_coherence = valid_coherence[np.isfinite(valid_coherence)]

        mean_coherence = (
            float(np.median(valid_coherence))
            if valid_coherence.size
            else 0.0
        )

    else:
        mean_coherence = 0.0

    # Fraction of the print area whose orientation was measured
    # reliably (grid blocks that are valid / blocks that touch the print).
    valid_grid = detection["valid_grid"]
    gh, gw = valid_grid.shape

    if gh > 0 and gw > 0:
        block_fg = (
            (mask[: gh * block_size, : gw * block_size] > 0)
            .reshape(gh, block_size, gw, block_size)
            .mean(axis=(1, 3))
        )
        touching = block_fg >= 0.35
        valid_fraction = (
            float(np.sum(valid_grid & touching) / max(1, np.sum(touching)))
        )
    else:
        valid_fraction = 0.0

    # Guard for the "no singular point => Arch" rule: a fragmented,
    # holey or tiny mask must never be allowed to prove an arch.
    coverage = float(np.mean(foreground))
    mask_u8 = (foreground.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        hull_area = cv2.contourArea(
            cv2.convexHull(max(contours, key=cv2.contourArea))
        )
        solidity = float(np.sum(foreground)) / max(1.0, hull_area)
    else:
        solidity = 0.0

    valid_fraction *= min(1.0, solidity)

    if coverage < 0.20:
        valid_fraction = 0.0

    pattern = infer_pattern(
        candidates=usable,
        mean_coherence=mean_coherence,
        image_shape=image_shape,
        valid_fraction=valid_fraction,
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
            "confidence": _safe_float(point["confidence"]),
        }

    return {
        "pattern": pattern,

        "singular_points": {
            "core": point_output(core),
            "delta": point_output(delta),
        },

        "candidates": candidates,

        "poincare": detection["poincare"],

        "orientation_grid": detection["orientation_grid"],

        "coherence_grid": detection["coherence_grid"],

        "valid_grid": detection["valid_grid"],
    }
