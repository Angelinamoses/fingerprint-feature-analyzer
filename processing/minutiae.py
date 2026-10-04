import cv2
import numpy as np
from skimage.morphology import skeletonize


# ============================================================
# Helpers
# ============================================================

def _empty_result():
    return {
        "total": None,
        "ridge_endings": None,
        "bifurcations": None,
        "points": [],
        "reliable": False,
        "reasons": ["detector_unavailable"],
    }


def _crossing_number(neighborhood):
    """
    Crossing Number (CN) of the centre pixel of a 3x3 neighbourhood.

        CN = 1 -> ridge ending,  CN = 3 -> bifurcation

    BUG FIXED: the original built the ring as
    [p0, p1, p2, p5, p3, p4, p6, p7, p0] from the flattened 3x3 block.
    That order is not a walk around the centre (it visits the centre
    pixel p4 and skips p8), so CN was wrong for most configurations:
    only ~3/8 true ending patterns and ~8/16 true bifurcation
    patterns were recognised, and several non-minutia staircase
    patterns were accepted. The correct clockwise ring is
    p0, p1, p2, p5, p8, p7, p6, p3.
    """
    p = np.asarray(neighborhood).flatten().astype(np.int16)

    ring = [p[0], p[1], p[2], p[5], p[8], p[7], p[6], p[3], p[0]]

    return int(sum(abs(ring[i + 1] - ring[i]) for i in range(8)) // 2)


def crossing_number_map(skeleton):
    """Vectorised crossing number for every skeleton pixel (0 elsewhere)."""
    sk = np.pad(skeleton.astype(np.int16), 1)

    ring = [
        sk[0:-2, 0:-2], sk[0:-2, 1:-1], sk[0:-2, 2:],
        sk[1:-1, 2:],
        sk[2:, 2:], sk[2:, 1:-1], sk[2:, 0:-2],
        sk[1:-1, 0:-2],
    ]

    total = np.zeros(skeleton.shape, dtype=np.int16)

    for i in range(8):
        total += np.abs(ring[i] - ring[(i + 1) % 8])

    cn = total // 2
    cn[~skeleton.astype(bool)] = 0

    return cn


def _distance_to_mask_boundary(mask):
    """Distance in pixels from foreground pixels to the mask boundary."""
    foreground = (mask > 0).astype(np.uint8)
    if not np.any(foreground):
        return np.zeros_like(mask, dtype=np.float32)

    return cv2.distanceTransform(
        foreground,
        cv2.DIST_L2,
        3,
    )


def _local_support(skeleton, y, x, radius=12):
    """Count skeleton pixels in a local circular-ish window."""
    h, w = skeleton.shape

    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)
    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)

    patch = skeleton[y1:y2, x1:x2]

    return int(np.count_nonzero(patch))


def _local_coherence(coherence, y, x, radius=4):
    """Median coherence around a candidate rather than one noisy pixel."""
    h, w = coherence.shape

    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)
    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)

    values = coherence[y1:y2, x1:x2]
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0

    return float(np.clip(np.median(values), 0.0, 1.0))


def _remove_small_components(binary, min_area):
    """Keep only sufficiently large connected ridge components."""
    number_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    clean = np.zeros_like(binary)

    for label in range(1, number_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area >= min_area:
            clean[labels == label] = 255

    return clean


def _remove_boundary_skeleton(skeleton, mask, boundary_margin=5):
    """
    Remove skeleton pixels close to the segmentation boundary.

    This is important because segmentation edges otherwise become
    thousands of fake ridge endings.
    """
    distance = _distance_to_mask_boundary(mask)

    cleaned = skeleton.copy()
    cleaned[distance < float(boundary_margin)] = False

    return cleaned


def _trace_endpoint_branch(skeleton, start_y, start_x, max_length=20):
    """
    Trace an endpoint until a junction or another endpoint.

    Returns the branch pixels. Used only for short-spur pruning.
    """
    h, w = skeleton.shape

    path = [(start_y, start_x)]
    previous = None
    current = (start_y, start_x)

    for _ in range(max_length):
        y, x = current

        neighbors = []

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue

                ny = y + dy
                nx = x + dx

                if (
                    0 <= ny < h
                    and 0 <= nx < w
                    and skeleton[ny, nx]
                ):
                    if previous is None or (ny, nx) != previous:
                        neighbors.append((ny, nx))

        if len(neighbors) != 1:
            break

        next_pixel = neighbors[0]
        path.append(next_pixel)
        previous = current
        current = next_pixel

    return path


def _prune_short_spurs(skeleton, min_branch_length=8, iterations=3):
    """
    Remove short endpoint-to-junction branches.

    Real fingerprint ridges can end, so pruning is deliberately
    conservative. Only branches that terminate quickly are removed.
    """
    cleaned = skeleton.copy()

    for _ in range(iterations):
        neighbors = neighbor_count_map(cleaned)
        endpoints = np.argwhere(neighbors == 1)

        remove_pixels = set()

        for y, x in endpoints:
            branch = _trace_endpoint_branch(
                cleaned,
                int(y),
                int(x),
                max_length=min_branch_length + 1,
            )

            if len(branch) <= min_branch_length:
                # Do not remove the junction pixel itself.
                for py, px in branch[:-1]:
                    remove_pixels.add((py, px))

        if not remove_pixels:
            break

        for y, x in remove_pixels:
            cleaned[y, x] = False

    return cleaned


# ============================================================
# Ridge binary
# ============================================================

def build_ridge_binary(image, mask):
    """
    Convert a fingerprint image into a conservative binary ridge map.

    The input is assumed to contain dark fingerprint ridges on a
    lighter background.
    """
    if image is None:
        raise ValueError("Image cannot be None.")

    if mask is None:
        raise ValueError("Mask cannot be None.")

    if image.shape != mask.shape:
        raise ValueError("Image and mask must have the same shape.")

    gray = image.astype(np.uint8)

    # Mild smoothing suppresses isolated pixel noise without
    # destroying normal ridge spacing.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Local thresholding is more robust than a single global Otsu
    # threshold when illumination varies across a fingerprint.
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5,
    )

    binary[mask == 0] = 0

    # Remove tiny isolated blobs.
    min_area = max(
        20,
        int(round(image.size * 0.00008)),
    )

    binary = _remove_small_components(
        binary,
        min_area=min_area,
    )

    # A very small closing operation repairs isolated one-pixel gaps.
    # Avoid large kernels because they can merge neighbouring ridges.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    binary[mask == 0] = 0

    return binary


# ============================================================
# Skeletonization
# ============================================================

def build_skeleton(ridge_binary, mask=None):
    """
    Skeletonize the binary ridge image and remove skeleton pixels
    close to the segmentation boundary.
    """
    if ridge_binary is None:
        raise ValueError("Ridge binary image cannot be None.")

    skeleton = skeletonize(ridge_binary > 0)

    if mask is not None:
        skeleton[mask == 0] = False
        skeleton = _remove_boundary_skeleton(
            skeleton,
            mask,
            boundary_margin=5,
        )

    # Remove very short artificial branches created by thresholding.
    skeleton = _prune_short_spurs(
        skeleton,
        min_branch_length=8,
        iterations=3,
    )

    return skeleton


# ============================================================
# Neighborhood analysis
# ============================================================

def neighbor_count_map(skeleton):
    """Count 8-connected skeleton neighbours for each pixel."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0

    return cv2.filter2D(
        skeleton.astype(np.uint8),
        -1,
        kernel,
    )


def local_ridge_support(skeleton, y, x, radius=12):
    """Return local skeleton support around a candidate."""
    return _local_support(
        skeleton,
        y,
        x,
        radius=radius,
    )


# ============================================================
# Candidate clustering
# ============================================================

def cluster_points(points, min_distance):
    """
    Spatially suppress duplicate detections.

    Higher-confidence candidates are retained first.
    """
    if not points:
        return []

    selected = []

    sorted_points = sorted(
        points,
        key=lambda point: (
            point["confidence"],
            point["support"],
        ),
        reverse=True,
    )

    min_distance_sq = float(min_distance ** 2)

    for point in sorted_points:
        too_close = False

        for existing in selected:
            distance_squared = (
                (point["x"] - existing["x"]) ** 2
                + (point["y"] - existing["y"]) ** 2
            )

            if distance_squared < min_distance_sq:
                too_close = True
                break

        if not too_close:
            selected.append(point)

    return selected


# ============================================================
# Minutiae detection
# ============================================================

def _trace_direction(skeleton, y, x, steps=10):
    """
    Walk along each skeleton branch leaving (y, x) for up to ``steps``
    pixels. Returns a list of angles (radians, image coords, y down)
    from the point to the end of every branch.
    """
    h, w = skeleton.shape
    angles = []

    starts = [
        (y + dy, x + dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy or dx)
        and 0 <= y + dy < h
        and 0 <= x + dx < w
        and skeleton[y + dy, x + dx]
    ]

    for sy, sx in starts:
        visited = {(y, x), (sy, sx)}
        cy, cx = sy, sx

        for _ in range(steps - 1):
            nxt = None
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if (
                        (dy or dx)
                        and 0 <= ny < h
                        and 0 <= nx < w
                        and skeleton[ny, nx]
                        and (ny, nx) not in visited
                        # do not cross into another branch of the
                        # starting junction
                        and max(abs(ny - y), abs(nx - x)) > 1
                    ):
                        nxt = (ny, nx)
                        break
                if nxt:
                    break
            if nxt is None:
                break
            visited.add(nxt)
            cy, cx = nxt

        if (cy, cx) != (y, x):
            angles.append(float(np.arctan2(cy - y, cx - x)))

    return angles


def _minutia_angle(skeleton, y, x, minutia_type):
    """
    Minutia direction in [0, 2*pi) (image coordinates, y down).

    ridge_ending: direction pointing from the ridge body OUT through
                  the ending (away from the ridge).
    bifurcation:  direction in which the two daughter branches open
                  (opposite to the single stem).

    Returns None if the local skeleton cannot be traced.
    """
    angles = _trace_direction(skeleton, y, x)

    if minutia_type == "ridge_ending":
        if not angles:
            return None
        return float((angles[0] + np.pi) % (2 * np.pi))

    if len(angles) < 3:
        return None

    vectors = np.exp(1j * np.asarray(angles[:3]))

    def separation(i):
        others = [j for j in range(3) if j != i]
        return min(
            abs(np.angle(vectors[i] / vectors[j])) for j in others
        )

    stem = max(range(3), key=separation)
    daughters = [j for j in range(3) if j != stem]

    return float(np.angle(vectors[daughters].sum()) % (2 * np.pi))


def _angle_difference(a, b):
    return abs(float(np.angle(np.exp(1j * (a - b)))))


def _remove_spurious_pairs(points, period):
    """
    Remove the classic false-minutiae signatures:

      * two endings facing each other across a short gap (broken ridge)
      * an ending next to a bifurcation (spur / bridge artefact)
      * two bifurcations very close together (bridge)
    """
    drop = set()
    gap = 1.3 * period
    near = 0.8 * period

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            a, b = points[i], points[j]
            d = np.hypot(a["x"] - b["x"], a["y"] - b["y"])

            if d > gap:
                continue

            both_ending = (
                a["type"] == "ridge_ending" and b["type"] == "ridge_ending"
            )

            if both_ending:
                if (
                    a.get("angle") is not None
                    and b.get("angle") is not None
                ):
                    if _angle_difference(a["angle"], b["angle"]) > 2.2:
                        drop.update((i, j))
                continue

            if d <= near:
                drop.update((i, j))

    return [p for k, p in enumerate(points) if k not in drop]


def detect_minutiae(
    skeleton,
    mask,
    orientation,
    coherence,
    ridge_spacing=None,
):
    """
    Detect fingerprint ridge endings and bifurcations.

    Stages: boundary rejection, correct crossing-number
    classification, local support, coherence, junction-noise test,
    duplicate suppression, spurious-pair removal, true minutia
    direction.

    Changes relative to the original
    --------------------------------
    * Correct crossing number (see ``_crossing_number``).
    * The "branching noise" test counted pixels with >= 3 skeleton
      neighbours. Ordinary staircase pixels satisfy that, so it
      rejected about half of all good candidates. It now counts real
      junctions (CN >= 3).
    * Distance thresholds scale with the measured ridge spacing
      instead of image size.
    * No more all-or-nothing discard. The original threw away EVERY
      minutia when the count fell outside [2, 120], but a rolled
      print legitimately has 100-200 minutiae, and the density /
      confidence checks hid everything else. Points are always
      returned; ``reliable`` and ``reasons`` describe the quality.
    * ``angle`` is a real minutia direction (0..2*pi) traced along the
      skeleton. The old ``orientation`` was the pi-periodic gradient
      angle sampled at the pixel (perpendicular to the ridge).
      ``orientation`` is now the ridge orientation at the point.
    """
    if skeleton is None or mask is None:
        return _empty_result()

    if orientation is None or coherence is None:
        return _empty_result()

    if (
        skeleton.shape != mask.shape
        or skeleton.shape != orientation.shape
        or skeleton.shape != coherence.shape
    ):
        return _empty_result()

    h, w = skeleton.shape

    if h < 32 or w < 32:
        return _empty_result()

    period = float(ridge_spacing) if ridge_spacing else 10.0 * max(
        0.75, min(h, w) / 512.0
    )
    period = float(np.clip(period, 4.0, 25.0))

    cleaned = skeleton.copy().astype(bool)
    cleaned[mask == 0] = False

    cleaned = _remove_boundary_skeleton(
        cleaned, mask, boundary_margin=max(4, int(round(0.6 * period)))
    )

    cleaned = _prune_short_spurs(
        cleaned,
        min_branch_length=max(6, int(round(0.8 * period))),
        iterations=2,
    )

    cn_map = crossing_number_map(cleaned)

    image_border = max(8, int(round(1.2 * period)))
    mask_margin = max(6, int(round(1.0 * period)))
    support_radius = max(8, int(round(1.2 * period)))
    min_support = max(10, int(round(1.2 * period)))
    min_distance = max(6, int(round(0.8 * period)))
    coherence_threshold = 0.30

    mask_distance = _distance_to_mask_boundary(mask)

    junctions = (cn_map >= 3).astype(np.uint8)
    junction_density = cv2.filter2D(
        junctions, -1, np.ones((9, 9), dtype=np.uint8)
    )

    candidates = []

    ys, xs = np.where((cn_map == 1) | (cn_map == 3))

    for y, x in zip(ys, xs):
        y = int(y)
        x = int(x)

        if (
            x < image_border
            or y < image_border
            or x >= w - image_border
            or y >= h - image_border
        ):
            continue

        if mask[y, x] == 0 or mask_distance[y, x] < mask_margin:
            continue

        minutia_type = "ridge_ending" if cn_map[y, x] == 1 else "bifurcation"

        support = local_ridge_support(cleaned, y, x, radius=support_radius)

        if support < min_support:
            continue

        local_coherence = _local_coherence(coherence, y, x, radius=4)

        if local_coherence < coherence_threshold:
            continue

        if int(junction_density[y, x]) > 4:
            continue

        support_score = float(
            np.clip(
                (support - min_support) / max(1.0, 4.0 * period - min_support),
                0.0,
                1.0,
            )
        )

        distance_score = float(
            np.clip(mask_distance[y, x] / (2.0 * period), 0.0, 1.0)
        )

        confidence = float(
            np.clip(
                0.55 * local_coherence
                + 0.25 * support_score
                + 0.20 * distance_score,
                0.0,
                1.0,
            )
        )

        theta = orientation[y, x]

        candidates.append(
            {
                "x": x,
                "y": y,
                "type": minutia_type,
                "support": support,
                "coherence": local_coherence,
                "confidence": confidence,
                "orientation": float(theta) if np.isfinite(theta) else None,
            }
        )

    candidates = cluster_points(candidates, min_distance=min_distance)

    for point in candidates:
        point["angle"] = _minutia_angle(
            cleaned, point["y"], point["x"], point["type"]
        )

    candidates = _remove_spurious_pairs(candidates, period)

    points = [
        {
            "x": int(p["x"]),
            "y": int(p["y"]),
            "type": p["type"],
            "orientation": p["orientation"],
            "angle": p["angle"],
            "confidence": float(np.clip(p["confidence"], 0.0, 1.0)),
        }
        for p in candidates
    ]

    endings = [p for p in points if p["type"] == "ridge_ending"]
    bifurcations = [p for p in points if p["type"] == "bifurcation"]

    skeleton_pixels = int(np.count_nonzero(cleaned))

    if skeleton_pixels == 0 or not points:
        result = _empty_result()
        result["reasons"] = ["no_minutiae_found"]
        return result

    reasons = []

    point_density = len(points) / float(skeleton_pixels)
    mean_confidence = float(np.mean([p["confidence"] for p in points]))

    ratio = len(endings) / max(1, len(bifurcations))

    if len(points) < 8:
        reasons.append("too_few_minutiae")

    if point_density > 0.04:
        reasons.append("minutiae_density_too_high_noisy_skeleton")

    if mean_confidence < 0.50:
        reasons.append("low_mean_confidence")

    if ratio > 6.0 or ratio < 1.0 / 6.0:
        reasons.append("implausible_ending_to_bifurcation_ratio")

    return {
        "total": int(len(points)),
        "ridge_endings": int(len(endings)),
        "bifurcations": int(len(bifurcations)),
        "points": points,
        "reliable": len(reasons) == 0,
        "reasons": reasons,
        "mean_confidence": mean_confidence,
    }


# ============================================================
# Complete minutiae analysis
# ============================================================

def analyze_minutiae(
    image,
    mask,
    orientation,
    coherence,
    ridge_spacing=None,
):
    """
    Complete minutiae-processing pipeline.

    Returns the intermediate ridge binary and skeleton so the
    frontend/debugging layer can visualize what the detector used.
    """
    ridge_binary = build_ridge_binary(
        image,
        mask,
    )

    skeleton = build_skeleton(
        ridge_binary,
        mask=mask,
    )

    minutiae = detect_minutiae(
        skeleton=skeleton,
        mask=mask,
        orientation=orientation,
        coherence=coherence,
        ridge_spacing=ridge_spacing,
    )

    return {
        "ridge_binary": ridge_binary,
        "skeleton": skeleton,
        "minutiae": minutiae,
    }
