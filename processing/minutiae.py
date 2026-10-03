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
    }


def _crossing_number(neighborhood):
    """
    Crossing Number (CN) for an 8-neighbourhood.

    CN = 1 -> ridge ending
    CN = 3 -> bifurcation
    """
    p = neighborhood.flatten().astype(np.uint8)

    # Clockwise order around the center pixel.
    ring = np.array([
        p[0], p[1], p[2],
        p[5],       p[3],
        p[4], p[6], p[7],
        p[0],
    ])

    return int(np.sum(np.abs(np.diff(ring.astype(np.int16)))) // 2)


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

def detect_minutiae(
    skeleton,
    mask,
    orientation,
    coherence,
):
    """
    Detect fingerprint ridge endings and bifurcations.

    Detection is deliberately conservative:
        - segmentation-boundary rejection
        - minimum distance from mask boundary
        - crossing-number classification
        - local ridge support
        - local orientation coherence
        - short-spur pruning
        - duplicate suppression
        - global sanity checks

    If the resulting set is not trustworthy, the function returns
    reliable=False and does not fabricate feature counts.
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

    # Work on a cleaned skeleton. This also protects callers that
    # invoke detect_minutiae directly rather than through analyze_minutiae.
    cleaned = skeleton.copy().astype(bool)
    cleaned[mask == 0] = False

    cleaned = _remove_boundary_skeleton(
        cleaned,
        mask,
        boundary_margin=6,
    )

    cleaned = _prune_short_spurs(
        cleaned,
        min_branch_length=8,
        iterations=3,
    )

    neighbors = neighbor_count_map(cleaned)

    # Scale thresholds for different image resolutions.
    scale = max(
        0.75,
        min(h, w) / 512.0,
    )

    # Stay well inside the image and foreground.
    image_border = max(
        12,
        int(round(16 * scale)),
    )

    # A candidate must be this far inside the actual fingerprint mask.
    mask_margin = max(
        6,
        int(round(8 * scale)),
    )

    support_radius = max(
        8,
        int(round(12 * scale)),
    )

    min_support = max(
        10,
        int(round(12 * scale)),
    )

    min_distance = max(
        8,
        int(round(12 * scale)),
    )

    # Conservative coherence threshold.
    coherence_threshold = 0.45

    mask_distance = _distance_to_mask_boundary(mask)

    endings = []
    bifurcations = []

    skeleton_y, skeleton_x = np.where(cleaned)

    for y, x in zip(skeleton_y, skeleton_x):
        y = int(y)
        x = int(x)

        # Image-boundary rejection.
        if (
            x < image_border
            or y < image_border
            or x >= w - image_border
            or y >= h - image_border
        ):
            continue

        # Foreground and segmentation-boundary rejection.
        if mask[y, x] == 0:
            continue

        if mask_distance[y, x] < mask_margin:
            continue

        neighbor_count = int(neighbors[y, x])

        # Only accept exact CN-compatible skeleton structures.
        if neighbor_count not in (1, 3):
            continue

        y1 = max(0, y - 1)
        y2 = min(h, y + 2)
        x1 = max(0, x - 1)
        x2 = min(w, x + 2)

        neighborhood = cleaned[y1:y2, x1:x2]

        # CN requires a complete 3x3 neighbourhood.
        if neighborhood.shape != (3, 3):
            continue

        cn = _crossing_number(neighborhood)

        if cn == 1:
            minutia_type = "ridge_ending"
        elif cn == 3:
            minutia_type = "bifurcation"
        else:
            continue

        support = local_ridge_support(
            cleaned,
            y,
            x,
            radius=support_radius,
        )

        if support < min_support:
            continue

        local_coherence = _local_coherence(
            coherence,
            y,
            x,
            radius=4,
        )

        if local_coherence < coherence_threshold:
            continue

        # Reject candidates where the local skeleton is implausibly
        # sparse or dominated by branching noise.
        local_neighbors = neighbors[
            max(0, y - 4):min(h, y + 5),
            max(0, x - 4):min(w, x + 5),
        ]

        local_branch_pixels = int(
            np.count_nonzero(local_neighbors >= 3)
        )

        if local_branch_pixels > 18:
            continue

        # Confidence is based on measurable properties only.
        support_score = float(
            np.clip(
                (support - min_support)
                / max(1.0, 40.0 * scale - min_support),
                0.0,
                1.0,
            )
        )

        distance_score = float(
            np.clip(
                mask_distance[y, x]
                / max(1.0, 20.0 * scale),
                0.0,
                1.0,
            )
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

        if np.isfinite(theta):
            theta = float(theta)
        else:
            theta = None

        point = {
            "x": x,
            "y": y,
            "type": minutia_type,
            "support": support,
            "coherence": local_coherence,
            "confidence": confidence,
            "orientation": theta,
        }

        if minutia_type == "ridge_ending":
            endings.append(point)
        else:
            bifurcations.append(point)

    # Remove nearby duplicates.
    endings = cluster_points(
        endings,
        min_distance=min_distance,
    )

    bifurcations = cluster_points(
        bifurcations,
        min_distance=min_distance,
    )

    # Also suppress an ending and bifurcation that represent the same
    # local artifact.
    all_candidates = cluster_points(
        endings + bifurcations,
        min_distance=min_distance,
    )

    endings = [
        point
        for point in all_candidates
        if point["type"] == "ridge_ending"
    ]

    bifurcations = [
        point
        for point in all_candidates
        if point["type"] == "bifurcation"
    ]

    points = []

    for point in all_candidates:
        points.append(
            {
                "x": int(point["x"]),
                "y": int(point["y"]),
                "type": point["type"],
                "orientation": (
                    float(point["orientation"])
                    if point["orientation"] is not None
                    else None
                ),
                "confidence": float(
                    np.clip(
                        point["confidence"],
                        0.0,
                        1.0,
                    )
                ),
            }
        )

    # --------------------------------------------------------
    # Global sanity checks
    # --------------------------------------------------------

    skeleton_pixels = int(np.count_nonzero(cleaned))
    point_count = len(points)

    if skeleton_pixels == 0:
        return _empty_result()

    point_density = point_count / float(skeleton_pixels)

    # Fingerprint minutiae should be sparse relative to the complete
    # skeleton. Extremely dense detections indicate threshold/skeleton
    # artifacts rather than genuine minutiae.
    density_limit = 0.025

    # Avoid accepting a tiny number of weak points as a meaningful
    # detector result.
    confidence_values = [
        point["confidence"]
        for point in points
    ]

    mean_confidence = (
        float(np.mean(confidence_values))
        if confidence_values
        else 0.0
    )

    reliable = bool(
        2 <= point_count <= 120
        and point_density <= density_limit
        and mean_confidence >= 0.55
    )

    if not reliable:
        return _empty_result()

    return {
        "total": int(point_count),
        "ridge_endings": int(len(endings)),
        "bifurcations": int(len(bifurcations)),
        "points": points,
        "reliable": True,
    }


# ============================================================
# Complete minutiae analysis
# ============================================================

def analyze_minutiae(
    image,
    mask,
    orientation,
    coherence,
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
    )

    return {
        "ridge_binary": ridge_binary,
        "skeleton": skeleton,
        "minutiae": minutiae,
    }
