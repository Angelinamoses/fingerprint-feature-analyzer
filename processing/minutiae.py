import cv2
import numpy as np
from skimage.morphology import skeletonize


# ============================================================
# Ridge binary
# ============================================================

def build_ridge_binary(
    image,
    mask,
):
    """
    Convert an enhanced fingerprint image into a binary
    ridge representation.

    Assumes dark ridges on a lighter background.

    Returns:
        Binary ridge image with values 0 or 255.
    """

    if image is None:
        raise ValueError(
            "Image cannot be None."
        )

    if mask is None:
        raise ValueError(
            "Mask cannot be None."
        )

    if image.shape != mask.shape:
        raise ValueError(
            "Image and mask must have the same shape."
        )

    blur = cv2.GaussianBlur(
        image,
        (3, 3),
        0,
    )

    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU,
    )

    # Never allow background pixels to become
    # artificial fingerprint ridges.
    binary[mask == 0] = 0

    # Remove tiny connected components.
    number_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    clean = np.zeros_like(
        binary
    )

    min_area = max(
        8,
        int(
            image.size
            * 0.00002
        ),
    )

    for label in range(
        1,
        number_labels,
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA,
        ]

        if area >= min_area:
            clean[
                labels == label
            ] = 255

    # Close tiny ridge gaps.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    clean = cv2.morphologyEx(
        clean,
        cv2.MORPH_CLOSE,
        kernel,
    )

    clean[mask == 0] = 0

    return clean


# ============================================================
# Skeletonization
# ============================================================

def build_skeleton(
    ridge_binary,
):
    """
    Skeletonize the binary ridge image.

    Returns:
        Boolean skeleton.
    """

    if ridge_binary is None:
        raise ValueError(
            "Ridge binary image cannot be None."
        )

    skeleton = skeletonize(
        ridge_binary > 0
    )

    return skeleton


# ============================================================
# Neighborhood analysis
# ============================================================

def neighbor_count_map(
    skeleton,
):
    """
    Count 8-connected skeleton neighbors
    for every skeleton pixel.
    """

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    kernel[1, 1] = 0

    return cv2.filter2D(
        skeleton.astype(np.uint8),
        -1,
        kernel,
    )


def local_ridge_support(
    skeleton,
    y,
    x,
    radius=7,
):
    """
    Count skeleton pixels around a candidate.
    """

    height, width = skeleton.shape

    y1 = max(
        0,
        y - radius,
    )

    y2 = min(
        height,
        y + radius + 1,
    )

    x1 = max(
        0,
        x - radius,
    )

    x2 = min(
        width,
        x + radius + 1,
    )

    return int(
        np.count_nonzero(
            skeleton[
                y1:y2,
                x1:x2
            ]
        )
    )


# ============================================================
# Candidate clustering
# ============================================================

def cluster_points(
    points,
    min_distance,
):
    """
    Spatially suppress duplicate minutiae candidates.

    Candidates with stronger local ridge support are retained.
    """

    if not points:
        return []

    selected = []

    sorted_points = sorted(
        points,
        key=lambda point: (
            point["support"],
            point["coherence"],
        ),
        reverse=True,
    )

    for point in sorted_points:

        is_too_close = False

        for existing in selected:

            distance_squared = (
                (
                    point["x"]
                    - existing["x"]
                ) ** 2
                +
                (
                    point["y"]
                    - existing["y"]
                ) ** 2
            )

            if (
                distance_squared
                < min_distance ** 2
            ):
                is_too_close = True
                break

        if not is_too_close:
            selected.append(
                point
            )

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
    Detect ridge endings and bifurcations using
    skeleton-neighbor analysis.

    The detector performs:
        - border rejection
        - segmentation rejection
        - local ridge-support filtering
        - spatial clustering
        - confidence estimation
        - final sanity checking

    Returns:
        {
            "total": ...,
            "ridge_endings": ...,
            "bifurcations": ...,
            "points": [...],
            "reliable": bool
        }
    """

    if skeleton is None:
        return {
            "total": None,
            "ridge_endings": None,
            "bifurcations": None,
            "points": [],
            "reliable": False,
        }

    if mask is None:
        return {
            "total": None,
            "ridge_endings": None,
            "bifurcations": None,
            "points": [],
            "reliable": False,
        }

    if orientation is None or coherence is None:
        return {
            "total": None,
            "ridge_endings": None,
            "bifurcations": None,
            "points": [],
            "reliable": False,
        }

    if (
        skeleton.shape != mask.shape
        or skeleton.shape != orientation.shape
        or skeleton.shape != coherence.shape
    ):
        return {
            "total": None,
            "ridge_endings": None,
            "bifurcations": None,
            "points": [],
            "reliable": False,
        }

    neighbors = neighbor_count_map(
        skeleton
    )

    height, width = skeleton.shape

    # Scale thresholds for different image resolutions.
    scale = max(
        1.0,
        min(height, width) / 512.0,
    )

    border = max(
        8,
        int(
            round(
                12 * scale
            )
        ),
    )

    min_support = max(
        6,
        int(
            round(
                8 * scale
            )
        ),
    )

    min_distance = max(
        6,
        int(
            round(
                10 * scale
            )
        ),
    )

    support_radius = max(
        4,
        int(
            round(
                7 * scale
            )
        ),
    )

    endings = []
    bifurcations = []

    skeleton_y, skeleton_x = np.where(
        skeleton
    )

    for y, x in zip(
        skeleton_y,
        skeleton_x,
    ):

        # Reject image boundaries.
        if (
            x < border
            or y < border
            or x >= width - border
            or y >= height - border
        ):
            continue

        # Reject outside segmentation.
        if mask[y, x] == 0:
            continue

        support = local_ridge_support(
            skeleton,
            y,
            x,
            radius=support_radius,
        )

        if support < min_support:
            continue

        local_coherence = float(
            coherence[y, x]
        )

        if not np.isfinite(
            local_coherence
        ):
            local_coherence = 0.0

        neighbor_count = int(
            neighbors[y, x]
        )

        if neighbor_count == 1:

            endings.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "support": support,
                    "coherence":
                        local_coherence,
                    "type":
                        "ridge_ending",
                }
            )

        elif neighbor_count >= 3:

            bifurcations.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "support": support,
                    "coherence":
                        local_coherence,
                    "type":
                        "bifurcation",
                }
            )

    # Remove clusters of nearby candidates.
    endings = cluster_points(
        endings,
        min_distance,
    )

    bifurcations = cluster_points(
        bifurcations,
        min_distance,
    )

    # --------------------------------------------------------
    # Convert to application format
    # --------------------------------------------------------

    points = []

    for point in (
        endings + bifurcations
    ):

        theta = orientation[
            point["y"],
            point["x"],
        ]

        if np.isfinite(theta):
            theta = float(theta)
        else:
            theta = None

        support_score = min(
            1.0,
            point["support"] / 25.0,
        )

        confidence = float(
            np.clip(
                0.5
                * point["coherence"]
                +
                0.5
                * support_score,
                0.0,
                1.0,
            )
        )

        points.append(
            {
                "x": point["x"],
                "y": point["y"],
                "type": point["type"],
                "orientation": theta,
                "confidence": confidence,
            }
        )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    skeleton_pixels = int(
        np.count_nonzero(
            skeleton
        )
    )

    point_count = len(
        points
    )

    density = (
        point_count
        / max(
            1,
            skeleton_pixels,
        )
    )

    reliable = bool(
        point_count >= 2
        and point_count
        <= max(
            300,
            int(
                skeleton_pixels
                * 0.08
            ),
        )
        and density < 0.12
    )

    if not reliable:

        return {
            "total": None,
            "ridge_endings": None,
            "bifurcations": None,
            "points": [],
            "reliable": False,
        }

    return {
        "total": int(
            point_count
        ),

        "ridge_endings": int(
            len(endings)
        ),

        "bifurcations": int(
            len(bifurcations)
        ),

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
    """

    ridge_binary = build_ridge_binary(
        image,
        mask,
    )

    skeleton = build_skeleton(
        ridge_binary
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