import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes


def _largest_component(binary):
    """Keep the largest 8-connected component of a boolean mask."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=8,
    )

    if count <= 1:
        return binary.astype(bool)

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    return labels == largest


def segment_fingerprint(
    image,
    texture_sigma=None,
    threshold_factor=1.0,
):
    """
    Segment the fingerprint foreground from the background.

    Method: local intensity standard deviation ("texture") ->
    Otsu threshold -> opening -> largest component -> hole filling
    -> closing -> hole filling.

    Changes relative to the original implementation
    -----------------------------------------------
    1. The local-std window was GaussianBlur((21, 21), 0), i.e.
       sigma ~ 3.5 px. That is smaller than one ridge period
       (~10 px on 512 px NIST prints), so the texture measure
       oscillated with ridge phase and low-contrast areas fell below
       the Otsu threshold. The window is now ~8 px (scaled with the
       image size), spanning about one ridge period.
    2. The texture map is normalised by its 99th percentile rather
       than its maximum, so a single dark blotch does not dictate
       the Otsu threshold.
    3. Interior holes are filled. Previously the mask was riddled
       with holes, typically right at the print centre (the
       low-contrast core / delta area) which made singular-point and
       minutiae detection impossible exactly where it matters.

    Args:
        image: uint8 grayscale image (denoised, not CLAHE'd, works best).
        texture_sigma: Gaussian sigma for the texture map in pixels.
        threshold_factor: scale applied to the Otsu threshold.

    Returns:
        mask: Binary foreground mask (0 or 255)
        segmented: Fingerprint image with background removed
    """

    image_float = image.astype(np.float32)

    height, width = image.shape[:2]

    if texture_sigma is None:
        texture_sigma = max(4.0, 8.0 * min(height, width) / 512.0)

    mean = cv2.GaussianBlur(image_float, (0, 0), texture_sigma)
    mean_squared = cv2.GaussianBlur(image_float ** 2, (0, 0), texture_sigma)

    local_std = np.sqrt(np.maximum(mean_squared - mean ** 2, 0.0))

    reference = float(np.percentile(local_std, 99))

    if reference <= 1e-6:
        empty = np.zeros_like(image, dtype=np.uint8)
        return empty, cv2.bitwise_and(image, image, mask=empty)

    texture_uint8 = np.clip(
        local_std / reference * 255.0, 0, 255
    ).astype(np.uint8)

    otsu, _ = cv2.threshold(
        texture_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    mask = texture_uint8 > (otsu * float(threshold_factor))

    scale = min(height, width) / 512.0

    open_k = max(3, int(round(15 * scale)) | 1)
    close_k = max(5, int(round(31 * scale)) | 1)

    mask = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)),
    ).astype(bool)

    if not np.any(mask):
        empty = np.zeros_like(image, dtype=np.uint8)
        return empty, cv2.bitwise_and(image, image, mask=empty)

    mask = _largest_component(mask)
    mask = binary_fill_holes(mask)

    mask = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
    ).astype(bool)

    mask = binary_fill_holes(mask)

    mask_uint8 = (mask.astype(np.uint8)) * 255

    segmented = cv2.bitwise_and(image, image, mask=mask_uint8)

    return mask_uint8, segmented
