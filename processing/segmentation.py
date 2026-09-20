import cv2
import numpy as np


def segment_fingerprint(image):
    """
    Segment the fingerprint foreground from the background.

    Returns:
        mask: Binary foreground mask (0 or 255)
        segmented: Fingerprint image with background removed
    """

    image_float = image.astype(np.float32)

    # Local texture estimation
    mean = cv2.GaussianBlur(image_float, (21, 21), 0)
    mean_squared = cv2.GaussianBlur(image_float ** 2, (21, 21), 0)

    local_variance = mean_squared - (mean ** 2)
    local_variance = np.maximum(local_variance, 0)

    local_std = np.sqrt(local_variance)

    # Normalize texture map
    texture_uint8 = cv2.normalize(
        local_std,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Automatic thresholding
    _, texture_mask = cv2.threshold(
        texture_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15)
    )

    cleaned_mask = cv2.morphologyEx(
        texture_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Keep largest connected region
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned_mask,
        connectivity=8
    )

    if num_labels <= 1:
        return cleaned_mask, cv2.bitwise_and(
            image,
            image,
            mask=cleaned_mask
        )

    largest_label = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    mask = np.zeros_like(cleaned_mask)
    mask[labels == largest_label] = 255

    # Apply mask
    segmented = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    return mask, segmented