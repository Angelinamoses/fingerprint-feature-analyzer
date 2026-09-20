import cv2
import numpy as np


def assess_image_quality(image, mask):
    """
    Calculate measurable image-quality indicators
    within the fingerprint foreground.

    Returns:
        Dictionary containing quality measurements.
    """

    foreground = image[mask > 0]

    if foreground.size == 0:
        return {
            "mean_intensity": None,
            "intensity_std": None,
            "local_variance": None,
            "foreground_pixels": 0,
        }

    mean_intensity = float(np.mean(foreground))
    intensity_std = float(np.std(foreground))

    image_float = image.astype(np.float32)

    mean = cv2.GaussianBlur(
        image_float,
        (21, 21),
        0
    )

    mean_squared = cv2.GaussianBlur(
        image_float ** 2,
        (21, 21),
        0
    )

    local_variance_map = (
        mean_squared - (mean ** 2)
    )

    local_variance_map = np.maximum(
        local_variance_map,
        0
    )

    local_variance = float(
        np.mean(local_variance_map[mask > 0])
    )

    return {
        "mean_intensity": mean_intensity,
        "intensity_std": intensity_std,
        "local_variance": local_variance,
        "foreground_pixels": int(foreground.size),
    }