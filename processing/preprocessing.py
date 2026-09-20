import cv2
import numpy as np


def load_image(image_path):
    """Load a fingerprint image from disk."""
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    return image


def convert_to_grayscale(image):
    """Convert an image to grayscale."""
    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_image(image):
    """Normalize pixel intensity values to 0-255."""
    normalized = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return normalized.astype(np.uint8)


def denoise_image(image):
    """Reduce small-scale noise while preserving image structure."""
    return cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )


def enhance_contrast(image):
    """Enhance local contrast using CLAHE."""
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(image)