import cv2
import numpy as np


def segment_fingerprint(image):
    """
    Create a binary mask separating fingerprint region
    from background using local image variance.
    """

    block_size = 16

    variance = cv2.blur(
        image.astype(np.float32) ** 2,
        (block_size, block_size)
    ) - cv2.blur(
        image.astype(np.float32),
        (block_size, block_size)
    ) ** 2

    threshold = np.mean(variance)

    mask = np.where(variance > threshold, 255, 0).astype(np.uint8)

    return mask