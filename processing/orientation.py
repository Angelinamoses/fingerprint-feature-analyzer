import cv2
import numpy as np


def _wrap_orientation(theta):
    """Wrap an orientation angle into [-pi/2, pi/2)."""
    return (theta + np.pi / 2.0) % np.pi - np.pi / 2.0


def estimate_orientation(
    image,
    block_size=16,
    mask=None,
    sigma_tensor=None,
    pre_sigma=1.0,
):
    """
    Estimate the local RIDGE orientation and orientation coherence.

    Convention (important)
    ----------------------
    The returned orientation is the direction of the ridge *lines*,
    measured from the +x axis in image coordinates (y points down),
    in radians, wrapped to [-pi/2, pi/2).

    The structure tensor gives the dominant *gradient* direction,
    which is perpendicular to the ridge, so pi/2 is added.
    (The previous implementation returned the gradient direction,
    i.e. the ridge orientation rotated by 90 degrees.)

    Args:
        image:
            Grayscale image. Pass the *unmasked* image together with
            ``mask`` -- feeding an already-masked image creates a
            strong artificial gradient along the mask border.
        block_size:
            Nominal analysis block. The structure tensor is averaged
            with a Gaussian of sigma = 0.4 * block_size unless
            ``sigma_tensor`` is given. This must span at least one
            ridge period; the old fixed 17x17/sigma~2.9 window was
            smaller than a ridge period on 512 px prints.
        mask:
            Optional foreground mask (0 / non-zero). Gradients outside
            the mask are excluded (normalised convolution).
        sigma_tensor:
            Override for the averaging sigma in pixels.
        pre_sigma:
            Small Gaussian pre-smoothing before taking gradients.

    Returns:
        orientation: ridge orientation map (radians, [-pi/2, pi/2))
        coherence:   0..1 map (1 = perfectly oriented neighbourhood)
    """
    img = image.astype(np.float32)

    if pre_sigma and pre_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), float(pre_sigma))

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    gxx = gx * gx
    gyy = gy * gy
    gxy = gx * gy

    sigma = float(sigma_tensor) if sigma_tensor else 0.4 * float(block_size)

    if mask is not None:
        w = (mask > 0).astype(np.float32)
        gxx *= w
        gyy *= w
        gxy *= w
        norm = cv2.GaussianBlur(w, (0, 0), sigma) + 1e-6
    else:
        norm = 1.0

    gxx = cv2.GaussianBlur(gxx, (0, 0), sigma) / norm
    gyy = cv2.GaussianBlur(gyy, (0, 0), sigma) / norm
    gxy = cv2.GaussianBlur(gxy, (0, 0), sigma) / norm

    gradient_orientation = 0.5 * np.arctan2(2.0 * gxy, gxx - gyy)

    # ridge direction is perpendicular to the dominant gradient
    orientation = _wrap_orientation(gradient_orientation + np.pi / 2.0)

    numerator = np.sqrt((gxx - gyy) ** 2 + (2.0 * gxy) ** 2)
    denominator = gxx + gyy + 1e-8
    coherence = np.clip(numerator / denominator, 0.0, 1.0)

    return orientation.astype(np.float32), coherence.astype(np.float32)
