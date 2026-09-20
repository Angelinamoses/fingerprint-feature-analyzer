from processing.preprocessing import (
    convert_to_grayscale,
    normalize_image,
    denoise_image,
    enhance_contrast,
)

from processing.segmentation import segment_fingerprint
from processing.orientation import estimate_orientation
from processing.frequency import estimate_ridge_frequency
from processing.density import estimate_ridge_density
from processing.quality import assess_image_quality
from processing.features import extract_features


def analyze_fingerprint(image):
    """
    Run the fingerprint image through the complete
    processing pipeline.

    Args:
        image: Input fingerprint image as a NumPy array.

    Returns:
        Dictionary containing processing results
        and extracted fingerprint features.
    """

    # --------------------------------------------------
    # 1. Preprocessing
    # --------------------------------------------------
    gray = convert_to_grayscale(image)
    normalized = normalize_image(gray)
    denoised = denoise_image(normalized)
    enhanced = enhance_contrast(denoised)

    # --------------------------------------------------
    # 2. Fingerprint segmentation
    # --------------------------------------------------
    mask, segmented = segment_fingerprint(enhanced)

    # --------------------------------------------------
    # 3. Image quality assessment
    # --------------------------------------------------
    quality = assess_image_quality(
        enhanced,
        mask
    )

    # --------------------------------------------------
    # 4. Orientation estimation
    # --------------------------------------------------
    orientation, coherence = estimate_orientation(
        segmented
    )

    # --------------------------------------------------
    # 5. Ridge frequency estimation
    # --------------------------------------------------
    frequency, spacing, frequency_results = (
        estimate_ridge_frequency(segmented)
    )

    # --------------------------------------------------
    # 6. Ridge density estimation
    # --------------------------------------------------
    density, density_results = estimate_ridge_density(
        segmented
    )

    # --------------------------------------------------
    # 7. Summary
    # --------------------------------------------------
    summary = {
        "image_shape": list(image.shape),

        "foreground_coverage": float(
            (mask > 0).mean()
        ),

        "mean_orientation_coherence": float(
            coherence.mean()
        ),

        "ridge_frequency": (
            float(frequency)
            if frequency is not None
            else None
        ),

        "ridge_spacing": (
            float(spacing)
            if spacing is not None
            else None
        ),

        "ridge_density": (
            float(density)
            if density is not None
            else None
        ),
    }

    # --------------------------------------------------
    # 8. Processing results
    # --------------------------------------------------
    results = {
        "summary": summary,

        "quality": quality,

        "preprocessing": {
            "gray": gray,
            "normalized": normalized,
            "denoised": denoised,
            "enhanced": enhanced,
        },

        "segmentation": {
            "mask": mask,
            "segmented": segmented,
        },

        "orientation": {
            "orientation": orientation,
            "coherence": coherence,
        },

        "ridge_frequency": {
            "frequency": frequency,
            "spacing": spacing,
            "row_estimates": frequency_results,
        },

        "ridge_density": {
            "density": density,
            "row_estimates": density_results,
        },
    }

    # --------------------------------------------------
    # 9. Extract application features
    # --------------------------------------------------
    features = extract_features(results)

    results["features"] = features

    return results