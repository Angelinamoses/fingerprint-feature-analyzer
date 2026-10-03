import numpy as np

from processing.preprocessing import (
    convert_to_grayscale,
    normalize_image,
    denoise_image,
    enhance_contrast,
)

from processing.segmentation import (
    segment_fingerprint,
)

from processing.orientation import (
    estimate_orientation,
)

from processing.frequency import (
    estimate_ridge_frequency,
)

from processing.density import (
    estimate_ridge_density,
)

from processing.quality import (
    assess_image_quality,
)

from processing.pattern import (
    analyze_pattern,
)

from processing.minutiae import (
    analyze_minutiae,
)

from processing.features import (
    extract_features,
)


def _safe_float(value):
    """
    Convert numeric values to JSON-safe Python floats.
    """

    if value is None:
        return None

    try:
        value = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(value):
        return None

    return value


def analyze_fingerprint(image):
    """
    Run the complete fingerprint-analysis pipeline.

    Pipeline:

        Input
          ↓
        Grayscale
          ↓
        Normalization
          ↓
        Denoising
          ↓
        CLAHE enhancement
          ↓
        Segmentation
          ↓
        Quality assessment
          ↓
        Orientation estimation
          ↓
        Ridge frequency
          ↓
        Ridge density
          ↓
        Pattern / singularity analysis
          ↓
        Minutiae analysis
          ↓
        Feature extraction

    Args:
        image:
            Input fingerprint image as a NumPy array.

    Returns:
        Dictionary containing intermediate processing
        results and final application features.
    """

    # ========================================================
    # Input validation
    # ========================================================

    if image is None:
        raise ValueError(
            "Input fingerprint image is None."
        )

    if not isinstance(
        image,
        np.ndarray,
    ):
        raise TypeError(
            "Input fingerprint image must be a NumPy array."
        )

    if image.size == 0:
        raise ValueError(
            "Input fingerprint image is empty."
        )

    if image.ndim not in (
        2,
        3,
    ):
        raise ValueError(
            "Input image must be a 2D grayscale "
            "or 3D color image."
        )

    # ========================================================
    # 1. Preprocessing
    # ========================================================

    gray = convert_to_grayscale(
        image
    )

    normalized = normalize_image(
        gray
    )

    denoised = denoise_image(
        normalized
    )

    enhanced = enhance_contrast(
        denoised
    )

    # ========================================================
    # 2. Fingerprint segmentation
    # ========================================================

    mask, segmented = (
        segment_fingerprint(
            enhanced
        )
    )

    foreground_pixels = int(
        np.count_nonzero(
            mask
        )
    )

    foreground_coverage = (
        foreground_pixels
        / max(
            1,
            mask.size,
        )
    )

    # ========================================================
    # 3. Image quality
    # ========================================================

    quality = assess_image_quality(
        enhanced,
        mask,
    )

    # ========================================================
    # 4. Orientation estimation
    # ========================================================

    orientation, coherence = (
        estimate_orientation(
            segmented
        )
    )

    # Calculate coherence only inside the
    # fingerprint foreground.
    valid_coherence = coherence[
        mask > 0
    ]

    valid_coherence = valid_coherence[
        np.isfinite(
            valid_coherence
        )
    ]

    if valid_coherence.size > 0:

        mean_orientation_coherence = (
            float(
                np.median(
                    valid_coherence
                )
            )
        )

    else:

        mean_orientation_coherence = None

    # ========================================================
    # 5. Ridge frequency
    # ========================================================

    (
        frequency,
        spacing,
        frequency_results,
    ) = estimate_ridge_frequency(
        segmented
    )

    # ========================================================
    # 6. Ridge density
    # ========================================================

    (
        density,
        density_results,
    ) = estimate_ridge_density(
        segmented
    )

    # ========================================================
    # 7. Pattern and singular-point analysis
    # ========================================================

    pattern_results = analyze_pattern(
        orientation=orientation,
        coherence=coherence,
        mask=mask,
        image_shape=enhanced.shape,
        block_size=16,
    )

    # ========================================================
    # 8. Minutiae analysis
    # ========================================================

    minutiae_results = analyze_minutiae(
        image=enhanced,
        mask=mask,
        orientation=orientation,
        coherence=coherence,
    )

    # ========================================================
    # 9. Summary
    # ========================================================

    summary = {
        "image_shape": [
            int(value)
            for value in image.shape
        ],

        "foreground_coverage":
            _safe_float(
                foreground_coverage
            ),

        "mean_orientation_coherence":
            _safe_float(
                mean_orientation_coherence
            ),

        "ridge_frequency":
            _safe_float(
                frequency
            ),

        "ridge_spacing":
            _safe_float(
                spacing
            ),

        "ridge_density":
            _safe_float(
                density
            ),
    }

    # ========================================================
    # 10. Complete intermediate results
    # ========================================================

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
            "row_estimates":
                frequency_results,
        },

        "ridge_density": {
            "density": density,
            "row_estimates":
                density_results,
        },

        # ----------------------------------------------------
        # Pattern / singularity outputs
        # ----------------------------------------------------

        "pattern": pattern_results[
            "pattern"
        ],

        "singular_points":
            pattern_results[
                "singular_points"
            ],

        "pattern_candidates":
            pattern_results[
                "candidates"
            ],

        "poincare":
            pattern_results[
                "poincare"
            ],

        "orientation_grid":
            pattern_results[
                "orientation_grid"
            ],

        "coherence_grid":
            pattern_results[
                "coherence_grid"
            ],

        "valid_orientation_grid":
            pattern_results[
                "valid_grid"
            ],

        # ----------------------------------------------------
        # Minutiae outputs
        # ----------------------------------------------------

        "minutiae":
            minutiae_results[
                "minutiae"
            ],

        "ridge_binary":
            minutiae_results[
                "ridge_binary"
            ],

        "skeleton":
            minutiae_results[
                "skeleton"
            ],
    }

    # ========================================================
    # 11. Final application features
    # ========================================================

    features = extract_features(
        results
    )

    results[
        "features"
    ] = features

    return results