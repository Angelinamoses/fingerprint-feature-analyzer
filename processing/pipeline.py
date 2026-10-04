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

from processing.enhancement import (
    gabor_enhance,
)

from processing.quality import (
    assess_image_quality,
)

from processing.pattern import (
    analyze_pattern,
)

from processing.classifier import (
    predict_pattern,
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
            denoised
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

    # Ridge orientation from the UNMASKED denoised image (a masked
    # image has an artificial gradient along the mask border); the
    # mask is applied inside the estimator.
    orientation, coherence = (
        estimate_orientation(
            denoised,
            mask=mask,
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
        denoised,
        orientation=orientation,
        mask=mask,
        coherence=coherence,
    )

    # ========================================================
    # 6. Ridge density
    # ========================================================

    (
        density,
        density_results,
    ) = estimate_ridge_density(
        denoised,
        orientation=orientation,
        mask=mask,
        coherence=coherence,
    )

    # ========================================================
    # 6b. Gabor ridge enhancement (used for minutiae)
    # ========================================================

    gabor, gabor_response = gabor_enhance(
        denoised,
        orientation,
        frequency,
        mask,
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

    # Optional learned classifier (models/pattern_svm.joblib).
    # When present and confident it decides the label; the
    # singular-point evidence is kept for transparency.
    learned = predict_pattern(gray)

    if learned is not None:
        rule_based = pattern_results["pattern"]
        pattern_results["pattern"] = {
            **learned,
            "subtype": rule_based.get("subtype"),
            "evidence": {
                **(rule_based.get("evidence") or {}),
                "rule_based_label": rule_based.get("label"),
            },
        }

    # ========================================================
    # 8. Minutiae analysis
    # ========================================================

    minutiae_results = analyze_minutiae(
        image=gabor,
        mask=mask,
        orientation=orientation,
        coherence=coherence,
        ridge_spacing=spacing,
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
            "gabor": gabor,
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