"""
Optional learned pattern classifier (HOG + RBF-SVM).

The repository's own notebook (03_segmentation_experiments) trained
this model (89.6 % hold-out, and it labelled class1_Arc_0001 as Arc
with p = 0.977), but it was never moved into ``processing/``, so the
API only had the singular-point heuristic.

Train with:   python scripts/train_pattern_classifier.py
The model is loaded from  models/pattern_svm.joblib  if it exists.
"""

import os

import cv2
import numpy as np

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "pattern_svm.joblib",
)

_MODEL = {"loaded": False, "model": None}


def hog_features(gray, size=(128, 128)):
    from skimage.feature import hog

    img = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )


def load_model(path=MODEL_PATH):
    if _MODEL["loaded"]:
        return _MODEL["model"]

    _MODEL["loaded"] = True

    if os.path.exists(path):
        import joblib

        _MODEL["model"] = joblib.load(path)

    return _MODEL["model"]


def predict_pattern(gray, min_confidence=0.60):
    """
    Returns {"label", "confidence", "method"} or None when no model
    is available / the prediction is not confident enough.
    """
    model = load_model()

    if model is None:
        return None

    probabilities = model.predict_proba([hog_features(gray)])[0]
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])

    if confidence < min_confidence:
        return None

    return {
        "label": str(model.classes_[index]).capitalize().replace("Arc", "Arch"),
        "confidence": confidence,
        "method": "hog_rbf_svm",
    }
