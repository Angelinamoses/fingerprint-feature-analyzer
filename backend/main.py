from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np

from processing.pipeline import analyze_fingerprint


app = FastAPI(
    title="Fingerprint Feature Analyzer API",
    description="AI-based fingerprint image feature extraction API",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Fingerprint Feature Analyzer API is running"
    }


@app.post("/analyze-fingerprint")
async def analyze_fingerprint_endpoint(
    file: UploadFile = File(...)
):
    contents = await file.read()

    image_array = np.frombuffer(
        contents,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image."
        )

    results = analyze_fingerprint(image)

    features = results["features"]

    return {
        "success": True,
        "filename": file.filename,
        "features": {
            "pattern_type": features["pattern_type"],

        "image_quality": features["image_quality"],

        "core": features["core"],

        "delta": features["delta"],

        "ridge": {
            "density": features["ridge"]["density"],
            "orientation_degrees": (
                np.degrees(features["ridge"]["orientation"])
                if features["ridge"]["orientation"] is not None
                else None
            ),
            "orientation_coherence": features["ridge"]["orientation_coherence"],
            "frequency": features["ridge"]["frequency"],
            "spacing_pixels": features["ridge"]["spacing"],
        },

        "minutiae": features["minutiae"],
    }
}