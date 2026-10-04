from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

import cv2
import numpy as np

from processing.pipeline import (
    analyze_fingerprint,
)


app = FastAPI(
    title="Fingerprint Feature Analyzer API",
    description=(
        "AI-based fingerprint image "
        "feature extraction API"
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://fingerprint-feature-analyzer.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message":
            "Fingerprint Feature Analyzer API is running"
    }


@app.post(
    "/analyze-fingerprint"
)
async def analyze_fingerprint_endpoint(
    file: UploadFile = File(...),
):
    """
    Analyze an uploaded fingerprint image.
    """

    # --------------------------------------------------------
    # Validate file type
    # --------------------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/png",
    }

    if (
        file.content_type
        not in allowed_types
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Upload a JPG, JPEG, or PNG image."
            ),
        )

    # --------------------------------------------------------
    # Read uploaded bytes
    # --------------------------------------------------------

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # --------------------------------------------------------
    # Decode image
    # --------------------------------------------------------

    image_array = np.frombuffer(
        contents,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unable to read the uploaded "
                "fingerprint image."
            ),
        )

    # --------------------------------------------------------
    # Run analysis
    # --------------------------------------------------------

    try:

        results = analyze_fingerprint(
            image
        )

    except (
        ValueError,
        TypeError,
    ) as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Fingerprint analysis failed."
            ),
        ) from exc

    # --------------------------------------------------------
    # Extract final features
    # --------------------------------------------------------

    features = results[
        "features"
    ]

    ridge = features[
        "ridge"
    ]

    orientation = ridge[
        "orientation"
    ]

    orientation_degrees = (
        float(
            np.degrees(
                orientation
            )
        )
        if orientation is not None
        else None
    )

    # --------------------------------------------------------
    # API response
    # --------------------------------------------------------

    return {
        "success": True,

        "filename": file.filename,

        "features": {

            "pattern_type":
                features[
                    "pattern_type"
                ],

            "pattern_details":
                features[
                    "pattern_details"
                ],

            "image_quality":
                features[
                    "image_quality"
                ],

            "core":
                features[
                    "core"
                ],

            "delta":
                features[
                    "delta"
                ],

            "ridge": {

                "density":
                    ridge[
                        "density"
                    ],

                "orientation_degrees":
                    orientation_degrees,

                "orientation_coherence":
                    ridge[
                        "orientation_coherence"
                    ],

                "frequency":
                    ridge[
                        "frequency"
                    ],

                "spacing_pixels":
                    ridge[
                        "spacing"
                    ],
            },

            "minutiae":
                features[
                    "minutiae"
                ],
        },
    }