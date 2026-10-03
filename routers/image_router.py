import numpy as np

from PIL import Image
from io import BytesIO

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException
)

from Services.model_loader import load_xray_model


router = APIRouter(
    prefix="/api/image",
    tags=["Image Analysis"]
)


# =========================
# LOAD X-RAY MODEL
# =========================

xray_model = load_xray_model()


# =========================
# CLASS NAMES
# =========================

XRAY_CLASSES = [
    "covid19",
    "normal",
    "pneumonia"
]


# =========================
# IMAGE PREPROCESSING
# =========================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (200, 200)
    )

    image = np.array(
        image,
        dtype=np.float32
    )

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# =========================
# ANALYZE X-RAY
# =========================

@router.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    image_type: str = Form(...)
):

    if image_type != "xray":

        raise HTTPException(
            status_code=400,
            detail="Only 'xray' image analysis is supported"
        )

    try:

        # Read image

        contents = await image.read()

        if not contents:

            raise HTTPException(
                status_code=400,
                detail="Image file is empty"
            )


        # Open image

        pil_image = Image.open(
            BytesIO(contents)
        )


        # Preprocess

        processed_image = preprocess_image(
            pil_image
        )


        # Prediction

        predictions = xray_model.predict(
            processed_image,
            verbose=0
        )


        # Predicted index

        predicted_index = int(
            np.argmax(predictions[0])
        )


        # Confidence

        confidence = float(
            np.max(predictions[0])
        )


        return {
            "image_type": "xray",
            "prediction": predicted_index,
            "class_name": XRAY_CLASSES[predicted_index],
            "confidence": round(
                confidence,
                4
            )
        }


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
