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

from Services.model_loader import load_models


router = APIRouter(
    prefix="/api/image",
    tags=["Image Analysis"]
)


# =========================
# LOAD MODELS
# =========================

mri_model, xray_model = load_models()


# =========================
# CLASS NAMES
# =========================

MRI_CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
    "unknown"
]


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
# ANALYZE IMAGE
# =========================

@router.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    image_type: str = Form(...)
):

    try:

        # Validate image type

        if image_type not in [
            "mri",
            "xray"
        ]:

            raise HTTPException(
                status_code=400,
                detail="image_type must be 'mri' or 'xray'"
            )


        # Read image

        contents = await image.read()


        # Open image

        pil_image = Image.open(
            BytesIO(contents)
        )


        # Preprocess

        processed_image = preprocess_image(
            pil_image
        )


        # Select model

        if image_type == "mri":

            model = mri_model
            classes = MRI_CLASSES

        else:

            model = xray_model
            classes = XRAY_CLASSES


        # Prediction

        predictions = model.predict(
            processed_image,
            verbose=0
        )


        # Predicted index

        predicted_index = int(
            np.argmax(
                predictions[0]
            )
        )


        # Confidence

        confidence = float(
            np.max(
                predictions[0]
            )
        )


        return {

            "image_type": image_type,

            "prediction": predicted_index,

            "class_name": classes[
                predicted_index
            ],

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