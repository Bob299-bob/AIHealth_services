import os
import gdown
import tensorflow as tf

from dotenv import load_dotenv


# =========================
# LOAD ENV
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

load_dotenv(
    os.path.join(BASE_DIR, ".env")
)


# =========================
# MODEL DIRECTORY
# =========================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "ml_models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# =========================
# X-RAY MODEL
# =========================

XRAY_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "Xraymodel.h5"
)

XRAY_MODEL_ID = os.getenv(
    "XRAY_MODEL_ID"
)


# =========================
# MODEL CACHE
# =========================

_xray_model = None


# =========================
# DOWNLOAD MODEL
# =========================

def download_model(
    file_id,
    output_path
):

    # Model already downloaded
    if os.path.exists(output_path):

        print(
            f"Model already exists: {output_path}"
        )

        return


    # Check Google Drive ID
    if not file_id:

        raise RuntimeError(
            "XRAY_MODEL_ID is missing from environment variables"
        )


    print(
        "Downloading X-Ray model..."
    )


    gdown.download(
        id=file_id,
        output=output_path,
        quiet=False
    )


    if not os.path.exists(output_path):

        raise RuntimeError(
            "Failed to download X-Ray model"
        )


# =========================
# GET X-RAY MODEL
# =========================

def get_xray_model():

    global _xray_model


    # Already loaded
    if _xray_model is not None:

        return _xray_model


    print(
        "Preparing X-Ray model..."
    )


    # Download only when required
    download_model(
        XRAY_MODEL_ID,
        XRAY_MODEL_PATH
    )


    print(
        "Loading X-Ray model..."
    )


    _xray_model = tf.keras.models.load_model(
        XRAY_MODEL_PATH
    )


    print(
        "X-Ray model loaded successfully!"
    )


    return _xray_model
