import os
import gdown
import tensorflow as tf

from dotenv import load_dotenv

load_dotenv()


# =========================
# BASE DIRECTORY
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
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
# MODEL PATHS
# =========================

MRI_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "MRImodel.h5"
)

XRAY_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "Xraymodel.h5"
)


# =========================
# GOOGLE DRIVE IDs
# =========================

MRI_MODEL_ID = os.getenv(
    "MRI_MODEL_ID"
)

XRAY_MODEL_ID = os.getenv(
    "XRAY_MODEL_ID"
)


# =========================
# DOWNLOAD MODEL
# =========================

def download_model(
    file_id,
    output_path
):

    if os.path.exists(output_path):

        print(
            f"Model already exists: {output_path}"
        )

        return


    if not file_id:

        raise RuntimeError(
            f"Google Drive ID missing for {output_path}"
        )


    print(
        f"Downloading model: {output_path}"
    )


    gdown.download(
        id=file_id,
        output=output_path,
        quiet=False
    )


    if not os.path.exists(output_path):

        raise RuntimeError(
            f"Failed to download model: {output_path}"
        )


# =========================
# LOAD BOTH MODELS
# =========================

def load_models():

    print("Checking MRI model...")

    download_model(
        MRI_MODEL_ID,
        MRI_MODEL_PATH
    )


    print("Checking X-Ray model...")

    download_model(
        XRAY_MODEL_ID,
        XRAY_MODEL_PATH
    )


    print("Loading MRI model...")

    mri_model = tf.keras.models.load_model(
        MRI_MODEL_PATH
    )


    print("Loading X-Ray model...")

    xray_model = tf.keras.models.load_model(
        XRAY_MODEL_PATH
    )


    print("Both models loaded successfully!")


    return (
        mri_model,
        xray_model
    )