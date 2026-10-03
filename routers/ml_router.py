import joblib
import os

from fastapi import APIRouter, HTTPException
from schemas import DiseaseInput


router = APIRouter(
    prefix="/api/ml",
    tags=["Machine Learning"]
)


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "modelDP.pkl"
)

model = joblib.load(MODEL_PATH)


@router.post("/disease-predict")
def disease_predict(data: DiseaseInput):

    features = [[
        data.age,
        data.sex,
        data.cp,
        data.trestbps,
        data.chol,
        data.fbs,
        data.restecg,
        data.thalach,
        data.exang,
        data.oldpeak,
        data.slope,
        data.ca,
        data.thal
    ]]

    try:

        prediction = model.predict(features)[0]

        result = {
            "prediction": int(prediction)
        }

        # Agar model probability support karta hai
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(features)[0]

            result["probability"] = float(
                max(probabilities)
            )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )