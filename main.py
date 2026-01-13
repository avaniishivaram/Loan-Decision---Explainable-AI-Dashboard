from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
import numpy as np

app = FastAPI()

# Correct model path
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

model = joblib.load(MODEL_PATH)

FEATURE_NAMES = [
    "Credit_History",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term"
]

class LoanInput(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str


@app.post("/predict")
def predict(data: LoanInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # Feature importance (from RandomForest)
    importances = model.named_steps["classifier"].feature_importances_

    explanation = sorted(
        zip(FEATURE_NAMES, importances),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "prediction": "Approved" if prediction == 1 else "Rejected",
        "approval_probability": float(prob),
        "feature_importance": explanation[:5],
        "input_values": input_df.to_dict(orient="records")[0]
    }
