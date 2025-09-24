from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from src.data import load_data, train_test_split_time_aware
from src.model import FraudModel
from src.evaluate import plot_confusion, plot_pr_curve
from src.features import add_simple_features 


app = FastAPI()

# Try to load model with error handling
try:
    model = joblib.load("models/rf_model.joblib")
except Exception as e:
    model = None
    print(f"Model loading error: {e}")


# Define all features expected by the model (example: V1-V28, Time, Amount)
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.post("/predict")
def predict(tx: Transaction):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        data = pd.DataFrame([tx.dict()])
        # Apply same feature engineering as training
        data = add_simple_features(data)
        prob = model.predict_proba(data)[:, 1][0]
        return {"fraud_prob": float(prob)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")