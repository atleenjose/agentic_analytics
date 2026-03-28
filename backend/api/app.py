from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI(title="High Cost Prediction API")

model = joblib.load("backend/models/high_cost_model_refined.pkl")


class PredictRequest(BaseModel):
    msg_count_5min: int
    model_tier_encoded: int   # 0=Basic, 1=Standard, 2=Premium
    user_tier_encoded: int    # 0=Free, 1=Pro, 2=Enterprise


@app.get("/")
def home():
    return {"status": "running", "model": "high_cost_model_refined"}


@app.post("/predict")
def predict(req: PredictRequest):
    features = [[req.msg_count_5min, req.model_tier_encoded, req.user_tier_encoded]]
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    return {
        "high_cost_flag": bool(prediction),
        "probability": round(float(probability), 4),
    }
