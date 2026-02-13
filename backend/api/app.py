from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("backend/models/knn_model.pkl")

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
def predict(features: list[float]):
    prediction = model.predict([features])[0]
    return {"predicted_price": float(prediction)}
