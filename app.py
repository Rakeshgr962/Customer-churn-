import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

model = joblib.load("customer_churn_model.joblib")
encoders = joblib.load("label_encoders.pkl")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # Encode categorical columns
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col].astype(str))

    prediction = model.predict(df)[0]

    return {
        "prediction": int(prediction)
    }
