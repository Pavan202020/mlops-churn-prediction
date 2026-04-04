from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.sklearn

print("API FILE LOADED")

app = FastAPI()

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/ChurnPredictionPipeline@champion")

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


@app.post("/predict")
def predict(customer: CustomerData):
    input_dict = customer.dict()
    df = pd.DataFrame([input_dict])

    probability = model.predict_proba(df)[0][1]

    threshold = 0.3
    prediction = 1 if probability > threshold else 0

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability),
        "threshold_used": threshold
    }