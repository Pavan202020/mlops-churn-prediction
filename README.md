# MLOps Churn Prediction Pipeline

## Problem
Customer churn prediction is critical for telecom companies to retain customers and reduce revenue loss.
This project builds an end-to-end ML system to:
1. Train a churn prediction model
2. Track experiments using MLflow
3. Register and version models
4. Serve predictions via a FastAPI API
5. Log real-time inference data

## Architecture
1. Train model using Scikit-learn pipeline
2. Log experiments in MLflow
3. Register best model in MLflow Model Registry
4. Assign alias (@champion)
5. Load model in FastAPI service
6. Serve predictions via /predict endpoint
7. Log inference data back to MLflow

## Tech Stack
1. Python
2. Scikit- Learn
3. Fast API
4. ML Flow

## Features
1. Experiment tracking with MLflow
2. Model versioning and registry
3. Model promotion using alias (@champion)
4. REST API for predictions
5. Input validation using Pydantic
6. Inference logging to MLflow

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train model
python src/models/train.py

### 3. Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

### 4. Start API
uvicorn src.api.app:app --reload

### 5. Test API
Open: http://127.0.0.1:8000/docs

## Sample Prediction Response

{
  "prediction": 1,
  "churn_probability": 0.31,
  "threshold_used": 0.3,
  "model_name": "ChurnPredictionPipeline",
  "model_alias": "champion"
}
