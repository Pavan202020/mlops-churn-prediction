import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


# -----------------------------
# MLflow configuration
# -----------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("churn_prediction_experiments")


# -----------------------------
# Config / hyperparameters
# -----------------------------
random_state = 42
test_size = 0.2
n_estimators = 100
max_depth = None
min_samples_split = 2
threshold = 0.3


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/raw/churn.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Target mapping
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop identifier column
df = df.drop("customerID", axis=1)

# Features / target split
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Identify feature types
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

print("Categorical features:", categorical_features)
print("Numeric features:", numeric_features)


# -----------------------------
# Preprocessing pipelines
# -----------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# -----------------------------
# Model pipeline
# -----------------------------
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split
    ))
])


# -----------------------------
# Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)


# -----------------------------
# Train + track with MLflow
# -----------------------------
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_name", "RandomForestClassifier")
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("threshold", threshold)

    # Train
    pipeline.fit(X_train, y_train)

    # Default predictions
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Default metrics
    accuracy_default = accuracy_score(y_test, predictions)
    precision_default = precision_score(y_test, predictions)
    recall_default = recall_score(y_test, predictions)
    f1_default = f1_score(y_test, predictions)

    print("\nDefault Prediction Metrics")
    print("--------------------------")
    print("Accuracy :", accuracy_default)
    print("Precision:", precision_default)
    print("Recall   :", recall_default)
    print("F1 Score :", f1_default)

    print("\nClassification Report")
    print("---------------------")
    print(classification_report(y_test, predictions))

    # Threshold-based predictions
    custom_preds = (probabilities > threshold).astype(int)

    precision_threshold = precision_score(y_test, custom_preds)
    recall_threshold = recall_score(y_test, custom_preds)
    f1_threshold = f1_score(y_test, custom_preds)

    print(f"\nCustom Threshold Metrics (threshold = {threshold})")
    print("-----------------------------------------------")
    print("Precision:", precision_threshold)
    print("Recall   :", recall_threshold)
    print("F1 Score :", f1_threshold)

    # Log metrics
    mlflow.log_metric("accuracy_default", accuracy_default)
    mlflow.log_metric("precision_default", precision_default)
    mlflow.log_metric("recall_default", recall_default)
    mlflow.log_metric("f1_default", f1_default)

    mlflow.log_metric("precision_threshold", precision_threshold)
    mlflow.log_metric("recall_threshold", recall_threshold)
    mlflow.log_metric("f1_threshold", f1_threshold)

    # Save pipeline locally
    joblib.dump(pipeline, "models/churn_pipeline.pkl")
    print("\nPipeline saved at models/churn_pipeline.pkl")

    # Log pipeline model to MLflow
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="churn_pipeline_model"
    )