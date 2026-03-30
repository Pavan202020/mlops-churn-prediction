import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


# Load dataset
df = pd.read_csv("data/raw/churn.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Target mapping
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop ID column
df = df.drop("customerID", axis=1)

# Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Identify column types
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

print("Categorical features:", categorical_features)
print("Numeric features:", numeric_features)

# Numeric preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Full pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train pipeline
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)
probabilities = pipeline.predict_proba(X_test)[:, 1]

threshold = 0.3
probs = pipeline.predict_proba(X_test)[:, 1]
custom_preds = (probs > threshold).astype(int)

print("\nCustom Threshold Evaluation (0.3):")
print("Precision:", precision_score(y_test, custom_preds))
print("Recall:", recall_score(y_test, custom_preds))
print("F1 Score:", f1_score(y_test, custom_preds))

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save whole pipeline
joblib.dump(pipeline, "models/churn_pipeline.pkl")

print("\nPipeline saved at models/churn_pipeline.pkl")