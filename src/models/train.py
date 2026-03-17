import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("data/raw/churn.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Convert target column to numeric
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop customerID because it's just an identifier
df = df.drop("customerID", axis=1)

# Fix TotalCharges column: convert to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Convert categorical columns into numeric dummy columns
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

print("Feature shape after encoding:", X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)