import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib


# -------------------------------------------------
# Load dataset
# -------------------------------------------------

dataframe = pd.read_csv("dataset/features.csv")


# -------------------------------------------------
# Feature columns
# -------------------------------------------------

feature_columns = [
    "mean_brightness",
    "brightness_std",
    "edge_ratio",
    "sharpness",

    "blue_mean",
    "green_mean",
    "red_mean",

    "blue_std",
    "green_std",
    "red_std",

    "gradient_mean",
    "gradient_std",

    "face_found",
    "face_area_ratio",
    "face_mean_brightness",
    "face_brightness_std",
    "face_sharpness"
]


# -------------------------------------------------
# Input and target
# -------------------------------------------------

X = dataframe[feature_columns]
y = dataframe["label"]


# -------------------------------------------------
# Train / test split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Number of features:", len(feature_columns))


# -------------------------------------------------
# Create Random Forest
# -------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# -------------------------------------------------
# Train model
# -------------------------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)


# -------------------------------------------------
# Predictions
# -------------------------------------------------

y_pred = model.predict(X_test)


# -------------------------------------------------
# Evaluation
# -------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# -------------------------------------------------
# Save model
# -------------------------------------------------

joblib.dump(
    model,
    "models/deepfake_model_v3.pkl"
)

print("\nV3 model saved successfully!")

print(
    "Location: models/deepfake_model_v3.pkl"
)