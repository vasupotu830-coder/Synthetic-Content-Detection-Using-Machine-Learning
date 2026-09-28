import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PROJECT INFORMATION
# ============================================================

print("=" * 60)
print("SYNTHETIC CONTENT DETECTION USING MACHINE LEARNING")
print("AUDIO MODEL TRAINING")
print("=" * 60)


# ============================================================
# FILE PATHS
# ============================================================

DATASET_FILE = "audio/dataset/features.csv"
MODEL_DIR = "audio/models"
MODEL_FILE = "audio/models/audio_model.pkl"


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(
        f"Audio feature dataset not found: {DATASET_FILE}"
    )


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading audio feature dataset...")

data = pd.read_csv(DATASET_FILE)

print("Dataset shape:", data.shape)


# ============================================================
# CHECK DATASET
# ============================================================

if "label" not in data.columns:
    raise ValueError("Column 'label' not found in dataset.")

if "filename" not in data.columns:
    raise ValueError("Column 'filename' not found in dataset.")


# ============================================================
# FEATURE / LABEL SEPARATION
# ============================================================

feature_columns = [
    column
    for column in data.columns
    if column not in ["label", "filename"]
]

X = data[feature_columns]
y = data["label"]


print("\nNumber of features:", len(feature_columns))

print("\nFeature columns:")
for index, feature in enumerate(feature_columns, start=1):
    print(f"{index}. {feature}")


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\nLabel distribution:")
print(y.value_counts().sort_index())

print("\nLabel meaning:")
print("0 = REAL")
print("1 = FAKE")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# CREATE RANDOM FOREST MODEL
# ============================================================

print("\nCreating Random Forest model...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed.")


# ============================================================
# PREDICTION
# ============================================================

print("\nMaking predictions...")

y_pred = model.predict(X_test)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("AUDIO MODEL RESULTS")
print("=" * 60)

print("\nAccuracy:", accuracy)
print("Accuracy percentage: {:.2f}%".format(accuracy * 100))


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

print("\nConfusion Matrix Meaning:")
print("Rows    = Actual")
print("Columns = Predicted")

print("\n              Predicted")
print("              REAL  FAKE")
print("Actual REAL   {:4d}  {:4d}".format(cm[0][0], cm[0][1]))
print("Actual FAKE   {:4d}  {:4d}".format(cm[1][0], cm[1][1]))


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["REAL", "FAKE"],
        digits=4
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\nFeature Importance:")

feature_importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print(feature_importance.to_string(index=False))


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# SAVE MODEL
# ============================================================

model_data = {
    "model": model,
    "feature_names": feature_columns
}

joblib.dump(model_data, MODEL_FILE)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("AUDIO MODEL TRAINING COMPLETED")
print("=" * 60)

print("\nModel saved successfully:")
print(MODEL_FILE)

print("\nModel type:")
print("Random Forest Classifier")

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))
print("Features        :", len(feature_columns))
print("Accuracy        : {:.2f}%".format(accuracy * 100))

print("\nNote:")
print(
    "This accuracy represents performance on the held-out "
    "test split of the prepared audio dataset."
)

print("\nNext step:")
print("Create audio prediction module.")