import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = os.path.join(
    "text",
    "dataset",
    "text_dataset.csv"
)

MODEL_FILE = os.path.join(
    "text",
    "models",
    "text_model.pkl"
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    os.path.dirname(MODEL_FILE),
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("TEXT DETECTION MODEL TRAINING")
print("=" * 60)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading text dataset...")

data = pd.read_csv(
    DATASET_FILE
)

print("Dataset loaded successfully.")

print("Total samples:", len(data))

print("\nLabel distribution:")
print(data["label"].value_counts())


# ============================================================
# PREPARE FEATURES AND LABEL
# ============================================================

X = data["text"].astype(str)

y = data["label"]


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
print("Testing samples:", len(X_test))


# ============================================================
# TF-IDF FEATURE EXTRACTION
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

print(
    "TF-IDF training shape:",
    X_train_tfidf.shape
)

print(
    "TF-IDF testing shape:",
    X_test_tfidf.shape
)


# ============================================================
# TRAIN RANDOM FOREST MODEL
# ============================================================

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(
    X_train_tfidf,
    y_train
)

print("Model training completed.")


# ============================================================
# PREDICTION
# ============================================================

print("\nMaking predictions...")

y_pred = model.predict(
    X_test_tfidf
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "REAL",
            "FAKE"
        ]
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("Confusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# SAVE MODEL + VECTORIZER
# ============================================================

model_data = {
    "model": model,
    "vectorizer": vectorizer
}

joblib.dump(
    model_data,
    MODEL_FILE
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("TEXT MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print("\nModel file:")
print(MODEL_FILE)

print("\nLabel mapping:")
print("0 = REAL / Human-written")
print("1 = FAKE / AI-generated")

print("\nModel:")
print("Random Forest")

print("Text features:")
print("TF-IDF")

print("=" * 60)