import os
import sys
import joblib
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = os.path.join(
    "text",
    "models",
    "text_model.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("TEXT DETECTION")
print("=" * 60)

print("\nLoading text model...")

model_data = joblib.load(
    MODEL_FILE
)

model = model_data["model"]
vectorizer = model_data["vectorizer"]

print("Text model loaded successfully.")


# ============================================================
# GET TEXT
# ============================================================

if len(sys.argv) > 1:

    text = " ".join(
        sys.argv[1:]
    )

else:

    text = input(
        "\nEnter text to analyze:\n"
    )


# ============================================================
# VALIDATE TEXT
# ============================================================

text = text.strip()

if not text:

    print("\nError: No text provided.")
    sys.exit(1)


# ============================================================
# TF-IDF TRANSFORMATION
# ============================================================

text_vector = vectorizer.transform(
    [text]
)


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(
    text_vector
)[0]

probabilities = model.predict_proba(
    text_vector
)[0]


# ============================================================
# LABEL MAPPING
# ============================================================

real_probability = probabilities[0] * 100
fake_probability = probabilities[1] * 100


if prediction == 0:

    label = "REAL"
    confidence = real_probability

else:

    label = "FAKE"
    confidence = fake_probability


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("TEXT ANALYSIS RESULT")
print("=" * 60)

print("\nInput text:")
print(text)

print("\nPrediction:", label)

print(
    f"Confidence: {confidence:.2f}%"
)

print(
    f"REAL / Human-written: {real_probability:.2f}%"
)

print(
    f"FAKE / AI-generated: {fake_probability:.2f}%"
)

print("\nLabel mapping:")
print("0 = REAL / Human-written")
print("1 = FAKE / AI-generated")

print("=" * 60)