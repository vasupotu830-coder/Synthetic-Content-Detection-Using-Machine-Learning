import os
import sys
import joblib
import pandas as pd

from audio_detector import extract_audio_features


# ============================================================
# PROJECT INFORMATION
# ============================================================

print("=" * 60)
print("SYNTHETIC CONTENT DETECTION USING MACHINE LEARNING")
print("AUDIO PREDICTION")
print("=" * 60)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_FILE = "audio/models/audio_model.pkl"


# ============================================================
# CHECK COMMAND-LINE ARGUMENT
# ============================================================

if len(sys.argv) != 2:
    print("\nUsage:")
    print("python src\\predict_audio.py <audio_file>")

    print("\nExample:")
    print(
        'python src\\predict_audio.py '
        '"audio\\uploads\\sample.wav"'
    )

    sys.exit(1)


audio_file = sys.argv[1]


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_FILE):
    print("\nERROR: Audio model not found.")
    print("Expected:", MODEL_FILE)
    sys.exit(1)


if not os.path.exists(audio_file):
    print("\nERROR: Audio file not found.")
    print("File:", audio_file)
    sys.exit(1)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading audio model...")

model_data = joblib.load(MODEL_FILE)

model = model_data["model"]
feature_names = model_data["feature_names"]

print("Model loaded successfully.")
print("Number of features:", len(feature_names))


# ============================================================
# EXTRACT AUDIO FEATURES
# ============================================================

print("\nAnalyzing audio...")
print("File:", audio_file)

try:
    features = extract_audio_features(audio_file)

except Exception as error:
    print("\nERROR: Could not extract audio features.")
    print("Reason:", error)
    sys.exit(1)


# ============================================================
# PREPARE FEATURES WITH FEATURE NAMES
# ============================================================

if len(features) != len(feature_names):
    print("\nERROR: Feature count mismatch.")
    print("Expected:", len(feature_names))
    print("Received:", len(features))
    sys.exit(1)


features_df = pd.DataFrame(
    [features],
    columns=feature_names
)


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(features_df)[0]

probabilities = model.predict_proba(features_df)[0]


# ============================================================
# PROBABILITY MAPPING
# ============================================================

real_probability = probabilities[0] * 100
fake_probability = probabilities[1] * 100


# ============================================================
# FINAL PREDICTION
# ============================================================

if prediction == 0:
    result = "REAL"
    confidence = real_probability
else:
    result = "FAKE"
    confidence = fake_probability


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("AUDIO DETECTION RESULT")
print("=" * 60)

print("\nFile:")
print(audio_file)

print("\nPrediction:")
print(result)

print("\nConfidence:")
print("{:.2f}%".format(confidence))

print("\nProbabilities:")
print("REAL: {:.2f}%".format(real_probability))
print("FAKE: {:.2f}%".format(fake_probability))

print("\n" + "=" * 60)
print("AUDIO ANALYSIS COMPLETED")
print("=" * 60)