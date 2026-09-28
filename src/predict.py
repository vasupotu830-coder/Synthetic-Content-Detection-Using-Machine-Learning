import sys
import os
import joblib
import pandas as pd

from preprocessing import preprocess_image
from feature_extraction import extract_features


MODEL_PATH = "models/deepfake_model_group_split.pkl"

FEATURE_COLUMNS = [
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


def predict_image(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Preprocess image
    image = preprocess_image(image_path)

    # Extract 17 features
    features = extract_features(image)

    # Create DataFrame with the same feature names
    features_dataframe = pd.DataFrame(
        [features],
        columns=FEATURE_COLUMNS
    )

    # Make prediction
    prediction = model.predict(features_dataframe)[0]

    # Get probabilities
    probabilities = model.predict_proba(
        features_dataframe
    )[0]

    # Handle model class labels
    classes = list(model.classes_)

    fake_probability = probabilities[
        classes.index("FAKE")
    ]

    real_probability = probabilities[
        classes.index("REAL")
    ]

    confidence = max(
        fake_probability,
        real_probability
    )

    print("\n==============================")
    print("      SYNTHETIC CONTENT DETECTOR")
    print("==============================")

    print(f"\nImage: {image_path}")

    print(f"\nPrediction: {prediction}")

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    print(
        f"REAL Probability: "
        f"{real_probability * 100:.2f}%"
    )

    print(
        f"FAKE Probability: "
        f"{fake_probability * 100:.2f}%"
    )

    print("==============================\n")


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print("Usage:")

        print(
            'python src\\predict.py "image_path"'
        )

        sys.exit(1)

    image_path = sys.argv[1]

    try:

        predict_image(image_path)

    except Exception as error:

        print(
            f"\nError: {error}"
        )