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

REAL_FOLDER = "dataset/real"
FAKE_FOLDER = "dataset/fake"


def predict(model, image_path):

    image = preprocess_image(image_path)

    features = extract_features(image)

    features_dataframe = pd.DataFrame(
        [features],
        columns=FEATURE_COLUMNS
    )

    prediction = model.predict(
        features_dataframe
    )[0]

    probabilities = model.predict_proba(
        features_dataframe
    )[0]

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

    return prediction, confidence


def test_folder(
    model,
    folder,
    actual_label,
    number_of_images=5
):

    if not os.path.exists(folder):

        print(
            f"Folder not found: {folder}"
        )

        return 0, 0

    files = [
        file
        for file in os.listdir(folder)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    files = files[:number_of_images]

    correct = 0

    for filename in files:

        image_path = os.path.join(
            folder,
            filename
        )

        try:

            prediction, confidence = predict(
                model,
                image_path
            )

            is_correct = (
                prediction == actual_label
            )

            if is_correct:
                correct += 1

            status = (
                "CORRECT"
                if is_correct
                else "WRONG"
            )

            print(
                f"{filename} | "
                f"Actual: {actual_label} | "
                f"Predicted: {prediction} | "
                f"Confidence: "
                f"{confidence * 100:.2f}% | "
                f"{status}"
            )

        except Exception as error:

            print(
                f"{filename} | "
                f"ERROR: {error}"
            )

    print(
        f"\n{actual_label} result: "
        f"{correct}/{len(files)} correct"
    )

    return correct, len(files)


def main():

    print("\n==============================")
    print("     CURRENT MODEL TEST")
    print("==============================\n")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Testing REAL images...\n"
    )

    real_correct, real_total = test_folder(
        model,
        REAL_FOLDER,
        "REAL"
    )

    print(
        "\n--------------------------------\n"
    )

    print(
        "Testing FAKE images...\n"
    )

    fake_correct, fake_total = test_folder(
        model,
        FAKE_FOLDER,
        "FAKE"
    )

    total_correct = (
        real_correct + fake_correct
    )

    total_images = (
        real_total + fake_total
    )

    print("\n==============================")
    print("          SUMMARY")
    print("==============================")

    print(
        f"\nCorrect: "
        f"{total_correct}/{total_images}"
    )

    if total_images > 0:

        accuracy = (
            total_correct
            / total_images
            * 100
        )

        print(
            f"Accuracy: {accuracy:.2f}%"
        )

    else:

        print(
            "Accuracy: No images tested"
        )

    print(
        "\n==============================\n"
    )


if __name__ == "__main__":

    main()