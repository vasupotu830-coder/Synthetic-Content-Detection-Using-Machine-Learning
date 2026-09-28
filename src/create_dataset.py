import os
import pandas as pd

from preprocessing import preprocess_image
from feature_extraction import extract_features


REAL_FOLDER = "dataset/real"
FAKE_FOLDER = "dataset/fake"


def process_folder(folder_path, label):
    data = []

    for filename in os.listdir(folder_path):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(folder_path, filename)

        try:
            # Load and preprocess image
            image = preprocess_image(image_path)

            # Extract 17 features
            features = extract_features(image)

            # 17 features + label = 18 values
            data.append([
                *features,
                label
            ])

            print(f"Processed: {filename}")

        except Exception as error:
            print(f"Could not process {filename}: {error}")

    return data


def create_dataset():

    # -------------------------------------------------
    # Process REAL images
    # -------------------------------------------------

    print("Processing REAL images...")

    real_data = process_folder(
        REAL_FOLDER,
        "REAL"
    )

    # -------------------------------------------------
    # Process FAKE images
    # -------------------------------------------------

    print("\nProcessing FAKE images...")

    fake_data = process_folder(
        FAKE_FOLDER,
        "FAKE"
    )

    # -------------------------------------------------
    # Combine data
    # -------------------------------------------------

    all_data = real_data + fake_data

    # -------------------------------------------------
    # Create DataFrame
    # -------------------------------------------------

    dataframe = pd.DataFrame(
        all_data,
        columns=[
            # Global image features
            "mean_brightness",
            "brightness_std",
            "edge_ratio",
            "sharpness",

            # Color features
            "blue_mean",
            "green_mean",
            "red_mean",

            # Color variation
            "blue_std",
            "green_std",
            "red_std",

            # Gradient / texture features
            "gradient_mean",
            "gradient_std",

            # Face features
            "face_found",
            "face_area_ratio",
            "face_mean_brightness",
            "face_brightness_std",
            "face_sharpness",

            # Target
            "label"
        ]
    )

    # -------------------------------------------------
    # Save dataset
    # -------------------------------------------------

    dataframe.to_csv(
        "dataset/features.csv",
        index=False
    )

    # -------------------------------------------------
    # Display results
    # -------------------------------------------------

    print("\nDataset created successfully!")

    print(
        f"Total images processed: {len(dataframe)}"
    )

    print(
        f"Number of features: {len(dataframe.columns) - 1}"
    )

    print("\nDataset shape:")
    print(dataframe.shape)

    print("\nLabel distribution:")
    print(dataframe["label"].value_counts())

    print("\nDataset preview:")
    print(dataframe.head())


if __name__ == "__main__":
    create_dataset()