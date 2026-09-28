import os
import pandas as pd

from audio_detector import extract_audio_features


# ============================================================
# Synthetic Content Detection Using Machine Learning
# Audio Feature Dataset Creation
# ============================================================

REAL_DIR = "audio/dataset/real"
FAKE_DIR = "audio/dataset/fake"

OUTPUT_FILE = "audio/dataset/features.csv"


# ============================================================
# Project Header
# ============================================================

print("=" * 60)
print("SYNTHETIC CONTENT DETECTION USING MACHINE LEARNING")
print("AUDIO FEATURE DATASET CREATION")
print("=" * 60)


# ============================================================
# Check Directories
# ============================================================

if not os.path.exists(REAL_DIR):
    raise FileNotFoundError(
        f"REAL audio directory not found: {REAL_DIR}"
    )

if not os.path.exists(FAKE_DIR):
    raise FileNotFoundError(
        f"FAKE audio directory not found: {FAKE_DIR}"
    )


# ============================================================
# Feature Names
# ============================================================

feature_names = [
    "mean_amplitude",
    "amplitude_std",
    "rms_mean",
    "rms_std",
    "zcr_mean",
    "zcr_std",
    "centroid_mean",
    "centroid_std",
    "bandwidth_mean",
    "bandwidth_std",
    "rolloff_mean",
    "rolloff_std"
]


# ============================================================
# Add 13 MFCC Features
# ============================================================

for i in range(1, 14):
    feature_names.append(
        f"mfcc_{i}"
    )


print("\nTotal features:", len(feature_names))


# ============================================================
# Dataset Storage
# ============================================================

dataset_rows = []


# ============================================================
# Process Audio Directory
# ============================================================

def process_audio_directory(directory, label):

    files = [
        file
        for file in os.listdir(directory)
        if file.lower().endswith(".wav")
    ]

    files.sort()

    print("\nProcessing:", directory)
    print("Audio files:", len(files))

    processed = 0
    failed = 0

    for filename in files:

        filepath = os.path.join(
            directory,
            filename
        )

        try:

            # ----------------------------------------------
            # Extract 25 audio features
            # ----------------------------------------------

            features = extract_audio_features(
                filepath
            )

            features = features.tolist()


            # ----------------------------------------------
            # Add label
            # ----------------------------------------------

            features.append(label)


            # ----------------------------------------------
            # Add filename
            # ----------------------------------------------

            features.append(filename)


            # ----------------------------------------------
            # Store row
            # ----------------------------------------------

            dataset_rows.append(
                features
            )

            processed += 1

            print(
                f"Processed: {processed}/{len(files)}",
                end="\r"
            )


        except Exception as error:

            failed += 1

            print(
                f"\nError processing {filename}:"
            )

            print(
                "Reason:",
                error
            )


    print("\nCompleted:", directory)
    print("Processed:", processed)
    print("Failed:", failed)


# ============================================================
# Process REAL Audio
# Label:
# 0 = REAL
# ============================================================

process_audio_directory(
    REAL_DIR,
    0
)


# ============================================================
# Process FAKE Audio
# Label:
# 1 = FAKE
# ============================================================

process_audio_directory(
    FAKE_DIR,
    1
)


# ============================================================
# Create DataFrame
# ============================================================

columns = feature_names + [
    "label",
    "filename"
]


data = pd.DataFrame(
    dataset_rows,
    columns=columns
)


# ============================================================
# Shuffle Dataset
# ============================================================

data = data.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)


# ============================================================
# Save Dataset
# ============================================================

data.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Display Results
# ============================================================

print("\n")
print("=" * 60)
print("AUDIO FEATURE DATASET CREATED")
print("=" * 60)

print(
    "Dataset shape:",
    data.shape
)

print("\nLabel distribution:")

print(
    data["label"]
    .value_counts()
    .sort_index()
)

print(
    "\nFeature count:",
    len(feature_names)
)

print("\nOutput file:")
print(OUTPUT_FILE)

print(
    "\nDataset creation completed successfully."
)