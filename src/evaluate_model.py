import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib


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


FEATURES_FILE = "dataset/features.csv"
METADATA_FILE = "dataset_source/metadata.csv"


print("Loading feature dataset...")

features_df = pd.read_csv(FEATURES_FILE)
metadata_df = pd.read_csv(METADATA_FILE)

print("Feature rows:", len(features_df))
print("Metadata rows:", len(metadata_df))


# --------------------------------------------------
# Create groups
# --------------------------------------------------

# The dataset contains paired REAL and FAKE images.
# We use the pair ID so that related images stay
# together in either training or testing.
#
# Our prepared dataset contains files:
# real_00000.jpg
# fake_00000.jpg
#
# Therefore we extract the numeric ID from filenames.

features_df["pair_id"] = (
    features_df.index
)

# Because create_dataset.py processed all REAL images
# first and then all FAKE images, calculate pair IDs
# from the row position.

real_count = len(
    features_df[features_df["label"] == "REAL"]
)

features_df.loc[
    features_df["label"] == "REAL",
    "pair_id"
] = features_df[
    features_df["label"] == "REAL"
].index

features_df.loc[
    features_df["label"] == "FAKE",
    "pair_id"
] = features_df[
    features_df["label"] == "FAKE"
].index - real_count


features_df["pair_id"] = (
    features_df["pair_id"].astype(int)
)


# --------------------------------------------------
# Features and labels
# --------------------------------------------------

X = features_df[FEATURE_COLUMNS]

y = features_df["label"]

groups = features_df["pair_id"]


# --------------------------------------------------
# Group-aware train/test split
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)


X_train = X.iloc[train_indices]

X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]

y_test = y.iloc[test_indices]


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print(
    "Training groups:",
    len(set(groups.iloc[train_indices]))
)

print(
    "Testing groups:",
    len(set(groups.iloc[test_indices]))
)


# --------------------------------------------------
# Train model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==============================")
print("GROUP-AWARE EVALUATION")
print("==============================")

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)

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


# --------------------------------------------------
# Save model
# --------------------------------------------------

joblib.dump(
    model,
    "models/deepfake_model_group_split.pkl"
)

print(
    "\nModel saved:"
    " models/deepfake_model_group_split.pkl"
)