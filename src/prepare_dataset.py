import os
import shutil
import pandas as pd


SOURCE_FOLDER = "dataset_source"
OUTPUT_REAL = "dataset/real"
OUTPUT_FAKE = "dataset/fake"
METADATA_FILE = "dataset_source/metadata.csv"


def prepare_dataset():

    print("Reading metadata...")

    dataframe = pd.read_csv(METADATA_FILE)

    os.makedirs(OUTPUT_REAL, exist_ok=True)
    os.makedirs(OUTPUT_FAKE, exist_ok=True)

    valid_pairs = 0

    for _, row in dataframe.iterrows():

        fake_source = os.path.join(SOURCE_FOLDER, row["fake_file"])
        real_source = os.path.join(SOURCE_FOLDER, row["real_file"])

        if not os.path.exists(fake_source):
            continue

        if not os.path.exists(real_source):
            continue

        fake_destination = os.path.join(
            OUTPUT_FAKE,
            os.path.basename(row["fake_file"])
        )

        real_destination = os.path.join(
            OUTPUT_REAL,
            os.path.basename(row["real_file"])
        )

        shutil.copy2(fake_source, fake_destination)
        shutil.copy2(real_source, real_destination)

        valid_pairs += 1

        if valid_pairs % 500 == 0:
            print(f"Processed pairs: {valid_pairs}")

    print("\nDataset preparation completed!")
    print(f"Complete pairs: {valid_pairs}")
    print(f"Real images: {valid_pairs}")
    print(f"Fake images: {valid_pairs}")


if __name__ == "__main__":
    prepare_dataset()