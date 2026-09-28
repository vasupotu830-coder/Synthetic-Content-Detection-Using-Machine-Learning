from huggingface_hub import hf_hub_download
import pandas as pd
import soundfile as sf
import io
import os


# ============================================================
# Synthetic Content Detection Using Machine Learning
# Audio Dataset Preparation
# ============================================================

DATASET_ID = "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA"
DATASET_SPLIT = "test"

OUTPUT_REAL = "audio/dataset/real"
OUTPUT_FAKE = "audio/dataset/fake"

REAL_LIMIT = 500
FAKE_LIMIT = 500

TOTAL_SHARDS = 9


# ============================================================
# Create output directories
# ============================================================

os.makedirs(OUTPUT_REAL, exist_ok=True)
os.makedirs(OUTPUT_FAKE, exist_ok=True)


print("=" * 60)
print("SYNTHETIC CONTENT DETECTION USING MACHINE LEARNING")
print("AUDIO DATASET PREPARATION")
print("=" * 60)

print("\nDataset:")
print(DATASET_ID)

print("\nTarget:")
print(f"REAL audio : {REAL_LIMIT}")
print(f"FAKE audio : {FAKE_LIMIT}")


# ============================================================
# Clear previously generated WAV files
# ============================================================

print("\nCleaning previous audio files...")

for directory in [OUTPUT_REAL, OUTPUT_FAKE]:

    for filename in os.listdir(directory):

        if filename.lower().endswith(".wav"):

            filepath = os.path.join(
                directory,
                filename
            )

            os.remove(filepath)


print("Previous WAV files removed.")


# ============================================================
# Counters
# ============================================================

real_count = 0
fake_count = 0

failed_count = 0


# ============================================================
# Process Parquet shards
# ============================================================

for shard_number in range(TOTAL_SHARDS):

    if (
        real_count >= REAL_LIMIT
        and fake_count >= FAKE_LIMIT
    ):
        break

    shard_filename = (
        f"data/test-{shard_number:05d}-of-{TOTAL_SHARDS:05d}.parquet"
    )

    print("\n" + "-" * 60)
    print("Processing shard:")
    print(shard_filename)

    try:

        parquet_file = hf_hub_download(
            repo_id=DATASET_ID,
            filename=shard_filename,
            repo_type="dataset"
        )

        dataframe = pd.read_parquet(
            parquet_file
        )

    except Exception as error:

        print("Could not load shard.")
        print("Reason:", error)

        continue


    print("Rows in shard:", len(dataframe))


    # ========================================================
    # Process rows
    # ========================================================

    for index in range(len(dataframe)):

        if (
            real_count >= REAL_LIMIT
            and fake_count >= FAKE_LIMIT
        ):
            break


        row = dataframe.iloc[index]

        label = int(row["label"])

        audio_info = row["audio"]

        try:

            # ------------------------------------------------
            # REAL / BONAFIDE
            # ------------------------------------------------

            if label == 1:

                if real_count >= REAL_LIMIT:
                    continue

                filename = (
                    f"real_{real_count:04d}.wav"
                )

                destination = os.path.join(
                    OUTPUT_REAL,
                    filename
                )


            # ------------------------------------------------
            # FAKE / SPOOF
            # ------------------------------------------------

            elif label == 0:

                if fake_count >= FAKE_LIMIT:
                    continue

                filename = (
                    f"fake_{fake_count:04d}.wav"
                )

                destination = os.path.join(
                    OUTPUT_FAKE,
                    filename
                )


            else:

                continue


            # =================================================
            # Extract embedded audio bytes
            # =================================================

            audio_bytes = audio_info["bytes"]

            audio_data, sample_rate = sf.read(
                io.BytesIO(audio_bytes)
            )


            # =================================================
            # Save as WAV
            # =================================================

            sf.write(
                destination,
                audio_data,
                sample_rate
            )


            # =================================================
            # Update counters
            # =================================================

            if label == 1:

                real_count += 1

                print(
                    f"REAL: {real_count}/{REAL_LIMIT}",
                    end="\r"
                )

            else:

                fake_count += 1

                print(
                    f"FAKE: {fake_count}/{FAKE_LIMIT}",
                    end="\r"
                )


        except Exception as error:

            failed_count += 1

            print(
                f"\nFailed to process row {index}"
            )

            print(
                "Reason:",
                error
            )


# ============================================================
# Final result
# ============================================================

print("\n")
print("=" * 60)
print("AUDIO DATASET PREPARATION COMPLETED")
print("=" * 60)

print(f"REAL audio files : {real_count}")
print(f"FAKE audio files : {fake_count}")
print(f"Failed files    : {failed_count}")

print("\nReal directory:")
print(OUTPUT_REAL)

print("\nFake directory:")
print(OUTPUT_FAKE)

print("\nDataset preparation finished.")