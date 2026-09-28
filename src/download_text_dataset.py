import os
import json
import pandas as pd
from huggingface_hub import hf_hub_download


# ============================================================
# CONFIGURATION
# ============================================================

REPO_ID = "Hello-SimpleAI/HC3"
DATASET_FILE = "all.jsonl"

OUTPUT_FILE = os.path.join(
    "text",
    "dataset",
    "text_dataset.csv"
)

MAX_REAL = 1000
MAX_FAKE = 1000


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("TEXT DATASET DOWNLOAD")
print("=" * 60)

print("\nDataset: Hello-SimpleAI/HC3")
print("Source file: all.jsonl")
print("Purpose: Human vs AI-generated text detection")

print("\nDownloading HC3 dataset...")


# ============================================================
# DOWNLOAD DATASET
# ============================================================

jsonl_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=DATASET_FILE,
    repo_type="dataset"
)

print("\nDataset downloaded successfully.")
print("File:")
print(jsonl_path)


# ============================================================
# READ JSONL
# ============================================================

print("\nReading dataset...")

real_texts = []
fake_texts = []

with open(
    jsonl_path,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        if not line.strip():
            continue

        row = json.loads(line)

        # ----------------------------------------------------
        # Human-written answers
        # ----------------------------------------------------

        human_answers = row.get(
            "human_answers",
            []
        )

        if isinstance(
            human_answers,
            str
        ):
            human_answers = [
                human_answers
            ]

        if isinstance(
            human_answers,
            list
        ):

            for text in human_answers:

                if (
                    isinstance(text, str)
                    and len(text.strip()) >= 30
                    and len(real_texts) < MAX_REAL
                ):

                    real_texts.append(
                        text.strip()
                    )

        # ----------------------------------------------------
        # ChatGPT-generated answers
        # ----------------------------------------------------

        chatgpt_answers = row.get(
            "chatgpt_answers",
            []
        )

        if isinstance(
            chatgpt_answers,
            str
        ):
            chatgpt_answers = [
                chatgpt_answers
            ]

        if isinstance(
            chatgpt_answers,
            list
        ):

            for text in chatgpt_answers:

                if (
                    isinstance(text, str)
                    and len(text.strip()) >= 30
                    and len(fake_texts) < MAX_FAKE
                ):

                    fake_texts.append(
                        text.strip()
                    )

        # ----------------------------------------------------
        # Stop once we have enough samples
        # ----------------------------------------------------

        if (
            len(real_texts) >= MAX_REAL
            and
            len(fake_texts) >= MAX_FAKE
        ):
            break


# ============================================================
# CREATE DATAFRAME
# ============================================================

rows = []

# REAL = 0
for text in real_texts:

    rows.append(
        {
            "text": text,
            "label": 0
        }
    )


# FAKE = 1
for text in fake_texts:

    rows.append(
        {
            "text": text,
            "label": 1
        }
    )


text_dataset = pd.DataFrame(
    rows,
    columns=[
        "text",
        "label"
    ]
)


# ============================================================
# SAVE DATASET
# ============================================================

text_dataset.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TEXT DATASET CREATED")
print("=" * 60)

print(
    "REAL / Human-written:",
    len(real_texts)
)

print(
    "FAKE / AI-generated:",
    len(fake_texts)
)

print(
    "Total texts:",
    len(text_dataset)
)

print("\nLabel mapping:")
print("0 = REAL / Human-written")
print("1 = FAKE / AI-generated")

print("\nDataset shape:")
print(text_dataset.shape)

print("\nSaved to:")
print(OUTPUT_FILE)

print("=" * 60)