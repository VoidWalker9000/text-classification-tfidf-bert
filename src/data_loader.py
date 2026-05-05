"""
data_loader.py
--------------
Loads Yahoo Answers dataset, performs stratified sampling (800 per class),
and saves to CSV.
"""

import os
import pandas as pd
from datasets import load_dataset

# ── Constants ────────────────────────────────────────────────────────────────
DATASET_NAME = "yahoo_answers_topics"
NUM_SAMPLES = 8000
RANDOM_SEED = 42
SAVE_PATH = "data/yahoo_answers.csv"

LABEL_NAMES = [
    "Society & Culture",
    "Science & Mathematics",
    "Health",
    "Education & Reference",
    "Computers & Internet",
    "Sports",
    "Business & Finance",
    "Entertainment & Music",
    "Family & Relationships",
    "Politics & Government"
]


def load_and_save_data():
    print("Loading dataset from HuggingFace...")

    dataset = load_dataset(DATASET_NAME, split="train")
    df = pd.DataFrame(dataset)

    print(f"Full dataset size: {len(df)} rows")

    # ── Safety Checks ─────────────────────────────────────────────────────────
    if "topic" not in df.columns:
        raise KeyError("'topic' column not found")

    if df["topic"].nunique() != 10:
        raise ValueError("Dataset does not contain 10 classes")

    # ── Stratified Sampling ───────────────────────────────────────────────────
    samples_per_class = NUM_SAMPLES // len(LABEL_NAMES)

    sampled_parts = []

    for topic in range(len(LABEL_NAMES)):
        class_df = df[df["topic"] == topic]

        if len(class_df) == 0:
            raise ValueError(f"No data for topic {topic}")

        sampled = class_df.sample(
            n=min(samples_per_class, len(class_df)),
            random_state=RANDOM_SEED
        )

        sampled_parts.append(sampled)

    sampled_df = pd.concat(sampled_parts).reset_index(drop=True)

    # ── Combine Text ──────────────────────────────────────────────────────────
    sampled_df["text"] = (
        sampled_df["question_title"].fillna("") + " " +
        sampled_df["question_content"].fillna("")
    )

    # ── Labels ────────────────────────────────────────────────────────────────
    sampled_df["label"] = sampled_df["topic"]
    label_map = dict(enumerate(LABEL_NAMES))
    sampled_df["label_name"] = sampled_df["topic"].map(label_map)

    # ── Final Output ──────────────────────────────────────────────────────────
    final_df = sampled_df[["text", "label", "label_name"]]

    os.makedirs("data", exist_ok=True)
    final_df.to_csv(SAVE_PATH, index=False)

    print(f"Saved {len(final_df)} samples to {SAVE_PATH}")
    print(final_df["label_name"].value_counts())

    return final_df


def load_from_csv():
    if not os.path.exists(SAVE_PATH):
        raise FileNotFoundError(f"{SAVE_PATH} not found")

    df = pd.read_csv(SAVE_PATH)
    print(f"Loaded {len(df)} samples")
    return df


if __name__ == "__main__":
    df = load_and_save_data()
    print(df.head(3))