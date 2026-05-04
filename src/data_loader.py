"""
data_loader.py
--------------
Loads the Yahoo Answers dataset from HuggingFace,
samples 8000 examples, and saves them locally as a CSV.
"""

import os
import pandas as pd
from datasets import load_dataset

# ── Constants ────────────────────────────────────────────────────────────────
DATASET_NAME   = "yahoo_answers_topics"  # HuggingFace dataset identifier
NUM_SAMPLES    = 8000                    # Total samples we want
RANDOM_SEED    = 42                      # For reproducibility
SAVE_PATH      = "data/yahoo_answers.csv"

# Yahoo Answers has 10 topic classes (0-9)
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

# ── Main Function ─────────────────────────────────────────────────────────────
def load_and_save_data():
    """
    Downloads Yahoo Answers from HuggingFace, samples 8000 rows,
    combines question title + content into one text column,
    and saves to a CSV file.
    """

    print("Loading dataset from HuggingFace...")

    # Load the train split of the dataset
    # HuggingFace datasets are like big tables — we load just the 'train' portion
    dataset = load_dataset(DATASET_NAME, split="train")

    # Convert to a pandas DataFrame so we can work with it easily
    # A DataFrame is like an Excel sheet in Python
    df = pd.DataFrame(dataset)

    print(f"Full dataset size: {len(df)} rows")

    # ── Sample 8000 rows evenly across all 10 classes ────────────────────────
    # We want 800 samples per class (800 × 10 = 8000)
    # This ensures no class dominates — called "stratified sampling"
    samples_per_class = NUM_SAMPLES // len(LABEL_NAMES)  # 8000 // 10 = 800

    sampled_df = (
        df.groupby("topic")                          # Group rows by their class label
          .apply(lambda x: x.sample(                  # From each group...
              n=min(samples_per_class, len(x)),       # ...sample 800 rows (or less if group is smaller)
              random_state=RANDOM_SEED                # Same seed = same sample every time
          ), include_groups=False)
          .reset_index(drop=True)                    # Flatten the grouped index back to normal
    )

    # ── Combine text columns into one ────────────────────────────────────────
    # Yahoo Answers has 3 text fields:
    #   - question_title   : short title of the question
    #   - question_content : full question body
    #   - best_answer      : the top answer (we won't use this — it would leak info)
    # We combine title + content into a single "text" column for our models

    sampled_df["text"] = (
        sampled_df["question_title"].fillna("") +    # Take the title (replace NaN with empty string)
        " " +                                         # Add a space between them
        sampled_df["question_content"].fillna("")    # Add the question body
    )

    # ── Add human-readable label names ───────────────────────────────────────
    # The dataset stores labels as numbers (0-9)
    # We map them to actual category names for readability
    sampled_df["label"]      = sampled_df["topic"]                        # Keep numeric label
    sampled_df["label_name"] = sampled_df["topic"].map(                   # Add text label
        lambda x: LABEL_NAMES[x]
    )

    # ── Keep only columns we need ─────────────────────────────────────────────
    final_df = sampled_df[["text", "label", "label_name"]]

    # ── Save to CSV ───────────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)   # Create 'data/' folder if it doesn't exist
    final_df.to_csv(SAVE_PATH, index=False)

    print(f"Saved {len(final_df)} samples to '{SAVE_PATH}'")
    print(f"Samples per class:\n{final_df['label_name'].value_counts()}")

    return final_df


# ── Helper Function ───────────────────────────────────────────────────────────
def load_from_csv():
    """
    Loads the already-saved CSV instead of re-downloading from HuggingFace.
    Use this after the first run to save time.
    """
    if not os.path.exists(SAVE_PATH):
        raise FileNotFoundError(
            f"'{SAVE_PATH}' not found. Run load_and_save_data() first."
        )
    df = pd.read_csv(SAVE_PATH)
    print(f"Loaded {len(df)} samples from '{SAVE_PATH}'")
    return df


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_and_save_data()
    print("\nFirst 3 rows:")
    print(df.head(3))