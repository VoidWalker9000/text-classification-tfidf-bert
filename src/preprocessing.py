"""
preprocessing.py
----------------
Cleans raw text and builds two feature representations:
  1. Unigram Bag-of-Words (BoW)
  2. Bigram TF-IDF
Saves the resulting feature matrices and labels to disk.
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import nltk
import scipy.sparse as sp
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

# ── Download NLTK data (only runs once) ──────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
TEST_SIZE     = 0.2        # 20% of data for testing, 80% for training
MAX_FEATURES  = 20000      # Vocabulary pruning: keep top 20000 words
MIN_DF        = 2          # Ignore words appearing in fewer than 2 documents
MAX_DF        = 0.95       # Ignore words appearing in more than 95% of documents
DATA_PATH     = "data/yahoo_answers.csv"
SAVE_DIR      = "data"

# ── Stop Words ────────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))


# ── Text Cleaning ─────────────────────────────────────────────────────────────
def clean_text(text):
    """
    Cleans a single text string:
      1. Lowercase
      2. Remove URLs
      3. Remove non-alphabetic characters
      4. Tokenize
      5. Remove stop words and very short words
    Returns a single cleaned string.
    """
    # Convert to string (safety net for any NaN that slipped through)
    text = str(text).lower()

    # Remove URLs (anything starting with http or https)
    text = re.sub(r"http\S+", "", text)

    # Remove everything that is NOT a letter or whitespace
    # ^ inside [] means "not", \s means whitespace, a-z means letters
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenize — split into individual words intelligently
    tokens = word_tokenize(text)

    # Remove stop words AND words with 2 or fewer characters
    # Short words like "ok", "hi", "us" are usually not meaningful
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 2]

    # Join tokens back into a single string
    return " ".join(tokens)


# ── Bag of Words ──────────────────────────────────────────────────────────────
def build_bow(train_texts, test_texts):
    """
    Builds a Unigram Bag-of-Words matrix.
    - Fits vocabulary on train data only
    - Transforms both train and test using that vocabulary
    Returns: (train_matrix, test_matrix, vectorizer)
    """
    vectorizer = CountVectorizer(
        max_features = MAX_FEATURES,  # Keep only top 20000 words (vocabulary pruning)
        min_df       = MIN_DF,        # Ignore very rare words (likely typos/noise)
        max_df       = MAX_DF,        # Ignore words that appear in almost every doc
        ngram_range  = (1, 1),        # Unigrams: single words only
    )

    # Learn vocabulary from TRAINING data only
    # Rule: never let the model "see" test data during fitting
    X_train = vectorizer.fit_transform(train_texts)

    # Transform test data using the SAME vocabulary learned from train
    X_test  = vectorizer.transform(test_texts)

    print(f"BoW matrix shape  — Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    return X_train, X_test, vectorizer


# ── TF-IDF ────────────────────────────────────────────────────────────────────
def build_tfidf(train_texts, test_texts):
    """
    Builds a Bigram TF-IDF matrix.
    - TF  = how often a word appears in THIS document
    - IDF = how rare the word is across ALL documents
    - TF-IDF = TF x IDF (high for words frequent here but rare overall)
    - Bigrams: pairs of consecutive words ("machine learning", "new york")
    Returns: (train_matrix, test_matrix, vectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features = MAX_FEATURES,  # Vocabulary pruning
        min_df       = MIN_DF,        # Ignore very rare bigrams
        max_df       = MAX_DF,        # Ignore very common bigrams
        ngram_range  = (2, 2),        # Bigrams: pairs of words only
        sublinear_tf = True,          # Apply log(TF) instead of raw TF — reduces impact of very frequent words
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)

    print(f"TF-IDF matrix shape — Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    return X_train, X_test, vectorizer


# ── Master Pipeline ───────────────────────────────────────────────────────────
def preprocess_and_save():
    """
    Full preprocessing pipeline:
      1. Load CSV
      2. Train/test split
      3. Clean text
      4. Build BoW and TF-IDF
      5. Save everything to disk
    """

    # ── Load Data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # ── Train / Test Split ────────────────────────────────────────────────────
    # test_size=0.2 → 20% test (1600 rows), 80% train (6400 rows)
    # stratify=df["label"] → maintains same class proportions in both splits
    # So each split has exactly 800*0.8=640 train and 800*0.2=160 test per class
    train_df, test_df = train_test_split(
        df,
        test_size    = TEST_SIZE,
        random_state = RANDOM_SEED,
        stratify     = df["label"]
    )

    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")

    # Save train/test labels as numpy arrays
    y_train = train_df["label"].values  # .values converts pandas Series to numpy array
    y_test  = test_df["label"].values

    # ── Clean Text ────────────────────────────────────────────────────────────
    print("Cleaning text... (this may take a minute)")

    # .apply(clean_text) runs clean_text() on every single row
    train_texts = train_df["text"].apply(clean_text)
    test_texts  = test_df["text"].apply(clean_text)

    print("Text cleaning done!")

    # ── Build Features ────────────────────────────────────────────────────────
    print("\nBuilding Bag-of-Words...")
    X_train_bow, X_test_bow, bow_vectorizer = build_bow(train_texts, test_texts)

    print("\nBuilding TF-IDF...")
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = build_tfidf(train_texts, test_texts)

    # ── Save to Disk ──────────────────────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Save sparse matrices as .npz (compressed format for sparse matrices)
    sp.save_npz(f"{SAVE_DIR}/X_train_bow.npz",   X_train_bow)
    sp.save_npz(f"{SAVE_DIR}/X_test_bow.npz",    X_test_bow)
    sp.save_npz(f"{SAVE_DIR}/X_train_tfidf.npz", X_train_tfidf)
    sp.save_npz(f"{SAVE_DIR}/X_test_tfidf.npz",  X_test_tfidf)

    # Save labels as .npy (numpy binary format — fast to load)
    np.save(f"{SAVE_DIR}/y_train.npy", y_train)
    np.save(f"{SAVE_DIR}/y_test.npy",  y_test)

    # Save vectorizers as .pkl (pickle = saves Python objects to disk)
    with open(f"{SAVE_DIR}/bow_vectorizer.pkl",   "wb") as f:
        pickle.dump(bow_vectorizer, f)
    with open(f"{SAVE_DIR}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf_vectorizer, f)

    # Save cleaned texts for later use in embeddings.py
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df["clean_text"] = train_texts.values
    test_df["clean_text"]  = test_texts.values
    train_df.to_csv(f"{SAVE_DIR}/train.csv", index=False)
    test_df.to_csv(f"{SAVE_DIR}/test.csv",   index=False)

    print("\n✓ All features saved to 'data/' folder")
    print(f"  - X_train_bow.npz   : {X_train_bow.shape}")
    print(f"  - X_test_bow.npz    : {X_test_bow.shape}")
    print(f"  - X_train_tfidf.npz : {X_train_tfidf.shape}")
    print(f"  - X_test_tfidf.npz  : {X_test_tfidf.shape}")
    print(f"  - y_train.npy       : {y_train.shape}")
    print(f"  - y_test.npy        : {y_test.shape}")

    return (
        X_train_bow, X_test_bow,
        X_train_tfidf, X_test_tfidf,
        y_train, y_test
    )


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    preprocess_and_save()