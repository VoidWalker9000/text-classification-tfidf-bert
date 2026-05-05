"""
embeddings.py
-------------
Extracts BERT/DistilBERT embeddings from text data.
Uses pretrained encoder as a FEATURE EXTRACTOR only — no fine-tuning.

Two embedding types:
  1. CLS token embedding     — first token, represents whole sentence
  2. Mean pooled embedding   — average of all token embeddings

NOTE: Run this on Kaggle/Colab with GPU for speed.
      Embeddings are computed ONCE and cached to disk.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from collections import defaultdict

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"   # DistilBERT: lighter and faster than BERT
                                            # "uncased" = treats "Hello" same as "hello"
BATCH_SIZE   = 64                           # How many sentences to process at once
MAX_LENGTH   = 128                          # Max tokens per sentence (longer gets truncated)
SAVE_DIR     = "embeddings"                 # Where to save the .npy files
DATA_DIR     = "data"                       # Where train.csv and test.csv are

# ── Device Setup ──────────────────────────────────────────────────────────────
# Automatically use GPU if available (Kaggle), otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── Custom Dataset Class ──────────────────────────────────────────────────────
class TextDataset(Dataset):
    """
    A PyTorch Dataset that holds our text data.
    PyTorch needs data in this format to use DataLoader (batching).
    """

    def __init__(self, texts, tokenizer, max_length):
        self.texts      = texts           # List of text strings
        self.tokenizer  = tokenizer       # DistilBERT tokenizer
        self.max_length = max_length      # Max token length

    def __len__(self):
        # Returns total number of samples
        # PyTorch calls this to know the size of the dataset
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Returns ONE tokenized sample at index idx.
        PyTorch DataLoader calls this repeatedly to build batches.
        """
        text = str(self.texts[idx])

        # Tokenizer converts raw text into numbers BERT understands
        # Example: "I love Python" → [101, 1045, 2293, 18750, 102]
        #   101  = [CLS] token
        #   102  = [SEP] token (end of sentence)
        #   rest = actual word tokens
        encoding = self.tokenizer(
            text,
            max_length      = self.max_length,
            padding         = "max_length",  # Pad shorter sentences with zeros to reach max_length
            truncation      = True,          # Cut sentences longer than max_length
            return_tensors  = "pt"           # Return PyTorch tensors (not lists)
        )

        return {
            # squeeze(0) removes the extra batch dimension the tokenizer adds
            "input_ids"      : encoding["input_ids"].squeeze(0),       # Token IDs
            "attention_mask" : encoding["attention_mask"].squeeze(0),  # 1 for real tokens, 0 for padding
        }


# ── Embedding Extraction ──────────────────────────────────────────────────────
def extract_embeddings(texts, tokenizer, model):
    """
    Extracts CLS and mean-pooled embeddings for a list of texts.

    Args:
        texts     : list of cleaned text strings
        tokenizer : DistilBERT tokenizer
        model     : DistilBERT model (frozen, no gradients)

    Returns:
        cls_embeddings  : numpy array of shape (n_samples, 768)
        mean_embeddings : numpy array of shape (n_samples, 768)
    """

    # Create dataset and dataloader
    dataset    = TextDataset(texts, tokenizer, MAX_LENGTH)

    # DataLoader batches the data — feeds BATCH_SIZE samples at a time to the model
    # num_workers=0 is safest on Windows (avoids multiprocessing issues)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    cls_embeddings  = []   # Will store CLS vectors
    mean_embeddings = []   # Will store mean-pooled vectors

    # torch.no_grad() tells PyTorch: don't compute gradients
    # We're not training — just extracting features — so no gradients needed
    # This saves memory and speeds things up significantly
    with torch.no_grad():

        # tqdm wraps the dataloader to show a progress bar
        for batch in tqdm(dataloader, desc="Extracting embeddings"):

            # Move batch tensors to GPU (or CPU if no GPU)
            input_ids      = batch["input_ids"].to(DEVICE)       # Shape: (batch_size, 128)
            attention_mask = batch["attention_mask"].to(DEVICE)  # Shape: (batch_size, 128)

            # Pass through DistilBERT
            # last_hidden_state shape: (batch_size, 128, 768)
            #   batch_size = number of sentences in this batch
            #   128        = number of tokens per sentence
            #   768        = embedding size for each token
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state

            # ── CLS Embedding ─────────────────────────────────────────────────
            # The CLS token is always at position 0
            # Shape: (batch_size, 768)
            cls_vec = last_hidden_state[:, 0, :]

            # ── Mean Pool Embedding ───────────────────────────────────────────
            # We average only the REAL tokens (not padding)
            # attention_mask = 1 for real tokens, 0 for padding
            # We use the mask to ignore padding positions in the average

            # unsqueeze(-1) expands mask from (batch_size, 128) to (batch_size, 128, 1)
            # so it can be multiplied with last_hidden_state (batch_size, 128, 768)
            mask_expanded = attention_mask.unsqueeze(-1).float()

            # Zero out padding positions by multiplying with mask
            sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)  # Sum over tokens

            # Count real tokens per sentence (sum of attention mask)
            # clamp(min=1e-9) prevents division by zero
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)

            # Divide sum by count to get mean
            mean_vec = sum_embeddings / sum_mask   # Shape: (batch_size, 768)

            # Move back to CPU and convert to numpy, then store
            cls_embeddings.append(cls_vec.cpu().numpy())
            mean_embeddings.append(mean_vec.cpu().numpy())

    # Concatenate all batches into one big array
    # np.vstack stacks arrays vertically (row by row)
    cls_embeddings  = np.vstack(cls_embeddings)   # Shape: (n_samples, 768)
    mean_embeddings = np.vstack(mean_embeddings)  # Shape: (n_samples, 768)

    return cls_embeddings, mean_embeddings


# ── Master Pipeline ───────────────────────────────────────────────────────────
def generate_and_save_embeddings():
    """
    Full pipeline:
      1. Load train/test CSVs
      2. Load DistilBERT tokenizer and model
      3. Extract CLS and mean-pool embeddings
      4. Save all 4 embedding files to disk
    """

    # ── Load Data ─────────────────────────────────────────────────────────────
    print("Loading train/test data...")
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_df  = pd.read_csv(f"{DATA_DIR}/test.csv")

    # Use clean_text column (already preprocessed by preprocessing.py)
    # Fill any NaN with empty string just in case
    train_texts = train_df["clean_text"].fillna("").tolist()
    test_texts  = test_df["clean_text"].fillna("").tolist()

    print(f"Train texts: {len(train_texts)} | Test texts: {len(test_texts)}")

    # ── Load DistilBERT ───────────────────────────────────────────────────────
    print(f"\nLoading {MODEL_NAME} tokenizer and model...")

    # AutoTokenizer automatically loads the right tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # AutoModel loads the base model WITHOUT a classification head
    # We want raw embeddings, not predictions
    model = AutoModel.from_pretrained(MODEL_NAME)

    # Move model to GPU if available
    model = model.to(DEVICE)

    # Set model to evaluation mode — disables dropout layers
    # Dropout is used during training to prevent overfitting
    # During inference (feature extraction) we want deterministic outputs
    model.eval()

    print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Extract Embeddings ────────────────────────────────────────────────────
    print("\nExtracting TRAIN embeddings...")
    train_cls, train_mean = extract_embeddings(train_texts, tokenizer, model)

    print("\nExtracting TEST embeddings...")
    test_cls, test_mean = extract_embeddings(test_texts, tokenizer, model)

    # ── Save to Disk ──────────────────────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)

    # np.save saves numpy arrays as .npy binary files (fast to load later)
    np.save(f"{SAVE_DIR}/train_cls.npy",  train_cls)
    np.save(f"{SAVE_DIR}/train_mean.npy", train_mean)
    np.save(f"{SAVE_DIR}/test_cls.npy",   test_cls)
    np.save(f"{SAVE_DIR}/test_mean.npy",  test_mean)

    print("\n✓ All embeddings saved!")
    print(f"  - train_cls.npy  : {train_cls.shape}")
    print(f"  - train_mean.npy : {train_mean.shape}")
    print(f"  - test_cls.npy   : {test_cls.shape}")
    print(f"  - test_mean.npy  : {test_mean.shape}")
    
    # ── GloVe Embeddings ──────────────────────────────────────────────────────
    print("\nGenerating GloVe embeddings...")
    
    # Download GloVe from: https://nlp.stanford.edu/projects/glove/
    # Download glove.6B.300d.txt from kaggle -> Glove6b dataset
    # Place glove.6B.300d.txt in data/glove/
    
    glove_path = "data/glove/glove.6B.300d.txt"  # local path
    generate_glove_embeddings(train_texts, test_texts, glove_path)

    return train_cls, train_mean, test_cls, test_mean

# ── GloVe Embeddings ──────────────────────────────────────────────────────────
def load_glove(glove_path):
    """
    Loads GloVe vectors from a .txt file into a dictionary.
    Format of each line: "word 0.123 0.456 ... 0.789"
    
    glove_path: path to glove.6B.300d.txt
    Returns: dict of {word: numpy array of shape (300,)}
    """
    print("Loading GloVe vectors...")
    glove = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()        # Split line into word + numbers
            word   = values[0]           # First element is the word
            vector = np.array(values[1:], dtype=np.float32)  # Rest are floats
            glove[word] = vector
    print(f"✓ Loaded {len(glove):,} GloVe vectors")
    return glove


def text_to_glove(texts, glove, dim=300):
    """
    Converts a list of texts to GloVe embeddings.
    Each text becomes the MEAN of its word vectors.
    Words not in GloVe vocabulary are skipped.
    
    texts : list of cleaned text strings
    glove : dict from load_glove()
    dim   : GloVe dimension (300 for glove.6B.300d)
    
    Returns: numpy array of shape (n_samples, 300)
    """
    embeddings = []

    for text in tqdm(texts, desc="Building GloVe embeddings"):
        words   = text.split()                    # Simple split (already cleaned)
        vectors = [glove[w] for w in words if w in glove]  # Only known words

        if vectors:
            # Mean of all word vectors in the sentence
            embeddings.append(np.mean(vectors, axis=0))
        else:
            # If no words found in GloVe (rare), use zero vector
            embeddings.append(np.zeros(dim))

    return np.array(embeddings)   # Shape: (n_samples, 300)


def generate_glove_embeddings(train_texts, test_texts, glove_path):
    """
    Full GloVe pipeline: load vectors → convert texts → save to disk
    """
    glove      = load_glove(glove_path)
    train_glove = text_to_glove(train_texts, glove)
    test_glove  = text_to_glove(test_texts, glove)

    np.save(f"{SAVE_DIR}/train_glove.npy", train_glove)
    np.save(f"{SAVE_DIR}/test_glove.npy",  test_glove)

    print(f"\n✓ GloVe embeddings saved!")
    print(f"  - train_glove.npy : {train_glove.shape}")
    print(f"  - test_glove.npy  : {test_glove.shape}")

    return train_glove, test_glove

# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_and_save_embeddings()