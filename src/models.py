# src/models.py

import numpy as np
import time
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ─────────────────────────────────────────────
# WHY THESE TWO MODELS?
# LogisticRegression: probabilistic, great baseline, works well on dense features (BERT)
# LinearSVC: margin-based, typically stronger on sparse/high-dim features (BOW, TF-IDF)
# Together they let us compare how model choice interacts with feature type
# ─────────────────────────────────────────────


def load_features(base_path=".."):
    """
    Loads all 4 feature sets (train + test) from embeddings/ directory.
    Returns a dict where each key is a feature set name.
    
    base_path: root of the project (one level up from src/)
    """
    emb = os.path.join(base_path, "embeddings")

    # np.load() reads .npy files back into numpy arrays
    # These are the dimensionality-reduced versions of our features
    features = {
        "BOW_SVD": {
            "X_train": np.load(os.path.join(emb, "X_train_bow_svd.npy")),
            "X_test":  np.load(os.path.join(emb, "X_test_bow_svd.npy")),
        },
        "TFIDF_SVD": {
            "X_train": np.load(os.path.join(emb, "X_train_tfidf_svd.npy")),
            "X_test":  np.load(os.path.join(emb, "X_test_tfidf_svd.npy")),
        },
        "BERT_CLS_PCA": {
            "X_train": np.load(os.path.join(emb, "X_train_cls_pca.npy")),
            "X_test":  np.load(os.path.join(emb, "X_test_cls_pca.npy")),
        },
        "BERT_MEAN_PCA": {
            "X_train": np.load(os.path.join(emb, "X_train_mean_pca.npy")),
            "X_test":  np.load(os.path.join(emb, "X_test_mean_pca.npy")),
        },
    }
    return features


def load_labels(base_path=".."):
    """
    Loads y_train and y_test from data/ directory.
    These are integer class labels (0–9) for the 10 Yahoo Answers topics.
    """
    data = os.path.join(base_path, "data")
    y_train = np.load(os.path.join(data, "y_train.npy"))
    y_test  = np.load(os.path.join(data, "y_test.npy"))
    return y_train, y_test


def get_models():
    """
    Returns a dict of model name → model instance.
    
    LogisticRegression:
      - max_iter=1000: default 100 often doesn't converge on text data, so we increase it
      - random_state=42: for reproducibility
    
    LinearSVC:
      - max_iter=2000: SVM also needs more iterations on text
      - random_state=42: reproducibility
    """
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "LinearSVC":          LinearSVC(max_iter=2000, random_state=42),
    }


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """
    Trains a model and returns a dict of evaluation metrics.
    
    time.time(): we record wall-clock time before and after fit()
    to measure how long training took in seconds.
    
    accuracy_score: fraction of correct predictions
    precision_score: of all predicted class X, how many were actually X
    recall_score: of all actual class X, how many did we correctly predict
    f1_score: harmonic mean of precision and recall
    average='weighted': accounts for class imbalance by weighting by support
    
    confusion_matrix: NxN matrix where entry [i,j] = 
      number of samples of true class i predicted as class j
    """
    # ── Training ──
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start  # seconds

    # ── Prediction ──
    y_pred = model.predict(X_test)

    # ── Metrics ──
    results = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "train_time": round(train_time, 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        # .tolist() converts numpy array → Python list so it's JSON serializable
        "classification_report": classification_report(y_test, y_pred, zero_division=0)
    }
    return results


def run_all_experiments(base_path=".."):
    """
    Runs all 8 experiments (2 models × 4 feature sets).
    Saves results to outputs/results.json for use in visualizations.
    
    Structure of results dict:
    {
      "BOW_SVD": {
        "LogisticRegression": { accuracy, precision, recall, f1, ... },
        "LinearSVC":          { ... }
      },
      "TFIDF_SVD": { ... },
      ...
    }
    """
    print("Loading features and labels...")
    features = load_features(base_path)
    y_train, y_test = load_labels(base_path)
    models = get_models()

    all_results = {}

    for feat_name, feat_data in features.items():
        print(f"\n── Feature set: {feat_name} ──")
        all_results[feat_name] = {}

        for model_name, model in models.items():
            print(f"  Training {model_name}...", end=" ")

            results = train_and_evaluate(
                model,
                feat_data["X_train"],
                feat_data["X_test"],
                y_train,
                y_test
            )

            # Print a quick summary to console
            print(f"Accuracy: {results['accuracy']:.4f} | F1: {results['f1']:.4f} | Time: {results['train_time']}s")

            # Store (without the long classification_report string, save separately)
            all_results[feat_name][model_name] = {
                k: v for k, v in results.items() if k != "classification_report"
            }

            # Print full classification report (per-class breakdown)
            print(f"\n  Classification Report ({feat_name} + {model_name}):")
            print(results["classification_report"])

    # ── Save to JSON ──
    # This file will be read by visualizations.py to generate charts
    out_path = os.path.join(base_path, "outputs", "results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    # json.dump: serializes Python dict → JSON string and writes to file
    # indent=2: pretty-prints with 2-space indentation (readable)

    print(f"\n✓ Results saved to {out_path}")
    return all_results


if __name__ == "__main__":
    # When you run `python src/models.py` directly,
    # base_path needs to point to project root (one level up from src/)
    results = run_all_experiments(base_path="..")