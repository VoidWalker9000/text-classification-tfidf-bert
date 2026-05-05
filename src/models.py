"""
models.py
---------
Trains and evaluates 4 models on 5 feature sets:
  Models: Logistic Regression, LinearSVC, KNN, KMeans
  Features: BOW+SVD, TF-IDF+SVD, BERT CLS+PCA, BERT Mean+PCA, GloVe+PCA
"""

import numpy as np
import time
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from scipy.stats import mode


def load_features(base_path=".."):
    """
    Loads all 5 feature sets (train + test) from embeddings/ directory.
    base_path: root of the project (one level up from src/)
    """
    emb = os.path.join(base_path, "embeddings")

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
        "GLOVE_PCA": {
            "X_train": np.load(os.path.join(emb, "X_train_glove_pca.npy")),
            "X_test":  np.load(os.path.join(emb, "X_test_glove_pca.npy")),
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


def get_supervised_models():
    """
    Returns supervised models dict.

    LogisticRegression:
      - max_iter=1000: default 100 often doesn't converge on text data
      - random_state=42: reproducibility

    LinearSVC:
      - max_iter=2000: SVM needs more iterations on text
      - random_state=42: reproducibility

    KNN:
      - n_neighbors=5: vote among 5 nearest neighbours
      - metric="cosine": cosine similarity works better than euclidean for text
        since it ignores magnitude and only considers direction
    """
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "LinearSVC":          LinearSVC(max_iter=2000, random_state=42),
        "KNN":                KNeighborsClassifier(n_neighbors=5, metric="cosine"),
    }


def compute_metrics(y_test, y_pred, train_time):
    """
    Computes all evaluation metrics for a given prediction.

    accuracy  : fraction of correct predictions
    precision : of all predicted class X, how many were actually X (weighted)
    recall    : of all actual class X, how many did we predict correctly (weighted)
    f1        : weighted harmonic mean of precision and recall
    f1_macro  : unweighted F1 across all classes — required by project spec
                macro treats all classes equally regardless of size
    confusion_matrix: NxN matrix where entry [i,j] =
                number of samples of true class i predicted as class j
    """
    return {
        "accuracy":         round(accuracy_score(y_test, y_pred), 4),
        "precision":        round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":           round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1":               round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1_macro":         round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "train_time":       train_time,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        # .tolist() converts numpy array → Python list so it's JSON serializable
    }


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """
    Trains a supervised model and returns evaluation metrics.
    """
    start = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - start, 4)

    y_pred = model.predict(X_test)

    results = compute_metrics(y_test, y_pred, train_time)

    # classification_report gives per-class breakdown — printed but not saved to JSON
    results["classification_report"] = classification_report(y_test, y_pred, zero_division=0)

    return results


def train_and_evaluate_kmeans(X_train, X_test, y_train, y_test):
    """
    KMeans is unsupervised — it has no concept of class labels during training.
    It just groups similar samples into n_clusters clusters.

    After clustering, we map each cluster ID to the most frequent true label
    in that cluster using majority voting. This is the standard way to evaluate
    KMeans on a classification task.

    n_init=10: runs KMeans 10 times with different seeds, picks the best result
               (lowest inertia) — makes results more stable
    """
    n_classes = len(np.unique(y_train))

    kmeans = KMeans(
        n_clusters=n_classes,
        random_state=42,
        n_init=10
    )

    start = time.time()
    kmeans.fit(X_train)
    train_time = round(time.time() - start, 4)

    # Map cluster ID → majority true label using training data
    train_clusters = kmeans.predict(X_train)
    cluster_to_label = {}
    for cluster_id in range(n_classes):
        mask = train_clusters == cluster_id
        if mask.sum() > 0:
            # mode() returns the most frequent value in y_train for this cluster
            cluster_to_label[cluster_id] = mode(y_train[mask], keepdims=True).mode[0]
        else:
            cluster_to_label[cluster_id] = 0

    # Convert cluster predictions → class label predictions
    cluster_labels = kmeans.predict(X_test)
    y_pred = np.array([cluster_to_label[c] for c in cluster_labels])

    results = compute_metrics(y_test, y_pred, train_time)
    results["classification_report"] = classification_report(y_test, y_pred, zero_division=0)

    return results


def run_all_experiments(base_path=".."):
    """
    Runs all 20 experiments (4 models × 5 feature sets).
    Saves results to outputs/results.json for use in visualizations.
    """
    print("Loading features and labels...")
    features = load_features(base_path)
    y_train, y_test = load_labels(base_path)
    supervised_models = get_supervised_models()

    all_results = {}

    # ── Supervised Models ──
    for feat_name, feat_data in features.items():
        print(f"\n── Feature set: {feat_name} ──")
        all_results[feat_name] = {}

        for model_name, model in supervised_models.items():
            print(f"  Training {model_name}...", end=" ")

            results = train_and_evaluate(
                model,
                feat_data["X_train"],
                feat_data["X_test"],
                y_train,
                y_test
            )

            print(f"Accuracy: {results['accuracy']} | F1 Macro: {results['f1_macro']} | Time: {results['train_time']}s")
            print(f"\n  Classification Report ({feat_name} + {model_name}):")
            print(results["classification_report"])

            # Save without classification_report string (too long for JSON)
            all_results[feat_name][model_name] = {
                k: v for k, v in results.items() if k != "classification_report"
            }

    # ── KMeans (Unsupervised) ──
    print("\n── KMeans (Unsupervised Clustering) ──")

    for feat_name, feat_data in features.items():
        print(f"\n  Feature set: {feat_name}...", end=" ")

        results = train_and_evaluate_kmeans(
            feat_data["X_train"],
            feat_data["X_test"],
            y_train,
            y_test
        )

        print(f"Accuracy: {results['accuracy']} | F1 Macro: {results['f1_macro']} | Time: {results['train_time']}s")
        print(results["classification_report"])

        all_results[feat_name]["KMeans"] = {
            k: v for k, v in results.items() if k != "classification_report"
        }

    # ── Save to JSON ──
    out_path = os.path.join(base_path, "outputs", "results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to {out_path}")
    return all_results


if __name__ == "__main__":
    results = run_all_experiments(base_path="..")