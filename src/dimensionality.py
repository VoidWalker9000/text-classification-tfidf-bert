"""
dimensionality.py
-----------------
Applies dimensionality reduction on all feature types:
  1. Truncated SVD on BoW (sparse)
  2. Truncated SVD on TF-IDF (sparse)
  3. PCA on CLS embeddings (dense)
  4. PCA on Mean-pool embeddings (dense)

Also analyzes explained variance to find optimal number of components.
Saves reduced features to disk.
"""

import os
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD, PCA
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend — saves plots to file instead of displaying
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────────────
N_COMPONENTS_SPARSE = 300       # Reduce BoW/TF-IDF to 300 dimensions
N_COMPONENTS_DENSE  = 100       # Reduce BERT embeddings to 100 dimensions
VARIANCE_THRESHOLD  = 0.95      # We want to explain 95% of variance
RANDOM_SEED         = 42
DATA_DIR            = "data"
EMB_DIR             = "embeddings"
SAVE_DIR            = "embeddings"
PLOT_DIR            = "outputs"


# ── Explained Variance Analysis ───────────────────────────────────────────────
def find_optimal_components(matrix, max_components, name, is_sparse=True):
    """
    Fits SVD/PCA with increasing number of components and plots
    cumulative explained variance to find the "elbow point" —
    where adding more components gives diminishing returns.

    Args:
        matrix         : feature matrix (sparse or dense)
        max_components : maximum components to try
        name           : label for the plot (e.g. "TF-IDF")
        is_sparse      : True for SVD (sparse), False for PCA (dense)

    Returns:
        n_components_95 : number of components needed for 95% variance
    """

    # We try increasing numbers of components
    # np.linspace creates evenly spaced numbers
    # e.g. np.linspace(10, 300, 20) → [10, 25, 40, ..., 300]
    component_range = np.linspace(10, max_components, 20, dtype=int)

    explained_variances = []   # Store cumulative explained variance for each n

    for n in component_range:
        if is_sparse:
            # TruncatedSVD for sparse matrices (BoW, TF-IDF)
            reducer = TruncatedSVD(n_components=n, random_state=RANDOM_SEED)
        else:
            # PCA for dense matrices (BERT embeddings)
            reducer = PCA(n_components=n, random_state=RANDOM_SEED)

        # fit() learns the components from the data
        reducer.fit(matrix)

        # explained_variance_ratio_ is an array of how much variance
        # each component explains. Summing them gives cumulative variance.
        cumulative_variance = np.sum(reducer.explained_variance_ratio_)
        explained_variances.append(cumulative_variance)

        print(f"  {name} — {n:4d} components → {cumulative_variance:.3f} variance explained")

    # ── Find components needed for 95% variance ───────────────────────────────
    # Fit one final time with max components to get full variance curve
    if is_sparse:
        reducer_full = TruncatedSVD(n_components=max_components, random_state=RANDOM_SEED)
    else:
        reducer_full = PCA(n_components=max_components, random_state=RANDOM_SEED)

    reducer_full.fit(matrix)

    # cumsum() computes cumulative sum — adds up values one by one
    # e.g. [0.1, 0.2, 0.15] → [0.1, 0.3, 0.45]
    cumulative = np.cumsum(reducer_full.explained_variance_ratio_)

    # np.searchsorted finds the first index where cumulative variance >= 95%
    n_components_95 = np.searchsorted(cumulative, VARIANCE_THRESHOLD) + 1

    print(f"\n  ✓ {name}: {n_components_95} components needed for {VARIANCE_THRESHOLD*100:.0f}% variance\n")

    # ── Plot ──────────────────────────────────────────────────────────────────
    os.makedirs(PLOT_DIR, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(component_range, explained_variances, marker="o", color="steelblue", linewidth=2)
    plt.axhline(y=VARIANCE_THRESHOLD, color="red", linestyle="--", label=f"{VARIANCE_THRESHOLD*100:.0f}% threshold")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(f"Explained Variance vs Components — {name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/variance_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()   # Close figure to free memory

    return n_components_95


# ── Apply Reduction ───────────────────────────────────────────────────────────
def apply_reduction(X_train, X_test, n_components, name, is_sparse=True):
    """
    Fits reducer on training data and transforms both train and test.

    IMPORTANT: Always fit on TRAIN only, then transform both.
    Same rule as with vectorizers — never let test data influence fitting.

    Args:
        X_train    : training feature matrix
        X_test     : test feature matrix
        n_components: number of dimensions to reduce to
        name       : label for printing
        is_sparse  : True for SVD, False for PCA

    Returns:
        X_train_reduced, X_test_reduced, fitted reducer
    """

    if is_sparse:
        reducer = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
    else:
        reducer = PCA(n_components=n_components, random_state=RANDOM_SEED)

    # Fit on training data only, then transform
    X_train_reduced = reducer.fit_transform(X_train)  # Fit + transform in one step
    X_test_reduced  = reducer.transform(X_test)        # Transform only (no fitting)

    print(f"  {name}: {X_train.shape} → {X_train_reduced.shape}")

    return X_train_reduced, X_test_reduced, reducer


# ── Master Pipeline ───────────────────────────────────────────────────────────
def reduce_and_save():
    """
    Full dimensionality reduction pipeline:
      1. Load all feature matrices
      2. Analyze explained variance
      3. Apply reduction
      4. Save reduced matrices
    """

    # ── Load Sparse Features (BoW, TF-IDF) ───────────────────────────────────
    print("Loading sparse features...")
    X_train_bow   = sp.load_npz(f"{DATA_DIR}/X_train_bow.npz")
    X_test_bow    = sp.load_npz(f"{DATA_DIR}/X_test_bow.npz")
    X_train_tfidf = sp.load_npz(f"{DATA_DIR}/X_train_tfidf.npz")
    X_test_tfidf  = sp.load_npz(f"{DATA_DIR}/X_test_tfidf.npz")

    # ── Load Dense Features (BERT embeddings) ────────────────────────────────
    print("Loading BERT embeddings...")
    train_cls  = np.load(f"{EMB_DIR}/train_cls.npy")
    test_cls   = np.load(f"{EMB_DIR}/test_cls.npy")
    train_mean = np.load(f"{EMB_DIR}/train_mean.npy")
    test_mean  = np.load(f"{EMB_DIR}/test_mean.npy")
    
    # ── Load GloVe Features ───────────────────────────────────────────────────
    print("Loading GloVe embeddings...")
    train_glove = np.load(f"{EMB_DIR}/train_glove.npy")
    test_glove  = np.load(f"{EMB_DIR}/test_glove.npy")

    # ── Explained Variance Analysis ───────────────────────────────────────────
    print("\n── Explained Variance Analysis ──────────────────────────────────")

    print("\nBoW (Truncated SVD):")
    n_bow = find_optimal_components(X_train_bow,   N_COMPONENTS_SPARSE, "BoW",       is_sparse=True)

    print("TF-IDF (Truncated SVD):")
    n_tfidf = find_optimal_components(X_train_tfidf, N_COMPONENTS_SPARSE, "TF-IDF",  is_sparse=True)

    print("CLS Embeddings (PCA):")
    n_cls = find_optimal_components(train_cls,     N_COMPONENTS_DENSE,  "CLS",       is_sparse=False)

    print("Mean Pool Embeddings (PCA):")
    n_mean = find_optimal_components(train_mean,   N_COMPONENTS_DENSE,  "Mean Pool", is_sparse=False)
    
    print("GloVe Embeddings (PCA):")
    n_glove = find_optimal_components(train_glove, N_COMPONENTS_DENSE, "GloVe", is_sparse=False)

    # ── Apply Reduction ───────────────────────────────────────────────────────
    print("\n── Applying Dimensionality Reduction ────────────────────────────")

    # Use the 95% variance components OR our preset max — whichever is smaller
    n_bow_final   = min(n_bow,   N_COMPONENTS_SPARSE)
    n_tfidf_final = min(n_tfidf, N_COMPONENTS_SPARSE)
    n_cls_final   = min(n_cls,   N_COMPONENTS_DENSE)
    n_mean_final  = min(n_mean,  N_COMPONENTS_DENSE)
    n_glove_final = min(n_glove, N_COMPONENTS_DENSE)

    X_train_bow_svd,   X_test_bow_svd,   _ = apply_reduction(
        X_train_bow,   X_test_bow,   n_bow_final,   "BoW SVD",       is_sparse=True)

    X_train_tfidf_svd, X_test_tfidf_svd, _ = apply_reduction(
        X_train_tfidf, X_test_tfidf, n_tfidf_final, "TF-IDF SVD",    is_sparse=True)

    X_train_cls_pca,   X_test_cls_pca,   _ = apply_reduction(
        train_cls,     test_cls,     n_cls_final,   "CLS PCA",       is_sparse=False)

    X_train_mean_pca,  X_test_mean_pca,  _ = apply_reduction(
        train_mean,    test_mean,    n_mean_final,  "Mean Pool PCA", is_sparse=False)
    
    X_train_glove_pca, X_test_glove_pca, _ = apply_reduction(
        train_glove, test_glove, n_glove_final, "GloVe PCA", is_sparse=False)

    # ── Save Reduced Features ─────────────────────────────────────────────────
    # Reduced features are now dense (numpy arrays) — save as .npy
    np.save(f"{SAVE_DIR}/X_train_bow_svd.npy",    X_train_bow_svd)
    np.save(f"{SAVE_DIR}/X_test_bow_svd.npy",     X_test_bow_svd)
    np.save(f"{SAVE_DIR}/X_train_tfidf_svd.npy",  X_train_tfidf_svd)
    np.save(f"{SAVE_DIR}/X_test_tfidf_svd.npy",   X_test_tfidf_svd)
    np.save(f"{SAVE_DIR}/X_train_cls_pca.npy",    X_train_cls_pca)
    np.save(f"{SAVE_DIR}/X_test_cls_pca.npy",     X_test_cls_pca)
    np.save(f"{SAVE_DIR}/X_train_mean_pca.npy",   X_train_mean_pca)
    np.save(f"{SAVE_DIR}/X_test_mean_pca.npy",    X_test_mean_pca)
    np.save(f"{SAVE_DIR}/X_train_glove_pca.npy", X_train_glove_pca)
    np.save(f"{SAVE_DIR}/X_test_glove_pca.npy",  X_test_glove_pca)

    print("\n✓ All reduced features saved!")
    print(f"  Variance plots saved to '{PLOT_DIR}/'")

    return {
        "bow_svd"   : (X_train_bow_svd,   X_test_bow_svd),
        "tfidf_svd" : (X_train_tfidf_svd, X_test_tfidf_svd),
        "cls_pca"   : (X_train_cls_pca,   X_test_cls_pca),
        "mean_pca"  : (X_train_mean_pca,  X_test_mean_pca),
        "glove_pca" : (X_train_glove_pca, X_test_glove_pca),
    }


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    reduce_and_save()