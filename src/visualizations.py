"""
visualizations.py
-----------------
Generates all plots and visualizations for the project:
  1. Accuracy & Macro F1 comparison bar charts
  2. Confusion matrix heatmaps (all 20 experiments)
  3. Training time comparison
  4. t-SNE feature space visualization
  5. EDA plots (class distribution, text length, word cloud)

All plots saved to outputs/ directory.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.manifold import TSNE
from wordcloud import WordCloud

# ── Constants ─────────────────────────────────────────────────────────────────
RESULTS_PATH = "outputs/results.json"
DATA_DIR     = "data"
EMB_DIR      = "embeddings"
PLOT_DIR     = "outputs"

# Yahoo Answers class names (0–9)
CLASS_NAMES = [
    "Society", "Science", "Health", "Education",
    "Computers", "Sports", "Business", "Entertainment",
    "Relationship", "Politics"
]

FEATURE_SETS = ["BOW_SVD", "TFIDF_SVD", "BERT_CLS_PCA", "BERT_MEAN_PCA", "GLOVE_PCA"]
MODELS       = ["LogisticRegression", "LinearSVC", "KNN", "KMeans"]

# ── Color Palette ─────────────────────────────────────────────────────────────
# One color per model — consistent across all charts
MODEL_COLORS = {
    "LogisticRegression": "#4C72B0",
    "LinearSVC":          "#DD8452",
    "KNN":                "#55A868",
    "KMeans":             "#C44E52",
}

os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ACCURACY & MACRO F1 BAR CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_metric_comparison(results, metric="accuracy", title="Accuracy Comparison"):
    """
    Grouped bar chart comparing all models across all feature sets.

    results : loaded results.json dict
    metric  : "accuracy" or "f1_macro"

    How grouped bars work:
    - x positions are evenly spaced (one group per feature set)
    - within each group, bars are offset by bar_width × model index
    - this creates the classic grouped bar chart look
    """

    fig, ax = plt.subplots(figsize=(14, 6))

    n_features = len(FEATURE_SETS)
    n_models   = len(MODELS)
    bar_width  = 0.18                          # width of each individual bar
    x          = np.arange(n_features)         # base x positions [0,1,2,3,4]

    for i, model in enumerate(MODELS):
        values = []
        for feat in FEATURE_SETS:
            try:
                values.append(results[feat][model][metric])
            except KeyError:
                values.append(0)

        # Offset each model's bars so they sit side by side
        # i - n_models/2 centers the group around the x tick
        offset = (i - n_models / 2 + 0.5) * bar_width

        bars = ax.bar(
            x + offset, values,
            width=bar_width,
            label=model,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.5
        )

        # Add value labels on top of each bar
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=7, rotation=45
            )

    ax.set_xticks(x)
    ax.set_xticklabels(["BOW+SVD", "TFIDF+SVD", "BERT CLS", "BERT Mean", "GloVe"], fontsize=11)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(0, 0.85)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fname = f"{PLOT_DIR}/{metric}_comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrices(results):
    """
    Plots all 20 confusion matrices in a 5×4 grid (feature sets × models).

    Each cell is a heatmap where:
    - rows = true class
    - cols = predicted class
    - diagonal = correct predictions (we want these high)
    - off-diagonal = errors

    fmt="d" displays integer counts
    annot=True writes the number inside each cell
    """

    fig, axes = plt.subplots(
        nrows=len(MODELS),
        ncols=len(FEATURE_SETS),
        figsize=(28, 20)
    )

    feat_labels = ["BOW+SVD", "TFIDF+SVD", "BERT CLS", "BERT Mean", "GloVe"]

    for j, feat in enumerate(FEATURE_SETS):
        for i, model in enumerate(MODELS):
            ax = axes[i][j]
            try:
                cm = np.array(results[feat][model]["confusion_matrix"])
                sns.heatmap(
                    cm, ax=ax,
                    annot=True, fmt="d",
                    cmap="Blues",
                    xticklabels=[c[:3] for c in CLASS_NAMES],  # Short labels
                    yticklabels=[c[:3] for c in CLASS_NAMES],
                    cbar=False,
                    annot_kws={"size": 6}
                )
            except KeyError:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")

            # Column headers (feature set names) on top row only
            if i == 0:
                ax.set_title(feat_labels[j], fontsize=11, fontweight="bold", pad=8)

            # Row headers (model names) on left column only
            if j == 0:
                ax.set_ylabel(model, fontsize=10, fontweight="bold")
            else:
                ax.set_ylabel("")

            ax.set_xlabel("")
            ax.tick_params(labelsize=6)

    fig.suptitle("Confusion Matrices — All Models × All Feature Sets",
                 fontsize=16, fontweight="bold", y=1.01)

    plt.tight_layout()
    fname = f"{PLOT_DIR}/confusion_matrices_all.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAINING TIME COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_time(results):
    """
    Horizontal bar chart of training times for all 20 experiments.

    We use log scale on x-axis because training times vary wildly:
    KNN trains in 0.001s (no real training, just stores data)
    KMeans can take 2s+
    Log scale makes all bars visible and comparable.
    """

    labels = []
    times  = []
    colors = []

    for feat in FEATURE_SETS:
        for model in MODELS:
            try:
                t = results[feat][model]["train_time"]
                labels.append(f"{model}\n{feat}")
                times.append(t)
                colors.append(MODEL_COLORS[model])
            except KeyError:
                pass

    # Sort by training time descending
    sorted_pairs = sorted(zip(times, labels, colors), reverse=True)
    times, labels, colors = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, 14))

    bars = ax.barh(labels, times, color=colors, edgecolor="white", linewidth=0.5)

    # Add time labels at end of each bar
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{t:.3f}s",
            va="center", fontsize=8
        )

    ax.set_xscale("log")   # Log scale for readability
    ax.set_xlabel("Training Time (seconds, log scale)", fontsize=12)
    ax.set_title("Training Time Comparison — All Experiments", fontsize=14, fontweight="bold")
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Add legend for model colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=m) for m, c in MODEL_COLORS.items()]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fname = f"{PLOT_DIR}/training_time_comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. t-SNE FEATURE SPACE VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def plot_tsne(y_train):
    """
    t-SNE reduces high-dimensional features to 2D for visualization.

    t-SNE (t-distributed Stochastic Neighbour Embedding):
    - Finds a 2D layout where similar points stay close together
    - Great for visualizing whether feature spaces are class-separable
    - perplexity: roughly how many neighbours each point considers (5–50)
    - n_iter: more iterations = better layout but slower

    We plot one t-SNE per feature set to compare separability visually.
    Well-separated clusters = good features for classification.
    """

    # Load all train feature sets
    feature_files = {
        "BOW + SVD":       f"{EMB_DIR}/X_train_bow_svd.npy",
        "TF-IDF + SVD":    f"{EMB_DIR}/X_train_tfidf_svd.npy",
        "BERT CLS + PCA":  f"{EMB_DIR}/X_train_cls_pca.npy",
        "BERT Mean + PCA": f"{EMB_DIR}/X_train_mean_pca.npy",
        "GloVe + PCA":     f"{EMB_DIR}/X_train_glove_pca.npy",
    }

    fig, axes = plt.subplots(1, 5, figsize=(30, 6))

    # Use a colormap with 10 distinct colors for 10 classes
    cmap = plt.cm.get_cmap("tab10", 10)

    for ax, (feat_name, fpath) in zip(axes, feature_files.items()):
        print(f"  Running t-SNE on {feat_name}...")

        X = np.load(fpath)

        # Subsample to 2000 points for speed — t-SNE is O(n²)
        # random sample so all classes are roughly represented
        idx = np.random.choice(len(X), size=min(2000, len(X)), replace=False)
        X_sub = X[idx]
        y_sub = y_train[idx]

        # Run t-SNE
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
        X_2d   = tsne.fit_transform(X_sub)

        # Scatter plot — one color per class
        for cls in range(10):
            mask = y_sub == cls
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=[cmap(cls)],
                label=CLASS_NAMES[cls],
                alpha=0.6, s=10
            )

        ax.set_title(feat_name, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.05), fontsize=9)

    fig.suptitle("t-SNE Feature Space Visualization (2000 samples)",
                 fontsize=16, fontweight="bold")

    plt.tight_layout()
    fname = f"{PLOT_DIR}/tsne_visualization.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. EDA PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_eda():
    """
    Exploratory Data Analysis plots:
      a) Class distribution — are classes balanced?
      b) Text length distribution — before and after cleaning
      c) Word clouds — most frequent words per class
    """

    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_df  = pd.read_csv(f"{DATA_DIR}/test.csv")
    df       = pd.concat([train_df, test_df], ignore_index=True)

    # ── a) Class Distribution ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    class_counts = df["label"].value_counts().sort_index()
    bars = ax.bar(
        [CLASS_NAMES[i] for i in class_counts.index],
        class_counts.values,
        color=sns.color_palette("husl", 10),
        edgecolor="white"
    )

    for bar, count in zip(bars, class_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(count),
            ha="center", va="bottom", fontsize=9
        )

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Class Distribution — Yahoo Answers Dataset", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/eda_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved eda_class_distribution.png")

    # ── b) Text Length Distribution ───────────────────────────────────────────
    # text length = number of words
    df["raw_length"]   = df["text"].fillna("").apply(lambda x: len(str(x).split()))
    df["clean_length"] = df["clean_text"].fillna("").apply(lambda x: len(str(x).split()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["raw_length"],   bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
    axes[0].set_title("Raw Text Length Distribution",   fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Number of Words")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(df["raw_length"].median(),   color="red", linestyle="--", label=f"Median: {df['raw_length'].median():.0f}")
    axes[0].legend()

    axes[1].hist(df["clean_length"], bins=50, color="#55A868", edgecolor="white", alpha=0.8)
    axes[1].set_title("Cleaned Text Length Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Number of Words")
    axes[1].set_ylabel("Frequency")
    axes[1].axvline(df["clean_length"].median(), color="red", linestyle="--", label=f"Median: {df['clean_length'].median():.0f}")
    axes[1].legend()

    plt.suptitle("Text Length Before vs After Preprocessing", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/eda_text_length.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved eda_text_length.png")

    # ── c) Word Clouds ────────────────────────────────────────────────────────
    # One word cloud per class — shows most frequent words
    # Word size ∝ frequency in that class
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()

    for cls in range(10):
        # Get all cleaned text for this class
        class_text = " ".join(
            df[df["label"] == cls]["clean_text"].fillna("").tolist()
        )

        wc = WordCloud(
            width=400, height=300,
            background_color="white",
            colormap="tab10",
            max_words=80,
            random_state=42
        ).generate(class_text)

        axes[cls].imshow(wc, interpolation="bilinear")
        axes[cls].set_title(CLASS_NAMES[cls], fontsize=13, fontweight="bold")
        axes[cls].axis("off")

    fig.suptitle("Word Clouds per Class — Cleaned Text", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/eda_wordclouds.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved eda_wordclouds.png")


# ─────────────────────────────────────────────────────────────────────────────
# MASTER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_visualizations():
    """
    Runs all visualization functions in order.
    Reads from outputs/results.json and embeddings/ directory.
    """

    # ── Load results ──
    print("Loading results.json...")
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    # ── Load labels ──
    y_train = np.load(f"{DATA_DIR}/y_train.npy")

    print("\n── 1. Accuracy & F1 Bar Charts ──")
    plot_metric_comparison(results, metric="accuracy",  title="Accuracy Comparison — All Models × Feature Sets")
    plot_metric_comparison(results, metric="f1_macro",  title="Macro F1 Comparison — All Models × Feature Sets")

    print("\n── 2. Confusion Matrices ──")
    plot_confusion_matrices(results)

    print("\n── 3. Training Time ──")
    plot_training_time(results)

    print("\n── 4. t-SNE Visualization ──")
    plot_tsne(y_train)

    print("\n── 5. EDA Plots ──")
    plot_eda()

    print("\n✓ All visualizations saved to outputs/")


if __name__ == "__main__":
    generate_all_visualizations()