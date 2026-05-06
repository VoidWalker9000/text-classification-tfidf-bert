"""
main.py
-------
Master pipeline for the SML NLP Classification Project.
Runs the entire project end-to-end with a single command.

Usage:
  python main.py                  # Run full pipeline
  python main.py --skip-data      # Skip data download (CSV already exists)
  python main.py --skip-embeddings # Skip BERT embedding extraction
  python main.py --viz-only       # Only regenerate visualizations
  python main.py --models-only    # Only run models + visualizations

Example (most common after first run):
  python main.py --skip-data --skip-embeddings
"""

import os
import sys
import json
import time
import argparse
import numpy as np

# Add src/ to Python path so all modules can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Argument Parser ───────────────────────────────────────────────────────────
# argparse lets us pass flags like --skip-data from the command line
# Each add_argument defines one flag
# action="store_true" means the flag is False by default, True if passed
parser = argparse.ArgumentParser(description="SML NLP Classification Pipeline")
parser.add_argument("--skip-data",        action="store_true", help="Skip data download")
parser.add_argument("--skip-embeddings",  action="store_true", help="Skip BERT embedding extraction")
parser.add_argument("--viz-only",         action="store_true", help="Only generate visualizations")
parser.add_argument("--models-only",      action="store_true", help="Only run models + visualizations")
args = parser.parse_args()


# ── Helper: Step Logger ───────────────────────────────────────────────────────
def log_step(step_num, title):
    """
    Prints a clean step header with timestamp.
    Makes the terminal output easy to follow during a demo.
    """
    print("\n" + "="*60)
    print(f"  STEP {step_num}: {title}")
    print(f"  Time: {time.strftime('%H:%M:%S')}")
    print("="*60)


def log_done(start_time):
    """Prints how long a step took."""
    elapsed = round(time.time() - start_time, 2)
    print(f"  ✓ Done in {elapsed}s")


# ── Helper: Check Files Exist ─────────────────────────────────────────────────
def check_file(path, name):
    """
    Checks if a required file exists before proceeding.
    Exits with a helpful error message if not found.
    """
    if not os.path.exists(path):
        print(f"\n  ✗ ERROR: {name} not found at '{path}'")
        print(f"    Please run the required step first.")
        sys.exit(1)    # sys.exit(1) stops the program with error code 1


# ── Step 1: Data Loading ──────────────────────────────────────────────────────
def run_data_loading():
    log_step(1, "Data Loading — Yahoo Answers from HuggingFace")
    t = time.time()

    # Import and run data_loader
    # We import here (not at top) so missing libraries only error at the step they're needed
    sys.path.insert(0, "src")   # Add src/ to Python path so imports work
    from data_loader import load_and_save_data
    load_and_save_data()

    log_done(t)


# ── Step 2: Preprocessing ─────────────────────────────────────────────────────
def run_preprocessing():
    log_step(2, "Preprocessing — Text Cleaning + BoW + TF-IDF")
    t = time.time()

    check_file("data/yahoo_answers.csv", "Yahoo Answers CSV")

    from preprocessing import preprocess_and_save
    preprocess_and_save()

    log_done(t)


# ── Step 3: Embeddings ────────────────────────────────────────────────────────
def run_embeddings():
    log_step(3, "Embeddings — DistilBERT CLS + Mean Pool + GloVe")
    t = time.time()

    check_file("data/train.csv", "train.csv")
    check_file("data/test.csv",  "test.csv")

    from embeddings import generate_and_save_embeddings
    generate_and_save_embeddings()

    log_done(t)


# ── Step 4: Dimensionality Reduction ─────────────────────────────────────────
def run_dimensionality():
    log_step(4, "Dimensionality Reduction — SVD + PCA")
    t = time.time()

    # Check all required input files exist
    check_file("data/X_train_bow.npz",    "BoW train matrix")
    check_file("data/X_train_tfidf.npz",  "TF-IDF train matrix")
    check_file("embeddings/train_cls.npy",  "BERT CLS embeddings")
    check_file("embeddings/train_mean.npy", "BERT Mean embeddings")
    check_file("embeddings/train_glove.npy","GloVe embeddings")

    from dimensionality import reduce_and_save
    reduce_and_save()

    log_done(t)


# ── Step 5: Models ────────────────────────────────────────────────────────────
def run_models():
    log_step(5, "Models — LR, LinearSVC, KNN, KMeans on all feature sets")
    t = time.time()

    check_file("embeddings/X_train_bow_svd.npy",   "BoW SVD features")
    check_file("embeddings/X_train_tfidf_svd.npy", "TF-IDF SVD features")
    check_file("embeddings/X_train_cls_pca.npy",   "CLS PCA features")
    check_file("embeddings/X_train_mean_pca.npy",  "Mean PCA features")
    check_file("embeddings/X_train_glove_pca.npy", "GloVe PCA features")

    from models import run_all_experiments
    run_all_experiments(base_path=".")

    log_done(t)


# ── Step 6: Visualizations ────────────────────────────────────────────────────
def run_visualizations():
    log_step(6, "Visualizations — Charts, Confusion Matrices, t-SNE, EDA")
    t = time.time()

    check_file("outputs/results.json", "results.json")

    from visualizations import generate_all_visualizations
    generate_all_visualizations()

    log_done(t)


# ── Results Summary ───────────────────────────────────────────────────────────
def print_summary():
    """
    Prints a clean summary table of all results after everything runs.
    Reads from outputs/results.json.
    """
    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)

    if not os.path.exists("outputs/results.json"):
        print("  No results.json found — run models first.")
        return

    with open("outputs/results.json", "r") as f:
        results = json.load(f)

    # Print accuracy table
    # f-string formatting: {value:<20} = left-align in 20 chars, {value:>10} = right-align in 10 chars
    print(f"\n  {'Feature Set':<20} {'Model':<22} {'Accuracy':>10} {'F1 Macro':>10}")
    print(f"  {'-'*20} {'-'*22} {'-'*10} {'-'*10}")

    best_acc   = 0
    best_combo = ""

    for feat, models in results.items():
        for model, metrics in models.items():
            acc      = metrics.get("accuracy", 0)
            f1_macro = metrics.get("f1_macro", 0)
            print(f"  {feat:<20} {model:<22} {acc:>10.4f} {f1_macro:>10.4f}")

            if acc > best_acc:
                best_acc   = acc
                best_combo = f"{feat} + {model}"

    print(f"\n  ✓ Best combination: {best_combo} → Accuracy: {best_acc:.4f}")
    print(f"\n  All plots saved to: outputs/")
    print("="*60)


# ── Main Entry Point ──────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  SML PROJECT — NLP Text Classification Pipeline")
    print("  Yahoo Answers | 10 Classes | 8000 Samples")
    print("="*60)

    # ── viz-only mode ─────────────────────────────────────────────────────────
    if args.viz_only:
        run_visualizations()
        print_summary()
        return

    # ── models-only mode ──────────────────────────────────────────────────────
    if args.models_only:
        run_models()
        run_visualizations()
        print_summary()
        return

    # ── Full pipeline ─────────────────────────────────────────────────────────

    # Step 1: Data
    if not args.skip_data:
        run_data_loading()
    else:
        print("\n  [SKIP] Step 1: Data Loading")
        check_file("data/yahoo_answers.csv", "Yahoo Answers CSV")

    # Step 2: Preprocessing
    run_preprocessing()

    # Step 3: Embeddings
    if not args.skip_embeddings:
        run_embeddings()
    else:
        print("\n  [SKIP] Step 3: Embeddings")
        check_file("embeddings/train_cls.npy", "BERT CLS embeddings")

    # Step 4: Dimensionality Reduction
    run_dimensionality()

    # Step 5: Models
    run_models()

    # Step 6: Visualizations
    run_visualizations()

    # Final Summary
    print_summary()


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()