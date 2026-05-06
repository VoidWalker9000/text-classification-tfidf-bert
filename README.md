# Text Classification: Sparse Features vs Contextual Embeddings

**Course:** Statistical Machine Learning | **Student:** Russel Pinkal Gandhi | **Roll No:** BA2025041

A systematic comparison of classical ML models across sparse and dense feature representations for multi-class text classification on the Yahoo Answers Topics dataset.

---

## Project Overview

This project compares 5 feature extraction strategies across 4 classical ML models (20 total experiments):

**Features:**
- Unigram Bag-of-Words + Truncated SVD
- Bigram TF-IDF + Truncated SVD
- DistilBERT CLS embeddings + PCA
- DistilBERT Mean-pooled embeddings + PCA
- GloVe 300d embeddings + PCA *(bonus)*

**Models:**
- Logistic Regression
- Linear SVC
- K-Nearest Neighbours (KNN)
- KMeans (unsupervised baseline)

**Best Result:** BERT Mean + Linear SVC → **63.44% accuracy, 62.6% Macro F1**

---

## Project Structure

text-classification-tfidf-bert/
├── data/                        # Raw and processed data (gitignored except .gitkeep)
│   ├── yahoo_answers.csv
│   ├── train.csv / test.csv
│   ├── X_train_bow.npz / X_test_bow.npz
│   ├── X_train_tfidf.npz / X_test_tfidf.npz
│   ├── y_train.npy / y_test.npy
│   ├── bow_vectorizer.pkl
│   └── tfidf_vectorizer.pkl
├── embeddings/                  # Reduced feature matrices (gitignored except .gitkeep)
│   ├── X_train_bow_svd.npy / X_test_bow_svd.npy
│   ├── X_train_tfidf_svd.npy / X_test_tfidf_svd.npy
│   ├── X_train_cls_pca.npy / X_test_cls_pca.npy
│   ├── X_train_mean_pca.npy / X_test_mean_pca.npy
│   ├── X_train_glove_pca.npy / X_test_glove_pca.npy
│   └── train_cls.npy / train_mean.npy / train_glove.npy (raw embeddings)
├── outputs/                     # Plots and results
│   ├── results.json
│   ├── accuracy_comparison.png
│   ├── f1_macro_comparison.png
│   ├── confusion_matrices_all.png
│   ├── training_time_comparison.png
│   ├── tsne_visualization.png
│   ├── eda_class_distribution.png
│   ├── eda_text_length.png
│   ├── eda_wordclouds.png
│   └── variance_*.png
├── src/
│   ├── data_loader.py           # Downloads and samples Yahoo Answers dataset
│   ├── preprocessing.py         # Text cleaning, BOW, TF-IDF
│   ├── embeddings.py            # DistilBERT + GloVe feature extraction
│   ├── dimensionality.py        # Truncated SVD + PCA reduction
│   ├── models.py                # Train and evaluate all models
│   └── visualizations.py        # Generate all plots
├── report/                      # LaTeX report
├── main.py                      # End-to-end pipeline orchestration
├── requirements.txt
└── README.md

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/text-classification-tfidf-bert.git
cd text-classification-tfidf-bert
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download GloVe vectors
Download `glove.6B.300d.txt` from [Stanford NLP](https://nlp.stanford.edu/projects/glove/) and place it at: data/glove/glove.6B.300d.txt

---

## Running the Pipeline

> **Note:** Steps 2 and 3 (embeddings + dimensionality reduction) are computationally heavy. Run them on Kaggle/Colab with GPU. All precomputed `.npy` files are included in the submission zip.

### Step 1 — Download and sample dataset
```bash
python src/data_loader.py
```

### Step 2 — Preprocess text + build BOW/TF-IDF
```bash
python src/preprocessing.py
```

### Step 3 — Extract embeddings (GPU recommended)
```bash
python src/embeddings.py
```

### Step 4 — Dimensionality reduction
```bash
python src/dimensionality.py
```

### Step 5 — Train and evaluate models
```bash
python src/models.py
```

### Step 6 — Generate visualizations
```bash
python src/visualizations.py
```

### Or run everything at once
```bash
python main.py
```

---

## Results Summary

| Feature Set | LR | SVM | KNN | KMeans |
|---|---|---|---|---|
| BOW + SVD | 0.4537 | 0.4669 | 0.2681 | 0.1344 |
| TF-IDF + SVD | 0.2275 | 0.2331 | 0.2056 | 0.1138 |
| BERT CLS + PCA | 0.6112 | 0.6206 | 0.5162 | 0.2512 |
| BERT Mean + PCA | 0.6238 | **0.6344** | 0.5494 | 0.2650 |
| GloVe + PCA | 0.6000 | 0.6119 | 0.5725 | 0.4025 |

---

## Dataset

Yahoo Answers Topics dataset via HuggingFace (`yahoo_answers_topics`).
- 8,000 samples (800 per class, stratified)
- 10 classes: Society, Science, Health, Education, Computers, Sports, Business, Entertainment, Relationship, Politics
- 80/20 train-test split

---

## External Dependencies

- **DistilBERT:** Downloaded automatically via HuggingFace Transformers
- **GloVe:** Download manually from https://nlp.stanford.edu/projects/glove/
- **Kaggle GPU:** Used for BERT embedding extraction

---

## Acknowledgements

DistilBERT embeddings computed using Kaggle GPU resources. GloVe vectors sourced from Stanford NLP.