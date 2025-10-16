# 🌸 Iris Classification Comparison

This project implements and compares multiple supervised machine learning classifiers on the **Iris dataset** from UCI Machine Learning Repository.
It evaluates model accuracy, precision, recall, and F1-score through both **test-set metrics** and **5-fold cross-validation**, providing visual performance analysis with bar charts and confusion-matrix heatmaps.

---

## 📘 Overview

The Iris dataset is a classic benchmark for multiclass classification, containing **150 samples** and **4 features** representing flower dimensions across **3 species** of Iris.

This project trains and evaluates four different models:

| Classifier | Type | Key Parameter |
|-------------|------|----------------|
| Logistic Regression | Linear | `max_iter=200` |
| k-Nearest Neighbors | Non-parametric | `k=5` |
| Decision Tree | Tree-based | `random_state=42` |
| Support Vector Machine | Kernel-based | RBF kernel |

---

## 🧩 Features

- Automated training, testing, and 5-fold cross-validation  
- Comparative evaluation table (accuracy, mean ± std)  
- Classification report and confusion matrix for each model  
- Visualization of:
  - Test accuracy (bar chart)
  - Confusion matrices (heatmaps)

---

## 🧠 Methodology

1. **Dataset Loading**  
   Loaded using `sklearn.datasets.load_iris()`.
2. **Preprocessing**  
   Standard 70/30 train–test split with stratified sampling to preserve class ratios.
3. **Model Training**  
   Each classifier trained on training data with consistent random seeds.
4. **Evaluation**  
   - `accuracy_score()` for test performance  
   - `cross_val_score()` for 5-fold CV  
   - `classification_report()` and `confusion_matrix()` for diagnostics
5. **Visualization**  
   `matplotlib` used to generate comparison and confusion-matrix plots.

---

## 📊 Example Output

**Model Comparison Summary**

[Logistic Regression Confusion Matrix](Figure1.png)
[knn Matrix](knn_matrix.png)
[knn Matrix](knn_matrix.png)
[logis Matrix](logis_matrix.png)
[svm Matrix](svm_matrix.png)
[tree Matrix](tree_matrix.png)
[test accuracy](test_acc.png)
---

## 📦 Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/dmok22/ml-iris.git
cd ml-iris

