#!/usr/bin/env python3
"""
Iris Classification Comparison

Trains and evaluates four classifiers on the Iris dataset:
  - Logistic Regression
  - k‑Nearest Neighbors (k=5)
  - Decision Tree
  - Support Vector Machine (RBF kernel)

For each model it reports:
  • Test‐set accuracy
  • 5‑fold cross‑validated accuracy (mean ± std)
  • Classification report (precision, recall, F1‑score)
  • Confusion matrix

Finally, it produces:
  • A bar chart of test accuracies
  • Confusion‐matrix heatmaps for each classifier
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    # ----------------------------
    # 1. Load the Iris dataset
    # ----------------------------
    iris = load_iris()
    X = iris.data  # shape (150, 4)
    y = iris.target  # shape (150,)
    feature_names = iris.feature_names
    target_names = iris.target_names

    # ----------------------------
    # 2. Train/Test split (70/30)
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # ----------------------------
    # 3. Define models
    # ----------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "k‑NN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM (RBF kernel)": SVC(kernel="rbf", probability=True, random_state=42),
    }

    # ----------------------------
    # 4. Train, evaluate, cross‑validate
    # ----------------------------
    results = {}
    for name, model in models.items():
        # Train on train set
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Test accuracy
        test_acc = accuracy_score(y_test, y_pred)

        # 5‑fold cross‑validation on entire dataset
        cv_scores = cross_val_score(model, X, y, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # Classification report & confusion matrix
        clf_report = classification_report(
            y_test, y_pred, target_names=target_names, output_dict=True
        )
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "test_accuracy": test_acc,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "report": clf_report,
            "confusion_matrix": cm,
        }

    # ----------------------------
    # 5. Print summary table
    # ----------------------------
    summary = pd.DataFrame(
        {
            name: {
                "Test Acc": res["test_accuracy"],
                "CV Mean Acc": res["cv_mean"],
                "CV Std Acc": res["cv_std"],
            }
            for name, res in results.items()
        }
    ).T

    print("\n=== Model Comparison Summary ===")
    print(summary.to_string(float_format="{:.4f}".format))

    # ----------------------------
    # 6. Detailed reports
    # ----------------------------
    for name, res in results.items():
        print(f"\n--- {name} ---")
        print("Confusion Matrix:")
        print(res["confusion_matrix"])
        print("\nClassification Report:")
        df_report = pd.DataFrame(res["report"]).T
        print(df_report.to_string(float_format="{:.4f}".format))

    # ----------------------------
    # 7. Plot test‐accuracy bar chart
    # ----------------------------
    plt.figure(figsize=(6, 4))
    summary["Test Acc"].plot(kind="bar", rot=45)
    plt.ylim(0, 1.05)
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy by Model")
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # 8. Plot confusion matrices
    # ----------------------------
    n_models = len(models)
    for name, res in results.items():
        cm = res["confusion_matrix"]
        plt.figure(figsize=(4, 3))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title(f"{name}: Confusion Matrix")
        plt.colorbar()
        ticks = np.arange(len(target_names))
        plt.xticks(ticks, target_names, rotation=45)
        plt.yticks(ticks, target_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
