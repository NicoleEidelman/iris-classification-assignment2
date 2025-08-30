# Home Assignment 2: Classification Models on the Iris Dataset
# ============================================================
# Structure:
#   Part 1: Data Preparation
#   Part 2: Model Implementation (Dummy, KNN, Decision Tree, Random Forest)
#   Part 3: Evaluation & Interpretation (classification reports + confusion matrix of BEST model)
#   Part 4: Bonus (GridSearchCV for Random Forest + Feature Importances)
#   Add-ons: KNN accuracy vs k plot, CV heatmap, REPORT.txt

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from io import StringIO
from datetime import datetime

from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler  # optional for distance-based models


# ------------------------------- Utilities ---------------------------------- #
def report_accuracy(name: str, y_true, y_pred) -> float:
    """Compute and print accuracy in a consistent format."""
    acc = accuracy_score(y_true, y_pred)
    print(f"{name} accuracy: {acc:.3f}")
    return acc


def save_confusion_matrix(y_true, y_pred, class_names, title, out_path):
    """Plot and save a confusion matrix as an image (no GUI needed)."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {out_path}")


def pick_best_model(acc_dict: dict[str, float]) -> tuple[str, float]:
    """Return (best_name, best_accuracy) by max accuracy; ties resolved by name order."""
    best_name = max(acc_dict, key=lambda k: (acc_dict[k], k))
    return best_name, acc_dict[best_name]


def save_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved text report to {path}")


def make_summary_text(accs: dict[str, float], best_name: str, best_acc: float) -> str:
    lines = [f"Summary (generated {datetime.now().isoformat(timespec='seconds')}):",
             "-"*72]
    for k, v in accs.items():
        lines.append(f"{k:25s} : {v:.3f}")
    lines.append("-"*72)
    lines.append(f"Best model on test: {best_name} ({best_acc:.3f})")
    lines.append("")
    lines.append("Notes:")
    lines.append("* Dummy ≈ 0.33 is just the baseline for balanced classes.")
    lines.append("* KNN performs exceptionally well on Iris due to clear separation in petal features.")
    lines.append("* Decision Tree is interpretable but prone to overfitting.")
    lines.append("* Random Forest is more stable; after GridSearch it nearly matched KNN performance.")
    return "\n".join(lines)



def write_full_report(path: str,
                      target_names: list[str],
                      y_test,
                      preds_by_model: dict[str, np.ndarray],
                      accs: dict[str, float],
                      best_name: str,
                      best_acc: float):
    buf = StringIO()
    buf.write(make_summary_text(accs, best_name, best_acc))
    buf.write("\n\nClassification reports\n")
    buf.write("="*72 + "\n")
    for name, y_pred in preds_by_model.items():
        buf.write(f"\n--- {name} ---\n")
        buf.write(classification_report(y_test, y_pred, target_names=target_names))
    save_text(path, buf.getvalue())


# --------------------------------- Main ------------------------------------- #
def main():
    # ======================================================
    # Part 1: Data Preparation
    # ======================================================
    iris = datasets.load_iris()
    X = iris.data          # features: shape (150, 4)
    y = iris.target        # labels:   shape (150,)

    print("=== Part 1: Data Preparation ===")
    print("Feature names:", iris.feature_names)
    print("Target names :", iris.target_names)
    print("X shape:", X.shape, "| y shape:", y.shape)

    print("\nFirst 5 rows of X:\n", X[:5])
    print("First 10 labels:", y[:10])

    classes, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:", dict(zip(iris.target_names[classes], counts)))

    df = pd.DataFrame(X, columns=iris.feature_names)
    print("\nBasic statistics (describe):\n", df.describe())

    # Train/Test split (stratified for class balance; random_state for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print("\nTrain shapes:", X_train.shape, y_train.shape)
    print("Test shapes :", X_test.shape, y_test.shape)

    # ======================================================
    # Part 2: Model Implementation
    # ======================================================
    print("\n=== Part 2: Model Implementation ===")

    # (1) Baseline: DummyClassifier (predicts the most frequent class)
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    acc_dummy = report_accuracy("Dummy (most_frequent)", y_test, y_pred_dummy)

    # (2) K-Nearest Neighbors (try different k values)
    print("\n-- KNN --")
    knn_results = {}
    for k in [3, 5, 7]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        knn_results[k] = report_accuracy(f"KNN (k={k})", y_test, y_pred_knn)

    # Add-on: plot KNN accuracy vs k (for interpretation in the report)
    k_values = [3, 5, 7, 9]
    knn_accuracies = []
    for k in k_values:
        knn_tmp = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        y_pred_tmp = knn_tmp.predict(X_test)
        knn_accuracies.append(accuracy_score(y_test, y_pred_tmp))

    plt.figure(figsize=(6, 4))
    plt.plot(k_values, knn_accuracies, marker="o", linewidth=2)
    plt.xlabel("k (neighbors)")
    plt.ylabel("Accuracy")
    plt.title("KNN Performance vs k")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("knn_k_curve.png", dpi=150)
    plt.close()
    print("Saved KNN k-curve to knn_k_curve.png")

    # (3) Decision Tree
    print("\n-- Decision Tree --")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    acc_dt = report_accuracy("Decision Tree", y_test, y_pred_dt)

    # Text view of the learned tree (useful for interpretability)
    print("\nDecision Tree (text view):")
    print(export_text(dt, feature_names=list(iris.feature_names)))

    # Plot and save the tree figure
    plt.figure(figsize=(10, 6))
    plot_tree(
        dt,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        fontsize=9,
    )
    plt.title("Decision Tree")
    plt.tight_layout()
    plt.savefig("decision_tree.png", dpi=150)
    plt.close()
    print("Saved tree plot to decision_tree.png")

    # (4) Random Forest
    print("\n-- Random Forest (n_estimators=100) --")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = report_accuracy("Random Forest (100)", y_test, y_pred_rf)

    print("\n=== Summary (accuracy) ===")
    print(f"Dummy (most_frequent): {acc_dummy:.3f}")
    for k, acc in knn_results.items():
        print(f"KNN (k={k})         : {acc:.3f}")
    print(f"Decision Tree        : {acc_dt:.3f}")
    print(f"Random Forest (100)  : {acc_rf:.3f}")

    # ======================================================
    # Part 3: Evaluation & Interpretation (updated)
    # ======================================================
    print("\n=== Part 3: Evaluation & Interpretation ===")

    # Rebuild models to ensure consistency in this section
    dummy = DummyClassifier(strategy="most_frequent", random_state=42).fit(X_train, y_train)
    knn5  = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    dt    = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    rf    = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

    # Predictions
    y_pred_dummy = dummy.predict(X_test)
    y_pred_knn5  = knn5.predict(X_test)
    y_pred_dt    = dt.predict(X_test)
    y_pred_rf    = rf.predict(X_test)

    # Accuracies dict
    accs = {
        "Dummy (most_frequent)": accuracy_score(y_test, y_pred_dummy),
        "KNN (k=5)"            : accuracy_score(y_test, y_pred_knn5),
        "Decision Tree"        : accuracy_score(y_test, y_pred_dt),
        "Random Forest (100)"  : accuracy_score(y_test, y_pred_rf),
    }
    print("\n=== Summary (accuracy) ===")
    for k, v in accs.items():
        print(f"{k:25s} : {v:.3f}")

    best_name, best_acc = pick_best_model(accs)
    print(f"\n>>> Best test accuracy: {best_name} = {best_acc:.3f}")

    # (1) Classification reports per model
    for name, y_pred in {
        "Dummy (most_frequent)": y_pred_dummy,
        "KNN (k=5)"            : y_pred_knn5,
        "Decision Tree"        : y_pred_dt,
        "Random Forest (100)"  : y_pred_rf,
    }.items():
        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # (2) Confusion matrix for the best-performing model
    best_pred = {
        "Dummy (most_frequent)": y_pred_dummy,
        "KNN (k=5)"            : y_pred_knn5,
        "Decision Tree"        : y_pred_dt,
        "Random Forest (100)"  : y_pred_rf,
    }[best_name]

    cm_title = f"Confusion Matrix - {best_name}"
    cm_path  = f"confusion_matrix_{best_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')}.png"
    save_confusion_matrix(y_test, best_pred, iris.target_names, cm_title, cm_path)

    # ======================================================
    # Part 4: Bonus Challenge (GridSearchCV + Feature Importances)
    # ======================================================
    print("\n=== Part 4: Bonus (GridSearchCV + Feature Importances) ===")

    rf_base = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 3, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("Best CV mean accuracy:", f"{grid.best_score_:.3f}")

    best_rf = grid.best_estimator_
    y_pred_best = best_rf.predict(X_test)
    print("Test accuracy (best RF):", f"{accuracy_score(y_test, y_pred_best):.3f}")

    # Heatmap-like visualization of CV mean accuracy (n_estimators × max_depth)
    results = grid.cv_results_
    means = results["mean_test_score"]
    n_list = param_grid["n_estimators"]
    d_list = [str(d) for d in param_grid["max_depth"]]
    score_matrix = np.zeros((len(d_list), len(n_list)))

    for i, md in enumerate(param_grid["max_depth"]):
        for j, ne in enumerate(param_grid["n_estimators"]):
            mask = (results["param_n_estimators"] == ne) & (results["param_max_depth"] == md)
            score_matrix[i, j] = means[mask][0]

    plt.figure(figsize=(6, 4))
    plt.imshow(score_matrix, aspect="auto")
    plt.colorbar(label="Mean CV Accuracy")
    plt.xticks(ticks=range(len(n_list)), labels=n_list)
    plt.yticks(ticks=range(len(d_list)), labels=d_list)
    plt.xlabel("n_estimators")
    plt.ylabel("max_depth")
    plt.title("GridSearchCV Mean Accuracy (Random Forest)")
    plt.tight_layout()
    plt.savefig("rf_gridsearch_heatmap.png", dpi=150)
    plt.close()
    print("Saved CV heatmap to rf_gridsearch_heatmap.png")

    # Feature importances (from tuned RF)
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(7, 4))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(
        ticks=range(len(importances)),
        labels=[iris.feature_names[i] for i in indices],
        rotation=30,
        ha="right",
    )
    plt.ylabel("Feature Importance")
    plt.title("Random Forest Feature Importances (tuned)")
    plt.tight_layout()
    plt.savefig("rf_feature_importances.png", dpi=150)
    plt.close()
    print("Saved feature importance plot to rf_feature_importances.png")

    # -------- REPORT.txt (summary + all classification reports) --------
    write_full_report(
        path="REPORT.txt",
        target_names=list(iris.target_names),
        y_test=y_test,
        preds_by_model={
            "Dummy (most_frequent)": y_pred_dummy,
            "KNN (k=5)"            : y_pred_knn5,
            "Decision Tree"        : y_pred_dt,
            "Random Forest (100)"  : y_pred_rf,
        },
        accs=accs,
        best_name=best_name,
        best_acc=best_acc,
    )

    print("\n✅ Assignment completed. Files saved:")
    print(" - knn_k_curve.png")
    print(" - decision_tree.png")
    print(f" - {cm_path}")
    print(" - rf_gridsearch_heatmap.png")
    print(" - rf_feature_importances.png")
    print(" - REPORT.txt")


if __name__ == "__main__":
    main()
