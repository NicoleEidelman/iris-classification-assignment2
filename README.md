# Assignment 2: Classification Models on the Iris Dataset

## Overview
This repository contains the implementation of **Assignment 2** for the Machine Learning course.  
The task focuses on applying different classification models on the **Iris dataset**, comparing their performance, and interpreting the results.

---

## Repository Structure
- `assignment2_classification.py` – Main Python script containing all parts of the assignment:
  - Part 1: Data Preparation
  - Part 2: Model Implementation (Dummy, KNN, Decision Tree, Random Forest)
  - Part 3: Evaluation & Interpretation (classification reports, confusion matrix)
  - Part 4: Bonus (GridSearchCV for Random Forest + feature importances)
- `REPORT.txt` – Summary of results and classification reports for all models (English only).
- **Figures**:
  - `knn_k_curve.png` – Accuracy vs. k for KNN
  - `decision_tree.png` – Visualization of the Decision Tree
  - `confusion_matrix_KNN_k=5.png` – Confusion Matrix of the best-performing model
  - `rf_gridsearch_heatmap.png` – Hyperparameter tuning results (Random Forest)
  - `rf_feature_importances.png` – Feature importance plot from the tuned Random Forest

---

## Models Implemented
1. **Baseline (DummyClassifier)**  
   - Strategy: `most_frequent`  
   - Accuracy ≈ 0.33 (expected baseline for balanced classes).

2. **K-Nearest Neighbors (KNN)**  
   - Tested values: k = 3, 5, 7, 9  
   - Best accuracy: **1.00** (perfect classification with k=3 or 5).  
   - Simple distance-based model works extremely well on Iris due to clear separation in *petal* features.

3. **Decision Tree**  
   - Accuracy ≈ 0.93  
   - Easy to interpret, but slightly prone to overfitting.  
   - First split on `petal length <= 2.45` perfectly separates **setosa**.

4. **Random Forest (100 estimators)**  
   - Accuracy ≈ 0.90  
   - After hyperparameter tuning with GridSearchCV: **0.967** accuracy on test set.  
   - Feature importances confirm that *petal length* and *petal width* dominate, while *sepal width* contributes very little.

---

## Results Summary
- **Dummy**: ~0.33 (baseline)  
- **KNN (k=3/5)**: **1.00** (best model on test set)  
- **Decision Tree**: ~0.93  
- **Random Forest (100)**: ~0.90  
- **Random Forest (tuned)**: ~0.967  

👉 **Best model:** KNN (k=3 or 5), achieving perfect classification on the given split.  
Random Forest is more robust and generalizes well, but slightly underperformed compared to KNN in this dataset.

---

## Requirements
To run the code, install the dependencies:

```bash
pip install -r requirements.txt
How to Run
Clone the repository:

bash
Copy code
git clone https://github.com/<your-username>/iris-classification-assignment2.git
cd iris-classification-assignment2
Run the script:

bash
Copy code
python assignment2_classification.py
Outputs will be saved as .png figures and REPORT.txt in the working directory.
