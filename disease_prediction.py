"""
CodeAlpha - Machine Learning Internship
Task 4: Disease Prediction from Medical Data
----------------------------------------
Objective: Predict the possibility of disease (malignant vs benign tumor)
based on patient diagnostic data.

Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset - built into
scikit-learn, a real, widely-used medical diagnostic dataset (UCI ML Repository).

Algorithms: Logistic Regression, SVM, Random Forest, XGBoost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import joblib

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ------------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="diagnosis")  # 0 = malignant, 1 = benign

print(f"Dataset shape: {X.shape}")
print("Class distribution:\n", y.value_counts().rename({0: "Malignant", 1: "Benign"}))

X.to_csv("breast_cancer_features.csv", index=False)

# ------------------------------------------------------------------
# 2. PREPROCESSING
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------
# 3. MODEL TRAINING
# ------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        eval_metric="logloss", random_state=RANDOM_STATE
    ),
}

scaled_models = {"Logistic Regression", "SVM (RBF Kernel)"}
results = []
roc_data = {}

for name, model in models.items():
    if name in scaled_models:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-Score": f1, "ROC-AUC": auc
    })

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, auc)

    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
print("\nModel comparison:\n", results_df.round(4))
results_df.to_csv("model_comparison_results.csv", index=False)

# ------------------------------------------------------------------
# 4. VISUALIZATIONS
# ------------------------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

if best_model_name in scaled_models:
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Malignant", "Benign"],
            yticklabels=["Malignant", "Benign"], ax=axes[0])
axes[0].set_title(f"Confusion Matrix - {best_model_name}")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

for name, (fpr, tpr, auc) in roc_data.items():
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[1].set_title("ROC Curves")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend(fontsize=8)

rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top10 = importances.sort_values(ascending=False).head(10).sort_values()
top10.plot(kind="barh", ax=axes[2], color="crimson")
axes[2].set_title("Top 10 Feature Importances (Random Forest)")

plt.tight_layout()
plt.savefig("disease_prediction_results.png", dpi=150)
print("\nSaved -> disease_prediction_results.png")

# ------------------------------------------------------------------
# 5. SAVE BEST MODEL
# ------------------------------------------------------------------
joblib.dump(best_model, "disease_prediction_best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print(f"\nBest model: {best_model_name} saved as disease_prediction_best_model.pkl")
