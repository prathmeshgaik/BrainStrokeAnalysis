import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import preprocess

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from imblearn.over_sampling import SMOTE

# Load Dataset

df = pd.read_csv(
    "../data/healthcare-dataset-stroke-data.csv"
)

# Preprocessing

df = preprocess(df)

print("\nMissing Values:\n")
print(df.isnull().sum())

# Features & Target

X = df.drop("stroke", axis=1)
y = df["stroke"]

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Scaling

scaler = StandardScaler()

numeric_cols = [
    "age",
    "avg_glucose_level",
    "bmi"
]

X_train[numeric_cols] = scaler.fit_transform(
    X_train[numeric_cols]
)

X_test[numeric_cols] = scaler.transform(
    X_test[numeric_cols]
)

# SMOTE

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

# Random Forest

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# Prediction

pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    pred
)

print("\nAccuracy:")
print(accuracy)

# Classification Report

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        pred
    )
)

# Confusion Matrix

cm = confusion_matrix(
    y_test,
    pred
)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(
    "../results/confusion_matrix.png"
)

plt.show()

# ROC Curve

y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {roc_auc:.3f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    linewidth=1
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)

plt.savefig(
    "../results/roc_curve.png"
)

plt.show()

print(f"\nROC AUC Score: {roc_auc:.4f}")

# Feature Importance

importance = model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,6))

plt.barh(
    feature_df["Feature"],
    feature_df["Importance"]
)

plt.title(
    "Feature Importance"
)

plt.xlabel(
    "Importance"
)

plt.tight_layout()

plt.savefig(
    "../results/feature_importance.png"
)

plt.show()

print("\nTop Features:\n")
print(feature_df.head(10))

# Save Model

joblib.dump(
    model,
    "../models/stroke_model.pkl"
)

joblib.dump(
    scaler,
    "../models/scaler.pkl"
)

print("\nModel Saved Successfully!")