import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import preprocess

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE

# Load Dataset
df = pd.read_csv(
    "../data/healthcare-dataset-stroke-data.csv"
)

df = preprocess(df)

X = df.drop("stroke", axis=1)
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# SMOTE
smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
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

models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Gaussian NB":
        GaussianNB(),

    "Bernoulli NB":
        BernoulliNB(),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(
        y_test,
        pred
    )

    results[name] = acc

    print(
        f"{name} Accuracy: {acc:.4f}"
    )

# Graph
plt.figure(figsize=(8,5))

plt.bar(
    results.keys(),
    results.values()
)

plt.title(
    "Model Comparison"
)

plt.ylabel(
    "Accuracy"
)

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "../results/model_comparison.png"
)

plt.show()