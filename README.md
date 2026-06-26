# 🧠 Brain Stroke Analysis and Prediction Using Machine Learning

## 📌 Project Overview

Brain Stroke Analysis and Prediction Using Machine Learning is a healthcare-focused machine learning project designed to predict the likelihood of stroke occurrence based on patient health indicators. The project applies data preprocessing, class balancing, feature analysis, and machine learning techniques to identify individuals at risk of stroke.

An interactive Streamlit web application is integrated to provide real-time stroke risk prediction based on user input.

---

## 🎯 Objectives

* Analyze healthcare data related to stroke occurrence.
* Identify important risk factors contributing to stroke.
* Compare multiple machine learning algorithms.
* Improve prediction performance using data balancing techniques.
* Build a user-friendly web application for stroke risk assessment.

---

## 📂 Project Structure

```text
BrainStrokePrediction/
│
├── app.py
├── README.md
│
├── data/
│   └── healthcare-dataset-stroke-data.csv
│
├── models/
│   ├── scaler.pkl
│   └── stroke_model.pkl
│
├── results/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   └── roc_curve.png
│
└── src/
    ├── preprocessing.py
    ├── train_model.py
    ├── model_comparison.py
    └── predict.py
```

---

## 📊 Visualizations

### Generated Outputs

* Model Comparison Graph
* Confusion Matrix
* ROC Curve
* Feature Importance Graph

Store all generated images inside:

```text
results/
├── confusion_matrix.png
├── feature_importance.png
├── model_comparison.png
└── roc_curve.png
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/BrainStrokeAnalysis.git
cd BrainStrokeAnalysis
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn streamlit joblib
```

### Train Model

```bash
cd src
python train_model.py
```

### Run Application

```bash
streamlit run app.py
```

---

## 📊 Dataset Information

**Resource :** https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

**Dataset Size:** 5,110 Records

**Target Variable:**

* 0 → No Stroke
* 1 → Stroke

### Features

| Feature           | Description           |
| ----------------- | --------------------- |
| gender            | Patient Gender        |
| age               | Patient Age           |
| hypertension      | Hypertension Status   |
| heart_disease     | Heart Disease Status  |
| ever_married      | Marital Status        |
| work_type         | Employment Type       |
| Residence_type    | Urban/Rural Residence |
| avg_glucose_level | Average Glucose Level |
| bmi               | Body Mass Index       |
| smoking_status    | Smoking Status        |
| stroke            | Target Variable       |

### Dataset Characteristics

* Total Records: 5110
* Stroke Cases: 249
* Non-Stroke Cases: 4861
* Missing Values: BMI Column

---

## 🛠 Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* Imbalanced-Learn (SMOTE)
* Matplotlib
* Seaborn
* Joblib
* Streamlit

### Machine Learning Algorithms

* Logistic Regression
* Gaussian Naive Bayes
* Bernoulli Naive Bayes
* Random Forest Classifier

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Removed unnecessary ID column
* Handled missing BMI values
* Label encoded categorical variables
* Feature scaling using StandardScaler

### 2. Data Balancing

The dataset was highly imbalanced.

Before SMOTE:

* Stroke Cases = 249
* Non-Stroke Cases = 4861

SMOTE (Synthetic Minority Oversampling Technique) was applied to balance the training dataset.

### 3. Model Training

The following models were trained and evaluated:

* Logistic Regression
* Gaussian Naive Bayes
* Bernoulli Naive Bayes
* Random Forest

### 4. Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix

---

## 📈 Model Performance

### Accuracy Comparison

| Model                 | Accuracy |
| --------------------- | -------- |
| Logistic Regression   | 78.47%   |
| Gaussian Naive Bayes  | 71.82%   |
| Bernoulli Naive Bayes | 72.90%   |
| Random Forest         | 90.22%   |

### Final Random Forest Results

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 86.79% |
| Precision | 16%    |
| Recall    | 40%    |
| F1-Score  | 23%    |
| ROC-AUC   | 79.60% |

---

## 🔍 Feature Importance

The Random Forest model identified the following key stroke risk factors:

| Rank | Feature               | Importance |
| ---- | --------------------- | ---------- |
| 1    | Age                   | 49.44%     |
| 2    | BMI                   | 15.07%     |
| 3    | Average Glucose Level | 14.89%     |
| 4    | Work Type             | 6.66%      |
| 5    | Smoking Status        | 4.28%      |

### Key Finding

Age was found to be the most significant factor influencing stroke prediction, contributing nearly 50% of the model's decision-making process.

---

## 💻 Streamlit Web Application

The project includes an interactive Streamlit application that allows users to:

* Enter patient details
* Predict stroke risk
* View prediction probability
* Categorize risk level (Low, Moderate, High)

---

## 🎓 Learning Outcomes

* Data Preprocessing Techniques
* Handling Missing Values
* Class Imbalance Handling Using SMOTE
* Machine Learning Model Development
* Model Evaluation and Validation
* Feature Importance Analysis
* Healthcare Data Analytics
* Streamlit Web Application Development

---

## ⚠️ Disclaimer

This project is developed for educational and research purposes only. It should not be considered a substitute for professional medical advice, diagnosis, or treatment.

---

## 👨‍💻 Author

**Prathmesh Gaikwad**

B.Tech Computer Science and Engineering

Passionate about Machine Learning, Full Stack Development, Artificial Intelligence, and Data Analytics.
