import pandas as pd
import joblib

# Load Model
model = joblib.load(
    "../models/stroke_model.pkl"
)

scaler = joblib.load(
    "../models/scaler.pkl"
)

# Example Patient

patient = pd.DataFrame([{

    "gender":1,
    "age":65,
    "hypertension":1,
    "heart_disease":0,
    "ever_married":1,
    "work_type":2,
    "Residence_type":1,
    "avg_glucose_level":180,
    "bmi":30,
    "smoking_status":2

}])

# Scale Numeric Features

numeric_cols = [
    "age",
    "avg_glucose_level",
    "bmi"
]

patient[numeric_cols] = scaler.transform(
    patient[numeric_cols]
)

prediction = model.predict(
    patient
)[0]

probability = model.predict_proba(
    patient
)[0][1]

print("\n====== RESULT ======\n")

if prediction == 1:

    print(
        f"Stroke Risk: HIGH"
    )

else:

    print(
        f"Stroke Risk: LOW"
    )

print(
    f"Probability: {probability*100:.2f}%"
)