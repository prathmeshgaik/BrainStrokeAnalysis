import streamlit as st
import pandas as pd
import joblib

# Load Model

model = joblib.load("models/stroke_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Page Config

st.set_page_config(
    page_title="Brain Stroke Prediction System",
    page_icon="🧠",
    layout="centered"
)

# Sidebar

st.sidebar.title("About")

st.sidebar.info(
    """
    Brain Stroke Analysis and Prediction
    using Machine Learning.

    By Prathmesh Bharat Gaikwad
    """
)

# Main UI

st.title("🧠 Brain Stroke Prediction System")

st.write(
    "Enter patient details to predict stroke risk."
)

# Inputs

gender = st.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

age = st.slider(
    "Age",
    1,
    100,
    45
)

hypertension = st.selectbox(
    "Hypertension",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

heart_disease = st.selectbox(
    "Heart Disease",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

ever_married = st.selectbox(
    "Ever Married",
    ["No", "Yes"]
)

work_type = st.selectbox(
    "Work Type",
    [
        "Private",
        "Self-employed",
        "Govt_job",
        "children",
        "Never_worked"
    ]
)

residence = st.selectbox(
    "Residence Type",
    ["Urban", "Rural"]
)

glucose = st.number_input(
    "Average Glucose Level",
    min_value=50.0,
    max_value=300.0,
    value=100.0
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

smoking = st.selectbox(
    "Smoking Status",
    [
        "never smoked",
        "formerly smoked",
        "smokes",
        "Unknown"
    ]
)

# Encoding Maps

gender_map = {
    "Female": 0,
    "Male": 1,
    "Other": 2
}

married_map = {
    "No": 0,
    "Yes": 1
}

work_map = {
    "Govt_job": 0,
    "Never_worked": 1,
    "Private": 2,
    "Self-employed": 3,
    "children": 4
}

residence_map = {
    "Rural": 0,
    "Urban": 1
}

smoking_map = {
    "Unknown": 0,
    "formerly smoked": 1,
    "never smoked": 2,
    "smokes": 3
}

# Prediction Button

if st.button("Predict Stroke Risk"):

    data = pd.DataFrame([{
        "gender": gender_map[gender],
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": married_map[ever_married],
        "work_type": work_map[work_type],
        "Residence_type": residence_map[residence],
        "avg_glucose_level": glucose,
        "bmi": bmi,
        "smoking_status": smoking_map[smoking]
    }])

    numeric_cols = [
        "age",
        "avg_glucose_level",
        "bmi"
    ]

    data[numeric_cols] = scaler.transform(
        data[numeric_cols]
    )

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    probability_percent = probability * 100

    st.subheader("Prediction Result")

    if probability_percent < 20:
        st.success(
            f"🟢 Low Stroke Risk ({probability_percent:.2f}%)"
        )

    elif probability_percent < 50:
        st.warning(
            f"🟡 Moderate Stroke Risk ({probability_percent:.2f}%)"
        )

    else:
        st.error(
            f"🔴 High Stroke Risk ({probability_percent:.2f}%)"
        )

    st.progress(float(probability))

    st.subheader("Important Risk Factors")

    st.write("""
    • Age  
    • BMI  
    • Average Glucose Level  
    • Smoking Status  
    • Hypertension  
    """)

    st.write("Model Prediction:", prediction)
    st.markdown("---")

st.caption(
    "⚠️ Disclaimer: This application is developed for educational purposes only. "
    "It should not be considered a substitute for professional medical advice, diagnosis, or treatment."
)