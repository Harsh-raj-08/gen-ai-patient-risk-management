import streamlit as st
import joblib
import numpy as np

# Load model & scaler
model = joblib.load("models/cardio_logreg_basic.joblib")
scaler = joblib.load("models/cardio_scaler_basic.joblib")

st.set_page_config(page_title="Cardio Predictor")

st.title("❤️ Cardiovascular Disease Predictor")
st.write("Enter patient details below:")

# Inputs
age = st.number_input("Age (years)", 18, 100)
height = st.number_input("Height (cm)", 100, 220)
weight = st.number_input("Weight (kg)", 30.0, 200.0)
ap_hi = st.number_input("Systolic BP")
ap_lo = st.number_input("Diastolic BP")

gender = st.selectbox("Gender (1:Female, 2:Male)", [1,2])
cholesterol = st.selectbox("Cholesterol (1:Normal,2:Above,3:High)", [1,2,3])
gluc = st.selectbox("Glucose (1:Normal,2:Above,3:High)", [1,2,3])
smoke = st.selectbox("Smoking (0:No,1:Yes)", [0,1])
alco = st.selectbox("Alcohol (0:No,1:Yes)", [0,1])
active = st.selectbox("Physically Active (0:No,1:Yes)", [0,1])

if st.button("Predict"):

    height_m = height / 100
    bmi = weight / (height_m ** 2)

    input_data = np.array([
        age,
        height,
        weight,
        ap_hi,
        ap_lo,
        bmi,
        gender,
        cholesterol,
        gluc,
        smoke,
        alco,
        active
    ]).reshape(1, -1)

    # Scale first 6 numeric features
    input_data[:, :6] = scaler.transform(input_data[:, :6])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Cardiovascular Disease")
    else:
        st.success("✅ Low Risk of Cardiovascular Disease")

    st.write(f"Risk Probability: {probability:.2f}")