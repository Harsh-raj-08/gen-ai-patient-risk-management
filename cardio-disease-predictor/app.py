import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "cardio_logreg_basic.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "cardio_scaler_basic.joblib")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
# Load model & scaler
# model = joblib.load("models/cardio_logreg_basic.joblib")
# scaler = joblib.load("models/cardio_scaler_basic.joblib")

st.set_page_config(page_title="Cardio Risk Predictor", layout="wide")

# ---------- HEADER ----------
st.markdown("""
# ❤️ Cardiovascular Risk Predictor
### Simple AI-powered heart risk assessment
""")

st.divider()

# ---------- LAYOUT ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧍 Basic Information")
    age = st.slider("Age (years)", 18, 80, 30)
    gender = st.selectbox("Gender", ["Female", "Male"])
    height = st.slider("Height (cm)", 140, 210, 170)
    weight = st.slider("Weight (kg)", 40, 150, 70)

with col2:
    st.subheader("🫀 Health Metrics")
    ap_hi = st.slider("Systolic BP", 90, 200, 120)
    ap_lo = st.slider("Diastolic BP", 60, 140, 80)
    cholesterol = st.selectbox(
        "Cholesterol Level",
        ["Normal", "Above Normal", "High"]
    )

st.divider()

# ---------- ADVANCED MODE ----------
with st.expander("Advanced Options (Optional)"):
    gluc = st.selectbox("Glucose Level", [1,2,3], index=0)
    smoke = st.selectbox("Smoking", [0,1], index=0)
    alco = st.selectbox("Alcohol", [0,1], index=0)
    active = st.selectbox("Physically Active", [0,1], index=1)

# Convert categorical to numeric
gender_val = 1 if gender == "Female" else 2
chol_map = {"Normal":1, "Above Normal":2, "High":3}
chol_val = chol_map[cholesterol]

# ---------- PREDICT ----------
if st.button("🔍 Assess Risk", use_container_width=True):

    height_m = height / 100
    bmi = weight / (height_m ** 2)

    input_data = np.array([
        age,
        height,
        weight,
        ap_hi,
        ap_lo,
        bmi,
        gender_val,
        chol_val,
        gluc,
        smoke,
        alco,
        active
    ]).reshape(1, -1)

    input_data[:, :6] = scaler.transform(input_data[:, :6])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

  v
    st.subheader("📊 Assessment Result")

    # ---------------- FINAL DISEASE RESULT ----------------
    if prediction == 1:
        st.error("⚠️ The model predicts that the patient is likely to have Cardiovascular Disease.")
        final_text = "LIKELY TO HAVE DISEASE"
    else:
        st.success("✅ The model predicts that the patient is NOT likely to have Cardiovascular Disease.")
        final_text = "NOT LIKELY TO HAVE DISEASE"

    # Risk Percentage
    st.metric("Predicted Risk Probability", f"{probability*100:.1f}%")

    # ---------------- RISK BAR GRAPH ----------------
    st.subheader("Risk Distribution")
    chart_data = {
        "Category": ["No Disease", "Disease"],
        "Probability": [1 - probability, probability]
    }

    st.bar_chart(chart_data, x="Category", y="Probability")

    # ---------------- SUMMARY BOX ----------------
    st.divider()
    st.markdown("### 🩺 Final Summary")

    st.info(f"""
    • **Final Prediction:** {final_text}  
    • **Risk Score:** {probability*100:.1f}%  
    • **Calculated BMI:** {bmi:.2f}  

    ⚠️ This prediction is based on a machine learning model and should not replace professional medical advice.
    """)
