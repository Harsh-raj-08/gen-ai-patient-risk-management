❤️ Cardiovascular Disease Risk Prediction System
<p align="center"> <b>End-to-End Machine Learning System with Training, Inference & Deployment</b><br> Built using Scikit-Learn & Streamlit </p>
📌 Overview

This project implements a complete Machine Learning pipeline to predict the likelihood of Cardiovascular Disease (CVD) using patient health data.

It demonstrates:

Data preprocessing & feature engineering

Model training & evaluation

Model artifact management

Production-ready inference pipeline

Streamlit web application

Cloud deployment

System design architecture

🏗 System Architecture

The system is divided into two major components:

🔹 1. Offline Model Training Environment

Dataset ingestion (Kaggle cardiovascular dataset)

Data cleaning & validation

Feature engineering (BMI, age in years)

Feature scaling (StandardScaler)

Logistic Regression model training

Model evaluation (Accuracy, ROC-AUC, Confusion Matrix)

Saving model artifacts (.joblib files)

🔹 2. Online Streamlit Application (Inference Layer)

User-friendly patient input form

Input validation & feature processing

BMI auto-calculation

Feature scaling using saved scaler

Real-time prediction & probability calculation

Risk classification & visualization

🧠 Machine Learning Model
🔍 Algorithm Used

Logistic Regression

✅ Why Logistic Regression?

Interpretable model

Provides probability output

Efficient for tabular data

Lightweight for deployment

Suitable for binary classification

📊 Evaluation Metrics

Accuracy

Precision

Recall

F1 Score

ROC-AUC

📊 Dataset Information

Source: Kaggle Cardiovascular Dataset

Records: ~70,000

Target Variable:

0 → No cardiovascular disease

1 → Cardiovascular disease

Key Features:

Age

Height

Weight

Blood Pressure (Systolic & Diastolic)

Cholesterol

Glucose

Smoking

Alcohol consumption

Physical activity

🚀 Streamlit Web Application

The deployed app allows users to:

Enter basic patient details

Automatically compute BMI

View predicted risk probability

Receive clear disease classification

Visualize risk distribution graphically

📂 Project Structure
gen-ai-patient-risk-management/
│
└── cardio-disease-predictor/
    │
    ├── app.py
    ├── requirements.txt
    ├── models/
    │   ├── cardio_logreg_basic.joblib
    │   ├── cardio_scaler_basic.joblib
    │   └── cardio_features_basic.json
⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone <your-repository-url>
cd cardio-disease-predictor
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Application
streamlit run app.py

The app will open in your browser automatically.

☁ Deployment

This project is deployed using:

Streamlit Cloud

Optional deployment methods:

Docker container

AWS EC2

Google Cloud Platform

🏛 Architecture Design

The architecture separates:

Offline model training

Model artifact storage

Online inference pipeline

Deployment layer

This separation ensures:

Reproducibility

Scalability

Clean ML workflow

Production readiness

🔮 Future Enhancements

SHAP explainability integration

REST API using FastAPI

Docker containerization

CI/CD automation

Model versioning

Monitoring & logging layer

Database-backed user session tracking

⚠ Disclaimer

This application is built for educational and demonstration purposes only.
It should not be used as a substitute for professional medical diagnosis.
