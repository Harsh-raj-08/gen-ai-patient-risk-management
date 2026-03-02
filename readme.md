Cardiovascular Disease Risk Prediction System

An end-to-end Machine Learning system that predicts the likelihood of cardiovascular disease using clinical and lifestyle data.

This project demonstrates:

Data preprocessing & feature engineering

Model training & evaluation

Model artifact management

Streamlit-based interactive UI

Cloud deployment

System design architecture

📌 Project Overview

Cardiovascular disease (CVD) is one of the leading causes of death globally.
This project builds a Logistic Regression model to predict whether a patient is likely to have cardiovascular disease based on medical indicators.

The system is divided into:

Offline Training Environment

Online Streamlit Inference Application

🏗 System Architecture

🔹 Offline Training Layer

Kaggle Cardiovascular Dataset

Data cleaning & validation

Feature engineering (BMI, age_years)

Feature scaling (StandardScaler)

Model training (Logistic Regression)

Model evaluation (Accuracy, ROC-AUC, Confusion Matrix)

Model artifacts saved:

logreg.joblib

scaler.joblib

features.json

🔹 Online Inference Layer (Streamlit App)

User-friendly patient input form

Real-time preprocessing

Feature scaling

Prediction & probability calculation

Risk visualization

Final disease classification

📊 Dataset

Source: Kaggle Cardiovascular Dataset

Records: 70,000+

Features:

Age

Height

Weight

Blood Pressure

Cholesterol

Glucose

Smoking

Alcohol

Physical activity

Target: cardio (0 = No disease, 1 = Disease)



🧠 Model Details
Algorithm Used

Logistic Regression (Primary Production Model)

Why Logistic Regression?

Interpretable

Probabilistic output

Suitable for binary classification

Efficient & lightweight for deployment

Evaluation Metrics

Accuracy

Precision

Recall

F1-score

ROC-AUC

🚀 Streamlit Web Application

The application allows users to:

Enter patient health data

Automatically compute BMI

Generate risk probability

View visual risk breakdown

Receive final disease prediction

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
1️⃣ Clone the repository
git clone <your-repo-url>
cd cardio-disease-predictor
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run locally
streamlit run app.py
☁ Deployment

Deployed using:

Streamlit Cloud

Optional:

Docker

AWS EC2

GCP


🔮 Future Improvements

SHAP explainability

Model versioning

REST API (FastAPI)

Docker containerization

CI/CD pipeline

Monitoring & logging
