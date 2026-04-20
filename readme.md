                    ❤️ Cardiovascular Disease Risk Prediction System

         End-to-End Machine Learning System with Training, Inference & Deployment
                        Built using Scikit-learn & Streamlit


📌 OVERVIEW

This project implements a complete Machine Learning pipeline to predict the
likelihood of Cardiovascular Disease (CVD) using patient health data.

It demonstrates:

• Data preprocessing & feature engineering
• Model training & evaluation
• Model artifact management
• Production-ready inference pipeline
• Interactive Streamlit web application
• Cloud deployment
• Clean system architecture


🏗 SYSTEM ARCHITECTURE

The system is divided into two major components:

1. Offline Model Training Environment

• Dataset ingestion (Kaggle cardiovascular dataset)
• Data cleaning & validation
• Feature engineering (BMI, age in years)
• Feature scaling using StandardScaler
• Logistic Regression model training
• Model evaluation
• Saving trained model artifacts (.joblib)

2. Online Streamlit Application (Inference Layer)

• User-friendly patient input form
• Input validation & preprocessing
• BMI auto-calculation
• Feature scaling using saved scaler
• Real-time prediction & probability output
• Risk classification & visualization


🧠 MACHINE LEARNING MODEL

Algorithm Used:
• Logistic Regression

Why Logistic Regression?

• Interpretable model
• Probability output available
• Efficient for structured/tabular data
• Lightweight and fast for deployment
• Ideal for binary classification


📊 EVALUATION METRICS

• Accuracy
• Precision
• Recall
• F1 Score
• ROC-AUC Score
• Confusion Matrix


📊 DATASET INFORMATION

Source  : Kaggle Cardiovascular Dataset
Records : ~70,000 rows

Target Variable:

0 = No Cardiovascular Disease
1 = Cardiovascular Disease

Key Features:

• Age
• Height
• Weight
• Systolic Blood Pressure
• Diastolic Blood Pressure
• Cholesterol
• Glucose
• Smoking
• Alcohol Consumption
• Physical Activity


🚀 STREAMLIT WEB APPLICATION

The deployed application allows users to:

• Enter patient health details
• Automatically compute BMI
• View predicted disease probability
• Receive clear classification result
• Visualize risk score interactively


📂 PROJECT STRUCTURE

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


⚙ INSTALLATION & SETUP

1. Clone Repository

git clone <your-repository-url>
cd cardio-disease-predictor

2. Install Dependencies

pip install -r requirements.txt

3. Run Application

streamlit run app.py


☁ DEPLOYMENT

Deployed using:

• Streamlit Cloud

Optional deployment methods:

• Docker
• AWS EC2
• Google Cloud Platform


🏛 ARCHITECTURE BENEFITS

This architecture separates:

• Offline model training
• Model artifact storage
• Online inference pipeline
• Deployment layer

Benefits:

• Reproducibility
• Scalability
• Clean ML workflow
• Production readiness


🔮 FUTURE ENHANCEMENTS

• SHAP explainability integration
• REST API using FastAPI
• Docker containerization
• CI/CD automation
• Model versioning
• Monitoring & logging system
• Database-backed patient sessions


⚠ DISCLAIMER

This project is built for educational and demonstration purposes only.
It should not be used as a substitute for professional medical diagnosis.
