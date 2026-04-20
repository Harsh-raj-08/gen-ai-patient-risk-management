# Cardiovascular Risk Intelligence System

This project is a multi-agent AI web application designed to predict cardiovascular disease risk and provide clinical evidence-based recommendations. It takes patient health data, predicts an initial risk score using a machine learning model, and then uses a LangGraph-based Retrieval-Augmented Generation (RAG) pipeline to curate personalized health advice.

## Features
- **Predictive ML Model:** A Logistic Regression model trained on patient records predicts cardiovascular disease probability based on 12 features (Age, Gender, Height, Weight, BMI, Blood Pressure, Cholesterol, Glucose, Smoking, Alcohol, Activity).
- **RAG Pipeline:** Interrogates clinical guidelines (WHO, AHA, ESC) from a customized vector store (ChromaDB) to ground AI responses in actual medical science.
- **Agentic AI Architecture:** Implemented via LangGraph. Different agents handle specific phases: Risk Analysis, Guideline Retrieval, Report Generation, and Quality Checking.
- **Modern User Interface:** Built with Streamlit, styled with custom CSS to provide an intuitive glassmorphic UI.
- **Contextual Chat:** Users can follow up with their AI Health Assistant directly inside the dashboard.

## Tech Stack
- **Frontend / UI:** Streamlit
- **Machine Learning:** Scikit-Learn (Logistic Regression, StandardScaler) 
- **LLM Orchestration:** LangChain & LangGraph
- **Vector Database:** ChromaDB 
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **LLM Provider:** Groq / Google Gemini 

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd cardio-disease-predictor
   ```

2. **Install dependencies:**
   Make sure you have Python 3.9+ installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API Keys:**
   Copy the example environment variables file and add your API key for Groq or Gemini.
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and fill in `GROQ_API_KEY` (or `GOOGLE_API_KEY`).

4. **Initialize RAG database:**
   Run this once to build the local vector embeddings from the medical guidelines:
   ```bash
   python setup_rag.py
   ```

5. **Start the application:**
   ```bash
   python -m streamlit run app.py
   ```

## Project Structure
- `app.py`: Main entry point containing the Streamlit application logic and routing.
- `agent/`: Contains the LangGraph configuration, node logic, TypedDict states, and prompt templates.
- `rag/`: Handles vector storage, document chunks, and the retrieval logic.
- `models/`: Pre-trained `scikit-learn` artifacts for the initial risk prediction.

## Disclaimer
This project is for educational purposes only and should not be used as a substitute for professional medical advice.
