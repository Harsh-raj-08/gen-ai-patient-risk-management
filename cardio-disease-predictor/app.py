"""
CardioAI — Agentic AI Health Support Assistant
Milestone 2: Full UI redesign + LangGraph multi-agent system

Pages: Landing → 3-step Intake Wizard → Results Dashboard → AI Chat
"""

import streamlit as st
import joblib
import numpy as np
import json
import os

# ─────────────────────────────────────────────────────────
# Config & paths
# ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cardio_logreg_basic.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "cardio_scaler_basic.joblib")

st.set_page_config(
    page_title="CardioAI — Heart Risk Intelligence",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────
# Design system — custom CSS
# ─────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&display=swap');

/* Reset and base */
* { font-family: 'Outfit', sans-serif !important; }
[data-testid="stAppViewContainer"] { 
    background: linear-gradient(135deg, #FAF8F5 0%, #EAE4DF 100%) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="block-container"] { padding: 2rem 2rem 4rem !important; max-width: 780px !important; }

/* Hide default streamlit elements */
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }

/* Typography */
h1 { color: #4A3E39 !important; font-size: 32px !important; font-weight: 600 !important; letter-spacing: -0.5px; }
h2 { color: #5C4F4A !important; font-size: 22px !important; font-weight: 500 !important; letter-spacing: -0.3px; }
h3 { color: #5C4F4A !important; font-size: 17px !important; font-weight: 500 !important; }
p, li, label { color: #695F5A !important; letter-spacing: 0.2px; }
.muted { color: #8A7F7A !important; font-size: 13px !important; }

/* Cards & Glassmorphism */
.card {
    background: rgba(255, 255, 255, 0.7); 
    border: 1px solid rgba(255, 255, 255, 0.5);
    backdrop-filter: blur(12px);
    border-radius: 16px; padding: 24px; margin-bottom: 16px;
    box-shadow: 0 10px 30px rgba(92, 79, 74, 0.04);
}
.card-cream {
    background: rgba(245, 232, 218, 0.4); 
    border: 1px solid rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(12px);
    border-radius: 16px; padding: 24px; margin-bottom: 16px;
    box-shadow: 0 10px 30px rgba(92, 79, 74, 0.05);
}

/* Primary button - Premium aesthetic */
.stButton > button {
    background: linear-gradient(135deg, #D4A373 0%, #C9996B 100%) !important;
    color: white !important;
    border: none !important; border-radius: 12px !important;
    font-weight: 500 !important; padding: 12px 28px !important;
    font-size: 15px !important; letter-spacing: 0.3px !important;
    box-shadow: 0 8px 20px rgba(201, 153, 107, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
}
.stButton > button:hover { 
    transform: translateY(-2px);
    box-shadow: 0 12px 25px rgba(201, 153, 107, 0.4) !important;
    background: linear-gradient(135deg, #dbb085 0%, #d1a478 100%) !important;
}

/* Sliders */
[data-testid="stSlider"] > div > div > div > div { background: #C9996B !important; }
.stSlider [data-testid="stThumbValue"] { color: #5C4F4A !important; font-weight: 500 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.8); border: 1px solid rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(10px);
    border-radius: 14px; padding: 16px;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.03);
}
[data-testid="stMetricLabel"] { color: #8A7F7A !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 0.5px; }
[data-testid="stMetricValue"] { color: #4A3E39 !important; font-weight: 600 !important; }

/* Alerts */
[data-testid="stAlert"] { border-radius: 12px !important; border: none !important; box-shadow: 0 4px 12px rgba(0,0,0,0.02); }

/* Step badges */
.step-badge {
    display: inline-flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #D4A373, #C9996B); color: white;
    border-radius: 50%; width: 28px; height: 28px; 
    font-size: 13px; font-weight: 600; margin-right: 6px; box-shadow: 0 4px 10px rgba(201,153,107,0.3);
}
.step-badge-done { 
    display: inline-flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #7A9B8F, #5C766D); color: white;
    border-radius: 50%; width: 28px; height: 28px; 
    font-size: 13px; font-weight: 600; margin-right: 6px; box-shadow: 0 4px 10px rgba(92,118,109,0.3);
}
.step-badge-pending { 
    display: inline-flex; align-items: center; justify-content: center;
    background: #EAE4DF; color: #A39B96;
    border-radius: 50%; width: 28px; height: 28px; 
    font-size: 13px; font-weight: 600; margin-right: 6px;
}

/* Agent tag */
.agent-tag {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(92, 118, 109, 0.1); color: #5C766D; padding: 6px 14px;
    border-radius: 20px; font-size: 12px; font-weight: 500; border: 1px solid rgba(92, 118, 109, 0.2);
}

/* Risk tags */
.risk-high { background: rgba(220, 38, 38, 0.1); color: #DC2626; padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 600; }
.risk-low  { background: rgba(92, 118, 109, 0.1); color: #5C766D; padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 600; }

/* Chat bubbles */
.chat-user {
    background: linear-gradient(135deg, #695F5A, #5C4F4A); color: white !important; 
    border-radius: 18px 18px 4px 18px; padding: 12px 18px; font-size: 14.5px; line-height: 1.6;
    box-shadow: 0 4px 15px rgba(92, 79, 74, 0.15);
}
.chat-user p { color: white !important; }
.chat-agent {
    background: rgba(255, 255, 255, 0.8); color: #4A3E39; border-radius: 18px 18px 18px 4px;
    padding: 12px 18px; font-size: 14.5px; border: 1px solid rgba(255, 255, 255, 0.5); line-height: 1.6;
    backdrop-filter: blur(10px); box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
}

/* Checkbox/Toggles */
[data-testid="stCheckbox"] label span { color: #5C4F4A !important; font-weight: 500; }

/* Divider */
hr { border-color: rgba(201, 153, 107, 0.2) !important; margin: 2rem 0 !important; }

/* Evidence note */
.evidence-note {
    background: rgba(92, 118, 109, 0.05); border: 1px solid rgba(92, 118, 109, 0.15); border-radius: 12px;
    padding: 10px 16px; font-size: 12.5px; color: #5C766D; margin-top: 6px; margin-bottom: 16px;
    font-weight: 400;
}

/* Feature badge */
.feature-badge {
    background: #fff; border: 1px solid #E0DBD6; padding: 4px 14px;
    border-radius: 20px; font-size: 12px; color: #5C4F4A; display: inline-block; margin: 3px;
}

/* BMI tag */
.bmi-normal { background: #E8F0EE; color: #5C766D; padding: 3px 12px; border-radius: 20px; font-size: 12px; }
.bmi-warning { background: #FFF3E0; color: #E67E22; padding: 3px 12px; border-radius: 20px; font-size: 12px; }
.bmi-danger { background: #FDE8E8; color: #C0392B; padding: 3px 12px; border-radius: 20px; font-size: 12px; }

/* BP info card */
.bp-card {
    background: #F5E8DA; border: 1px solid #E8D5C0; border-radius: 10px;
    padding: 10px 14px; font-size: 12px; color: #8A6A4A; margin-top: 8px; margin-bottom: 8px;
}
</style>
"""


# ─────────────────────────────────────────────────────────
# RAG auto-initialization
# ─────────────────────────────────────────────────────────
def initialize_rag_if_needed():
    if not st.session_state.get("rag_initialized"):
        chroma_path = os.path.join(BASE_DIR, "chroma_db")
        if not os.path.exists(chroma_path) or not os.listdir(chroma_path):
            with st.spinner("Initializing knowledge base (first run only)..."):
                from setup_rag import build_vector_store
                build_vector_store()
        st.session_state["rag_initialized"] = True


# ─────────────────────────────────────────────────────────
# ML prediction (preserves Milestone 1 logic exactly)
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_ml_models():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def run_ml_prediction(patient_data: dict) -> tuple:
    """
    Run the Milestone 1 logistic regression prediction.
    Returns (prediction, probability).
    """
    model, scaler = load_ml_models()

    input_data = np.array([
        patient_data["age"],
        patient_data["height"],
        patient_data["weight"],
        patient_data["ap_hi"],
        patient_data["ap_lo"],
        patient_data["bmi"],
        patient_data["gender"],
        patient_data["cholesterol"],
        patient_data["gluc"],
        patient_data["smoke"],
        patient_data["alco"],
        patient_data["active"],
    ]).reshape(1, -1)

    input_data[:, :6] = scaler.transform(input_data[:, :6])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return int(prediction), float(probability)


# ─────────────────────────────────────────────────────────
# Step indicator component
# ─────────────────────────────────────────────────────────
def render_step_indicator(current_step):
    steps = ["Basic info", "Vitals & labs", "Lifestyle"]
    cols = st.columns(3)
    for i, (col, label) in enumerate(zip(cols, steps)):
        step_num = i + 1
        if step_num < current_step:
            badge_class = "step-badge-done"
            symbol = "✓"
        elif step_num == current_step:
            badge_class = "step-badge"
            symbol = str(step_num)
        else:
            badge_class = "step-badge-pending"
            symbol = str(step_num)
        with col:
            st.markdown(
                f'<div style="text-align:center">'
                f'<span class="{badge_class}">{symbol}</span><br>'
                f'<span style="font-size:12px;color:#8A7F7A;">{label}</span></div>',
                unsafe_allow_html=True,
            )

    # Progress bar
    progress = (current_step - 1) / 2
    st.markdown(
        f'<div style="background:#E0DBD6;height:4px;border-radius:2px;margin:12px 0 24px;">'
        f'<div style="background:#C9996B;height:100%;width:{progress*100:.0f}%;border-radius:2px;'
        f'transition:width 0.3s ease;"></div></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────
# Critical values check
# ─────────────────────────────────────────────────────────
def check_critical_values(patient_data):
    warnings = []
    if patient_data["ap_hi"] >= 180 or patient_data["ap_lo"] >= 120:
        warnings.append(
            "🚨 HYPERTENSIVE CRISIS: Systolic ≥180 or diastolic ≥120 mmHg. "
            "Seek emergency medical care immediately."
        )
    if patient_data["ap_hi"] >= 160:
        warnings.append(
            "⚠️ Stage 2 hypertension detected. Medical consultation strongly recommended."
        )
    if patient_data["bmi"] >= 40:
        warnings.append(
            "⚠️ Severe obesity (Class III) detected. Comprehensive weight management program recommended."
        )
    return warnings


# ═════════════════════════════════════════════════════════
# PAGE 1: LANDING
# ═════════════════════════════════════════════════════════
def render_landing():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Hero section
    st.markdown(
        """
        <div class="card-cream" style="text-align:center; padding: 48px 32px;">
            <div style="font-size:36px;margin-bottom:8px;">🫀</div>
            <div style="background:#F5E8DA;display:inline-block;padding:5px 16px;
                        border-radius:20px;font-size:12px;color:#C9996B;margin-bottom:16px;
                        font-weight:500;">
                Cardiovascular Risk Intelligence
            </div>
            <h1 style="font-size:32px !important;line-height:1.3;margin-bottom:12px;">
                Know your heart risk.<br>Get a plan to fix it.
            </h1>
            <p class="muted" style="max-width:420px;margin:0 auto 8px;line-height:1.7;">
                Enter your health data once. Our AI analyzes your risk using clinical ML,
                retrieves evidence-based guidelines, and builds a personalized prevention plan.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Single CTA button to Chat with AI Agent
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Check your health with AI →", use_container_width=True, key="cta_start"):
            st.session_state["page"] = "intake"
            st.rerun()

    # Disclaimer
    st.markdown(
        '<p class="muted" style="text-align:center;font-size:11px;margin-top:16px;">'
        "For educational purposes only. Not a substitute for professional medical advice.</p>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════
# PAGE 2: PATIENT INTAKE WIZARD
# ═════════════════════════════════════════════════════════
def render_intake():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize wizard state
    if "intake_step" not in st.session_state:
        st.session_state["intake_step"] = 1
    if "patient_data" not in st.session_state:
        st.session_state["patient_data"] = {}

    step = st.session_state["intake_step"]

    # Header
    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
        '<span style="font-size:20px;">🫀</span>'
        '<span style="font-size:16px;font-weight:500;color:#5C4F4A;">CardioAI</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("## Health Assessment")
    render_step_indicator(step)

    # ───── Step 1: Basic Information ─────
    if step == 1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🧍 Basic Information")

        age = st.slider("Age (years)", 18, 80, 45, key="inp_age")

        st.markdown("**Gender**")
        gender_choice = st.radio(
            "Gender",
            ["Female", "Male"],
            horizontal=True,
            label_visibility="collapsed",
            key="inp_gender",
        )

        height = st.slider("Height (cm)", 130, 210, 168, key="inp_height")
        weight = st.slider("Weight (kg)", 40, 160, 75, key="inp_weight")

        # Live BMI preview
        bmi = weight / (height / 100) ** 2
        if bmi < 18.5:
            bmi_cat, bmi_class = "Underweight", "bmi-warning"
        elif bmi < 25:
            bmi_cat, bmi_class = "Normal", "bmi-normal"
        elif bmi < 30:
            bmi_cat, bmi_class = "Overweight", "bmi-warning"
        else:
            bmi_cat, bmi_class = "Obese", "bmi-danger"

        st.markdown(
            f'<div style="margin-top:8px;">BMI: <strong>{bmi:.1f}</strong> '
            f'<span class="{bmi_class}">{bmi_cat}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Navigation
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("← Back to home", key="back_to_home"):
                st.session_state["page"] = "landing"
                st.rerun()
        with col_b:
            if st.button("Continue →", key="step1_next", use_container_width=True):
                st.session_state["patient_data"].update({
                    "age": age,
                    "gender": 1 if gender_choice == "Female" else 2,
                    "height": height,
                    "weight": weight,
                    "bmi": bmi,
                })
                st.session_state["intake_step"] = 2
                st.rerun()

    # ───── Step 2: Vitals & Labs ─────
    elif step == 2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🫀 Vitals & Lab Results")

        ap_hi = st.slider("Systolic blood pressure (mmHg)", 80, 220, 120, key="inp_ap_hi")
        ap_lo = st.slider("Diastolic blood pressure (mmHg)", 50, 140, 80, key="inp_ap_lo")

        # BP validation
        if ap_lo >= ap_hi:
            st.warning("⚠️ Diastolic BP should be lower than systolic BP.")

        # BP category card
        if ap_hi < 120 and ap_lo < 80:
            bp_cat, bp_color = "Normal", "#5C766D"
        elif ap_hi < 130 and ap_lo < 80:
            bp_cat, bp_color = "Elevated", "#E67E22"
        elif ap_hi < 140 or ap_lo < 90:
            bp_cat, bp_color = "Stage 1 Hypertension", "#E67E22"
        elif ap_hi < 180 and ap_lo < 120:
            bp_cat, bp_color = "Stage 2 Hypertension", "#C0392B"
        else:
            bp_cat, bp_color = "Hypertensive Crisis", "#C0392B"

        st.markdown(
            f'<div class="bp-card">'
            f'BP Category: <strong style="color:{bp_color}">{bp_cat}</strong> '
            f"({ap_hi}/{ap_lo} mmHg)</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Cholesterol
        st.markdown("**Cholesterol level**")
        chol_options = ["Normal", "Above Normal", "High"]
        cholesterol = st.select_slider(
            "Cholesterol",
            options=chol_options,
            value="Normal",
            label_visibility="collapsed",
            key="inp_chol",
        )
        chol_val = chol_options.index(cholesterol) + 1

        # Glucose
        st.markdown("**Glucose level**")
        gluc_options = ["Normal", "Above Normal", "High"]
        glucose = st.select_slider(
            "Glucose",
            options=gluc_options,
            value="Normal",
            label_visibility="collapsed",
            key="inp_gluc",
        )
        gluc_val = gluc_options.index(glucose) + 1

        st.markdown("</div>", unsafe_allow_html=True)

        # Navigation
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("← Back", key="step2_back"):
                st.session_state["intake_step"] = 1
                st.rerun()
        with col_b:
            if st.button("Continue →", key="step2_next", use_container_width=True):
                st.session_state["patient_data"].update({
                    "ap_hi": ap_hi,
                    "ap_lo": ap_lo,
                    "cholesterol": chol_val,
                    "gluc": gluc_val,
                })
                st.session_state["intake_step"] = 3
                st.rerun()

    # ───── Step 3: Lifestyle ─────
    elif step == 3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🏃 Lifestyle Factors")

        smoke = st.toggle("🚬 Smoker", value=False, key="inp_smoke")
        if smoke:
            st.markdown(
                '<div class="evidence-note">Smoking increases CVD risk by 2-4x — WHO 2023</div>',
                unsafe_allow_html=True,
            )

        alco = st.toggle("🍷 Regular alcohol consumption", value=False, key="inp_alco")
        if alco:
            st.markdown(
                '<div class="evidence-note">Heavy drinking is associated with hypertension and cardiomyopathy — ESC 2023</div>',
                unsafe_allow_html=True,
            )

        active = st.toggle(
            "🏃 Physically active (150+ min/week)", value=True, key="inp_active"
        )
        if active:
            st.markdown(
                '<div class="evidence-note">Regular exercise reduces CVD risk by 20-30% — WHO 2023</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="evidence-note">⚠️ Physical inactivity is a major modifiable CVD risk factor — AHA 2023</div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Navigation
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("← Back", key="step3_back"):
                st.session_state["intake_step"] = 2
                st.rerun()
        with col_b:
            if st.button("Analyze My Risk →", key="step3_submit", use_container_width=True):
                st.session_state["patient_data"].update({
                    "smoke": 1 if smoke else 0,
                    "alco": 1 if alco else 0,
                    "active": 1 if active else 0,
                })

                # Run ML prediction
                pd = st.session_state["patient_data"]
                prediction, probability = run_ml_prediction(pd)
                st.session_state["ml_result"] = {
                    "prediction": prediction,
                    "probability": probability,
                }

                st.session_state["intake_step"] = 1  # Reset for next time
                st.session_state["page"] = "results"
                st.rerun()


# ═════════════════════════════════════════════════════════
# PAGE 3: RESULTS DASHBOARD
# ═════════════════════════════════════════════════════════
def render_results():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    patient_data = st.session_state.get("patient_data", {})
    ml_result = st.session_state.get("ml_result", {})
    prediction = ml_result.get("prediction", 0)
    probability = ml_result.get("probability", 0.0)

    # Header
    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
        '<span style="font-size:20px;">🫀</span>'
        '<span style="font-size:16px;font-weight:500;color:#5C4F4A;">CardioAI</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("## Your Results")

    # ──── Section A: ML Risk Score ────
    risk_pct = probability * 100
    color = "#C0392B" if prediction == 1 else "#5C766D"
    tag = "High Risk" if prediction == 1 else "Low Risk"
    tag_class = "risk-high" if prediction == 1 else "risk-low"

    st.markdown(
        f"""<div class="card-cream" style="text-align:center;padding:32px;">
            <p class="muted" style="margin-bottom:4px;">Cardiovascular risk probability</p>
            <div style="font-size:52px;font-weight:600;color:{color};line-height:1;margin-bottom:8px;">
                {risk_pct:.1f}%
            </div>
            <span class="{tag_class}">{tag}</span>
            <div style="background:#E0DBD6;border-radius:6px;height:10px;margin:20px 32px 0;">
                <div style="background:linear-gradient(to right,#5C766D,#C9996B,#C0392B);
                            width:{min(risk_pct, 100):.0f}%;height:100%;border-radius:6px;
                            transition:width 0.5s ease;"></div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Metric cards
    bmi = patient_data.get("bmi", 0)
    bmi_cat = (
        "Underweight" if bmi < 18.5
        else "Normal" if bmi < 25
        else "Overweight" if bmi < 30
        else "Obese"
    )
    bp_cat = (
        "Normal" if patient_data.get("ap_hi", 0) < 120
        else "Elevated" if patient_data.get("ap_hi", 0) < 130
        else "Stage 1" if patient_data.get("ap_hi", 0) < 140
        else "Stage 2"
    )
    chol_labels = {1: "Normal", 2: "Above Normal", 3: "High"}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BMI", f"{bmi:.1f}", bmi_cat)
    c2.metric("Blood Pressure", f"{patient_data.get('ap_hi', 0)}/{patient_data.get('ap_lo', 0)}", bp_cat)
    c3.metric("Age", f"{patient_data.get('age', 0)} yrs")
    c4.metric("Cholesterol", chol_labels.get(patient_data.get("cholesterol", 1), "Normal"))

    # ──── Section B: Emergency Escalation ────
    critical = check_critical_values(patient_data)
    for w in critical:
        if "🚨" in w:
            st.error(w)
        else:
            st.warning(w)

    st.markdown("---")

    # ──── Section C: AI Health Report ────
    if "health_report" not in st.session_state:
        st.markdown("### 🤖 AI Health Report")
        st.markdown(
            '<p class="muted">Generate a comprehensive, evidence-based health report using our multi-agent AI system.</p>',
            unsafe_allow_html=True,
        )

        if st.button("Generate AI Health Report →", key="gen_report", use_container_width=True):
            initialize_rag_if_needed()

            from agent.graph import run_agent_graph

            status_placeholder = st.empty()

            with st.spinner(""):
                status_placeholder.markdown(
                    '<div class="agent-tag">● Checking critical values...</div>',
                    unsafe_allow_html=True,
                )
                import time
                time.sleep(0.5)

                status_placeholder.markdown(
                    '<div class="agent-tag">● Analyzing risk profile...</div>',
                    unsafe_allow_html=True,
                )

                ml_pred = {"prediction": prediction, "probability": probability}
                result = run_agent_graph(patient_data, ml_pred)

                status_placeholder.markdown(
                    '<div class="agent-tag">● Retrieving clinical guidelines...</div>',
                    unsafe_allow_html=True,
                )
                time.sleep(0.3)

                status_placeholder.markdown(
                    '<div class="agent-tag">● Generating health report...</div>',
                    unsafe_allow_html=True,
                )
                time.sleep(0.3)

                status_placeholder.markdown(
                    '<div class="agent-tag">● Validating output...</div>',
                    unsafe_allow_html=True,
                )
                time.sleep(0.3)

            status_placeholder.empty()

            st.session_state["health_report"] = result["health_report"]
            st.session_state["agent_status"] = result["status"]
            st.session_state["agent_errors"] = result.get("error_log", [])
            st.rerun()

    # Display report if generated
    if "health_report" in st.session_state:
        report = st.session_state["health_report"]

        # Status indicator
        status = st.session_state.get("agent_status", "completed")
        if status == "completed":
            st.markdown(
                '<div class="agent-tag" style="margin-bottom:12px;">✓ Report generated successfully</div>',
                unsafe_allow_html=True,
            )
        elif status == "partial":
            st.warning("⚠️ Report generated with partial data. Some AI analysis was unavailable.")
        elif status == "failed":
            st.error("❌ AI analysis failed. Showing fallback recommendations.")

        # Risk summary card
        st.markdown(
            f"""<div class="card">
                <h3 style="margin-bottom:8px;">📊 Risk Summary</h3>
                <p style="line-height:1.7;font-size:14px;">{report.get('risk_summary', 'N/A')}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        # Risk factor tags
        risk_factors = report.get("key_risk_factors", [])
        if risk_factors:
            tags_html = " ".join(
                f'<span style="background:#FDE8E8;color:#C0392B;padding:3px 10px;'
                f'border-radius:20px;font-size:12px;margin:2px;display:inline-block;">{f}</span>'
                for f in risk_factors
            )
            st.markdown(
                f'<div style="margin-bottom:16px;">'
                f'<span class="muted" style="font-size:12px;">Key risk factors: </span>{tags_html}</div>',
                unsafe_allow_html=True,
            )

        # Recommendations
        recs = report.get("recommendations", [])
        if recs:
            st.markdown("### 📋 Personalized Recommendations")
            for i, rec in enumerate(recs, 1):
                st.markdown(
                    f"""<div class="card" style="display:flex;gap:12px;padding:14px 18px;">
                        <div style="background:#F5E8DA;color:#C9996B;width:24px;height:24px;
                                    border-radius:50%;text-align:center;line-height:24px;
                                    font-size:12px;font-weight:600;flex-shrink:0;">{i}</div>
                        <p style="margin:0;font-size:14px;line-height:1.6;">{rec}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # Lifestyle changes
        lifestyle = report.get("lifestyle_changes", [])
        if lifestyle:
            st.markdown("### 🏃 Lifestyle Changes")
            for change in lifestyle:
                st.markdown(
                    f'<div class="evidence-note">✓ {change}</div>',
                    unsafe_allow_html=True,
                )

        # Follow-up actions
        followup = report.get("follow_up_actions", [])
        if followup:
            st.markdown("### 📅 Follow-up Actions")
            for action in followup:
                st.markdown(
                    f'<div class="card" style="padding:12px 18px;"><p style="margin:0;font-size:14px;">→ {action}</p></div>',
                    unsafe_allow_html=True,
                )

        # Sources
        sources = report.get("sources", [])
        if sources:
            st.markdown("**Sources:**")
            src_html = " ".join(
                f'<span class="agent-tag" style="margin:2px;">{src}</span>'
                for src in sources
            )
            st.markdown(src_html, unsafe_allow_html=True)

        # Disclaimer
        disclaimer = report.get("disclaimer", "")
        if disclaimer:
            st.markdown(
                f'<div style="background:#FFF8F0;border:1px solid #F0D9BE;border-radius:10px;'
                f'padding:12px;font-size:12px;color:#8A6A4A;margin-top:16px;">{disclaimer}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ──── Section D: What-If Simulator ────
    st.markdown("### 🔄 What-if Scenario Simulator")
    st.markdown(
        '<p class="muted">Explore how lifestyle changes affect your cardiovascular risk.</p>',
        unsafe_allow_html=True,
    )

    scenario_col1, scenario_col2 = st.columns(2)
    with scenario_col1:
        wt_change = st.slider("Weight change (kg)", -30, 10, 0, key="wt_delta")
        bp_change = st.slider("Systolic BP change (mmHg)", -40, 20, 0, key="bp_delta")
    with scenario_col2:
        quit_smoking = st.checkbox("Quit smoking", key="quit_smoke")
        start_exercise = st.checkbox("Start exercising (150+ min/week)", key="start_ex")

    if st.button("Simulate Scenario →", key="sim_btn"):
        # Build modified patient data
        modified = patient_data.copy()
        modified["weight"] = max(40, patient_data.get("weight", 75) + wt_change)
        modified["ap_hi"] = max(80, patient_data.get("ap_hi", 120) + bp_change)
        modified["bmi"] = modified["weight"] / (patient_data.get("height", 168) / 100) ** 2
        if quit_smoking:
            modified["smoke"] = 0
        if start_exercise:
            modified["active"] = 1

        # Re-run ML prediction
        new_pred, new_prob = run_ml_prediction(modified)

        # Show delta comparison
        delta = new_prob - probability
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Original Risk", f"{probability * 100:.1f}%")
        col_m2.metric("New Risk", f"{new_prob * 100:.1f}%", f"{delta * 100:+.1f}%")

        # Run agent for scenario-specific advice
        if "health_report" in st.session_state:
            initialize_rag_if_needed()
            from agent.graph import run_agent_graph

            with st.spinner("Agent analyzing your scenario..."):
                scenario_result = run_agent_graph(
                    modified,
                    {"prediction": new_pred, "probability": new_prob},
                    context="What-if scenario analysis",
                )
            scenario_report = scenario_result["health_report"]
            st.markdown(
                f'<div class="card"><h3>Scenario Analysis</h3>'
                f'<p style="line-height:1.7;">{scenario_report.get("risk_summary", "N/A")}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Navigation
    col_n1, col_n2 = st.columns(2)
    with col_n1:
        if st.button("← Edit Inputs", key="back_to_intake"):
            st.session_state["page"] = "intake"
            st.rerun()
    with col_n2:
        if st.button("Chat with AI →", key="go_to_chat"):
            st.session_state["page"] = "chat"
            st.rerun()


# ═════════════════════════════════════════════════════════
# PAGE 4: AI CHAT
# ═════════════════════════════════════════════════════════
def render_chat():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
        '<span style="font-size:20px;">🫀</span>'
        '<span style="font-size:16px;font-weight:500;color:#5C4F4A;">CardioAI</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("## Health Assistant Chat")

    # Context banner
    st.markdown(
        """<div class="card-cream" style="display:flex;align-items:center;gap:12px;padding:14px 18px;">
            <span class="agent-tag">● Agent ready</span>
            <p style="margin:0;font-size:13px;">The agent has your full health profile and report. Ask anything.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # Check prerequisites
    if "ml_result" not in st.session_state:
        st.warning("Please complete the health assessment first.")
        if st.button("← Go to Assessment"):
            st.session_state["page"] = "intake"
            st.rerun()
        return

    # Initialize chat history with greeting
    if "chat_history" not in st.session_state:
        risk_pct = st.session_state["ml_result"]["probability"] * 100
        prediction = st.session_state["ml_result"]["prediction"]
        risk_word = "elevated" if prediction == 1 else "within a lower range"

        greeting = (
            f"Hello! I've analyzed your cardiovascular risk report. Your {risk_pct:.1f}% risk score "
            f"is {risk_word}. I can answer questions about your specific results, "
            f"lifestyle changes, or general heart health. What would you like to know?"
        )
        st.session_state["chat_history"] = [
            {"role": "assistant", "content": greeting}
        ]

    # Render chat history
    for msg in st.session_state["chat_history"]:
        css_class = "chat-user" if msg["role"] == "user" else "chat-agent"
        align = "flex-end" if msg["role"] == "user" else "flex-start"
        st.markdown(
            f'<div style="display:flex;justify-content:{align};margin-bottom:10px;">'
            f'<div class="{css_class}" style="max-width:80%;">{msg["content"]}</div></div>',
            unsafe_allow_html=True,
        )

    # Quick prompt chips
    st.markdown("**Quick questions:**")
    chip_cols = st.columns(3)
    quick_prompts = [
        "What foods should I avoid?",
        "Best exercise for my heart?",
        "When should I see a doctor?",
    ]
    for col, prompt in zip(chip_cols, quick_prompts):
        with col:
            if st.button(prompt, key=f"chip_{prompt}"):
                handle_chat_message(prompt)

    # Chat input
    user_input = st.chat_input("Ask about your health report...")
    if user_input:
        handle_chat_message(user_input)

    st.markdown("---")

    # Back button
    if st.button("← Back to Results", key="back_to_results"):
        st.session_state["page"] = "results"
        st.rerun()


def handle_chat_message(message: str):
    """Process a chat message through the chat agent."""
    st.session_state["chat_history"].append({"role": "user", "content": message})

    try:
        from agent.chat_node import build_chat_context, run_chat_agent

        patient_data = st.session_state.get("patient_data", {})
        risk_score = st.session_state["ml_result"]["probability"] * 100
        health_report = st.session_state.get("health_report", {})
        chat_history = st.session_state["chat_history"]

        context = build_chat_context(
            patient_data=patient_data,
            risk_score=risk_score,
            health_report=health_report,
            chat_history=chat_history,
            user_question=message,
        )

        response = run_chat_agent(context)
    except Exception as e:
        response = (
            "I'm having trouble connecting to the AI service right now. "
            "Your health data and report are still available on the Results page. "
            "Please try again in a moment."
        )

    st.session_state["chat_history"].append({"role": "assistant", "content": response})
    st.rerun()


# ═════════════════════════════════════════════════════════
# MAIN — Page router
# ═════════════════════════════════════════════════════════
def main():
    if "page" not in st.session_state:
        st.session_state["page"] = "landing"

    page = st.session_state["page"]

    if page == "landing":
        render_landing()
    elif page == "intake":
        render_intake()
    elif page == "results":
        render_results()
    elif page == "chat":
        render_chat()
    else:
        st.session_state["page"] = "landing"
        render_landing()


if __name__ == "__main__":
    main()
