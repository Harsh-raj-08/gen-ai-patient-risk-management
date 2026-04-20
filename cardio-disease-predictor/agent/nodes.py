"""
LangGraph node functions — 5 nodes for the cardiovascular risk agent pipeline.
Each node receives AgentState, performs its task, and returns an updated state dict.

Node execution order:
  emergency_checker → risk_analyzer → guideline_retriever → report_generator → quality_checker
"""

import json
import os
from dotenv import load_dotenv
from agent.state import AgentState
from agent.prompts import SYSTEM_BASE, RISK_ANALYZER_PROMPT, REPORT_GENERATOR_PROMPT
from rag.retriever import retrieve_guidelines

load_dotenv()

# ─────────────────────────────────────────────────
# LLM provider setup (Groq or Gemini)
# ─────────────────────────────────────────────────

def _get_llm():
    """Return the configured LLM based on .env settings."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
        )
    else:  # groq (default)
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
        )


# ─────────────────────────────────────────────────
# Fallback report (used when all LLM calls fail)
# ─────────────────────────────────────────────────

FALLBACK_REPORT = {
    "risk_summary": "ML model prediction is available. AI analysis unavailable due to a service error.",
    "key_risk_factors": ["See your ML risk score above"],
    "recommendations": [
        "Consult a healthcare professional for a full evaluation",
        "Monitor your blood pressure regularly",
        "Maintain physical activity of 150+ minutes per week",
        "Follow a heart-healthy diet low in sodium and saturated fat"
    ],
    "lifestyle_changes": ["Regular physical activity", "Balanced Mediterranean or DASH diet"],
    "follow_up_actions": ["Schedule a medical checkup", "Request a full lipid panel blood test"],
    "sources": ["WHO CVD Guidelines 2023"],
    "disclaimer": "This is a fallback report. AI analysis was unavailable. Please consult a qualified doctor."
}


# ─────────────────────────────────────────────────
# Node 1: Emergency Checker
# ─────────────────────────────────────────────────

def emergency_checker(state: AgentState) -> dict:
    """Check for critical/emergency values before running full analysis."""
    pd = state["patient_data"]
    flags = []

    # Hypertensive crisis
    if pd.get("ap_hi", 0) >= 180 or pd.get("ap_lo", 0) >= 120:
        flags.append("HYPERTENSIVE_CRISIS")

    # Triple high risk: Stage 2 hypertension + high cholesterol + smoker
    if (pd.get("ap_hi", 0) >= 140
            and pd.get("cholesterol", 1) == 3
            and pd.get("smoke", 0) == 1):
        flags.append("TRIPLE_HIGH_RISK")

    # Severe obesity
    bmi = pd.get("bmi", 0)
    if bmi >= 40:
        flags.append("SEVERE_OBESITY")

    return {"critical_flags": flags}


# ─────────────────────────────────────────────────
# Node 2: Risk Analyzer (chain-of-thought LLM)
# ─────────────────────────────────────────────────

def risk_analyzer(state: AgentState) -> dict:
    """Analyze patient data with chain-of-thought prompting to identify top risk factors."""
    try:
        llm = _get_llm()

        scenario_line = ""
        if state.get("scenario_context"):
            scenario_line = f"\nScenario context: {state['scenario_context']}"

        prompt = RISK_ANALYZER_PROMPT.format(
            system_base=SYSTEM_BASE,
            patient_data=json.dumps(state["patient_data"], indent=2),
            ml_prediction=json.dumps(state["ml_prediction"]),
            scenario_line=scenario_line,
        )

        response = llm.invoke(prompt)
        risk_profile = response.content.strip()

        return {"risk_profile": risk_profile}

    except Exception as e:
        return {
            "risk_profile": (
                "Unable to perform detailed risk analysis (LLM error). "
                "Basic ML prediction is available. Key inputs: "
                f"BP {state['patient_data'].get('ap_hi', '?')}/{state['patient_data'].get('ap_lo', '?')}, "
                f"BMI {state['patient_data'].get('bmi', '?'):.1f}, "
                f"Age {state['patient_data'].get('age', '?')}."
            ),
            "error_log": [f"risk_analyzer failed: {str(e)}"],
        }


# ─────────────────────────────────────────────────
# Node 3: Guideline Retriever (RAG)
# ─────────────────────────────────────────────────

def guideline_retriever(state: AgentState) -> dict:
    """Query ChromaDB for relevant clinical guidelines based on risk profile."""
    try:
        query = state.get("risk_profile", "cardiovascular risk assessment")
        guidelines = retrieve_guidelines(query, top_k=4)
        return {"retrieved_guidelines": guidelines}

    except Exception as e:
        return {
            "retrieved_guidelines": [{
                "topic": "general_prevention",
                "source": "WHO CVD Guidelines 2023",
                "content": (
                    "Maintain blood pressure below 140/90 mmHg, exercise 150+ min/week, "
                    "follow a heart-healthy diet, avoid tobacco, limit alcohol."
                )
            }],
            "error_log": [f"guideline_retriever failed: {str(e)}"],
        }


# ─────────────────────────────────────────────────
# Node 4: Report Generator (structured JSON output)
# ─────────────────────────────────────────────────

def report_generator(state: AgentState) -> dict:
    """Generate a structured 7-field health report using retrieved guidelines."""
    try:
        llm = _get_llm()

        scenario_line = ""
        if state.get("scenario_context"):
            scenario_line = f"\nScenario context: {state['scenario_context']}"

        guidelines_text = json.dumps(state.get("retrieved_guidelines", []), indent=2)

        prompt = REPORT_GENERATOR_PROMPT.format(
            system_base=SYSTEM_BASE,
            risk_profile=state.get("risk_profile", "N/A"),
            guidelines=guidelines_text,
            scenario_line=scenario_line,
        )

        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        report = json.loads(raw)
        return {"health_report": report}

    except json.JSONDecodeError as e:
        return {
            "health_report": FALLBACK_REPORT,
            "error_log": [f"report_generator JSON parse failed: {str(e)}"],
        }
    except Exception as e:
        return {
            "health_report": FALLBACK_REPORT,
            "error_log": [f"report_generator failed: {str(e)}"],
        }


# ─────────────────────────────────────────────────
# Node 5: Quality Checker (validation + retry gate)
# ─────────────────────────────────────────────────

REQUIRED_KEYS = [
    "risk_summary", "key_risk_factors", "recommendations",
    "lifestyle_changes", "follow_up_actions", "sources", "disclaimer"
]


def quality_checker(state: AgentState) -> dict:
    """Validate the health report structure and trigger retry if needed."""
    report = state.get("health_report", {})
    retry_count = state.get("retry_count", 0)

    # Check all required keys
    missing_keys = [k for k in REQUIRED_KEYS if k not in report]

    # Check minimum recommendations count
    recs = report.get("recommendations", [])
    has_enough_recs = len(recs) >= 3

    if not missing_keys and has_enough_recs:
        return {
            "quality_passed": True,
            "status": "completed",
        }

    # Quality failed — check if retry is allowed
    if retry_count < 1:
        return {
            "quality_passed": False,
            "retry_count": retry_count + 1,
            "error_log": [f"quality_checker: missing keys={missing_keys}, recs={len(recs)}. Retrying..."],
        }

    # Max retries exceeded — accept partial report
    # Fill in any missing keys with fallback values
    for key in REQUIRED_KEYS:
        if key not in report:
            report[key] = FALLBACK_REPORT[key]

    return {
        "health_report": report,
        "quality_passed": False,
        "status": "partial",
        "error_log": [f"quality_checker: max retries exceeded. Partial report accepted."],
    }
