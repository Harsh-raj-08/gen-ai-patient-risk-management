"""
Chat agent — stateless single-LLM-call function for the conversational follow-up page.
Receives full patient context + chat history and returns a grounded response.
"""

import os
import json
from dotenv import load_dotenv
from agent.prompts import CHAT_AGENT_PROMPT, SYSTEM_BASE

load_dotenv()


def _get_chat_llm():
    """Return the configured LLM for chat (separate instance, slightly higher temperature)."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.5,
        )
    else:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.5,
        )


def run_chat_agent(context: str) -> str:
    """
    Run a single-shot chat agent call with full patient context.

    Args:
        context: Pre-formatted context string containing patient data,
                 risk score, health report, chat history, and current question.

    Returns:
        Agent response text (2-4 sentences).
    """
    try:
        llm = _get_chat_llm()

        # Parse context components (passed as a formatted string from app.py)
        response = llm.invoke(context)
        return response.content.strip()

    except Exception as e:
        return (
            "I apologize, but I'm having trouble processing your question right now. "
            "Your health data and report are still available above. "
            "Please try again in a moment, or consult a healthcare professional "
            f"for personalized medical advice. (Error: {type(e).__name__})"
        )


def build_chat_context(patient_data: dict, risk_score: float,
                        health_report: dict, chat_history: list,
                        user_question: str) -> str:
    """
    Build the full prompt for the chat agent using the template.

    Args:
        patient_data: Patient intake dict.
        risk_score: ML risk probability (0-100).
        health_report: Generated health report dict.
        chat_history: Last N messages as list of {"role", "content"}.
        user_question: Current user question.

    Returns:
        Formatted prompt string.
    """
    # Format chat history for the prompt
    history_lines = []
    for msg in chat_history[-6:]:
        role = "Patient" if msg["role"] == "user" else "Assistant"
        history_lines.append(f"{role}: {msg['content']}")
    history_text = "\n".join(history_lines) if history_lines else "(No previous messages)"

    prompt = CHAT_AGENT_PROMPT.format(
        system_base=SYSTEM_BASE,
        patient_data=json.dumps(patient_data, indent=2),
        risk_score=f"{risk_score:.1f}",
        health_report=json.dumps(health_report, indent=2),
        chat_history=history_text,
        user_question=user_question,
    )

    return prompt
