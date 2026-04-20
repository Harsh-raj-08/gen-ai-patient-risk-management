"""
AgentState — Typed state schema for the LangGraph cardiovascular risk agent.
Passed through all graph nodes. Each node reads and writes to this state.
"""

from typing import TypedDict, Optional, Annotated
import operator


class AgentState(TypedDict):
    # --- Inputs ---
    patient_data: dict                          # Full patient intake dict
    ml_prediction: dict                         # {"prediction": 0|1, "probability": float}
    scenario_context: Optional[str]             # "what-if" or None for normal run

    # --- Emergency escalation ---
    critical_flags: list                        # e.g. ["HYPERTENSIVE_CRISIS"]

    # --- Node outputs ---
    risk_profile: str                           # Output of risk_analyzer node
    retrieved_guidelines: list                  # Output of guideline_retriever (list of dicts)
    health_report: dict                         # Output of report_generator (7-field JSON)

    # --- Quality control ---
    quality_passed: bool                        # Output of quality_checker
    retry_count: int                            # Max 1 retry allowed

    # --- Observability ---
    error_log: Annotated[list, operator.add]    # Accumulates errors across nodes
    status: str                                 # "running" | "completed" | "partial" | "failed"
