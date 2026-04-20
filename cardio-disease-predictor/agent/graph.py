"""
LangGraph graph definition — compiles the 5-node cardiovascular risk agent pipeline.

Graph topology:
  START → emergency_checker → risk_analyzer → guideline_retriever → report_generator → quality_checker
  quality_checker → report_generator  (conditional retry, max 1)
  quality_checker → END
"""

from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    emergency_checker,
    risk_analyzer,
    guideline_retriever,
    report_generator,
    quality_checker,
)


def _should_retry(state: AgentState) -> str:
    """Conditional edge: retry report generation or finish."""
    if not state.get("quality_passed", False) and state.get("retry_count", 0) <= 1 and state.get("status") != "partial":
        return "report_generator"
    return END


def build_graph():
    """Build and compile the LangGraph agent graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("emergency_checker", emergency_checker)
    graph.add_node("risk_analyzer", risk_analyzer)
    graph.add_node("guideline_retriever", guideline_retriever)
    graph.add_node("report_generator", report_generator)
    graph.add_node("quality_checker", quality_checker)

    # Linear edges
    graph.set_entry_point("emergency_checker")
    graph.add_edge("emergency_checker", "risk_analyzer")
    graph.add_edge("risk_analyzer", "guideline_retriever")
    graph.add_edge("guideline_retriever", "report_generator")
    graph.add_edge("report_generator", "quality_checker")

    # Conditional edge for retry loop
    graph.add_conditional_edges(
        "quality_checker",
        _should_retry,
        {
            "report_generator": "report_generator",
            END: END,
        }
    )

    return graph.compile()


# Pre-compile the graph (singleton)
_compiled_graph = None


def get_graph():
    """Return the compiled graph (lazy singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent_graph(patient_data: dict, ml_prediction: dict,
                    context: str = None) -> dict:
    """
    Execute the full agent pipeline.

    Args:
        patient_data: Patient intake dict with all 12 features.
        ml_prediction: {"prediction": 0|1, "probability": float}
        context: Optional scenario context string (e.g., "What-if scenario analysis").

    Returns:
        dict with keys: health_report, critical_flags, status, error_log
    """
    graph = get_graph()

    initial_state = {
        "patient_data": patient_data,
        "ml_prediction": ml_prediction,
        "scenario_context": context,
        "critical_flags": [],
        "risk_profile": "",
        "retrieved_guidelines": [],
        "health_report": {},
        "quality_passed": False,
        "retry_count": 0,
        "error_log": [],
        "status": "running",
    }

    try:
        result = graph.invoke(initial_state)
        return {
            "health_report": result.get("health_report", {}),
            "critical_flags": result.get("critical_flags", []),
            "status": result.get("status", "completed"),
            "error_log": result.get("error_log", []),
        }
    except Exception as e:
        from agent.nodes import FALLBACK_REPORT
        return {
            "health_report": FALLBACK_REPORT,
            "critical_flags": [],
            "status": "failed",
            "error_log": [f"Graph execution failed: {str(e)}"],
        }
