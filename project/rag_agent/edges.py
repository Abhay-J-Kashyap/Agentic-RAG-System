from typing import Literal
from langgraph.types import Send
from langchain_core.messages import HumanMessage
from .graph_state import State, AgentState
from config import MAX_ITERATIONS, MAX_TOOL_CALLS

def route_after_rewrite(state: State) -> Literal["request_clarification", "agent"]:
    if not state.get("questionIsClear", False):
        return "request_clarification"
    else:
        return [
                Send("agent", {"question": query, "question_index": idx, "messages": []})
                for idx, query in enumerate(state["rewrittenQuestions"])
            ]
    
def route_after_orchestrator_call(state: AgentState) -> Literal["tools", "fallback_response", "integrity_check"]:
    iteration = state.get("iteration_count", 0)
    tool_count = state.get("tool_call_count", 0)

    if iteration >= MAX_ITERATIONS or tool_count > MAX_TOOL_CALLS:
        return "fallback_response"

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    # CHANGED: Route to integrity_check instead of collect_answer
    if not tool_calls:
        return "integrity_check"
    
    return "tools"

# --- ADDED: Routing for the Integrity Check ---
def route_after_integrity_check(state: AgentState) -> Literal["orchestrator", "collect_answer"]:
    """Routes back to the orchestrator if the check failed, otherwise finalizes the answer."""
    last_message = state["messages"][-1]
    
    # If the last message is our HumanMessage guardrail feedback, loop back to the Llama model
    if isinstance(last_message, HumanMessage) and "SYSTEM GUARDRAIL TRIGGERED" in last_message.content:
        return "orchestrator"
        
    return "collect_answer"
# ----------------------------------------------