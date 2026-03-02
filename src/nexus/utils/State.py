from operator import add
from typing import List, Literal, Annotated, Optional, Dict, Any
from copilotkit import CopilotKitState
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

def flexible_reducer(existing: list, new: list) -> list:
    # If the new value is explicitly None, wipe the state
    if new is None:
        return []
    # If existing is None (first run), initialize as list
    if existing is None:
        return new or []
    return existing + new

class ExtractedFacts(TypedDict):
    lessons_learned: Annotated[str, "Technical insights or root causes of failure discovered during the session."]
    user_preferences: Annotated[str, "Specific stylistic or behavioral constraints requested by the user."]
    is_novel: Annotated[bool, "True if this information is not already present in existing long-term memory."]


class SafetySignal(TypedDict, total=False):
    is_safe: bool
    blocked_reason: Optional[str]


class TaskClassification(TypedDict):
    intent: Annotated[
        Literal["technical_support", "billing", "feature_request", "account_access", "other"],
        "The primary category of the user's request."
    ]
    severity: Annotated[Literal["low", "medium", "high", "critical"], "The technical impact of the issue."]
    urgency: Annotated[Literal["low", "medium", "high"], "How quickly the user requires a response."]
    summary: Annotated[Optional[str], "A concise, 1-sentence summary of the core problem."]


class MemoryResults(TypedDict):
    episodic: Optional[str]
    semantic: Optional[str]


# Added this to match your State class
class PlanDetails(TypedDict):
    selected_nodes: Annotated[
        List[Literal["episodic_memory", "semantic_memory", "aggregator","tool_calls"]],
        "List of specialized memory or processing nodes to activate in the graph."
    ]
    priority: Annotated[Literal["standard", "expedited"], "The processing speed/priority level."]
    strategy_notes: Annotated[str, "The internal reasoning for choosing this specific retrieval path."]


# Added this to match your State class
class ReasoningData(TypedDict):
    analysis: Annotated[str, "Internal cross-referencing between user history and technical documentation."]
    resolution_steps: Annotated[List[str], "Ordered list of actionable steps for the user to follow."]
    confidence: Annotated[float, "Confidence score (0.0 to 1.0) regarding the accuracy of the proposed solution."]
    source_used: Annotated[str, "Identifies if the answer came from episodic, semantic, or hybrid sources."]
    context:str

# --- Main Agent State ---

class AgentMemoryState(CopilotKitState):
    # --- Input & Context ---
    task_description: str
    user_id: str

    # --- Processed State ---
    safety: Annotated[SafetySignal, "Guardrails safety signal"]
    classification: TaskClassification
    planning_strategy: PlanDetails

    # Merges parallel results from Episodic and Semantic nodes
    memory_context: Annotated[MemoryResults, lambda old, new: {**(old or {}), **(new or {})}]

    # Results from Reasoning node
    reasoning_output: ReasoningData

    # --- Final Output & History ---
    final_report: Optional[str]
    is_complete: bool
    extracted_facts: Optional[ExtractedFacts]

    # Reducers for list appending
    past_steps: Annotated[List[str], flexible_reducer]
    messages: Annotated[List[BaseMessage], add]
    internal_steps: Annotated[List[str], flexible_reducer]
    node_trajectory: Annotated[List[str], flexible_reducer]
    past_nodes: Annotated[List[str], flexible_reducer]

    expected_nodes: Annotated[List[str], add]
    completed_nodes: Annotated[List[str], add]
    has_joined:bool
