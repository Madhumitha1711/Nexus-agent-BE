from typing import List

from langgraph.constants import END

from src.nexus.utils.State import AgentMemoryState


def should_continue(state: AgentMemoryState):
    last_message = state['messages'][-1]
    print("Last message:", last_message)
    if isinstance(last_message, dict):
        # In case any upstream converted to dict form
        tool_calls = last_message.get("tool_calls")
    else:
        tool_calls = getattr(last_message, "tool_calls", None)

    if tool_calls:
        return "tools"

    return "join_node"

def route_planner(state: AgentMemoryState) -> List[str]:
    """
    Returns a list of node keys to trigger parallel execution.
    """
    strategy = state.get("planning_strategy", {})
    # This must match the 'targets' from your LLM response
    selected = strategy.get("selected_nodes", [])

    if not selected:
        return ["aggregator"]  # Fallback if no plan generated

    # Map the LLM's strings to the Graph's node keys
    mapping = {
        "episodic_memory": "episodic_memory_node",
        "semantic_memory": "semantic_memory_node",
        "tool_calls": "tool_call_model"
    }

    routes = [mapping[s] for s in selected if s in mapping]
    return routes


def route_from_join(state: AgentMemoryState):
    if state.get("has_joined"):
        return "reasoning"
    return END # This kills THIS specific branch, but the graph stays alive



def route_after_reasoning(state: AgentMemoryState):
    # Access the confidence from your reasoning_output
    output = state.get("reasoning_output", {})
    confidence = output.get("confidence", 0)
    print("Routing based on confidence:", confidence)

    if confidence < 0.4:
        return "human_review"
    return "aggregator"
