from src.nexus.utils.State import AgentMemoryState


def clear_state_node(state: AgentMemoryState):
    """
    Resets ephemeral state for a fresh request cycle,
    preserving 'messages' and 'user_id'.
    """
    return {
        # Resetting List[str] fields that use 'add'
        # by passing an empty list (if the reducer handles it)
        # or by using a custom reducer that allows overwriting.
        "past_steps": None,
        "internal_steps": None,
        "node_trajectory": None,
        "past_nodes": None,

        # Non-annotated fields are simply overwritten
        "classification": None,
        "planning_strategy": None,
        "memory_context": {},
        "reasoning_output": None,
        "is_complete": False,
        "final_report": None,
        "extracted_facts": None,
        "expected_nodes":[],
        "completed_nodes": [],
        "has_joined": False
    }




def join_node(state: AgentMemoryState):
    # 1. If reasoning has already started, stop this branch immediately
    if state.get("has_joined"):
        return {}

    expected = set(state.get("expected_nodes") or [])
    completed = set(state.get("completed_nodes") or [])

    print("JOIN NODE :: Expected:", expected, "Completed:", completed)
    # 2. Wait for all parallel branches to report 'done'
    if expected.issubset(completed) and len(expected) > 0:
        print("parallel branches complete; proceeding.")
        return {
            "has_joined": True,
            "past_steps": ["Parallel branches synchronized."]
        }
    print("JOIN NODE :: Not ready yet.")
    # 3. Branch not ready; kill this execution path
    return {}