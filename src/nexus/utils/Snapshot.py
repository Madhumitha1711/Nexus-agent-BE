def ui_snapshot(state: dict, overrides: dict | None = None):
    snapshot = {
        "task_description": state.get("task_description"),
        "plan": state.get("plan", []),
        "semantic_context": state.get("semantic_context", []),
        "episodic_context": state.get("episodic_context", []),
        "worker_logs": state.get("worker_logs", []),
        "current_step_index": state.get("current_step_index"),
        "is_complete": state.get("is_complete"),
    }

    if overrides:
        snapshot.update(overrides)

    return snapshot