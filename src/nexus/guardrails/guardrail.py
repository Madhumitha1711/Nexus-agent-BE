import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from nemoguardrails import RailsConfig, LLMRails

from src.nexus.utils.State import AgentMemoryState, SafetySignal

CURRENT_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = (CURRENT_DIR / ".." / ".." / "api" / "config").resolve()

if not os.path.exists(CONFIG_PATH):
    raise ValueError(f"CRITICAL: Config directory not found at {CONFIG_PATH}")

load_dotenv()
config = RailsConfig.from_path(str(CONFIG_PATH))
rails = LLMRails(config)

from langchain_core.messages import AIMessage, HumanMessage


async def guardrails_in_node(state: AgentMemoryState, config: RunnableConfig):
    user_message = state["messages"][-1].content

    result = await rails.generate_async(
        messages=[{"role": "user", "content": user_message}]
    )

    print("Guardrails IN Result:", result)

    bot_message = result.get("content", "")
    context = result.get("context", {})
    print("Guardrails IN Context:", context)

    status = context.get("resolution_status", "safe")

    if bot_message != user_message and status == "safe":
        if any(phrase in bot_message.lower() for phrase in ["sorry", "cannot", "can't", "policy"]):
            status = "blocked"

    updates = {
        "resolution_status": status,
        "internal_steps": [f"Guardrail Check: {status}"]
    }

    print("Guardrails IN Updates:", updates)
    report_status = ""
    if status == "blocked":
        state["messages"] = [AIMessage(content=bot_message)]
        safetySignal: SafetySignal = {
            "is_safe": False,
            "blocked_reason": "Content violated guardrail policies."
        }
        report_status = "Request blocked due to policy violation."
    else:
        safetySignal: SafetySignal = {
            "is_safe": True,
            "blocked_reason": None,
        }

    print("Safefy Signal",safetySignal)
    return {"safety": safetySignal,"node_trajectory": ["guardrails_in_node"],"final_report": report_status}


