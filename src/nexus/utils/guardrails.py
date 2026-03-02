from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from src.nexus.utils.State import AgentMemoryState, SafetySignal

load_dotenv()


async def check_guardrails(state: AgentMemoryState, config: RunnableConfig):
    messages = state.get("messages", [])
    if not messages:
        return {"safety": {"is_safe": True}}

    # 1. Get only the last message content to check for safety
    # This avoids the "ToolMessage must follow AIMessage" error entirely
    last_message = messages[-1].content

    llm_guard = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(SafetySignal)

    # 2. Construct a clean, simple prompt for the guardrail
    prompt = [
        SystemMessage(content="Determine if the following text contains violence, illegal acts, or hate speech."),
        HumanMessage(content=f"Text to check: {last_message}")
    ]

    response = await llm_guard.ainvoke(prompt)

    safety_signal = response if isinstance(response, dict) else response.dict()
    return {"safety": safety_signal}