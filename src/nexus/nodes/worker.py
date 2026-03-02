from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from src.nexus.memory.episodic_memory import EpisodicMemory
from src.nexus.memory.rag_semantic_memory import MultimodalSemanticMemory
from src.nexus.tools.tools import get_app_resources
from src.nexus.utils.State import AgentMemoryState


async def episodic_agent_node(state: AgentMemoryState):
    memory_system = EpisodicMemory()
    # We use the summary/message from classification for better search
    query = state["classification"].get("summary") or state["messages"][-1].content

    episodic_docs = await memory_system.retrieve_episodic_memory(query, state.get("user_id", ""))
    print("EPISODIC ::", episodic_docs)
    return {
        "memory_context": {"episodic": str(episodic_docs)},
        "internal_steps": ["Retrieved past interaction history"],
        "completed_nodes": ["episodic_memory"],
        "node_trajectory": ["episodic_agent_node"]
    }


async def tool_call_model(state: AgentMemoryState):
    messages = state["messages"]
    _, mcp_tools = await get_app_resources()
    model = ChatOpenAI(model="gpt-4o").bind_tools(mcp_tools)

    response = await model.ainvoke(messages)

    # Check if the LLM wants to call more tools
    has_tool_calls = bool(getattr(response, "tool_calls", None))

    update = {
        "messages": [AIMessage(content=response.content, tool_calls=getattr(response, "tool_calls", None))],
        "internal_steps": ["Executed tool call model."],
        "node_trajectory": ["tool_call_model"]
    }

    # IMPORTANT: Only signal completion if we are NOT calling more tools
    if not has_tool_calls:
        update["completed_nodes"] = ["tool_calls"]

    return update


async def semantic_agent_node(state: AgentMemoryState):
    memory_system = MultimodalSemanticMemory()
    query = state["classification"].get("summary") or state["messages"][-1].content

    semantic_docs = memory_system.semantic_similarity_search(query)
    print("SEMANTIC ::", semantic_docs)
    return {
        "memory_context": {"semantic": semantic_docs},
        "internal_steps": ["Retrieved technical documentation"],
        "completed_nodes": ["semantic_memory"],
        "node_trajectory": ["semantic_agent_node"]
    }
