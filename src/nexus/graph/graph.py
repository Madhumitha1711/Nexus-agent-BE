from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.nexus.nodes.helper import clear_state_node, join_node
from src.nexus.nodes.router import route_planner, should_continue, route_from_join, route_after_reasoning
from src.nexus.nodes.worker import tool_call_model, episodic_agent_node, semantic_agent_node
# Ensure these imports match your project structure
from src.nexus.tools.tools import get_tools_node
from src.nexus.utils.State import AgentMemoryState
from src.nexus.nodes.nodes import (
    guardrails_in_node,
    task_classifier_node,
    planning_agent_node,
    reasoning_node,
    aggregator_node,
    reflection_node,
    human_review_node,
)


async def initialize_graph():
    # 1. Initialize Tools and State
    tool_node, tools = await get_tools_node()
    builder = StateGraph(AgentMemoryState)

    # 2. Define All Nodes

    builder.add_node("cleanup", clear_state_node)
    builder.add_node("guard_in", guardrails_in_node)
    builder.add_node("classifier", task_classifier_node)
    builder.add_node("planner", planning_agent_node)
    builder.add_node("tool_agent", tool_call_model)
    builder.add_node("tools", tool_node)
    builder.add_node("episodic_mem", episodic_agent_node)
    builder.add_node("semantic_mem", semantic_agent_node)
    builder.add_node("reasoning", reasoning_node)
    builder.add_node("aggregator", aggregator_node)
    builder.add_node("reflection", reflection_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("join_node", join_node)  # Pass-through join node

    # Simple lambda node for blocked requests
    builder.add_node("blocked", lambda state: {
        "messages": [AIMessage(content="I'm sorry, but I can’t help with that request.")]
    })

    # 3. Define Main Flow (Edges)

    # Entry Point
    builder.add_edge(START, "cleanup")
    builder.add_edge("cleanup", "guard_in")

    builder.add_conditional_edges(
        "guard_in",
        lambda state: "classifier" if state.get("safety", {}).get("is_safe") else "blocked",
        {
            "classifier": "classifier",
            "blocked": "blocked"
        }
    )

    builder.add_edge("classifier", "planner")

    # 4. Parallel Execution (Fan-out)
    # Planner decides which memory systems to consult
    builder.add_conditional_edges(
        "planner",
        route_planner,
        {
            "episodic_memory_node": "episodic_mem",
            "semantic_memory_node": "semantic_mem",
            "tool_call_model": "tool_agent"
        }
    )

    # 5. Convergence (Fan-in)
    builder.add_edge("episodic_mem", "join_node")
    builder.add_edge("semantic_mem", "join_node")

    builder.add_conditional_edges("tool_agent", should_continue,{
        "tools": "tools",
        "join_node": "join_node"
    })

    builder.add_edge("tools", "tool_agent")
    builder.add_conditional_edges(
        "join_node",
        route_from_join,
        {
            "reasoning": "reasoning",
            "__end__": END
        }
    )
    # Reasoning to Review or Aggregation
    builder.add_conditional_edges(
        "reasoning",
        route_after_reasoning,
        {
            "human_review": "human_review",
            "aggregator": "aggregator"
        }
    )

    builder.add_edge("human_review", "aggregator")
    builder.add_edge("aggregator", "reflection")

    # 6. Exit Points
    builder.add_edge("reflection", END)
    builder.add_edge("blocked", END)



    # Compile with persistence
    return builder.compile(checkpointer=MemorySaver())

