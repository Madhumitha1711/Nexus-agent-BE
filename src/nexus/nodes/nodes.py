import json
from typing import List
from datetime import datetime

from dotenv import load_dotenv
from copilotkit.langgraph import copilotkit_emit_state
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage, SystemMessage, ToolMessage
)
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END
from langgraph.types import interrupt

from pydantic import BaseModel, Field

from src.nexus.guardrails.guardrail import guardrails_in_node
from src.nexus.memory.episodic_memory import EpisodicMemory
from src.nexus.memory.rag_semantic_memory import MultimodalSemanticMemory
from src.nexus.tools.tools import get_app_resources
from src.nexus.utils.State import AgentMemoryState, TaskClassification, PlanDetails

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ---------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------
from langchain_core.utils.function_calling import convert_to_openai_tool





async def ingestion_node(state: AgentMemoryState):
    # 1. Prefer chat message
    user_id = "anonymous"
    if state.get("messages"):
        text = state["messages"][-1].content
        source = "chat"
        user_id = state.get("user_id", "anonymous")

    # 2. Support ticket input
    elif state.get("raw_input", {}).get("ticket"):
        ticket = state["raw_input"]["ticket"]
        text = (
            f"Support Ticket\n"
            f"Subject: {ticket.get('subject')}\n"
            f"Description: {ticket.get('description')}\n"
            f"Priority: {ticket.get('priority')}\n"
            f"Product: {ticket.get('product')}"
        )
        source = "support_ticket"

    # 3. Fallback text
    else:
        text = state.get("raw_input", {}).get("text", "")
        source = "raw_text"

    text = " ".join(text.strip().split())[:2000]

    return {
        "task_description": text,
        "user_id": user_id,
        "past_steps": [f"Ingested input from {source}"],
        "internal_steps": [f"Normalized {source} input"],
        "node_trajectory": ["ingestion_node"]
    }




async def guardrails_in(state: AgentMemoryState, config: RunnableConfig):
    return await guardrails_in_node(state, config)


async def task_classifier_node(state: AgentMemoryState):
    structured_llm = llm.with_structured_output(TaskClassification)
    user_input = state["messages"][-1].content

    res = await structured_llm.ainvoke(f"Classify this ticket: {user_input}")

    print("CLASSIFIER RESPONSE ::", res)

    return {
        "classification": {
            "intent": res["intent"],
            "severity": res["severity"],
            "urgency": res["urgency"],
            "summary": res["summary"],
        },
        "task_description": user_input,
        "internal_steps": ["Classified task intent and urgency."],
        "node_trajectory": ["task_classifier_node"]
    }


async def planning_agent_node(state: AgentMemoryState):
    structured_planner = llm.with_structured_output(PlanDetails)

    classif = state["classification"]
    user_message = state["messages"][-1].content

    prompt = f"""
      You are a Strategic Planning Agent for a Support System. 
    Your goal is to route the request to the correct processing nodes.

    ### SELECTION RULES:
    1. **ACTIVATE ['tool_calls']**: Whenever the user references a specific ticket (e.g., "INC-123", "ticket #45") or asks to:
       - Get details/status of a ticket.
       - Create, update, or close a ticket.
       - List tickets assigned to someone.
    2. **ACTIVATE ['episodic_memory']**: Use when the user refers to past interactions (e.g., "What did we do last time?", "The ticket from yesterday").
    3. **ACTIVATE ['semantic_memory']**: Use for general documentation,get details/status of a ticket or any details about incident.List tickets assigned to someone, "How-to" guides, or technical definitions and any incidents.
    4. **ACTIVATE ['aggregator']**: Use ONLY for greetings or basic conversational fluff.

    ### LOGIC EXAMPLES:
    - "What is the status of ticket 505 or any incident?" -> ["tool_calls","semantic_memory"]
    - "Update the ticket we talked about earlier with 'Fixed'." -> ["tool_calls", "episodic_memory"]
    - "How do I reset my password?" -> ["semantic_memory"]

    ### CURRENT TASK:
    User Message: "{user_message}"
    Intent: {classif.get('intent')}
    Summary: {classif.get('summary')}
    """

    res = await structured_planner.ainvoke(prompt)
    selected = res.get("selected_nodes", [])

    # Fallback: Ensure aggregator is always present if no memories are selected
    if not selected:
        selected = ["aggregator"]

    selected = [n for n in selected if n != "aggregator"]

    return {
        "planning_strategy": {
            "selected_nodes": selected,
            "priority": res.get("priority", "standard"),
            "strategy_notes": res.get("strategy_notes", "")
        },
        "expected_nodes": selected,
        "completed_nodes": [],
        "internal_steps": [f"Planner selected: {', '.join(selected)}"],
        "node_trajectory": ["planning_agent_node"]
    }





class ReasoningSchema(BaseModel):
    analysis: str = Field(description="Internal logic cross-referencing history and docs.")
    resolution_steps: List[str] = Field(description="Final steps for the user.")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    context:str = Field(description="Raw context used for reasoning.")


async def reasoning_node(state: AgentMemoryState, config: RunnableConfig):
    """
    Synthesizes results into a final reasoning path with real-time state streaming.
    """
    # Use the Pydantic model with structured output
    print("Reasoning State Expected ::", state["expected_nodes"])
    print("Reasoning State Completed::", state["expected_nodes"])

    structured_reasoner = llm.with_structured_output(ReasoningSchema)

    # 1. Gather context
    tool_results = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage):
            # Extract actual text from the ToolMessage content list
            if isinstance(m.content, list):
                for item in m.content:
                    if isinstance(item, dict) and "text" in item:
                        tool_results.append(item["text"])
            else:
                tool_results.append(str(m.content))

    clean_tool_context = "\n".join(tool_results) if tool_results else "Ticket not found in live system."

    # --- 2. CLEAN SEMANTIC CONTEXT (RAG Documents) ---
    semantic_data = state.get("memory_context", {}).get("semantic")
    clean_semantic = ""

    if semantic_data:
        # If it was accidentally stringified elsewhere
        if isinstance(semantic_data, str) and semantic_data != "[]":
            clean_semantic = semantic_data
        elif isinstance(semantic_data, list):
            docs = []
            for item in semantic_data:
                # Handle Document objects or tuples from similarity search
                content = item[0].page_content if isinstance(item, tuple) else getattr(item, 'page_content', str(item))
                docs.append(content)
            clean_semantic = "\n---\n".join(docs)

    if not clean_semantic:
        clean_semantic = "No technical documentation available."

    # --- 3. CLEAN EPISODIC CONTEXT ---
    episodic = state.get("memory_context", {}).get("episodic") or "No previous history available."
    prompt = (
        "### TASK\n"
        f"Request: {state.get('task_description')}\n"
        f"Summary: {state.get('classification', {}).get('summary')}\n\n"

        "### DATA SOURCES\n"
        "1.TOOL DATA:\n"
        f"{clean_tool_context}\n\n"

        "2. TECHNICAL DOCUMENTATION (Knowledge of Incidents):\n"
        f"{clean_semantic}\n\n"

        "3. USER HISTORY (Previous Interactions):\n"
        f"{episodic}\n\n"

        "### REASONING PROTOCOL\n"
        "- **Prioritize Resolution:** Provide the most comprehensive answer by synthesizing all available context. If information exists in Documentation, treat the Documentation as the valid historical record.\n"
        "- Use the 'Acknowledgment' and 'Resolution Time' from Technical Docs if available.\n"
    )

    print("REASONING PROMPT ::", prompt)
    print("Tool Results ::", tool_results)
    print("Semantic ::", clean_semantic)
    print("Episodic ::", episodic)

    # 2. Execute LLM with streaming
    final_res = None

    # We use astream to capture the partial structured data
    async for partial_res in structured_reasoner.astream(prompt, config):
        final_res = partial_res  # The last chunk will be the complete object

        # Emit partial updates so the UI can show the analysis growing
        await copilotkit_emit_state(config, {
            "reasoning_output": {
                "analysis": getattr(partial_res, 'analysis', "Analyzing..."),
                "confidence": getattr(partial_res, 'confidence', 0),
                "source_used": "processing..."
            }
        })

    # 3. Final metadata calculation
    has_tool = any('"success": true' in res.lower() for res in tool_results)

    # 2. Check if memory systems returned valid data (not the fallback strings)
    has_episodic = "No previous history" not in episodic
    has_semantic = "No technical documentation" not in clean_semantic

    source = "hybrid"
    if has_tool and not (has_episodic or has_semantic):
        source = "tool_only"
    elif has_episodic and not (has_tool or has_semantic):
        source = "episodic_only"
    elif has_semantic and not (has_tool or has_episodic):
        source = "semantic_only"

    # 4. Final State Update
    output_data = {
        "analysis": final_res.analysis,
        "resolution_steps": final_res.resolution_steps,
        "confidence": final_res.confidence,
        "source_used": source,
        "context": final_res.context,
    }

    # Final emit to ensure the UI has the complete, clean data
    await copilotkit_emit_state(config, {"reasoning_output": output_data})

    return {
        "reasoning_output": output_data,
        "past_steps": [f"Reasoning completed via {source}."],
        "internal_steps": ["Synthesized reasoning from retrieved contexts."],
        "node_trajectory": ["reasoning_node"]
    }


async def aggregator_node(state: AgentMemoryState, config: RunnableConfig):
    """
    Final node that streams the formatted response and updates the UI in real-time.
    """
    reasoning = state.get("reasoning_output", {})

    analysis = reasoning.get("analysis", "I have analyzed your request.")
    context = reasoning.get("context", {})

    prompt = [
        SystemMessage(content=(
            "You are a concise Technical Support Assistant. "
            "Your responses must be strictly grounded in the provided ANALYSIS, and CONTEXT. "
            "Do not hallucinate features, UI elements, or causes not explicitly stated."
        )),
        HumanMessage(content=(
            f"CONTEXT: {context}\n"
            f"ANALYSIS: {analysis}\n"
            f"ORIGINAL ISSUE: {state['task_description']}\n\n"
            "Requirements:\n"
            "- Tone: Professional and empathetic.\n"
            "- Length: Maximum 2-3 sentences for the opening. Keep it brief.\n"
            "- Formatting: Use Markdown bullet points."
        ))
    ]

    full_content = ""

    # Use astream to get chunks as they are generated
    async for chunk in llm.astream(prompt, config):
        # Extract content from the chunk (handling different possible chunk formats)
        content_chunk = chunk.content if hasattr(chunk, 'content') else str(chunk)
        full_content += content_chunk

        # Emit partial state to CopilotKit so the user sees the 'typing' effect
        await copilotkit_emit_state(config, {
            "final_report": full_content,
            "is_complete": False
        })

    # Final emission to mark completion
    await copilotkit_emit_state(config, {
        "final_report": full_content,
        "is_complete": True
    })

    return {
        "final_report": full_content,
        "messages": [AIMessage(content=full_content)],
        "past_steps": ["Final response streamed and delivered."],
        "is_complete": True,
        "internal_steps": ["Streamed final response to the user."],
        "node_trajectory": ["aggregator_node"]
    }


async def human_review_node(state: AgentMemoryState, config: RunnableConfig):
    """
    Pause execution to ask the user if they want to proceed or email.
    """
    thread_id = config["configurable"].get("thread_id", "unknown")
    print(f"--- Entering Human Review for Thread: {thread_id} ---")

    # The interrupt pauses the graph and waits for a user response.
    # We structure the 'human_command' as the expected input.
    human_command = interrupt({
        "message": "Human Review Required: Choose your next action.",
        "options": ["proceed", "send_email"],
        "context": {
            "current_analysis": state.get("reasoning_output", {}),
            "thread_id": thread_id
        }
    })

    # Extract data from the human's response
    # We expect human_command to be a dict like:
    # {"action": "send_email", "reason": "...", "description": "..."}
    action = human_command.get("action")
    reason = human_command.get("reason", "No reason provided")
    description = human_command.get("task_description", "No description")

    if action == "send_email":
        analysis_result = f"Action: Email Sent. Reason: {reason}. Task: {description}"
    else:
        analysis_result = f"Action: Proceeded. Reason: {reason}. Task: {description}"

    return {
        "reasoning_output": {
            "analysis": analysis_result,
            "human_action": action
        },
        "past_steps": [f"Human chose to {action}"],
        "internal_steps": ["Processed human feedback via interrupt."],
        "node_trajectory": ["human_review_node"]
    }

    #
    # return {
    #     "messages": [AIMessage(content="Confidence is low. Escalating to human review.")],
    #     "past_steps": ["Escalated to human review due to low confidence."],
    #     "internal_steps":["Routed to human review node."],
    #     "node_trajectory": ["human_review_node"]
    # }


async def reflection_node(state: AgentMemoryState, config: RunnableConfig):
    memory_system = EpisodicMemory()
    facts = await memory_system.run_reflection(state, config)
    print("Facts extracted:", facts)
    return {"extracted_facts": facts,
            "is_complete": True, "node_trajectory": ["reflection_node"],
            " internal_steps": ["Reflected and extracted new facts into episodic memory."]}
