import sys
import os
import pytest
import uuid
from typing import Annotated, TypedDict
from unittest.mock import patch, AsyncMock

from dotenv import load_dotenv
from langsmith import Client
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import json

# Load environment and set up graph
load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.nexus.graph.graph import initialize_graph

pytest_plugins = "pytest_asyncio"


# --- 1. MOCK FIXTURE ---
@pytest.fixture(autouse=True)
def mock_episodic():
    with patch("src.nexus.nodes.nodes.reflection_node", new_callable=AsyncMock) as mocked:
        mocked.return_value = {"extracted_facts": "Mocked Fact", "is_complete": True}
        yield mocked


# --- 2. TARGET FUNCTION ---
async def target(inputs: dict) -> dict:
    """Runs the graph in debug stream mode to capture full node trajectory."""
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": "user_xyz",  # Mandatory field for Nexus
            "env": "test"
        }
    }

    initial_input = {
        "messages": [HumanMessage(content=inputs['question'])],
        "user_id": "user_xyz"
    }

    trajectory = []
    graph = await initialize_graph()
    async def _consume_stream(stream_input, stream_config):
        """Helper to process chunks from the debug stream."""
        async for namespace, chunk in graph.astream(
                stream_input,
                config=stream_config,
                subgraphs=True,
                stream_mode="debug",
        ):
            # Capture the start of every node task
            if chunk["type"] == "task":
                node_name = chunk["payload"]["name"]
                trajectory.append(node_name)

                # If the node is 'tools', extract the specific tool name
                if node_name == "tools":
                    messages = chunk["payload"]["input"].get("messages", [])
                    if messages and hasattr(messages[-1], 'tool_calls'):
                        for tc in messages[-1].tool_calls:
                            trajectory.append(tc["name"])

    try:
        # Phase 1: Initial invocation
        await _consume_stream(initial_input, config)
        print("Completed Phase 1 with trajectory:", trajectory)

        # Phase 2: Check for Interrupt (Human Review)
        state_snapshot = await graph.aget_state(config)
        if state_snapshot.next:
            # Provide feedback to resume from the interrupt
            # We pass None as input to continue the existing thread
            await _consume_stream({"human_feedback": "Yes, proceed."}, config)

        # Phase 3: Final State Extraction
        final_state = await graph.aget_state(config)
        vals = final_state.values

        # Priority list for finding the final answer in your state
        final_answer = (
                vals.get("final_report") or
                vals.get("answer") or
                (vals["messages"][-1].content if vals.get("messages") else "No response")
        )

        return {
            "answer": final_answer,
            "trajectory": trajectory
        }
    except Exception as e:
        return {"answer": f"ERROR: {str(e)}", "trajectory": trajectory}


# --- 3. EVALUATORS ---
class Grade(TypedDict):
    reasoning: Annotated[str, "Reasoning"]
    is_correct: Annotated[bool, "True/False"]


# Create grader LLM if API key exists; otherwise we'll fall back to string match
grader_llm = None
if os.getenv("OPENAI_API_KEY"):
    grader_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(Grade, strict=True)


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


async def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    prediction = outputs.get("answer", "")
    reference = reference_outputs.get("answer") or reference_outputs.get("response") or ""

    # Prefer LLM grading when available
    if grader_llm is not None:
        prompt = f"QUESTION: {inputs['question']}\nGROUND TRUTH: {reference}\nSTUDENT: {prediction}"
        try:
            grade = await grader_llm.ainvoke([
                SystemMessage(content="Grade the student response based on factual accuracy."),
                HumanMessage(content=prompt)
            ])
            return {"key": "correctness", "score": 1 if grade["is_correct"] else 0, "comment": grade["reasoning"]}
        except Exception as e:
            # Fall through to deterministic check
            pass

    # Deterministic fallback: simple normalized equality check
    eq = _normalize(prediction) == _normalize(reference)
    return {"key": "correctness", "score": 1 if eq else 0, "comment": "Fallback equality check used."}


def trajectory_match(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    actual = outputs.get("trajectory", [])
    expected = reference_outputs.get("trajectory") or reference_outputs.get("reference.trajectory") or []
    if not expected:
        return {"key": "trajectory_match", "score": 1, "comment": "No reference trajectory provided."}
    success = all(node in actual for node in expected)
    return {"key": "trajectory_match", "score": 1 if success else 0, "comment": f"Actual: {actual}"}


# --- 4. DATASET CHECK & EXPERIMENT ---
@pytest.mark.asyncio
async def test_e2e_experiment():
    client = Client()

    # Load dataset config and examples from external JSON file
    data_file = os.path.join(os.path.dirname(__file__), "data", "e2e_dataset.json")
    with open(data_file, "r", encoding="utf-8") as f:
        dataset_cfg = json.load(f)

    dataset_name = dataset_cfg.get("dataset_name", "nexus-e2e-default")
    examples = dataset_cfg.get("examples", [])

    # Ensure dataset exists
    if not client.has_dataset(dataset_name=dataset_name):
        print(f"Creating dataset: {dataset_name}")
        dataset = client.create_dataset(dataset_name=dataset_name)

        # Convert examples into separate inputs/outputs lists for create_examples
        inputs_list = [ex.get("inputs", {}) for ex in examples]
        outputs_list = [ex.get("outputs", {}) for ex in examples]
        client.create_examples(
            inputs=inputs_list,
            outputs=outputs_list,
            dataset_id=dataset.id
        )

    print(f"\n🚀 Evaluating {dataset_name}...")

    results = await client.aevaluate(
        target,
        data=dataset_name,
        evaluators=[correctness_evaluator, trajectory_match],
        experiment_prefix="nexus-trajectory-fix",
        max_concurrency=1,
    )

    df = results.to_pandas()

    correct_col = "feedback.correctness"
    traj_col = "feedback.trajectory_match"

    # Ensure evaluator columns exist
    assert correct_col in df.columns, f"Missing evaluator column: {correct_col}. Columns: {list(df.columns)}"
    assert traj_col in df.columns, f"Missing evaluator column: {traj_col}. Columns: {list(df.columns)}"

    # Both metrics must be 1 for all rows
    assert df[correct_col].all(), "Factual correctness check failed."
    assert df[traj_col].all(), f"Trajectory failed. Found: {df.get('outputs.trajectory', []).tolist()}"
