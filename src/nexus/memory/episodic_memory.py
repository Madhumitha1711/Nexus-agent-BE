from datetime import datetime, timezone
from mem0 import MemoryClient, AsyncMemoryClient
from copilotkit.langgraph import copilotkit_emit_state
from langchain_core.runnables import RunnableConfig

# Assuming these are imported from your state file
from src.nexus.utils.State import AgentMemoryState, ExtractedFacts


class EpisodicMemory:
    def __init__(self):
        """
        Initializes Mem0.
        Uses OpenAI by default for fact extraction and local storage.
        """
        self.memory = AsyncMemoryClient(api_key="m0-i62Fjg8nY7njcPmWgmyyzi4WsXIPaLtWTpPy3IXy")

    async def run_reflection(self, state: AgentMemoryState, config: RunnableConfig) -> ExtractedFacts:
        """
        Captures the conversation outcome and saves it to Mem0.
        Extracts the specific 'memory' string from Mem0 results.
        """
        user_id = state.get("user_id", "default_user")
        task = state.get("task_description", "Unknown Task")
        final_report = state.get("final_report", "")

        content_to_save = [
            {"role": "user", "content": f"Task: {task}"},
            {"role": "assistant", "content": f"Final Resolution: {final_report}"}
        ]

        print("Episodic Memory - Content to Save:", content_to_save)

        # 1. Save to Mem0
        save_results = await self.memory.add(
            content_to_save,
            user_id=user_id,
            metadata={"timestamp": datetime.utcnow().isoformat(), "task": task}
        )

        print("Episodic Memory - Save Results:", save_results)


        # 2. Extract the actual memory strings from the results list
        # We join them in case Mem0 extracted multiple atomic facts from one interaction
        memories_list = [res.get("memory", "") for res in save_results.get("results", [])]
        memory_content = "; ".join(memories_list) if memories_list else "No new facts extracted."

        # 3. Format result to match your State (Changing lessons_learned to the memory content)
        extracted: ExtractedFacts = {
            "lessons_learned": memory_content,
            "user_preferences": "Preferences synchronized with long-term memory.",
            "is_novel": len(save_results.get("results", [])) > 0
        }

        # Update and emit state
        state["extracted_facts"] = extracted
        state["is_complete"] = True
        await copilotkit_emit_state(config, state)

        return extracted

    async def retrieve_episodic_memory(self, task_description: str, user_id: str, top_k: int = 5) -> str:
        """
        Retrieves relevant extracted facts from the user's history.
        """
        print(f"DEBUG: Retrieval Attempt - UserID: {user_id} | Query: {task_description}")

        search_results = await self.memory.search(
            query=task_description,
            filters={"user_id": user_id}
        )
        print("Episodic Memory - Search Results:", search_results)


        facts = [res["memory"] for res in search_results.get("results", [])]

        if not facts:
            return "No relevant past interactions found."

        return "\n".join([f"- {fact}" for fact in facts])