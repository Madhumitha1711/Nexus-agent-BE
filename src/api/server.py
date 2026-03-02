import json
import os
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import uvicorn

from ag_ui_langgraph import add_langgraph_fastapi_endpoint
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File

# Updated Imports for the latest SDK
from copilotkit import LangGraphAGUIAgent
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, StreamingResponse

from src.api.mcptools import mcp
from src.nexus.graph.graph import initialize_graph
from src.nexus.memory.rag_semantic_memory import MultimodalSemanticMemory
from langgraph.types import Command

load_dotenv()
import asyncio
import ag_ui_langgraph.utils as utils

_original_make_json_safe = utils.make_json_safe


def patched_make_json_safe(value, _seen=None):
    if isinstance(value, asyncio.Future):
        return None
    if asyncio.iscoroutine(value):
        return None
    try:
        return _original_make_json_safe(value, _seen)
    except TypeError:
        return str(value)


# Apply monkey patch
utils.make_json_safe = patched_make_json_safe


@asynccontextmanager
async def lifespan(app1: FastAPI):
    # 1. Initialize the graph
    compiled_graph = await initialize_graph()

    # 2. Register the endpoint dynamically
    add_langgraph_fastapi_endpoint(
        app=app1,
        agent=LangGraphAGUIAgent(
            name="sample_agent",
            description="Agent with MCP tools",
            graph=compiled_graph,
            config={"recursion_limit": 20}
        ),
        path="/copilotkit",
    )
    yield


app = FastAPI(lifespan=lifespan)


origins = [
    "http://43.205.108.251:3000",
    "http://localhost:3000",
    "http://localhost:8123",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    # Option A: List specific origins
    allow_origins=[
        "http://43.205.108.251:3000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    # OR Option B: Use a regex to allow any origin (only if you truly want "*" + credentials)
    # allow_origin_regex="https?://.*",

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


memory_manager = MultimodalSemanticMemory()

# Temporary directory for processing uploads
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/ingest")
async def ingest_files(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...)  # Changed to 'files' and added List
):
    results = []
    for file in files:
        file_path = UPLOAD_DIR / file.filename
        # Save file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Ingest
        memory_manager.ingest_file(str(file_path))
        results.append(file.filename)

        # Cleanup
        if file_path.exists():
            os.remove(file_path)

    return {"status": "success", "uploaded": results}


@app.get("/graph/mermaid-text")
async def get_mermaid_text():
    """
    Returns the raw mermaid string.
    Use this if your frontend has a mermaid.js library to render it.
    """
    graph = await initialize_graph()
    mermaid_code = graph.get_graph().draw_mermaid()
    return {"code": mermaid_code}


@app.get("/graph/html", response_class=HTMLResponse)
async def get_graph_html():
    """
    Returns a full HTML page that renders the graph using a CDN.
    Useful for quick debugging in a browser.
    """
    graph = await initialize_graph()
    mermaid_code = graph.get_graph().draw_mermaid()
    return f"""
    <html>
        <body>
            <pre class="mermaid">
                {mermaid_code}
            </pre>
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ startOnLoad: true }});
            </script>
        </body>
    </html>
    """

@app.post("/approve")
async def approve_endpoint(request: Request):
    data = await request.json()
    print("data received at /approve:", data)
    thread_id = data.get("thread_id")
    human_feedback = data.get("feedback")
    graph = await initialize_graph()

    config = {"configurable": {"thread_id": thread_id}}

    async def resume_generator():
        # Resume the graph by sending a 'Command' with the human's input
        # This replaces the return value of the 'interrupt()' call in your node
        async for chunk in graph.astream(
            Command(resume=human_feedback),
            config,
            stream_mode="values"
        ):
            # Stream the remaining nodes (e.g., Aggregator)
            content = chunk.get("final_report", "")
            yield f"data: {json.dumps({'type': 'final', 'content': content})}\n\n"

    return StreamingResponse(resume_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    print("Combined Server running at http://localhost:8123")
    uvicorn.run(app, host="0.0.0.0", port=8123)
