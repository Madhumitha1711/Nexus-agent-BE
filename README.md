# Nexus Agent

Nexus Agent is a sophisticated, multi-faceted AI agent designed to handle complex support tasks by leveraging a combination of memory systems, planning capabilities, and tool usage.

## Features

- **Multi-Modal Input:** Ingests data from various sources including chat, support tickets, and raw text.
- **Task Classification:** Automatically classifies incoming tasks to determine intent, severity, and urgency.
- **Dynamic Planning:** A planning agent that dynamically selects the appropriate processing nodes based on the task requirements.
- **Hybrid Memory System:**
    - **Episodic Memory:** Remembers past interactions.
    - **Semantic Memory:** Accesses technical documentation and knowledge bases.
- **Tool Integration:** Utilizes external tools to perform actions like fetching ticket details.
- **Reasoning and Synthesis:** A reasoning engine that synthesizes information from all sources to generate a comprehensive resolution plan.
- **Human-in-the-Loop:** Includes a human review node to handle low-confidence situations or for escalations.
- **Real-Time UI Updates:** Streams state and final reports to a user interface in real-time.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- `uv` for package management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd nexus-agent
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install dependencies using `uv`:**
    ```bash
    uv pip install -r requirements.txt
    ```

### Configuration

1.  Create a `.env` file in the root of the project.
2.  Add the necessary environment variables to the `.env` file. At a minimum, you will need your `OPENAI_API_KEY`:
    ```
    OPENAI_API_KEY="your-openai-api-key"
    ```

## Usage

### Running the Agent

To start the Nexus Agent, run the `main.py` script:

```bash
python main.py
```

This will start the FastAPI server.

### API

Once the server is running, you can access the API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Project Structure

```
nexus-agent/
├── src/
│   ├── nexus/
│   │   ├── nodes/         # Core logic for each node in the graph
│   │   ├── memory/        # Memory systems (episodic, semantic)
│   │   ├── guardrails/    # Input/output validation
│   │   ├── tools/         # External tool integrations
│   │   └── utils/         # Utility functions and state definitions
│   └── api/             # FastAPI server and endpoints
├── tests/                 # Test suite
├── main.py                # Main application entry point
├── pyproject.toml         # Project metadata and dependencies
└── README.md              # This file
```

## Running Tests

To run the test suite, use `pytest`:

```bash
pytest
```

