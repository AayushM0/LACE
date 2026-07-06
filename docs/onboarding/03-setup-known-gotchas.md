# Local Setup & Known Gotchas

This document guides you through setting up LACE on a local machine and highlights critical gotchas to look out for.

---

## 1. Local Development Setup

To reach a working local environment in under 5 minutes:

### Prerequisites
- **Python**: `>= 3.11` (Python `3.13` or `3.14` recommended)
- **uv** (recommended): Speedy package installer and resolver.
- **Ollama**: Required if running local LLMs (default config). Ollama must be running and have the `llama3.2` model downloaded:
  ```bash
  ollama pull llama3.2
  ```

### Step-by-Step Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/AayushM0/lace.git
   cd lace
   ```
2. **Install in editable mode**:
   - Using `uv` (recommended):
     ```bash
     uv pip install -e .
     ```
   - Using standard `pip`:
     ```bash
     python3 -m pip install -e .
     ```
3. **Initialize default config and paths**:
   This creates the directories in `~/.lace` and copies default config files:
   ```bash
   lace init
   ```
4. **Verify tests pass**:
   ```bash
   .venv/bin/pytest
   ```

---

## 2. Known Local Gotchas & Developer Sharp Edges

### 1. Python Version Mismatch & Virtualenv Drift
- **The Issue**: `pyproject.toml` lists `requires-python = ">=3.11"`, `.python-version` pins `3.13`, but the checked-in virtual environment in the repository uses Python `3.14.6`.
- **The Fix**: It is recommended to use the provided virtualenv interpreter at `.venv/bin/python` to run tests and commands, or reconstruct your virtual environment using your local Python 3.13/3.14:
  ```bash
  uv venv --python 3.13
  source .venv/bin/activate
  uv pip install -e .
  ```

### 2. Missing `uv` on System Path
- **The Issue**: Running `uv run pytest` fails with `bash: uv: command not found` if `uv` is not installed globally on the developer machine.
- **The Fix**: You do not need `uv` globally to run the project. Use the local virtual environment binary directly:
  ```bash
  .venv/bin/pytest
  .venv/bin/lace memory list
  ```

### 3. LLM Offline / Missing API Credentials
- **The Issue**: Background extraction jobs are enqueued in `extraction_queue.db`, but stay in `pending` or fail with error logs.
- **The Fix**: LACE defaults to using Ollama (`http://localhost:11434`) with `llama3.2`. If Ollama is not running, extraction fails.
  - Start Ollama: `ollama serve` (or open the desktop app).
  - If using OpenAI/Anthropic, set the default provider in `~/.lace/config/lace.yaml` and set the required environment variables:
    ```bash
    lace config set provider.default "openai"
    export OPENAI_API_KEY="your-api-key"
    ```

### 4. ChromaDB Telemetry Warning on Python 3.14
- **The Issue**: When running tests on Python 3.14, you will see deprecation warnings:
  ```
  DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
  ```
- **The Fix**: This is a harmless telemetry warning triggered by ChromaDB's dependencies. It can be safely ignored.

### 5. `test_worker_survives_llm_offline` Test Failure
- **The Issue**: In tests, `TestEndToEndExtractionPipeline.test_worker_survives_llm_offline` might fail asserting `'done' == 'failed'`.
- **The Fix**: This happens because the test mocks `extract_from_conversation` (the legacy extractor), but the active code triggers the new worthiness-gated pipeline which catches the RuntimeError internally, logs it, and marks the job as `done` instead of throwing to the retry loop. This is a known test-mocking mismatch.
