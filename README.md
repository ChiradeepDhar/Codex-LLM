# LLM + Agentic AI Project Assistant

This is a compact Python project for learning and using LLM + agentic AI concepts in a practical way. The main use case is a local project assistant that can inspect safe text files, answer questions about the workspace, and generate saved project briefs.

## What is included

- `basic`: send a direct prompt to a model and print the response
- `demo`: run a no-cost offline walkthrough and save a demo report
- `reports`: list saved Markdown reports and JSON traces
- `ask`: ask questions about the current workspace
- `agent`: run a tool-using assistant with optional JSON trace output
- `project-brief`: generate a saved Markdown report under `runs/`
- Read-only tools for listing files, reading safe text files, searching text, summarizing the workspace, simple arithmetic, and current time
- Production-oriented basics: environment config, log level, request timeout, retry count, file-size guardrails, secret handling, and local artifacts

## Project structure

```text
.
|-- .env.example
|-- pyproject.toml
|-- README.md
`-- src/
    `-- llm_agent_starter/
        |-- agentic.py
        |-- client.py
        |-- config.py
        |-- llm_basic.py
        |-- logging_config.py
        |-- main.py
        |-- reporting.py
        `-- tools.py
```

## OpenAI setup

Create a `.env` file and keep your real API key there only. Do not put real keys in `.env.example`.

```text
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_key_here

APP_ENV=local
LOG_LEVEL=INFO
OUTPUT_DIR=runs
MAX_FILE_BYTES=20000
REQUEST_TIMEOUT_SECONDS=60
OPENAI_MAX_RETRIES=2
ALLOWED_EXTENSIONS=.py,.md,.txt,.toml,.json,.yaml,.yml,.csv,.env.example
```

Install and check the project:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
python -m llm_agent_starter.main doctor
```

## Useful commands

```powershell
python -m llm_agent_starter.main show-config
python -m llm_agent_starter.main doctor
python -m llm_agent_starter.main demo
python -m llm_agent_starter.main reports
python -m llm_agent_starter.main basic --prompt "Explain agentic AI in practical terms."
python -m llm_agent_starter.main ask "What are the important files in this project?"
python -m llm_agent_starter.main agent --task "Read README.md and summarize how to run this project." --save-trace
python -m llm_agent_starter.main project-brief
```

## Ollama alternative

If you prefer local models, install Ollama from `https://ollama.com/`, pull a model, and switch `.env` to the local endpoint:

```powershell
ollama pull llama3.2:3b
```

```text
MODEL_PROVIDER=ollama
MODEL_NAME=llama3.2:3b
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
```

## Safety notes

- The agent blocks direct reads of `.env`.
- Tools are intentionally read-only.
- File reads are limited by `MAX_FILE_BYTES`.
- Output artifacts are written to `OUTPUT_DIR`, which defaults to `runs/`.
- Use `python -m llm_agent_starter.main reports` to find generated Markdown reports and JSON traces.
- `.env.example` is a template only and must never contain real secrets.

## Next production steps

- Add authentication and a job queue before exposing this beyond your machine
- Add tests around tool permissions and path traversal
- Add retrieval over a document folder or database
- Add a web UI with FastAPI or Streamlit
- Add observability for token usage, latency, and tool execution errors
