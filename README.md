---
title: AETHER Voice Agent
emoji: "🤖"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# AETHER Voice Agent

Personal voice-first AI assistant project by `alwaysprince05`, built with FastAPI, Groq-compatible OpenAI SDK calls, long-term memory, and task management.

## Live Deployment

- **View Live App**: [AETHER Voice Agent on Hugging Face](https://huggingface.co/spaces/alwaysprince05/aether_voice_agent_with_memory)

## What This Project Does

`AETHER` lets you talk to an AI agent through a browser UI and get:

- conversational replies from an LLM
- speech-to-text transcription for voice input
- text-to-speech playback for AI responses
- persistent memory storage for user facts/context
- built-in to-do management via natural language

The app runs as a full-stack system:

- backend: `FastAPI` (`src/api.py`)
- frontend: `HTML/CSS/JS` (`frontend/`)
- storage: local JSON files in the configured data directory

## Core Features

- **Voice chat pipeline**: Browser microphone -> `/api/voice` -> transcription -> agent response -> optional TTS.
- **Text chat pipeline**: UI text input -> `/api/chat` -> agent response + base64 audio.
- **Long-term memory**: Stores and recalls user details through `MemorySystem`.
- **Task manager**: Create/list/update/complete tasks using `ToDoManager`.
- **Web dashboard**: Visual cards for interactions, tasks, and memory context.
- **Docker-ready deployment**: Run locally with `docker-compose`.

## Tech Stack

- Python `3.10+` (3.11+ recommended)
- FastAPI + Uvicorn + Gunicorn
- OpenAI Python SDK (used with Groq-compatible base URL)
- gTTS for response audio generation
- Vanilla JavaScript + HTML + CSS frontend
- Docker + Docker Compose

## Project Structure

```text
Aether-Voice-Agent-With-Memory/
├── src/
│   ├── api.py
│   ├── agent_core.py
│   ├── memory_system.py
│   ├── todo_manager.py
│   ├── voice_agent.py
│   ├── voice_interface.py
│   ├── models.py
│   └── config.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── tests/
├── docs/
├── examples/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick Start

### 1) Prerequisites

- Python `3.14+` (for local run)
- Docker + Docker Compose (optional, recommended for container run)
- A Groq API key

### 2) Environment Setup

Create `.env` in the repo root (see `.env.example` for all options):

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys).
> The app refuses to start without it (clear error message at startup).

### 3) Run with Docker

```bash
docker-compose up --build
```

Open: [http://localhost:8000](http://localhost:8000)

### 4) Run Locally (Without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=your_groq_api_key_here
uvicorn src.api:app --reload --port 8000
```

Open: [http://localhost:8000](http://localhost:8000)

## API Endpoints

- `GET /health` - service health check
- `POST /api/chat` - text chat endpoint
- `POST /api/voice` - voice upload/transcribe/chat endpoint
- `GET /api/tts?text=...` - generate/stream TTS audio
- `GET /api/todos` - list stored to-dos
- `GET /api/memories` - list stored memories

## Deploying to Hugging Face Spaces

The Space runs the Docker SDK with `app_port: 8000`.

1. Create a Space (SDK: **Docker**) and push this repo to it.
2. Go to **Space → Settings → Variables and secrets** and add a **secret**:
   - Name: `GROQ_API_KEY` — Value: your Groq key.
3. Restart the Space (Settings → **Restart Space**) after changing secrets.

### If the deployed app returns errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| Chat replies with `401 - Invalid API Key` | Missing/expired `GROQ_API_KEY` secret in the Space | Re-add the secret under Settings → Variables and secrets, then restart the Space |
| Space shows `Runtime error` / builds forever | Dependency or startup failure | Check the Space **Logs** tab; the app now fails fast with a clear config message |
| Frontend loads but voice features fail | Browser blocked the microphone | Allow mic access for the `*.hf.space` domain |

## Development Notes

- Main backend entrypoint: `src/api.py`
- Static frontend is mounted by FastAPI from `frontend/`
- Local data persistence defaults to the configured `data_dir`
- Unit tests are available in `tests/unit`

## Running Tests

```bash
pytest
```

## Roadmap

- improve memory relevance/ranking
- add robust auth and multi-user sessions
- enhance frontend state management and error UX
- add CI pipeline and production deployment templates
- expand test coverage for API routes

## Author

- **Prince Kumar Maurya** (`alwaysprince05`)
- GitHub: [alwaysprince05](https://github.com/alwaysprince05)
- Hugging Face: [alwaysprince05e](https://huggingface.co/alwaysprince05e)

## License

Use and distribution terms should follow the project license file in this repository.
