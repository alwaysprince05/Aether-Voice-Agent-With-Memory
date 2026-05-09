# Voice AI Agent with Memory — Implementation Guide

This document provides complete, detailed instructions for an LLM to implement all remaining tasks in the project. It includes the full context of what already exists, what needs to be built, exact file paths, function signatures, and behavioral contracts.

---

## Project State Summary

### Already Implemented (do not recreate)

| File | Status | What it contains |
|------|--------|-----------------|
| `src/models.py` | Complete | `ToDoItem`, `Memory`, `ToolCall`, `ToolResult`, `AgentResponse` dataclasses |
| `src/todo_manager.py` | Complete | `ToDoManager` with full CRUD + JSON persistence at `~/.voice-agent/todos.json` |
| `src/memory_system.py` | Complete | `MemorySystem` with store/retrieve/search + JSON persistence + optional OpenAI embeddings |
| `src/voice_interface.py` | Complete | `VoiceInterface` with `capture_audio`, `speech_to_text`, `text_to_speech`, `play_audio` |
| `src/agent_core.py` | Complete | `AgentCore` with OpenAI function calling, tool execution, conversation context, memory integration |
| `tests/unit/test_todo_manager.py` | Complete | Full unit test suite for ToDoManager |
| `tests/unit/test_memory_system.py` | Complete | Full unit test suite for MemorySystem |
| `tests/unit/test_agent_core.py` | Complete | Full unit test suite for AgentCore |

### Remaining Tasks (implement these)

| Task | File(s) to create |
|------|------------------|
| 8. Checkpoint | Run `pytest tests/` — all existing tests must pass |
| 9.1 System prompt | `src/prompts.py` |
| 10.1 VoiceAgent orchestrator | `src/voice_agent.py` |
| 10.2 Configuration management | `src/config.py` |
| 11.1 CLI demo script | `demo.py` |
| 11.2 README update | `README.md` (update existing) |
| 12. Final checkpoint | Run `pytest tests/` — all tests must pass |

---

## Runtime Environment

- Python 3.10+
- Dependencies in `requirements.txt`:
  ```
  openai>=1.0.0
  hypothesis>=6.0.0
  pytest>=7.0.0
  python-dotenv>=1.0.0
  pydantic>=2.0.0
  sounddevice>=0.4.6
  numpy>=1.24.0
  pydub>=0.25.0
  ```
- All imports use `from src.X import Y` style (project root is on `sys.path`)
- OpenAI API key loaded from `.env` file via `python-dotenv`

---

## Task 8: Checkpoint — Ensure All Tests Pass

Before writing any new code, run the existing test suite:

```bash
pytest tests/ -v
```

All tests in `tests/unit/` must pass. If any fail, fix the underlying source file before proceeding. Do not modify test files to make them pass — fix the implementation.

---

## Task 9.1: Create `src/prompts.py`

### Purpose

A standalone module exporting a single `SYSTEM_PROMPT` constant used by `AgentCore`. The prompt already exists inline in `AgentCore._get_default_system_prompt()` — this task externalises it into a dedicated file and expands it.

### File: `src/prompts.py`

```python
"""System prompts for the Voice AI Agent."""

SYSTEM_PROMPT = """..."""  # see full content below
```

### `SYSTEM_PROMPT` content requirements

The prompt must cover all of the following (Requirements 15.1–15.4):

1. **Agent identity and personality** — friendly, helpful, voice-optimised assistant
2. **Capabilities list** — to-do management and memory
3. **Tool usage rules** — when to call each of the 6 tools
4. **Memory decision rules** — when to store vs. retrieve
5. **Response style** — concise, natural, conversational (suitable for TTS)

#### Exact tool names the prompt must reference

| Tool name | When to use |
|-----------|-------------|
| `create_todo` | User wants to add a new task |
| `list_todos` | User wants to see their tasks |
| `update_todo` | User wants to change a task's description or status |
| `delete_todo` | User wants to remove a task |
| `store_memory` | User shares a preference, fact, or important information |
| `recall_memories` | Context from past conversations is needed to answer well |

#### Memory storage decision rules the prompt must encode

- Store when: user shares a preference, personal fact, recurring schedule, or explicit "remember this"
- Do NOT store: transient requests, one-off questions, to-do operations themselves
- Retrieve automatically before responding when the query is personal or context-dependent

#### Response style rules the prompt must encode

- Keep responses short — they will be spoken aloud via TTS
- Avoid markdown, bullet points, or formatting in responses
- Confirm tool actions naturally ("Done, I've added that to your list")
- When a tool fails, explain briefly and offer an alternative

### Integration point

`AgentCore.__init__` already calls `self._get_default_system_prompt()` when no `system_prompt` is passed. After creating `src/prompts.py`, update `src/agent_core.py` to import and use `SYSTEM_PROMPT`:

```python
# In src/agent_core.py — replace _get_default_system_prompt body:
from src.prompts import SYSTEM_PROMPT

def _get_default_system_prompt(self) -> str:
    return SYSTEM_PROMPT
```

This change must not break any existing tests in `tests/unit/test_agent_core.py`.

---

## Task 10.2: Create `src/config.py`

### Purpose

Centralised configuration loaded from environment variables (via `.env`). All other components read config from here instead of calling `os.getenv` directly.

### File: `src/config.py`

```python
"""Configuration management for the Voice AI Agent."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    openai_api_key: str
    openai_model: str
    openai_tts_model: str
    openai_tts_voice: str
    openai_whisper_model: str
    data_dir: str

def load_config() -> Config:
    """Load and validate configuration from environment variables.
    
    Raises:
        ValueError: If OPENAI_API_KEY is not set.
    """
    ...

# Module-level singleton
config = load_config()
```

### Environment variables

| Variable | Default | Required |
|----------|---------|----------|
| `OPENAI_API_KEY` | — | Yes — raise `ValueError` if missing |
| `OPENAI_MODEL` | `"gpt-4"` | No |
| `OPENAI_TTS_MODEL` | `"tts-1"` | No |
| `OPENAI_TTS_VOICE` | `"alloy"` | No |
| `OPENAI_WHISPER_MODEL` | `"whisper-1"` | No |
| `DATA_DIR` | `"~/.voice-agent"` | No |

### Validation rules

- `OPENAI_API_KEY` must be non-empty string — raise `ValueError("OPENAI_API_KEY environment variable is required")` if missing or empty
- `DATA_DIR` must be expanded with `os.path.expanduser()`
- All other variables use their defaults silently if not set

### `.env.example` (already exists — verify it contains these keys)

```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_TTS_MODEL=tts-1
OPENAI_TTS_VOICE=alloy
OPENAI_WHISPER_MODEL=whisper-1
DATA_DIR=~/.voice-agent
```

---

## Task 10.1: Create `src/voice_agent.py`

### Purpose

The top-level orchestrator that wires `ToDoManager`, `MemorySystem`, `VoiceInterface`, and `AgentCore` together into a single `VoiceAgent` class with a conversation loop.

### File: `src/voice_agent.py`

#### Imports

```python
import uuid
import signal
import sys
from typing import Optional
from openai import OpenAI

from src.config import config
from src.todo_manager import ToDoManager
from src.memory_system import MemorySystem
from src.voice_interface import VoiceInterface
from src.agent_core import AgentCore
```

#### Class: `VoiceAgent`

```python
class VoiceAgent:
    def __init__(self, text_mode: bool = False):
        """Initialise all components using values from config.
        
        Args:
            text_mode: If True, use stdin/stdout instead of microphone/speakers.
                       Useful for testing without audio hardware.
        """

    def run(self) -> None:
        """Start the main conversation loop.
        
        Loops indefinitely until the user says "quit", "exit", "bye",
        or presses Ctrl+C (SIGINT).
        
        In voice mode:
          1. Call voice_interface.capture_audio()
          2. Call voice_interface.speech_to_text(audio)
          3. If text is empty, print "I didn't catch that, please try again." and loop
          4. Call agent_core.process_input(text, session_id=self._session_id)
          5. Call voice_interface.text_to_speech(response.text)
          6. Call voice_interface.play_audio(audio)
        
        In text mode:
          1. Print "> " prompt and read from stdin
          2. Call agent_core.process_input(text, session_id=self._session_id)
          3. Print response.text to stdout
        
        On any exception during a turn: print the error and continue the loop
        (do not crash the agent on a single bad turn).
        """

    def _handle_exit(self, text: str) -> bool:
        """Return True if text is an exit command (quit/exit/bye, case-insensitive)."""

    def _setup_signal_handler(self) -> None:
        """Register SIGINT handler to exit gracefully with a farewell message."""
```

#### Initialisation details

Inside `__init__`, construct components in this order:

```python
self._session_id = str(uuid.uuid4())
self._text_mode = text_mode

openai_client = OpenAI(api_key=config.openai_api_key)

self._todo_manager = ToDoManager(
    storage_path=f"{config.data_dir}/todos.json"
)
self._memory_system = MemorySystem(
    storage_path=f"{config.data_dir}/memories.json",
    openai_client=openai_client
)
self._agent_core = AgentCore(
    openai_client=openai_client,
    todo_manager=self._todo_manager,
    memory_system=self._memory_system
)

if not text_mode:
    self._voice_interface = VoiceInterface(api_key=config.openai_api_key)
else:
    self._voice_interface = None
```

#### Session management

- Each `VoiceAgent` instance gets a unique `session_id` (UUID4) at construction time
- Pass this `session_id` to every `agent_core.process_input()` call
- This ensures conversation context is isolated per agent instance

#### Error handling in the loop

Wrap each turn in a `try/except Exception`:
- In voice mode: speak the error message via TTS if possible, otherwise print it
- In text mode: print the error message
- Always continue the loop — never let a single turn crash the agent

---

## Task 11.1: Create `demo.py`

### Purpose

A runnable CLI entry point. Supports both voice mode (default) and text mode (`--text-mode` flag).

### File: `demo.py` (project root)

```python
#!/usr/bin/env python3
"""Voice AI Agent demo application."""

import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Voice AI Agent with Memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                  # Voice mode (microphone + speakers)
  python demo.py --text-mode      # Text mode (keyboard + terminal)
        """
    )
    parser.add_argument(
        "--text-mode",
        action="store_true",
        help="Use text input/output instead of voice"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Import here so config errors surface cleanly
    try:
        from src.voice_agent import VoiceAgent
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set OPENAI_API_KEY in your .env file.")
        sys.exit(1)
    
    agent = VoiceAgent(text_mode=args.text_mode)
    
    mode = "text" if args.text_mode else "voice"
    print(f"Voice AI Agent started in {mode} mode.")
    if args.text_mode:
        print("Type your message and press Enter. Type 'quit' to exit.")
    else:
        print("Speak to interact. Say 'quit' or press Ctrl+C to exit.")
    print()
    
    agent.run()

if __name__ == "__main__":
    main()
```

### Behaviour requirements

- `--text-mode` flag passes `text_mode=True` to `VoiceAgent`
- Configuration errors (missing API key) must print a clear message and exit with code 1
- Startup prints the mode so the user knows what to expect
- `main()` is the only entry point — no logic outside it

---

## Task 11.2: Update `README.md`

The `README.md` already exists with a good structure. Update the following sections:

### Section: "Usage" — update the demo command

```markdown
### Running the Agent

Start in voice mode (requires microphone and speakers):
```bash
python demo.py
```

Start in text mode (no audio hardware required):
```bash
python demo.py --text-mode
```
```

### Section: "Example Voice Commands" — add memory examples

```markdown
**Memory:**
- "Remember that I prefer morning meetings"
- "I like to work in 2-hour focused blocks"
- "What do you know about my preferences?"
- "Do you remember what I told you about my schedule?"
```

### Section: "Project Structure" — verify it lists all new files

Ensure the structure block includes:
```
│   ├── prompts.py         # System prompts for the agent
│   ├── config.py          # Configuration management
│   └── voice_agent.py     # Main application orchestrator
```
and at the root:
```
├── demo.py                # CLI entry point
```

### Section: "Troubleshooting" — add a new subsection

```markdown
### Agent Issues

**Agent doesn't use tools:**
- Ensure your request clearly mentions a task action ("add", "list", "delete", "mark as done")
- The agent uses conversational responses for general questions

**Memory not persisting:**
- Check that DATA_DIR is writable
- Verify ~/.voice-agent/ directory exists after first run
- Check for IOError messages in the console output
```

---

## Task 12: Final Checkpoint — Ensure All Tests Pass

Run the full test suite:

```bash
pytest tests/ -v
```

All tests must pass. The test suite covers:
- `tests/unit/test_models.py` — data model validation
- `tests/unit/test_todo_manager.py` — CRUD + persistence
- `tests/unit/test_memory_system.py` — memory storage + semantic search
- `tests/unit/test_voice_interface.py` — STT/TTS with mocked APIs
- `tests/unit/test_agent_core.py` — LLM orchestration with mocked dependencies

If any test fails after your changes, fix the implementation (not the test).

---

## Key Interfaces Reference

These are the exact signatures of the already-implemented components you will wire together. Do not change these.

### `AgentCore.process_input`

```python
def process_input(self, user_input: str, session_id: str = "default") -> AgentResponse:
```

Returns `AgentResponse(text: str, tool_calls: List[ToolCall], success: bool, error: Optional[str])`.

### `VoiceInterface.capture_audio`

```python
def capture_audio(self, duration: Optional[float] = None, ...) -> AudioData:
```

### `VoiceInterface.speech_to_text`

```python
def speech_to_text(self, audio: AudioData) -> str:
```

Returns empty string for silent audio. Raises `STTError` on API failure.

### `VoiceInterface.text_to_speech`

```python
def text_to_speech(self, text: str, voice: str = "alloy") -> AudioData:
```

Raises `TTSError` if text is empty or API fails.

### `VoiceInterface.play_audio`

```python
def play_audio(self, audio: AudioData) -> None:
```

### `ToDoManager.__init__`

```python
def __init__(self, storage_path: Optional[str] = None):
```

Default storage: `~/.voice-agent/todos.json`

### `MemorySystem.__init__`

```python
def __init__(self, storage_path: Optional[str] = None, openai_client=None):
```

Default storage: `~/.voice-agent/memories.json`. Pass `openai_client` to enable semantic search.

---

## Implementation Order

Follow this order to avoid import errors:

1. `src/config.py` — no internal dependencies
2. `src/prompts.py` — no internal dependencies
3. Update `src/agent_core.py` — import `SYSTEM_PROMPT` from `src/prompts.py`
4. `src/voice_agent.py` — depends on config, all four components
5. `demo.py` — depends on `VoiceAgent`
6. Update `README.md` — documentation only

---

## Correctness Checklist

Before marking any task complete, verify:

- [ ] `pytest tests/ -v` passes with zero failures
- [ ] `python demo.py --text-mode` starts without error (requires valid `.env`)
- [ ] `src/config.py` raises `ValueError` when `OPENAI_API_KEY` is absent
- [ ] `src/prompts.py` exports `SYSTEM_PROMPT` as a module-level string constant
- [ ] `src/voice_agent.py` uses a unique `session_id` per instance
- [ ] `demo.py` exits with code 1 on config error, not an unhandled exception
- [ ] `VoiceAgent.run()` never crashes on a single bad turn — it catches and continues
