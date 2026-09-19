"""Configuration management for the Voice AI Agent."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    groq_api_key: str
    openai_api_key: str
    openai_model: str
    openai_tts_voice: str
    openai_whisper_model: str
    data_dir: str

def load_config() -> Config:
    """Load configuration from environment variables without raising at import.

    The API key is validated lazily via `config.groq_api_key` so that unit tests
    and offline tooling can import the package without a real key. Deployment
    surfaces (FastAPI startup, voice agent CLI) must call `config.validate()`
    to fail fast with a clear message when GROQ_API_KEY is missing.
    """
    return Config(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_key=os.getenv("GROQ_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b"),
        openai_tts_voice=os.getenv("OPENAI_TTS_VOICE", "en"),
        openai_whisper_model=os.getenv("OPENAI_WHISPER_MODEL", "whisper-large-v3"),
        data_dir=os.path.expanduser(os.getenv("DATA_DIR", "~/.voice-agent"))
    )

def validate_config(cfg: Config) -> None:
    """Raise ValueError if required configuration is missing or empty."""
    if not cfg.groq_api_key or not str(cfg.groq_api_key).strip():
        raise ValueError("GROQ_API_KEY environment variable is required")

# Module-level singleton (validated lazily; see validate_config)
config = load_config()
