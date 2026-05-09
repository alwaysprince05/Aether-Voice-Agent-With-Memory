"""Configuration management for the Voice AI Agent."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    groq_api_key: str
    openai_model: str
    openai_tts_voice: str
    openai_whisper_model: str
    data_dir: str

def load_config() -> Config:
    """Load and validate configuration from environment variables.
    
    Raises:
        ValueError: If GROQ_API_KEY is not set.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")
        
    return Config(
        groq_api_key=groq_api_key,
        openai_model=os.getenv("OPENAI_MODEL", "llama-3.1-8b-instant"),
        openai_tts_voice=os.getenv("OPENAI_TTS_VOICE", "en"),
        openai_whisper_model=os.getenv("OPENAI_WHISPER_MODEL", "whisper-large-v3"),
        data_dir=os.path.expanduser(os.getenv("DATA_DIR", "~/.voice-agent"))
    )

# Module-level singleton
config = load_config()
