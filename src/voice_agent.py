"""Main application orchestrator for Voice AI Agent."""

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

class VoiceAgent:
    def __init__(self, text_mode: bool = False):
        """Initialise all components using values from config.
        
        Args:
            text_mode: If True, use stdin/stdout instead of microphone/speakers.
                       Useful for testing without audio hardware.
        """
        self._session_id = str(uuid.uuid4())
        self._text_mode = text_mode
        
        openai_client = OpenAI(
            api_key=config.groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        
        self._todo_manager = ToDoManager(
            storage_path=f"{config.data_dir}/todos.json"
        )
        self._memory_system = MemorySystem(
            storage_path=f"{config.data_dir}/memories.json"
        )
        self._agent_core = AgentCore(
            openai_client=openai_client,
            todo_manager=self._todo_manager,
            memory_system=self._memory_system
        )
        
        if not text_mode:
            self._voice_interface = VoiceInterface(api_key=config.groq_api_key)
        else:
            self._voice_interface = None
            
        self._setup_signal_handler()

    def run(self) -> None:
        """Start the main conversation loop."""
        while True:
            try:
                if self._text_mode:
                    print("> ", end="", flush=True)
                    user_input = sys.stdin.readline().strip()
                    if not user_input:
                        continue
                    if self._handle_exit(user_input):
                        break
                    
                    response = self._agent_core.process_input(user_input, session_id=self._session_id)
                    print(response.text)
                else:
                    audio = self._voice_interface.capture_audio()
                    user_input = self._voice_interface.speech_to_text(audio)
                    if not user_input:
                        print("I didn't catch that, please try again.")
                        continue
                    
                    print(f"\n[YOU]: {user_input}")
                        
                    if self._handle_exit(user_input):
                        break
                        
                    response = self._agent_core.process_input(user_input, session_id=self._session_id)
                    print(f"[AI]: {response.text}")
                    
                    audio_out = self._voice_interface.text_to_speech(response.text, voice=config.openai_tts_voice)
                    self._voice_interface.play_audio(audio_out)
                    
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                if self._text_mode:
                    print(error_msg)
                else:
                    try:
                        audio_out = self._voice_interface.text_to_speech("I encountered an error.", voice=config.openai_tts_voice)
                        self._voice_interface.play_audio(audio_out)
                    except Exception:
                        print(error_msg)
                continue
                
        print("\nGoodbye!")

    def _handle_exit(self, text: str) -> bool:
        """Return True if text is an exit command (quit/exit/bye, case-insensitive)."""
        clean_text = text.strip().lower().rstrip('.!')
        return clean_text in ("quit", "exit", "bye", "goodbye")

    def _setup_signal_handler(self) -> None:
        """Register SIGINT handler to exit gracefully with a farewell message."""
        def handler(signum, frame):
            print("\nGoodbye!")
            sys.exit(0)
        signal.signal(signal.SIGINT, handler)
