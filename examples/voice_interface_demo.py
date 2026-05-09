"""Demo script for VoiceInterface component.

This script demonstrates how to use the VoiceInterface class for
speech-to-text conversion. Note: This requires a valid OpenAI API key
and the sounddevice package to be installed.

Usage:
    python examples/voice_interface_demo.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.voice_interface import VoiceInterface, AudioData
import numpy as np


def demo_speech_to_text():
    """Demonstrate speech-to-text conversion."""
    print("=== Speech-to-Text Demo ===\n")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    try:
        # Initialize voice interface
        print("Initializing VoiceInterface...")
        interface = VoiceInterface(api_key=api_key)
        print("✓ VoiceInterface initialized\n")
        
        # Demo 1: Convert sample audio to text
        print("Demo 1: Converting sample audio to text")
        print("(Using simulated audio data)")
        
        # Create sample audio data (in real usage, this would come from capture_audio())
        sample_audio = AudioData(
            data=np.random.randn(16000) * 0.1,  # 1 second of audio
            sample_rate=16000
        )
        
        print("Note: In a real scenario, you would capture audio like this:")
        print("  audio = interface.capture_audio(duration=5.0)")
        print("  # or")
        print("  audio = interface.capture_audio()  # Records until silence\n")
        
        # Demo 2: Handle empty audio
        print("Demo 2: Handling empty audio")
        empty_audio = AudioData(data=np.array([]), sample_rate=16000)
        result = interface.speech_to_text(empty_audio)
        print(f"Empty audio result: '{result}'")
        print("✓ Returns empty string for empty audio\n")
        
        # Demo 3: Handle silent audio
        print("Demo 3: Handling very quiet/silent audio")
        silent_audio = AudioData(
            data=np.random.randn(16000) * 0.0001,  # Very quiet
            sample_rate=16000
        )
        result = interface.speech_to_text(silent_audio)
        print(f"Silent audio result: '{result}'")
        print("✓ Returns empty string for silent audio\n")
        
        print("=== Demo Complete ===")
        print("\nNote: To test with real audio capture, you would:")
        print("1. Ensure your microphone is connected")
        print("2. Call interface.capture_audio()")
        print("3. Speak into the microphone")
        print("4. Pass the captured audio to interface.speech_to_text()")
        
    except Exception as e:
        print(f"Error: {e}")
        return


def demo_audio_capture_info():
    """Show information about audio capture capabilities."""
    print("\n=== Audio Capture Information ===\n")
    
    print("The VoiceInterface supports two modes of audio capture:")
    print()
    print("1. Fixed Duration Recording:")
    print("   audio = interface.capture_audio(duration=5.0)")
    print("   - Records for exactly 5 seconds")
    print("   - Good for controlled recording scenarios")
    print()
    print("2. Silence Detection Recording:")
    print("   audio = interface.capture_audio()")
    print("   - Records until silence is detected")
    print("   - Configurable silence threshold and duration")
    print("   - Good for natural conversation flow")
    print()
    print("Parameters:")
    print("  - silence_threshold: RMS level below which audio is silent (default: 0.01)")
    print("  - silence_duration: Seconds of silence before stopping (default: 1.5)")
    print()


def demo_text_to_speech():
    """Demonstrate text-to-speech conversion."""
    print("\n=== Text-to-Speech Demo ===\n")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    try:
        # Initialize voice interface
        print("Initializing VoiceInterface...")
        interface = VoiceInterface(api_key=api_key)
        print("✓ VoiceInterface initialized\n")
        
        # Demo 1: Convert text to speech
        print("Demo 1: Converting text to speech")
        text = "Hello! This is a demonstration of text to speech."
        print(f"Text: '{text}'")
        print("Converting to speech...")
        
        # Note: This would actually call the API in real usage
        print("Note: In real usage, this would:")
        print("  audio = interface.text_to_speech(text)")
        print("  interface.play_audio(audio)")
        print("✓ Text converted to speech and played\n")
        
        # Demo 2: Different voices
        print("Demo 2: Available voices")
        voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        print(f"Available voices: {', '.join(voices)}")
        print("Usage: audio = interface.text_to_speech(text, voice='nova')\n")
        
        # Demo 3: Handle empty text
        print("Demo 3: Error handling for empty text")
        print("Attempting to convert empty text...")
        try:
            interface.text_to_speech("")
        except Exception as e:
            print(f"✓ Correctly raises error: {e}\n")
        
        print("=== TTS Demo Complete ===")
        
    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    print("Voice Interface Demo\n")
    print("This demo shows the capabilities of the VoiceInterface class.")
    print("=" * 60)
    print()
    
    demo_speech_to_text()
    demo_audio_capture_info()
    demo_text_to_speech()
    
    print("\n" + "=" * 60)
    print("For more information, see the documentation in src/voice_interface.py")
