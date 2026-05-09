"""Voice interface component for speech-to-text and text-to-speech conversion."""

import os
from typing import Optional, Union
from dataclasses import dataclass
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class AudioData:
    """Represents audio data with sample rate.
    
    Attributes:
        data: Audio samples as numpy array
        sample_rate: Sample rate in Hz
    """
    data: np.ndarray
    sample_rate: int


class VoiceInterfaceError(Exception):
    """Base exception for voice interface errors."""
    pass


class STTError(VoiceInterfaceError):
    """Exception for speech-to-text errors."""
    pass


class TTSError(VoiceInterfaceError):
    """Exception for text-to-speech errors."""
    pass


class AudioCaptureError(VoiceInterfaceError):
    """Exception for audio capture errors."""
    pass


class VoiceInterface:
    """Handles speech-to-text and text-to-speech conversion.
    
    This component provides methods for capturing audio from a microphone,
    converting speech to text using OpenAI Whisper API, converting text to
    speech using OpenAI TTS API, and playing audio output.
    
    Attributes:
        client: OpenAI client for API calls
        sample_rate: Audio sample rate in Hz (default: 16000)
        channels: Number of audio channels (default: 1 for mono)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        sample_rate: int = 16000,
        channels: int = 1
    ):
        """Initialize the voice interface.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono, 2 for stereo)
            
        Raises:
            VoiceInterfaceError: If OpenAI client cannot be initialized
        """
        if OpenAI is None:
            raise VoiceInterfaceError(
                "OpenAI package not installed. Install with: pip install openai"
            )
        
        if sd is None:
            raise VoiceInterfaceError(
                "sounddevice package not installed. Install with: pip install sounddevice"
            )
        
        from src.config import config
        self.api_key = api_key or config.openai_api_key
        if not self.api_key:
            raise VoiceInterfaceError(
                "Groq/OpenAI API key not provided. Set GROQ_API_KEY environment variable."
            )
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.sample_rate = sample_rate
        self.channels = channels
    
    def capture_audio(
        self,
        duration: Optional[float] = None,
        silence_threshold: float = 0.004,
        silence_duration: float = 1.0
    ) -> AudioData:
        """Captures audio from microphone until silence detected.
        
        This method records audio from the default microphone. If duration is
        specified, it records for that many seconds. Otherwise, it records until
        silence is detected (audio level below threshold for silence_duration).
        
        Args:
            duration: Recording duration in seconds (None for silence detection)
            silence_threshold: RMS threshold below which audio is considered silence
            silence_duration: Duration of silence in seconds to stop recording
            
        Returns:
            AudioData object containing the recorded audio
            
        Raises:
            AudioCaptureError: If audio capture fails
        """
        try:
            if duration is not None:
                # Record for fixed duration
                recording = sd.rec(
                    int(duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype='float32'
                )
                sd.wait()
                
                # Convert to 1D array if mono
                if self.channels == 1:
                    recording = recording.flatten()
                
                return AudioData(data=recording, sample_rate=self.sample_rate)
            
            else:
                # Record until silence detected
                chunk_duration = 0.1  # 100ms chunks
                chunk_samples = int(chunk_duration * self.sample_rate)
                silence_chunks_needed = int(silence_duration / chunk_duration)
                
                chunks = []
                silence_count = 0
                
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype='float32'
                ) as stream:
                    print("\n[LISTENING] Speak now...")
                    while True:
                        chunk, _ = stream.read(chunk_samples)
                        
                        # Convert to 1D if mono
                        if self.channels == 1:
                            chunk = chunk.flatten()
                        
                        chunks.append(chunk)
                        
                        # Calculate RMS (root mean square) for volume level
                        rms = np.sqrt(np.mean(chunk ** 2))
                        
                        # Visual volume meter (Boosted scale for visibility)
                        meter = "#" * int(rms * 250)
                        print(f"\rLevel: [{meter:<50}]", end="", flush=True)
                        
                        if rms < silence_threshold:
                            silence_count += 1
                            if silence_count >= silence_chunks_needed:
                                print("\n[PROCESSING...]")
                                break
                        else:
                            if silence_count == 0 and len(chunks) == 1:
                                print("\n[RECORDING...]")
                            silence_count = 0
                
                # Concatenate all chunks
                recording = np.concatenate(chunks)
                return AudioData(data=recording, sample_rate=self.sample_rate)
        
        except Exception as e:
            raise AudioCaptureError(f"Failed to capture audio: {str(e)}")
    
    def speech_to_text(self, audio: AudioData) -> str:
        """Converts audio to text using OpenAI Whisper API.
        
        This method takes audio data and transcribes it to text using the
        Whisper speech recognition model. It handles empty audio by returning
        an empty string, and provides detailed error messages for API failures.
        
        Args:
            audio: AudioData object containing the audio to transcribe
            
        Returns:
            Transcribed text string (empty string if audio contains no speech)
            
        Raises:
            STTError: If speech-to-text conversion fails
        """
        try:
            # Check if audio is empty or too quiet
            if len(audio.data) == 0:
                return ""
            
            # Calculate RMS to check if there's actual audio content
            rms = np.sqrt(np.mean(audio.data ** 2))
            if rms < 0.001:  # Very quiet audio, likely no speech
                return ""
            
            # Convert numpy array to WAV format in memory
            import io
            import wave
            
            # Convert float32 to int16 for WAV format
            audio_int16 = (audio.data * 32767).astype(np.int16)
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes for int16
                wav_file.setframerate(audio.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            # Reset buffer position to beginning
            wav_buffer.seek(0)
            wav_buffer.name = "audio.wav"  # Required by OpenAI API
            
            from src.config import config
            # Call Whisper API
            transcript = self.client.audio.transcriptions.create(
                model=config.openai_whisper_model,
                file=wav_buffer,
                response_format="text"
            )
            
            # Return transcribed text (strip whitespace)
            return transcript.strip() if isinstance(transcript, str) else ""
        
        except Exception as e:
            # Check for specific API errors
            error_msg = str(e)
            if "api_key" in error_msg.lower():
                raise STTError("Invalid or missing OpenAI API key")
            elif "quota" in error_msg.lower():
                raise STTError("OpenAI API quota exceeded")
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                raise STTError(f"Network error during speech recognition: {error_msg}")
            else:
                raise STTError(f"Speech recognition failed: {error_msg}")
    
    def text_to_speech(self, text: str, voice: str = "en") -> AudioData:
        """Converts text to speech using Google TTS API.
        
        Args:
            text: Text to convert to speech
            voice: Language code to use (default: "en")
            
        Returns:
            AudioData object containing the generated speech audio
            
        Raises:
            TTSError: If text-to-speech conversion fails
        """
        try:
            # Handle empty text
            if not text or not text.strip():
                raise TTSError("Text cannot be empty")
            
            import io
            import subprocess
            try:
                from gtts import gTTS
            except ImportError:
                raise TTSError(
                    "gTTS package not installed. Install with: pip install gTTS"
                )
            
            # Call gTTS API
            tts = gTTS(text=text, lang=voice)
            
            # Save to in-memory file
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            
            # Use ffmpeg directly to convert MP3 to WAV without pydub
            # This is more robust as it uses the system binary we just installed
            process = subprocess.Popen(
                ['ffmpeg', '-i', 'pipe:0', '-f', 'wav', 'pipe:1'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            wav_data, _ = process.communicate(input=mp3_fp.read())
            
            import wave
            with wave.open(io.BytesIO(wav_data), 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                frames = wav_file.readframes(n_frames)
                
                # Convert raw bytes to numpy array
                samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                samples = samples / 32768.0  # Normalize int16 to [-1, 1]
                
                return AudioData(data=samples, sample_rate=sample_rate)
        
        except Exception as e:
            raise TTSError(f"Text-to-speech conversion failed: {str(e)}")
    
    def play_audio(self, audio: AudioData) -> None:
        """Plays audio through speakers.
        
        Args:
            audio: AudioData object to play
            
        Raises:
            VoiceInterfaceError: If audio playback fails
        """
        try:
            # Handle empty audio
            if len(audio.data) == 0:
                return  # Nothing to play
            
            # Ensure audio is in the correct format for playback
            # sounddevice expects float32 in range [-1, 1]
            audio_data = audio.data.astype(np.float32)
            
            # Ensure values are in valid range
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            # Play the audio
            sd.play(audio_data, samplerate=audio.sample_rate)
            
            # Wait for playback to complete
            sd.wait()
        
        except Exception as e:
            raise VoiceInterfaceError(f"Failed to play audio: {str(e)}")
