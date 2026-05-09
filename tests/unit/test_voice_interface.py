"""Unit tests for VoiceInterface component."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.voice_interface import (
    VoiceInterface,
    AudioData,
    VoiceInterfaceError,
    STTError,
    TTSError,
    AudioCaptureError
)


class TestVoiceInterfaceInitialization:
    """Tests for VoiceInterface initialization."""
    
    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        with patch('src.voice_interface.OpenAI'), \
             patch('src.voice_interface.sd', Mock()):
            interface = VoiceInterface(api_key="test-key")
            assert interface.api_key == "test-key"
            assert interface.sample_rate == 16000
            assert interface.channels == 1
    
    def test_init_with_env_var(self):
        """Test initialization with API key from environment."""
        with patch('src.voice_interface.OpenAI'), \
             patch('src.voice_interface.sd', Mock()), \
             patch('os.getenv', return_value="env-key"):
            interface = VoiceInterface()
            assert interface.api_key == "env-key"
    
    def test_init_without_api_key_raises_error(self):
        """Test initialization fails without API key."""
        with patch('src.voice_interface.OpenAI'), \
             patch('src.voice_interface.sd', Mock()), \
             patch('os.getenv', return_value=None):
            with pytest.raises(VoiceInterfaceError, match="API key not provided"):
                VoiceInterface()
    
    def test_init_with_custom_sample_rate(self):
        """Test initialization with custom sample rate."""
        with patch('src.voice_interface.OpenAI'), \
             patch('src.voice_interface.sd', Mock()):
            interface = VoiceInterface(api_key="test-key", sample_rate=44100)
            assert interface.sample_rate == 44100


class TestSpeechToText:
    """Tests for speech_to_text method."""
    
    @pytest.fixture
    def interface(self):
        """Create a VoiceInterface instance for testing."""
        with patch('src.voice_interface.OpenAI') as mock_openai, \
             patch('src.voice_interface.sd', Mock()):
            mock_client = Mock()
            mock_openai.return_value = mock_client
            interface = VoiceInterface(api_key="test-key")
            interface.client = mock_client
            return interface
    
    def test_speech_to_text_success(self, interface):
        """Test successful speech-to-text conversion."""
        # Create sample audio data
        audio = AudioData(
            data=np.random.randn(16000) * 0.1,  # 1 second of audio
            sample_rate=16000
        )
        
        # Mock the API response
        interface.client.audio.transcriptions.create = Mock(
            return_value="Hello, world!"
        )
        
        result = interface.speech_to_text(audio)
        
        assert result == "Hello, world!"
        interface.client.audio.transcriptions.create.assert_called_once()
    
    def test_speech_to_text_empty_audio(self, interface):
        """Test speech-to-text with empty audio returns empty string."""
        audio = AudioData(data=np.array([]), sample_rate=16000)
        
        result = interface.speech_to_text(audio)
        
        assert result == ""
    
    def test_speech_to_text_silent_audio(self, interface):
        """Test speech-to-text with very quiet audio returns empty string."""
        # Create very quiet audio (below threshold)
        audio = AudioData(
            data=np.random.randn(16000) * 0.0001,
            sample_rate=16000
        )
        
        result = interface.speech_to_text(audio)
        
        assert result == ""
    
    def test_speech_to_text_api_error(self, interface):
        """Test speech-to-text handles API errors."""
        audio = AudioData(
            data=np.random.randn(16000) * 0.1,
            sample_rate=16000
        )
        
        # Mock API error
        interface.client.audio.transcriptions.create = Mock(
            side_effect=Exception("API error")
        )
        
        with pytest.raises(STTError, match="Speech recognition failed"):
            interface.speech_to_text(audio)
    
    def test_speech_to_text_invalid_api_key(self, interface):
        """Test speech-to-text handles invalid API key error."""
        audio = AudioData(
            data=np.random.randn(16000) * 0.1,
            sample_rate=16000
        )
        
        # Mock API key error
        interface.client.audio.transcriptions.create = Mock(
            side_effect=Exception("Invalid api_key provided")
        )
        
        with pytest.raises(STTError, match="Invalid or missing OpenAI API key"):
            interface.speech_to_text(audio)
    
    def test_speech_to_text_quota_exceeded(self, interface):
        """Test speech-to-text handles quota exceeded error."""
        audio = AudioData(
            data=np.random.randn(16000) * 0.1,
            sample_rate=16000
        )
        
        # Mock quota error
        interface.client.audio.transcriptions.create = Mock(
            side_effect=Exception("Quota exceeded")
        )
        
        with pytest.raises(STTError, match="OpenAI API quota exceeded"):
            interface.speech_to_text(audio)
    
    def test_speech_to_text_network_error(self, interface):
        """Test speech-to-text handles network errors."""
        audio = AudioData(
            data=np.random.randn(16000) * 0.1,
            sample_rate=16000
        )
        
        # Mock network error
        interface.client.audio.transcriptions.create = Mock(
            side_effect=Exception("Network connection failed")
        )
        
        with pytest.raises(STTError, match="Network error during speech recognition"):
            interface.speech_to_text(audio)


class TestCaptureAudio:
    """Tests for capture_audio method."""
    
    @pytest.fixture
    def interface(self):
        """Create a VoiceInterface instance for testing."""
        with patch('src.voice_interface.OpenAI'), \
             patch('src.voice_interface.sd', Mock()):
            return VoiceInterface(api_key="test-key")
    
    @patch('src.voice_interface.sd')
    def test_capture_audio_fixed_duration(self, mock_sd, interface):
        """Test audio capture with fixed duration."""
        # Mock recording
        mock_recording = np.random.randn(16000, 1).astype('float32')
        mock_sd.rec.return_value = mock_recording
        
        result = interface.capture_audio(duration=1.0)
        
        assert isinstance(result, AudioData)
        assert len(result.data) == 16000
        assert result.sample_rate == 16000
        mock_sd.rec.assert_called_once()
        mock_sd.wait.assert_called_once()
    
    @patch('src.voice_interface.sd')
    def test_capture_audio_error_handling(self, mock_sd, interface):
        """Test audio capture handles errors."""
        mock_sd.rec.side_effect = Exception("Audio device error")
        
        with pytest.raises(AudioCaptureError, match="Failed to capture audio"):
            interface.capture_audio(duration=1.0)


class TestTextToSpeech:
    """Tests for text_to_speech method."""
    
    @pytest.fixture
    def interface(self):
        """Create a VoiceInterface instance for testing."""
        with patch('src.voice_interface.OpenAI') as mock_openai, \
             patch('src.voice_interface.sd', Mock()):
            mock_client = Mock()
            mock_openai.return_value = mock_client
            interface = VoiceInterface(api_key="test-key")
            interface.client = mock_client
            return interface
    
    def test_text_to_speech_success(self, interface):
        """Test successful text-to-speech conversion."""
        # Mock gTTS
        mock_gtts = MagicMock()
        mock_tts_instance = Mock()
        mock_gtts.return_value = mock_tts_instance
        
        # Mock pydub module and AudioSegment
        mock_pydub = MagicMock()
        mock_segment = Mock()
        mock_segment.frame_rate = 24000
        mock_segment.get_array_of_samples.return_value = np.array([1000, 2000, 3000], dtype=np.int16)
        mock_pydub.AudioSegment.from_mp3.return_value = mock_segment
        
        with patch.dict('sys.modules', {'pydub': mock_pydub, 'gtts': MagicMock(gTTS=mock_gtts)}):
            result = interface.text_to_speech("Hello, world!")
            
            assert isinstance(result, AudioData)
            assert result.sample_rate == 24000
            assert len(result.data) == 3
            mock_gtts.assert_called_once_with(text="Hello, world!", lang="en")
            mock_tts_instance.write_to_fp.assert_called_once()
    
    def test_text_to_speech_empty_text(self, interface):
        """Test text-to-speech with empty text raises error."""
        with pytest.raises(Exception, match="Text cannot be empty"):
            interface.text_to_speech("")
    
    def test_text_to_speech_whitespace_only(self, interface):
        """Test text-to-speech with whitespace-only text raises error."""
        with pytest.raises(Exception, match="Text cannot be empty"):
            interface.text_to_speech("   ")
    
    def test_text_to_speech_custom_voice(self, interface):
        """Test text-to-speech with custom voice/lang."""
        mock_gtts = MagicMock()
        mock_tts_instance = Mock()
        mock_gtts.return_value = mock_tts_instance
        
        # Mock pydub module and AudioSegment
        mock_pydub = MagicMock()
        mock_segment = Mock()
        mock_segment.frame_rate = 24000
        mock_segment.get_array_of_samples.return_value = np.array([1000], dtype=np.int16)
        mock_pydub.AudioSegment.from_mp3.return_value = mock_segment
        
        with patch.dict('sys.modules', {'pydub': mock_pydub, 'gtts': MagicMock(gTTS=mock_gtts)}):
            interface.text_to_speech("Hello", voice="es")
            mock_gtts.assert_called_once_with(text="Hello", lang="es")
    
    def test_text_to_speech_api_error(self, interface):
        """Test text-to-speech handles API errors."""
        mock_gtts = MagicMock(side_effect=Exception("API error"))
        
        with patch.dict('sys.modules', {'gtts': MagicMock(gTTS=mock_gtts)}):
            with pytest.raises(Exception, match="Text-to-speech conversion failed"):
                interface.text_to_speech("Hello")
    
    def test_text_to_speech_network_error(self, interface):
        """Test text-to-speech handles network errors."""
        mock_gtts = MagicMock(side_effect=Exception("Network connection failed"))
        
        with patch.dict('sys.modules', {'gtts': MagicMock(gTTS=mock_gtts)}):
            with pytest.raises(Exception, match="Network error during text-to-speech"):
                interface.text_to_speech("Hello")


class TestPlayAudio:
    """Tests for play_audio method."""
    
    @pytest.fixture
    def interface(self):
        """Create a VoiceInterface instance for testing."""
        with patch('src.voice_interface.OpenAI'), \
             patch('src.voice_interface.sd', Mock()):
            return VoiceInterface(api_key="test-key")
    
    @patch('src.voice_interface.sd')
    def test_play_audio_success(self, mock_sd, interface):
        """Test successful audio playback."""
        audio = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000
        )
        
        interface.play_audio(audio)
        
        mock_sd.play.assert_called_once()
        mock_sd.wait.assert_called_once()
    
    @patch('src.voice_interface.sd')
    def test_play_audio_empty(self, mock_sd, interface):
        """Test playing empty audio does nothing."""
        audio = AudioData(data=np.array([]), sample_rate=16000)
        
        interface.play_audio(audio)
        
        # Should not call play for empty audio
        mock_sd.play.assert_not_called()
    
    @patch('src.voice_interface.sd')
    def test_play_audio_clips_values(self, mock_sd, interface):
        """Test that audio values are clipped to valid range."""
        # Create audio with values outside [-1, 1]
        audio = AudioData(
            data=np.array([2.0, -2.0, 0.5], dtype=np.float32),
            sample_rate=16000
        )
        
        interface.play_audio(audio)
        
        # Verify play was called
        mock_sd.play.assert_called_once()
        
        # Get the audio data that was passed to play
        call_args = mock_sd.play.call_args
        played_audio = call_args[0][0]
        
        # Verify values are clipped
        assert np.all(played_audio >= -1.0)
        assert np.all(played_audio <= 1.0)
    
    @patch('src.voice_interface.sd')
    def test_play_audio_error_handling(self, mock_sd, interface):
        """Test audio playback handles errors."""
        audio = AudioData(
            data=np.random.randn(16000).astype(np.float32),
            sample_rate=16000
        )
        
        mock_sd.play.side_effect = Exception("Audio device error")
        
        with pytest.raises(VoiceInterfaceError, match="Failed to play audio"):
            interface.play_audio(audio)
