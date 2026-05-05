"""
Tests for AudioPlayerWidget Component

Story 1.4: Generate Single Voice Variation
- Audio player widget appears with play button and duration on completion
- Play/stop functionality works for preview audio
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QPushButton, QLabel

from myvoice.ui.dialogs.voice_design_studio.audio_player_widget import AudioPlayerWidget


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def widget(qapp):
    """Create an AudioPlayerWidget instance."""
    widget = AudioPlayerWidget()
    yield widget
    widget.deleteLater()


@pytest.fixture
def temp_audio_file():
    """Create a temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Write minimal WAV header (44 bytes) + some data
        # RIFF header
        f.write(b'RIFF')
        f.write((44 + 1000).to_bytes(4, 'little'))  # File size - 8
        f.write(b'WAVE')
        # fmt chunk
        f.write(b'fmt ')
        f.write((16).to_bytes(4, 'little'))  # Chunk size
        f.write((1).to_bytes(2, 'little'))   # Audio format (PCM)
        f.write((2).to_bytes(2, 'little'))   # Num channels
        f.write((44100).to_bytes(4, 'little'))  # Sample rate
        f.write((176400).to_bytes(4, 'little'))  # Byte rate
        f.write((4).to_bytes(2, 'little'))   # Block align
        f.write((16).to_bytes(2, 'little'))  # Bits per sample
        # data chunk
        f.write(b'data')
        f.write((1000).to_bytes(4, 'little'))  # Data size
        f.write(b'\x00' * 1000)  # Audio data
        temp_path = Path(f.name)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


class TestAudioPlayerWidgetInit:
    """Tests for AudioPlayerWidget initialization."""

    def test_widget_creation(self, widget):
        """Test widget creates successfully."""
        assert widget is not None
        assert isinstance(widget, AudioPlayerWidget)

    def test_play_button_exists(self, widget):
        """Test play button exists."""
        assert hasattr(widget, 'play_button')
        assert isinstance(widget.play_button, QPushButton)

    def test_duration_label_exists(self, widget):
        """Test duration label exists."""
        assert hasattr(widget, 'duration_label')
        assert isinstance(widget.duration_label, QLabel)

    def test_play_button_initially_disabled(self, widget):
        """Test play button is disabled when no audio file set."""
        assert not widget.play_button.isEnabled()

    def test_initial_duration_is_zero(self, widget):
        """Test initial duration shows 0:00."""
        assert widget.duration_label.text() == "0:00"


class TestAudioFileManagement:
    """Tests for audio file setting and management."""

    def test_set_audio_file_enables_play(self, widget, temp_audio_file):
        """Test setting valid audio file enables play button."""
        widget.set_audio_file(temp_audio_file)
        assert widget.play_button.isEnabled()

    def test_set_audio_file_updates_duration(self, widget, temp_audio_file):
        """Test setting audio file updates duration display."""
        widget.set_audio_file(temp_audio_file)
        # Duration should be non-zero
        assert widget.duration_label.text() != "0:00" or widget.get_duration() > 0

    def test_set_invalid_file_disables_play(self, widget):
        """Test setting invalid file path disables play button."""
        widget.set_audio_file("/nonexistent/path/audio.wav")
        assert not widget.play_button.isEnabled()

    def test_set_none_disables_play(self, widget, temp_audio_file):
        """Test setting None disables play button."""
        widget.set_audio_file(temp_audio_file)
        assert widget.play_button.isEnabled()

        widget.set_audio_file(None)
        assert not widget.play_button.isEnabled()

    def test_get_audio_path(self, widget, temp_audio_file):
        """Test get_audio_path returns correct path."""
        widget.set_audio_file(temp_audio_file)
        assert widget.get_audio_path() == temp_audio_file

    def test_get_audio_path_none_initially(self, widget):
        """Test get_audio_path returns None initially."""
        assert widget.get_audio_path() is None

    def test_clear_resets_state(self, widget, temp_audio_file):
        """Test clear resets all state."""
        widget.set_audio_file(temp_audio_file)
        widget.clear()

        assert widget.get_audio_path() is None
        assert not widget.play_button.isEnabled()
        assert widget.duration_label.text() == "0:00"


class TestPlaybackState:
    """Tests for playback state management."""

    def test_is_playing_initially_false(self, widget):
        """Test is_playing returns False initially."""
        assert not widget.is_playing()

    def test_get_duration(self, widget, temp_audio_file):
        """Test get_duration returns duration in seconds."""
        widget.set_audio_file(temp_audio_file)
        duration = widget.get_duration()
        assert isinstance(duration, float)
        assert duration >= 0


class TestSignals:
    """Tests for widget signals."""

    def test_playback_started_signal_exists(self, widget):
        """Test playback_started signal exists."""
        assert hasattr(widget, 'playback_started')

    def test_playback_stopped_signal_exists(self, widget):
        """Test playback_stopped signal exists."""
        assert hasattr(widget, 'playback_stopped')


class TestAccessibility:
    """Tests for accessibility features."""

    def test_play_button_has_accessible_name(self, widget):
        """Test play button has accessible name."""
        assert widget.play_button.accessibleName() == "Play audio"

    def test_duration_label_has_accessible_name(self, widget):
        """Test duration label has accessible name."""
        assert widget.duration_label.accessibleName() == "Duration"


class TestDurationFormatting:
    """Tests for duration formatting."""

    def test_format_duration_seconds(self, widget):
        """Test duration formatting for seconds only."""
        formatted = widget._format_duration(45.0)
        assert formatted == "0:45"

    def test_format_duration_minutes(self, widget):
        """Test duration formatting with minutes."""
        formatted = widget._format_duration(125.0)
        assert formatted == "2:05"

    def test_format_duration_zero(self, widget):
        """Test duration formatting for zero."""
        formatted = widget._format_duration(0.0)
        assert formatted == "0:00"
