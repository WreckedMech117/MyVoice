"""
Tests for VoiceVariantRow Widget

Story 1.7: Generate 5 Variations with Progressive Reveal
Story 1.8: Audition and Select Variant
"""

import pytest
from pathlib import Path

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from myvoice.ui.dialogs.voice_design_studio.voice_variant_row import (
    VoiceVariantRow, VariantState
)


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def row(qapp):
    """Create a VoiceVariantRow instance."""
    row = VoiceVariantRow(0)
    yield row
    row.deleteLater()


class TestVoiceVariantRowInit:
    """Tests for VoiceVariantRow initialization."""

    def test_row_creation(self, row):
        """Test row creates successfully."""
        assert row is not None
        assert isinstance(row, VoiceVariantRow)

    def test_row_has_variant_index(self, row):
        """Test row has correct variant index."""
        assert row.get_variant_index() == 0

    def test_row_initial_state_is_waiting(self, row):
        """Test row starts in waiting state."""
        assert row.get_state() == VariantState.WAITING

    def test_row_has_radio_button(self, row):
        """Test row has radio button."""
        assert row.radio_button is not None

    def test_row_has_variant_label(self, row):
        """Test row has variant label."""
        assert row.variant_label is not None
        assert "1" in row.variant_label.text()  # Variant 1 for index 0

    def test_row_has_status_label(self, row):
        """Test row has status label."""
        assert row.status_label is not None

    def test_row_has_play_button(self, row):
        """Test row has play button."""
        assert row.play_button is not None


class TestVoiceVariantRowWaitingState:
    """Tests for waiting state."""

    def test_waiting_state_shows_waiting_text(self, row):
        """Test waiting state shows 'Waiting...' text."""
        row.set_waiting()
        assert "Waiting" in row.status_label.text()

    def test_waiting_state_disables_radio_button(self, row):
        """Test waiting state disables radio button."""
        row.set_waiting()
        assert not row.radio_button.isEnabled()

    def test_waiting_state_hides_play_button(self, row):
        """Test waiting state hides play button."""
        row.set_waiting()
        assert not row.play_button.isVisible()

    def test_waiting_state_hides_duration(self, row):
        """Test waiting state hides duration label."""
        row.set_waiting()
        assert not row.duration_label.isVisible()


class TestVoiceVariantRowGeneratingState:
    """Tests for generating state."""

    def test_generating_state_shows_generating_text(self, row):
        """Test generating state shows 'Generating...' text."""
        row.set_generating()
        assert "Generating" in row.status_label.text()

    def test_generating_state_disables_radio_button(self, row):
        """Test generating state disables radio button."""
        row.set_generating()
        assert not row.radio_button.isEnabled()

    def test_generating_state_hides_play_button(self, row):
        """Test generating state hides play button."""
        row.set_generating()
        assert not row.play_button.isVisible()


class TestVoiceVariantRowCompleteState:
    """Tests for complete state."""

    def test_complete_state_shows_ready(self, row, tmp_path):
        """Test complete state shows 'Ready' status."""
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")

        row.set_complete(audio, embedding, duration=5.0)

        assert "Ready" in row.status_label.text()

    def test_complete_state_enables_radio_button(self, row, tmp_path):
        """Test complete state enables radio button."""
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")

        row.set_complete(audio, embedding)

        assert row.radio_button.isEnabled()

    def test_complete_state_shows_play_button(self, row, tmp_path):
        """Test complete state shows play button."""
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")

        row.set_complete(audio, embedding)

        # Use isHidden() instead of isVisible() because parent isn't shown
        assert not row.play_button.isHidden()

    def test_complete_state_shows_duration(self, row, tmp_path):
        """Test complete state shows duration."""
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")

        row.set_complete(audio, embedding, duration=5.0)

        # Use isHidden() instead of isVisible() because parent isn't shown
        assert not row.duration_label.isHidden()
        assert "0:05" in row.duration_label.text()

    def test_complete_state_stores_paths(self, row, tmp_path):
        """Test complete state stores audio and embedding paths."""
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")

        row.set_complete(audio, embedding)

        assert row.get_audio_path() == audio
        assert row.get_embedding_path() == embedding

    def test_is_complete_returns_true(self, row, tmp_path):
        """Test is_complete returns True in complete state."""
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")

        row.set_complete(audio, embedding)

        assert row.is_complete()


class TestVoiceVariantRowErrorState:
    """Tests for error state."""

    def test_error_state_shows_failed(self, row):
        """Test error state shows 'Failed' status."""
        row.set_error("Test error")
        assert "Failed" in row.status_label.text() or "error" in row.status_label.text().lower()

    def test_error_state_disables_radio_button(self, row):
        """Test error state disables radio button."""
        row.set_error("Test error")
        assert not row.radio_button.isEnabled()

    def test_error_state_hides_play_button(self, row):
        """Test error state hides play button."""
        row.set_error("Test error")
        assert not row.play_button.isVisible()

    def test_is_error_returns_true(self, row):
        """Test is_error returns True in error state."""
        row.set_error("Test error")
        assert row.is_error()


class TestVoiceVariantRowSelection:
    """Tests for selection functionality."""

    def test_set_selected_true(self, row, tmp_path):
        """Test setting selection to true."""
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")
        row.set_complete(audio, embedding)

        row.set_selected(True)

        assert row.is_selected()
        assert row.radio_button.isChecked()

    def test_set_selected_false(self, row, tmp_path):
        """Test setting selection to false."""
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")
        row.set_complete(audio, embedding)
        row.set_selected(True)

        row.set_selected(False)

        assert not row.is_selected()
        assert not row.radio_button.isChecked()


class TestVoiceVariantRowPlayback:
    """Tests for playback state."""

    def test_set_playing_true(self, row, tmp_path):
        """Test setting playing state to true."""
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")
        row.set_complete(audio, embedding)

        row.set_playing(True)

        assert row.is_playing()
        assert row.play_button.text() == "Stop"

    def test_set_playing_false(self, row, tmp_path):
        """Test setting playing state to false."""
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")
        row.set_complete(audio, embedding)
        row.set_playing(True)

        row.set_playing(False)

        assert not row.is_playing()
        assert row.play_button.text() == "Play"


class TestVoiceVariantRowClear:
    """Tests for clear functionality."""

    def test_clear_resets_to_waiting(self, row, tmp_path):
        """Test clear resets row to waiting state."""
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")
        row.set_complete(audio, embedding)
        row.set_selected(True)

        row.clear()

        assert row.get_state() == VariantState.WAITING
        assert not row.is_selected()
        assert row.get_audio_path() is None
        assert row.get_embedding_path() is None


class TestVoiceVariantRowSignals:
    """Tests for signals."""

    def test_selection_changed_signal_exists(self, row):
        """Test selection_changed signal exists."""
        assert hasattr(row, 'selection_changed')

    def test_play_requested_signal_exists(self, row):
        """Test play_requested signal exists."""
        assert hasattr(row, 'play_requested')

    def test_stop_requested_signal_exists(self, row):
        """Test stop_requested signal exists."""
        assert hasattr(row, 'stop_requested')


class TestVoiceVariantRowAccessibility:
    """Tests for accessibility."""

    def test_radio_button_has_accessible_name(self, row):
        """Test radio button has accessible name."""
        assert row.radio_button.accessibleName() != ""

    def test_play_button_has_accessible_name(self, row):
        """Test play button has accessible name."""
        assert row.play_button.accessibleName() != ""
