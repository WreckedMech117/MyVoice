"""
Tests for SamplePathPanel Component

Story 2.1: Upload and Validate Audio File
Covers:
- Browse button for file selection
- File dialog filters for WAV, MP3, M4A formats
- Valid file shows filename, duration, format info
- Warning for audio < 3 seconds or > 30 seconds (not blocking)
- Play/stop button for uploaded audio preview
- Error message for unsupported formats

Story 2.2: Auto-Transcribe with Whisper
Covers:
- Multi-line transcript field appears after file upload
- "Auto-Transcribe" button triggers Whisper transcription
- Progress indicator during transcription (indeterminate)
- Completed transcription populates text area (editable)
- Error handling with retry option

Story 2.3: Extract Embedding and Preview
Covers:
- Preview Text field and Extract Embedding button visible
- Button disabled if preview text empty
- Clicking shows progress "Extracting...", loads Base model if needed
- Preview audio auto-plays on extraction completion
- Play/stop button for replay
- Voice Name field and Save Voice button available on success
- Errors show clear message with suggestions

Story 2.4: Re-Extract After Edit
Covers:
- Visual indicator when transcript changed since last extraction
- Extract Embedding button remains available after first extraction
- Re-extraction clears previous preview, shows new result
- Only most recent extraction retained
- Changing Preview Text allows testing same embedding with different text

Acceptance Criteria (2.1):
- "From Sample" tab shows Browse button for file selection
- File dialog filters for WAV, MP3, M4A formats
- Valid file shows filename, duration, format info
- Warning for audio < 3 seconds or > 30 seconds (not blocking)
- Play/stop button for uploaded audio preview
- Error message for unsupported formats

Acceptance Criteria (2.2):
- Multi-line "Transcript" text area appears after file upload
- User can manually enter/edit transcript
- "Auto-Transcribe" button visible for automatic transcription
- Clicking Auto-Transcribe shows progress indicator (indeterminate)
- Transcribed text appears in editable field
- Transcription errors show message with "Retry" button

Acceptance Criteria (2.3):
- Preview Text field and Extract Embedding button visible after file upload
- Extract Embedding button disabled if preview text empty
- Clicking shows progress "Extracting..." with indeterminate progress
- Preview audio auto-plays on extraction completion
- Play/stop button available for replay
- Voice Name field and Save Voice button available on success
- Errors show clear message with suggestions (cleaner sample, check transcript)

Acceptance Criteria (2.4):
- Visual indicator shows when transcript changed since last extraction
- Extract Embedding button remains available after first extraction
- Re-extraction clears previous preview, shows new result
- Only most recent extraction is retained
- Changing Preview Text works with same embedding
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QPushButton, QLineEdit, QLabel

from myvoice.ui.dialogs.voice_design_studio.sample_path_panel import SamplePathPanel


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def panel(qapp):
    """Create a SamplePathPanel instance."""
    panel = SamplePathPanel()
    panel.show()  # Show panel so isVisible() works correctly for child widgets
    yield panel
    panel.close()
    panel.deleteLater()


@pytest.fixture
def sample_wav_file(tmp_path):
    """Create a sample WAV file for testing."""
    # Create a minimal WAV file header
    wav_file = tmp_path / "test_sample.wav"
    # Minimal WAV header for a ~5 second file at 44100Hz mono 16-bit
    wav_header = bytes([
        0x52, 0x49, 0x46, 0x46,  # "RIFF"
        0x24, 0xB0, 0x06, 0x00,  # File size - 8
        0x57, 0x41, 0x56, 0x45,  # "WAVE"
        0x66, 0x6D, 0x74, 0x20,  # "fmt "
        0x10, 0x00, 0x00, 0x00,  # Chunk size (16)
        0x01, 0x00,              # Audio format (PCM)
        0x01, 0x00,              # Num channels (1)
        0x44, 0xAC, 0x00, 0x00,  # Sample rate (44100)
        0x88, 0x58, 0x01, 0x00,  # Byte rate
        0x02, 0x00,              # Block align
        0x10, 0x00,              # Bits per sample (16)
        0x64, 0x61, 0x74, 0x61,  # "data"
        0x00, 0xB0, 0x06, 0x00,  # Data chunk size
    ])
    # Add some dummy audio data (~5 seconds worth)
    wav_file.write_bytes(wav_header + b'\x00' * (44100 * 2 * 5))
    return wav_file


class TestSamplePathPanelInit:
    """Tests for SamplePathPanel initialization."""

    def test_panel_creation(self, panel):
        """Test panel creates successfully."""
        assert panel is not None
        assert isinstance(panel, SamplePathPanel)

    def test_browse_button_exists(self, panel):
        """Test browse button exists."""
        assert hasattr(panel, 'browse_button')
        assert isinstance(panel.browse_button, QPushButton)
        assert "Browse" in panel.browse_button.text()

    def test_file_label_exists(self, panel):
        """Test file label exists."""
        assert hasattr(panel, 'file_label')
        assert "No file selected" in panel.file_label.text()

    def test_voice_name_edit_exists(self, panel):
        """Test voice name field exists."""
        assert hasattr(panel, 'voice_name_edit')
        assert isinstance(panel.voice_name_edit, QLineEdit)

    def test_audio_player_exists(self, panel):
        """Test audio player widget exists."""
        assert hasattr(panel, 'audio_player')

    def test_info_group_initially_hidden(self, panel):
        """Test file info group is hidden initially."""
        assert not panel.info_group.isVisible()


class TestSupportedFormats:
    """Tests for supported audio formats."""

    def test_supported_formats_constant(self, panel):
        """Test SUPPORTED_FORMATS constant exists."""
        assert hasattr(panel, 'SUPPORTED_FORMATS')
        assert '.wav' in panel.SUPPORTED_FORMATS
        assert '.mp3' in panel.SUPPORTED_FORMATS
        assert '.m4a' in panel.SUPPORTED_FORMATS

    def test_file_filter_includes_all_formats(self, panel):
        """Test FILE_FILTER includes all supported formats."""
        assert '*.wav' in panel.FILE_FILTER
        assert '*.mp3' in panel.FILE_FILTER
        assert '*.m4a' in panel.FILE_FILTER


class TestDurationThresholds:
    """Tests for duration warning thresholds."""

    def test_min_duration_threshold(self, panel):
        """Test MIN_DURATION_WARNING is 3 seconds."""
        assert panel.MIN_DURATION_WARNING == 3.0

    def test_max_duration_threshold(self, panel):
        """Test MAX_DURATION_WARNING is 30 seconds."""
        assert panel.MAX_DURATION_WARNING == 30.0


class TestFileLoading:
    """Tests for file loading functionality."""

    def test_load_valid_wav_file(self, panel, sample_wav_file):
        """Test loading a valid WAV file."""
        panel._load_audio_file(sample_wav_file)

        assert panel.has_audio_file()
        assert panel.get_audio_path() == sample_wav_file
        # QA3: info_group is on Sample tab; after file load we auto-switch to Clone tab
        # Switch back to Sample tab to verify info_group visibility
        panel.sub_tabs.setCurrentIndex(0)  # Sample tab
        assert panel.info_group.isVisible()

    def test_load_file_updates_file_label(self, panel, sample_wav_file):
        """Test loading file updates the file label."""
        panel._load_audio_file(sample_wav_file)

        assert sample_wav_file.name in panel.file_label.text()

    def test_load_file_emits_file_loaded_signal(self, panel, sample_wav_file):
        """Test loading file emits file_loaded signal."""
        signal_received = []
        panel.file_loaded.connect(lambda path: signal_received.append(path))

        panel._load_audio_file(sample_wav_file)

        assert len(signal_received) == 1
        assert signal_received[0] == str(sample_wav_file)

    def test_load_nonexistent_file_shows_error(self, panel, tmp_path):
        """Test loading non-existent file shows error."""
        fake_path = tmp_path / "nonexistent.wav"

        panel._load_audio_file(fake_path)

        assert panel.has_error()
        assert not panel.has_audio_file()


class TestUnsupportedFormats:
    """Tests for unsupported format handling."""

    def test_unsupported_format_shows_error(self, panel, tmp_path):
        """Test loading unsupported format shows error."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not audio")

        panel._load_audio_file(txt_file)

        assert panel.has_error()
        assert "Unsupported" in panel.file_label.text() or "Error" in panel.file_label.text()

    def test_unsupported_ogg_format(self, panel, tmp_path):
        """Test OGG format is not supported."""
        ogg_file = tmp_path / "test.ogg"
        ogg_file.write_bytes(b"OggS" + b"\x00" * 100)

        panel._load_audio_file(ogg_file)

        assert panel.has_error()


class TestDurationWarnings:
    """Tests for duration warnings."""

    def test_short_audio_shows_warning(self, panel, tmp_path):
        """Test audio < 3 seconds shows warning."""
        # Create short audio file
        wav_file = tmp_path / "short.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        # Mock duration detection to return 2 seconds
        with patch.object(panel, '_get_audio_duration', return_value=2.0):
            panel._load_audio_file(wav_file)

        # QA3: warning_label is on Sample tab; switch back to view it
        panel.sub_tabs.setCurrentIndex(0)  # Sample tab
        assert panel.warning_label.isVisible()
        assert "3 seconds" in panel.warning_label.text()

    def test_long_audio_shows_warning(self, panel, tmp_path):
        """Test audio > 30 seconds shows warning."""
        wav_file = tmp_path / "long.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        # Mock duration detection to return 45 seconds
        with patch.object(panel, '_get_audio_duration', return_value=45.0):
            panel._load_audio_file(wav_file)

        # QA3: warning_label is on Sample tab; switch back to view it
        panel.sub_tabs.setCurrentIndex(0)  # Sample tab
        assert panel.warning_label.isVisible()
        assert "30 seconds" in panel.warning_label.text()

    def test_good_duration_no_warning(self, panel, tmp_path):
        """Test audio between 3-30 seconds shows no warning."""
        wav_file = tmp_path / "good.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        # Mock duration detection to return 10 seconds
        with patch.object(panel, '_get_audio_duration', return_value=10.0):
            panel._load_audio_file(wav_file)

        assert not panel.warning_label.isVisible()


class TestFileInfo:
    """Tests for file information display."""

    def test_duration_displayed(self, panel, tmp_path):
        """Test duration is displayed."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=8.5):
            panel._load_audio_file(wav_file)

        assert "0:08" in panel.duration_value.text()

    def test_format_displayed(self, panel, tmp_path):
        """Test format is displayed."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            with patch.object(panel, '_get_audio_format', return_value="WAV, 44.1kHz"):
                panel._load_audio_file(wav_file)

        assert "WAV" in panel.format_value.text()


class TestSaveReadiness:
    """Tests for save readiness state."""

    def test_initially_not_save_ready(self, panel):
        """Test panel is not save ready initially."""
        assert not panel.is_save_ready()

    def test_save_ready_with_file_and_name(self, panel, tmp_path):
        """Test save ready when file loaded, name entered, and extraction complete."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview = tmp_path / "preview.wav"
        audio_preview.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_voice_name("Test Voice")
        # Story 2.3: Extraction must be complete for save to be ready
        panel.set_extraction_complete(audio_preview, auto_play=False)

        assert panel.is_save_ready()

    def test_save_not_ready_without_name(self, panel, tmp_path):
        """Test not save ready without voice name."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        assert not panel.is_save_ready()

    def test_save_ready_changed_signal(self, panel, tmp_path):
        """Test save_ready_changed signal emits."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview = tmp_path / "preview.wav"
        audio_preview.write_bytes(b"RIFF" + b"\x00" * 40)

        signal_received = []
        panel.save_ready_changed.connect(lambda ready: signal_received.append(ready))

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_voice_name("Test Voice")
        # Story 2.3: Extraction must be complete for save to be ready
        panel.set_extraction_complete(audio_preview, auto_play=False)

        # Should have received at least one True signal
        assert any(signal_received)


class TestVoiceName:
    """Tests for voice name field."""

    def test_get_voice_name_empty(self, panel):
        """Test get_voice_name returns empty string initially."""
        assert panel.get_voice_name() == ""

    def test_set_voice_name(self, panel):
        """Test set_voice_name sets the field."""
        panel.set_voice_name("My Voice")
        assert panel.get_voice_name() == "My Voice"

    def test_get_voice_name_strips_whitespace(self, panel):
        """Test get_voice_name strips whitespace."""
        panel.voice_name_edit.setText("  Test  ")
        assert panel.get_voice_name() == "Test"


class TestClear:
    """Tests for clear functionality."""

    def test_clear_resets_state(self, panel, tmp_path):
        """Test clear resets all state."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_voice_name("Test")

        panel.clear()

        assert not panel.has_audio_file()
        assert panel.get_voice_name() == ""
        assert not panel.info_group.isVisible()


class TestAccessibility:
    """Tests for accessibility features."""

    def test_browse_button_has_accessible_name(self, panel):
        """Test browse button has accessible name."""
        assert panel.browse_button.accessibleName() != ""

    def test_voice_name_has_accessible_name(self, panel):
        """Test voice name field has accessible name."""
        assert panel.voice_name_edit.accessibleName() == "Voice Name"


class TestSignals:
    """Tests for panel signals."""

    def test_file_loaded_signal_exists(self, panel):
        """Test file_loaded signal exists."""
        assert hasattr(panel, 'file_loaded')

    def test_content_changed_signal_exists(self, panel):
        """Test content_changed signal exists."""
        assert hasattr(panel, 'content_changed')

    def test_save_ready_changed_signal_exists(self, panel):
        """Test save_ready_changed signal exists."""
        assert hasattr(panel, 'save_ready_changed')

    def test_content_changed_emits_on_file_load(self, panel, tmp_path):
        """Test content_changed signal emits when file loaded."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        signal_received = []
        panel.content_changed.connect(lambda: signal_received.append(True))

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        assert len(signal_received) >= 1


# Story 2.2: Auto-Transcribe with Whisper Tests

class TestTranscriptSectionInit:
    """Tests for transcript section initialization (Story 2.2)."""

    def test_transcript_group_exists(self, panel):
        """Test transcript group box exists."""
        assert hasattr(panel, 'transcript_group')

    def test_transcript_group_initially_hidden(self, panel):
        """Test transcript section hidden until file loaded."""
        assert not panel.transcript_group.isVisible()

    def test_transcript_edit_exists(self, panel):
        """Test transcript text area exists."""
        assert hasattr(panel, 'transcript_edit')

    def test_transcribe_button_exists(self, panel):
        """Test Auto-Transcribe button exists."""
        assert hasattr(panel, 'transcribe_button')
        assert "Auto-Transcribe" in panel.transcribe_button.text()

    def test_transcribe_progress_exists(self, panel):
        """Test transcription progress bar exists."""
        assert hasattr(panel, 'transcribe_progress')
        assert not panel.transcribe_progress.isVisible()

    def test_transcribe_retry_button_exists(self, panel):
        """Test retry button exists and is hidden initially."""
        assert hasattr(panel, 'transcribe_retry_button')
        assert not panel.transcribe_retry_button.isVisible()


class TestTranscriptVisibility:
    """Tests for transcript section visibility (Story 2.2)."""

    def test_transcript_visible_after_file_load(self, panel, tmp_path):
        """Test transcript section shows after file loaded."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        assert panel.transcript_group.isVisible()

    def test_transcript_hidden_after_clear(self, panel, tmp_path):
        """Test transcript section hidden after clear."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.clear()

        assert not panel.transcript_group.isVisible()


class TestTranscriptionSignals:
    """Tests for transcription signals (Story 2.2)."""

    def test_transcribe_requested_signal_exists(self, panel):
        """Test transcribe_requested signal exists."""
        assert hasattr(panel, 'transcribe_requested')

    def test_transcript_changed_signal_exists(self, panel):
        """Test transcript_changed signal exists."""
        assert hasattr(panel, 'transcript_changed')

    def test_transcribe_button_emits_signal(self, panel, tmp_path):
        """Test Auto-Transcribe button emits transcribe_requested signal."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        signal_received = []
        panel.transcribe_requested.connect(lambda path: signal_received.append(path))

        panel.transcribe_button.click()

        assert len(signal_received) == 1
        assert signal_received[0] == str(wav_file)

    def test_transcript_changed_emits_on_text_change(self, panel):
        """Test transcript_changed signal emits when text changes."""
        signal_received = []
        panel.transcript_changed.connect(lambda: signal_received.append(True))

        panel.transcript_edit.setText("Hello world")

        assert len(signal_received) >= 1


class TestTranscriptionState:
    """Tests for transcription state management (Story 2.2)."""

    def test_set_transcribing_shows_progress(self, panel, tmp_path):
        """Test set_transcribing shows progress indicator."""
        # First load a file to make transcript_group visible
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_transcribing(True, "Transcribing...")

        assert panel.transcribe_progress.isVisible()
        assert not panel.transcribe_button.isEnabled()
        assert not panel.transcript_edit.isEnabled()
        assert "Transcribing" in panel.transcribe_status.text()

    def test_set_transcribing_false_hides_progress(self, panel):
        """Test set_transcribing(False) hides progress."""
        panel.set_transcribing(True)
        panel.set_transcribing(False)

        assert not panel.transcribe_progress.isVisible()
        assert panel.transcribe_button.isEnabled()
        assert panel.transcript_edit.isEnabled()

    def test_is_transcribing_tracks_state(self, panel):
        """Test is_transcribing returns correct state."""
        assert not panel.is_transcribing()

        panel.set_transcribing(True)
        assert panel.is_transcribing()

        panel.set_transcribing(False)
        assert not panel.is_transcribing()


class TestTranscriptionComplete:
    """Tests for transcription completion (Story 2.2)."""

    def test_set_transcription_complete_sets_text(self, panel):
        """Test set_transcription_complete populates text area."""
        panel.set_transcription_complete("Hello, this is the transcription.")

        assert panel.transcript_edit.toPlainText() == "Hello, this is the transcription."
        assert not panel.is_transcribing()

    def test_set_transcription_complete_hides_progress(self, panel):
        """Test set_transcription_complete hides progress."""
        panel.set_transcribing(True)
        panel.set_transcription_complete("Test")

        assert not panel.transcribe_progress.isVisible()
        assert panel.transcribe_button.isEnabled()
        assert panel.transcript_edit.isEnabled()

    def test_set_transcription_complete_shows_success_status(self, panel):
        """Test set_transcription_complete shows success message."""
        panel.set_transcription_complete("Test transcript")

        assert "complete" in panel.transcribe_status.text().lower()


class TestTranscriptionError:
    """Tests for transcription error handling (Story 2.2)."""

    def test_set_transcription_error_shows_error(self, panel):
        """Test set_transcription_error shows error message."""
        panel.set_transcription_error("Model not found")

        assert "Error" in panel.transcribe_status.text()
        assert "Model not found" in panel.transcribe_status.text()

    def test_set_transcription_error_shows_retry(self, panel, tmp_path):
        """Test set_transcription_error shows retry button."""
        # First load a file to make transcript_group visible
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_transcription_error("Connection failed")

        assert panel.transcribe_retry_button.isVisible()

    def test_set_transcription_error_enables_controls(self, panel):
        """Test set_transcription_error re-enables controls."""
        panel.set_transcribing(True)
        panel.set_transcription_error("Failed")

        assert not panel.is_transcribing()
        assert panel.transcribe_button.isEnabled()
        assert panel.transcript_edit.isEnabled()

    def test_has_transcription_error(self, panel):
        """Test has_transcription_error tracks error state."""
        assert not panel.has_transcription_error()

        panel.set_transcription_error("Error")
        assert panel.has_transcription_error()

    def test_retry_button_emits_transcribe_requested(self, panel, tmp_path):
        """Test retry button emits transcribe_requested signal."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_transcription_error("Failed")

        signal_received = []
        panel.transcribe_requested.connect(lambda path: signal_received.append(path))

        panel.transcribe_retry_button.click()

        assert len(signal_received) == 1


class TestTranscriptText:
    """Tests for transcript text manipulation (Story 2.2)."""

    def test_get_transcript_empty_initially(self, panel):
        """Test get_transcript returns empty string initially."""
        assert panel.get_transcript() == ""

    def test_set_transcript(self, panel):
        """Test set_transcript sets the text."""
        panel.set_transcript("This is a test")
        assert panel.get_transcript() == "This is a test"

    def test_get_transcript_strips_whitespace(self, panel):
        """Test get_transcript strips whitespace."""
        panel.transcript_edit.setText("  Test  ")
        assert panel.get_transcript() == "Test"

    def test_has_transcript_false_when_empty(self, panel):
        """Test has_transcript returns False when empty."""
        assert not panel.has_transcript()

    def test_has_transcript_true_when_text(self, panel):
        """Test has_transcript returns True when text exists."""
        panel.set_transcript("Some transcript")
        assert panel.has_transcript()


class TestTranscriptionClear:
    """Tests for clearing transcription state (Story 2.2)."""

    def test_clear_resets_transcript_text(self, panel):
        """Test clear resets transcript text."""
        panel.set_transcript("Test transcript")
        panel.clear()
        assert panel.get_transcript() == ""

    def test_clear_resets_transcribing_state(self, panel):
        """Test clear resets transcribing state."""
        panel.set_transcribing(True)
        panel.clear()
        assert not panel.is_transcribing()

    def test_clear_resets_error_state(self, panel):
        """Test clear resets transcription error state."""
        panel.set_transcription_error("Error")
        panel.clear()
        assert not panel.has_transcription_error()

    def test_clear_hides_retry_button(self, panel):
        """Test clear hides retry button."""
        panel.set_transcription_error("Error")
        panel.clear()
        assert not panel.transcribe_retry_button.isVisible()

    def test_clear_hides_progress(self, panel):
        """Test clear hides progress bar."""
        panel.set_transcribing(True)
        panel.clear()
        assert not panel.transcribe_progress.isVisible()


class TestTranscriptionAccessibility:
    """Tests for transcription accessibility (Story 2.2)."""

    def test_transcript_edit_has_accessible_name(self, panel):
        """Test transcript text area has accessible name."""
        assert panel.transcript_edit.accessibleName() == "Transcript"

    def test_transcribe_button_has_accessible_name(self, panel):
        """Test Auto-Transcribe button has accessible name."""
        assert panel.transcribe_button.accessibleName() == "Auto-Transcribe"

    def test_retry_button_has_accessible_name(self, panel):
        """Test retry button has accessible name."""
        assert panel.transcribe_retry_button.accessibleName() != ""


# Story 2.3: Extract Embedding and Preview Tests

class TestExtractSectionInit:
    """Tests for extract section initialization (Story 2.3)."""

    def test_extract_group_exists(self, panel):
        """Test extract group box exists."""
        assert hasattr(panel, 'extract_group')

    def test_extract_group_initially_hidden(self, panel):
        """Test extract section hidden until file loaded."""
        assert not panel.extract_group.isVisible()

    def test_preview_text_edit_exists(self, panel):
        """Test preview text field exists."""
        assert hasattr(panel, 'preview_text_edit')

    def test_extract_button_exists(self, panel):
        """Test Extract Embedding button exists."""
        assert hasattr(panel, 'extract_button')
        assert "Extract" in panel.extract_button.text()

    def test_extract_button_initially_disabled(self, panel):
        """Test Extract button disabled when no preview text."""
        assert not panel.extract_button.isEnabled()

    def test_extract_progress_exists(self, panel):
        """Test extraction progress bar exists."""
        assert hasattr(panel, 'extract_progress')
        assert not panel.extract_progress.isVisible()

    def test_extract_retry_button_exists(self, panel):
        """Test retry button exists and is hidden initially."""
        assert hasattr(panel, 'extract_retry_button')
        assert not panel.extract_retry_button.isVisible()

    def test_preview_audio_group_exists(self, panel):
        """Test preview audio group exists and is hidden."""
        assert hasattr(panel, 'preview_audio_group')
        assert not panel.preview_audio_group.isVisible()

    def test_preview_audio_player_exists(self, panel):
        """Test preview audio player exists."""
        assert hasattr(panel, 'preview_audio_player')


class TestExtractVisibility:
    """Tests for extract section visibility (Story 2.3)."""

    def test_extract_visible_after_file_load(self, panel, tmp_path):
        """Test extract section shows after file loaded."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        assert panel.extract_group.isVisible()

    def test_extract_hidden_after_clear(self, panel, tmp_path):
        """Test extract section hidden after clear."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.clear()

        assert not panel.extract_group.isVisible()


class TestExtractButtonState:
    """Tests for Extract button enabled/disabled state (Story 2.3)."""

    def test_extract_button_disabled_without_preview_text(self, panel, tmp_path):
        """Test Extract button disabled when preview text empty."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        assert not panel.extract_button.isEnabled()

    def test_extract_button_enabled_with_preview_text(self, panel, tmp_path):
        """Test Extract button enabled when preview text entered."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_preview_text("Hello world")

        assert panel.extract_button.isEnabled()

    def test_extract_button_disabled_when_text_cleared(self, panel, tmp_path):
        """Test Extract button disabled when preview text cleared."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_preview_text("Hello")
        assert panel.extract_button.isEnabled()

        panel.set_preview_text("")
        assert not panel.extract_button.isEnabled()


class TestExtractionSignals:
    """Tests for extraction signals (Story 2.3)."""

    def test_extract_requested_signal_exists(self, panel):
        """Test extract_requested signal exists."""
        assert hasattr(panel, 'extract_requested')

    def test_preview_text_changed_signal_exists(self, panel):
        """Test preview_text_changed signal exists."""
        assert hasattr(panel, 'preview_text_changed')

    def test_extract_button_emits_signal(self, panel, tmp_path):
        """Test Extract button emits extract_requested signal."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_transcript("This is the transcript")
        panel.set_preview_text("Hello world")

        signal_received = []
        panel.extract_requested.connect(
            lambda a, t, p: signal_received.append((a, t, p))
        )

        panel.extract_button.click()

        assert len(signal_received) == 1
        assert signal_received[0][0] == str(wav_file)  # audio_path
        assert signal_received[0][1] == "This is the transcript"  # transcript
        assert signal_received[0][2] == "Hello world"  # preview_text

    def test_preview_text_changed_emits_on_change(self, panel):
        """Test preview_text_changed signal emits when text changes."""
        signal_received = []
        panel.preview_text_changed.connect(lambda: signal_received.append(True))

        panel.preview_text_edit.setText("Test text")

        assert len(signal_received) >= 1


class TestExtractionState:
    """Tests for extraction state management (Story 2.3)."""

    def test_set_extracting_shows_progress(self, panel, tmp_path):
        """Test set_extracting shows progress indicator."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_extracting(True, "Extracting...")

        assert panel.extract_progress.isVisible()
        assert not panel.extract_button.isEnabled()
        assert not panel.preview_text_edit.isEnabled()
        assert "Extracting" in panel.extract_status.text()

    def test_set_extracting_false_hides_progress(self, panel, tmp_path):
        """Test set_extracting(False) hides progress."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_preview_text("Test")
        panel.set_extracting(True)
        panel.set_extracting(False)

        assert not panel.extract_progress.isVisible()
        assert panel.extract_button.isEnabled()  # Re-enabled with preview text
        assert panel.preview_text_edit.isEnabled()

    def test_is_extracting_tracks_state(self, panel):
        """Test is_extracting returns correct state."""
        assert not panel.is_extracting()

        panel.set_extracting(True)
        assert panel.is_extracting()

        panel.set_extracting(False)
        assert not panel.is_extracting()


class TestExtractionComplete:
    """Tests for extraction completion (Story 2.3)."""

    def test_set_extraction_complete_shows_preview(self, panel, tmp_path):
        """Test set_extraction_complete shows preview audio section."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview = tmp_path / "preview.wav"
        audio_preview.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        # auto_play=False skips playback
        panel.set_extraction_complete(audio_preview, auto_play=False)

        assert panel.preview_audio_group.isVisible()
        assert not panel.is_extracting()

    def test_set_extraction_complete_sets_audio_path(self, panel, tmp_path):
        """Test set_extraction_complete stores audio path."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview = tmp_path / "preview.wav"
        audio_preview.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_extraction_complete(audio_preview, auto_play=False)

        assert panel.get_generated_audio_path() == audio_preview

    def test_set_extraction_complete_enables_save(self, panel, tmp_path):
        """Test save ready after successful extraction."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview = tmp_path / "preview.wav"
        audio_preview.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_voice_name("My Voice")
        panel.set_extraction_complete(audio_preview, auto_play=False)

        assert panel.is_save_ready()

    def test_has_generated_preview(self, panel, tmp_path):
        """Test has_generated_preview tracks extraction success."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview = tmp_path / "preview.wav"
        audio_preview.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        assert not panel.has_generated_preview()

        panel.set_extraction_complete(audio_preview, auto_play=False)

        assert panel.has_generated_preview()


class TestExtractionError:
    """Tests for extraction error handling (Story 2.3)."""

    def test_set_extraction_error_shows_error(self, panel, tmp_path):
        """Test set_extraction_error shows error message."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_extraction_error("Model not found")

        assert "Error" in panel.extract_status.text()
        assert "Model not found" in panel.extract_status.text()

    def test_set_extraction_error_shows_suggestions(self, panel, tmp_path):
        """Test set_extraction_error shows suggestions."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_extraction_error(
            "Extraction failed",
            suggestions="Try using a cleaner audio sample"
        )

        assert "cleaner audio sample" in panel.extract_status.text()

    def test_set_extraction_error_shows_retry(self, panel, tmp_path):
        """Test set_extraction_error shows retry button."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_extraction_error("Failed")

        assert panel.extract_retry_button.isVisible()

    def test_set_extraction_error_hides_preview(self, panel, tmp_path):
        """Test set_extraction_error hides preview section."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_extraction_error("Failed")

        assert not panel.preview_audio_group.isVisible()

    def test_has_extraction_error(self, panel):
        """Test has_extraction_error tracks error state."""
        assert not panel.has_extraction_error()

        panel.set_extraction_error("Error")
        assert panel.has_extraction_error()

    def test_retry_button_emits_extract_requested(self, panel, tmp_path):
        """Test retry button emits extract_requested signal."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_preview_text("Hello")
        panel.set_extraction_error("Failed")

        signal_received = []
        panel.extract_requested.connect(
            lambda a, t, p: signal_received.append((a, t, p))
        )

        panel.extract_retry_button.click()

        assert len(signal_received) == 1


class TestPreviewText:
    """Tests for preview text manipulation (Story 2.3)."""

    def test_get_preview_text_empty_initially(self, panel):
        """Test get_preview_text returns empty string initially."""
        assert panel.get_preview_text() == ""

    def test_set_preview_text(self, panel):
        """Test set_preview_text sets the text."""
        panel.set_preview_text("Hello world")
        assert panel.get_preview_text() == "Hello world"

    def test_get_preview_text_strips_whitespace(self, panel):
        """Test get_preview_text strips whitespace."""
        panel.preview_text_edit.setText("  Test  ")
        assert panel.get_preview_text() == "Test"

    def test_has_preview_text_false_when_empty(self, panel):
        """Test has_preview_text returns False when empty."""
        assert not panel.has_preview_text()

    def test_has_preview_text_true_when_text(self, panel):
        """Test has_preview_text returns True when text exists."""
        panel.set_preview_text("Some text")
        assert panel.has_preview_text()


class TestExtractionClear:
    """Tests for clearing extraction state (Story 2.3)."""

    def test_clear_resets_preview_text(self, panel):
        """Test clear resets preview text."""
        panel.set_preview_text("Test text")
        panel.clear()
        assert panel.get_preview_text() == ""

    def test_clear_resets_extracting_state(self, panel):
        """Test clear resets extracting state."""
        panel.set_extracting(True)
        panel.clear()
        assert not panel.is_extracting()

    def test_clear_resets_extraction_error_state(self, panel):
        """Test clear resets extraction error state."""
        panel.set_extraction_error("Error")
        panel.clear()
        assert not panel.has_extraction_error()

    def test_clear_hides_extract_retry_button(self, panel):
        """Test clear hides retry button."""
        panel.set_extraction_error("Error")
        panel.clear()
        assert not panel.extract_retry_button.isVisible()

    def test_clear_hides_extract_progress(self, panel):
        """Test clear hides progress bar."""
        panel.set_extracting(True)
        panel.clear()
        assert not panel.extract_progress.isVisible()

    def test_clear_hides_preview_audio(self, panel, tmp_path):
        """Test clear hides preview audio section."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview = tmp_path / "preview.wav"
        audio_preview.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_extraction_complete(audio_preview, auto_play=False)
        panel.clear()

        assert not panel.preview_audio_group.isVisible()

    def test_clear_resets_generated_audio_path(self, panel, tmp_path):
        """Test clear resets generated audio path."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview = tmp_path / "preview.wav"
        audio_preview.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_extraction_complete(audio_preview, auto_play=False)
        panel.clear()

        assert panel.get_generated_audio_path() is None


class TestExtractionAccessibility:
    """Tests for extraction accessibility (Story 2.3)."""

    def test_preview_text_edit_has_accessible_name(self, panel):
        """Test preview text field has accessible name."""
        assert panel.preview_text_edit.accessibleName() == "Preview Text"

    def test_extract_button_has_accessible_name(self, panel):
        """Test Extract Embedding button has accessible name."""
        assert panel.extract_button.accessibleName() == "Extract Embedding"

    def test_extract_retry_button_has_accessible_name(self, panel):
        """Test retry button has accessible name."""
        assert panel.extract_retry_button.accessibleName() != ""


class TestSaveReadinessWithExtraction:
    """Tests for save readiness requiring extraction (Story 2.3)."""

    def test_not_save_ready_without_extraction(self, panel, tmp_path):
        """Test not save ready without extraction even with file and name."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_voice_name("My Voice")

        # Should not be save ready without extraction
        assert not panel.is_save_ready()

    def test_save_ready_after_extraction_with_name(self, panel, tmp_path):
        """Test save ready after extraction and voice name entered."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview = tmp_path / "preview.wav"
        audio_preview.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_voice_name("My Voice")
        panel.set_extraction_complete(audio_preview, auto_play=False)

        assert panel.is_save_ready()


# Story 2.4: Re-Extract After Edit Tests

class TestTranscriptModifiedIndicator:
    """Tests for transcript modified indicator (Story 2.4)."""

    def test_transcript_modified_label_exists(self, panel):
        """Test transcript modified label exists."""
        assert hasattr(panel, 'transcript_modified_label')

    def test_transcript_modified_label_initially_hidden(self, panel):
        """Test transcript modified indicator hidden initially."""
        assert not panel.transcript_modified_label.isVisible()

    def test_no_modified_state_before_extraction(self, panel, tmp_path):
        """Test no modified state before any extraction."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_transcript("Hello world")

        # No extraction done yet, so no modified state
        assert not panel.is_transcript_modified_since_extraction()
        assert not panel.transcript_modified_label.isVisible()

    def test_modified_indicator_after_transcript_change(self, panel, tmp_path):
        """Test modified indicator shows after changing transcript post-extraction."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview = tmp_path / "preview.wav"
        audio_preview.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_transcript("Original transcript")
        panel.set_extraction_complete(audio_preview, auto_play=False)

        # Not modified yet
        assert not panel.is_transcript_modified_since_extraction()
        assert not panel.transcript_modified_label.isVisible()

        # Now modify the transcript
        panel.set_transcript("Modified transcript")

        assert panel.is_transcript_modified_since_extraction()
        assert panel.transcript_modified_label.isVisible()

    def test_modified_indicator_hidden_after_re_extraction(self, panel, tmp_path):
        """Test modified indicator hidden after re-extraction."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview = tmp_path / "preview.wav"
        audio_preview.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_transcript("Original")
        panel.set_extraction_complete(audio_preview, auto_play=False)

        # Modify transcript
        panel.set_transcript("Modified")
        assert panel.transcript_modified_label.isVisible()

        # Re-extract
        panel.set_extraction_complete(audio_preview, auto_play=False)

        # Indicator should be hidden after new extraction
        assert not panel.is_transcript_modified_since_extraction()
        assert not panel.transcript_modified_label.isVisible()

    def test_modified_indicator_false_when_reverted(self, panel, tmp_path):
        """Test modified indicator hidden when transcript reverted to original."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview = tmp_path / "preview.wav"
        audio_preview.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_transcript("Original transcript")
        panel.set_extraction_complete(audio_preview, auto_play=False)

        # Modify
        panel.set_transcript("Modified")
        assert panel.transcript_modified_label.isVisible()

        # Revert to original
        panel.set_transcript("Original transcript")
        assert not panel.transcript_modified_label.isVisible()


class TestLastExtractedTranscript:
    """Tests for tracking last extracted transcript (Story 2.4)."""

    def test_get_last_extracted_transcript_initially_none(self, panel):
        """Test get_last_extracted_transcript returns None initially."""
        assert panel.get_last_extracted_transcript() is None

    def test_get_last_extracted_transcript_after_extraction(self, panel, tmp_path):
        """Test get_last_extracted_transcript returns transcript used."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview = tmp_path / "preview.wav"
        audio_preview.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_transcript("Test transcript")
        panel.set_extraction_complete(audio_preview, auto_play=False)

        assert panel.get_last_extracted_transcript() == "Test transcript"

    def test_has_had_extraction_false_initially(self, panel):
        """Test has_had_extraction returns False initially."""
        assert not panel.has_had_extraction()

    def test_has_had_extraction_true_after_extraction(self, panel, tmp_path):
        """Test has_had_extraction returns True after extraction."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview = tmp_path / "preview.wav"
        audio_preview.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_extraction_complete(audio_preview, auto_play=False)

        assert panel.has_had_extraction()


class TestReExtractionBehavior:
    """Tests for re-extraction behavior (Story 2.4)."""

    def test_extract_button_remains_available_after_extraction(self, panel, tmp_path):
        """Test Extract button remains available after first extraction."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview = tmp_path / "preview.wav"
        audio_preview.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_preview_text("Hello world")
        panel.set_extraction_complete(audio_preview, auto_play=False)

        # Button should still be enabled for re-extraction
        assert panel.extract_button.isEnabled()

    def test_re_extraction_clears_previous_preview(self, panel, tmp_path):
        """Test re-extraction clears previous preview."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview1 = tmp_path / "preview1.wav"
        audio_preview1.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview2 = tmp_path / "preview2.wav"
        audio_preview2.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_preview_text("Hello")

        # First extraction
        panel.set_extraction_complete(audio_preview1, auto_play=False)
        assert panel.get_generated_audio_path() == audio_preview1

        # Simulate re-extraction starting
        panel.set_extracting(True)
        assert not panel.preview_audio_group.isVisible()

        # Second extraction completes
        panel.set_extraction_complete(audio_preview2, auto_play=False)
        assert panel.get_generated_audio_path() == audio_preview2

    def test_only_most_recent_extraction_retained(self, panel, tmp_path):
        """Test only most recent extraction is retained."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview1 = tmp_path / "preview1.wav"
        audio_preview1.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview2 = tmp_path / "preview2.wav"
        audio_preview2.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        # First extraction with transcript A
        panel.set_transcript("Transcript A")
        panel.set_extraction_complete(audio_preview1, auto_play=False)
        assert panel.get_last_extracted_transcript() == "Transcript A"

        # Second extraction with transcript B
        panel.set_transcript("Transcript B")
        panel.set_extraction_complete(audio_preview2, auto_play=False)
        assert panel.get_last_extracted_transcript() == "Transcript B"
        assert panel.get_generated_audio_path() == audio_preview2

    def test_changing_preview_text_after_extraction(self, panel, tmp_path):
        """Test changing preview text works with same embedding."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview = tmp_path / "preview.wav"
        audio_preview.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_preview_text("First preview text")
        panel.set_transcript("My transcript")
        panel.set_extraction_complete(audio_preview, auto_play=False)

        # Change preview text
        panel.set_preview_text("Different preview text")

        # Extract button still enabled
        assert panel.extract_button.isEnabled()

        # Transcript hasn't changed, so no modified indicator
        assert not panel.is_transcript_modified_since_extraction()


class TestReExtractionClear:
    """Tests for clearing re-extraction state (Story 2.4)."""

    def test_clear_resets_last_extracted_transcript(self, panel, tmp_path):
        """Test clear resets last extracted transcript."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview = tmp_path / "preview.wav"
        audio_preview.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_transcript("Test")
        panel.set_extraction_complete(audio_preview, auto_play=False)

        panel.clear()

        assert panel.get_last_extracted_transcript() is None
        assert not panel.has_had_extraction()

    def test_clear_resets_transcript_modified_state(self, panel, tmp_path):
        """Test clear resets transcript modified state."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview = tmp_path / "preview.wav"
        audio_preview.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_transcript("Original")
        panel.set_extraction_complete(audio_preview, auto_play=False)
        panel.set_transcript("Modified")

        panel.clear()

        assert not panel.is_transcript_modified_since_extraction()

    def test_clear_hides_modified_indicator(self, panel, tmp_path):
        """Test clear hides transcript modified indicator."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
        audio_preview = tmp_path / "preview.wav"
        audio_preview.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(panel, '_get_audio_duration', return_value=5.0):
            panel._load_audio_file(wav_file)

        panel.set_transcript("Original")
        panel.set_extraction_complete(audio_preview, auto_play=False)
        panel.set_transcript("Modified")
        assert panel.transcript_modified_label.isVisible()

        panel.clear()

        assert not panel.transcript_modified_label.isVisible()


class TestTranscriptModifiedAccessibility:
    """Tests for transcript modified indicator accessibility (Story 2.4)."""

    def test_transcript_modified_label_has_accessible_name(self, panel):
        """Test transcript modified label has accessible name."""
        assert panel.transcript_modified_label.accessibleName() != ""
