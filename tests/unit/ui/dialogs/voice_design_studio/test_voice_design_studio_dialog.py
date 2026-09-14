"""
Tests for VoiceDesignStudioDialog Component

Story 1.2: Voice Design Studio Dialog Shell
Covers:
- FR1: User can access Voice Design Studio from the main application
- FR2: User can switch between "From Description" and "From Sample" tabs
- FR3: System displays the currently active tab with visual indication
- FR4: User can close the dialog and return to main application

Acceptance Criteria:
- Menu action opens modal dialog titled "Voice Design Studio"
- Dialog displays two tabs: "From Description" and "From Sample"
- "From Description" tab is active by default with visual indicator
- Clicking tabs switches content panels
- Close button (X) closes dialog and returns to main app
"""

import pytest
from unittest.mock import Mock, MagicMock

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QTabWidget
from PyQt6.QtCore import Qt

from myvoice.ui.dialogs.voice_design_studio import VoiceDesignStudioDialog


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def dialog(qapp):
    """Create a VoiceDesignStudioDialog instance."""
    dialog = VoiceDesignStudioDialog()
    yield dialog
    dialog.deleteLater()


@pytest.fixture
def tts_service_stub():
    """A TTS service stub that passes _on_generate_requested's two guards
    (tooling-4, Site B).

    The regenerate tests exist to assert that regenerate routes into the
    generate flow. That flow guards on ``self._tts_service`` and
    ``self._tts_service.is_running()`` and pops a *modal* QMessageBox.warning
    when either fails -- which is what happened when the tests constructed
    the dialog with no service at all (3 of tooling-3's 27 timeouts).
    """
    from myvoice.services.qwen_tts_service import QwenTTSService

    stub = MagicMock(spec=QwenTTSService)
    stub.is_running.return_value = True
    return stub


def _capture_async_dispatch(dialog):
    """Replace ``dialog._run_async_task`` so the generate flow's per-variant
    coroutine is recorded and closed instead of being scheduled with
    ``asyncio.ensure_future`` on whatever loop is current under pytest.

    Without this a pending task carrying a MagicMock TTS call is left on the
    main-thread loop and can run inside a later test after the dialog is gone.
    Returns the list the dispatched coroutines are appended to.
    """
    dispatched = []

    def _record(coro, on_success=None, on_error=None):
        dispatched.append(coro)
        coro.close()

    dialog._run_async_task = _record
    return dispatched


class TestVoiceDesignStudioDialogInit:
    """Tests for VoiceDesignStudioDialog initialization (FR1)."""

    def test_dialog_creation(self, dialog):
        """Test dialog creates successfully."""
        assert dialog is not None
        assert isinstance(dialog, VoiceDesignStudioDialog)

    def test_dialog_title(self, dialog):
        """Test dialog has correct title 'Voice Design Studio'."""
        assert dialog.windowTitle() == "Voice Design Studio"

    def test_dialog_is_modal(self, dialog):
        """Test dialog is modal."""
        assert dialog.isModal()

    def test_dialog_has_close_button(self, dialog):
        """Test dialog has close button in window flags."""
        flags = dialog.windowFlags()
        assert flags & Qt.WindowType.WindowCloseButtonHint


class TestVoiceDesignStudioTabs:
    """Tests for tab widget functionality (FR2, FR3)."""

    def test_tab_widget_exists(self, dialog):
        """Test tab widget exists."""
        assert hasattr(dialog, 'tab_widget')
        assert isinstance(dialog.tab_widget, QTabWidget)

    def test_has_two_tabs(self, dialog):
        """Test dialog has exactly two tabs."""
        assert dialog.tab_widget.count() == 2

    def test_first_tab_is_from_description(self, dialog):
        """Test first tab is labeled 'From Description'."""
        tab_text = dialog.tab_widget.tabText(0)
        assert tab_text == "From Description"

    def test_second_tab_is_from_sample(self, dialog):
        """Test second tab is labeled 'From Sample'."""
        tab_text = dialog.tab_widget.tabText(1)
        assert tab_text == "From Sample"

    def test_from_description_tab_active_by_default(self, dialog):
        """Test 'From Description' tab is active by default."""
        assert dialog.tab_widget.currentIndex() == 0

    def test_tab_switching_works(self, dialog):
        """Test clicking tabs switches content panels."""
        # Initially on tab 0
        assert dialog.tab_widget.currentIndex() == 0

        # Switch to tab 1 (From Sample)
        dialog.tab_widget.setCurrentIndex(1)
        assert dialog.tab_widget.currentIndex() == 1

        # Switch back to tab 0 (From Description)
        dialog.tab_widget.setCurrentIndex(0)
        assert dialog.tab_widget.currentIndex() == 0

    def test_get_active_path_description(self, dialog):
        """Test get_active_path returns 'description' when on first tab."""
        dialog.tab_widget.setCurrentIndex(0)
        assert dialog.get_active_path() == "description"

    def test_get_active_path_sample(self, dialog):
        """Test get_active_path returns 'sample' when on second tab."""
        dialog.tab_widget.setCurrentIndex(1)
        assert dialog.get_active_path() == "sample"


class TestVoiceDesignStudioButtons:
    """Tests for dialog buttons."""

    def test_cancel_button_exists(self, dialog):
        """Test cancel button exists."""
        assert hasattr(dialog, 'cancel_button')
        assert dialog.cancel_button is not None

    def test_save_button_exists(self, dialog):
        """Test save button exists."""
        assert hasattr(dialog, 'save_button')
        assert dialog.save_button is not None

    def test_save_button_initially_disabled(self, dialog):
        """Test save button is disabled initially (no voice ready)."""
        assert not dialog.save_button.isEnabled()


class TestVoiceDesignStudioSignals:
    """Tests for dialog signals."""

    def test_voice_saved_signal_exists(self, dialog):
        """Test voice_saved signal exists."""
        assert hasattr(dialog, 'voice_saved')

    def test_dialog_closing_signal_exists(self, dialog):
        """Test dialog_closing signal exists."""
        assert hasattr(dialog, 'dialog_closing')

    def test_dialog_closing_emits_on_cancel(self, dialog):
        """Test dialog_closing signal emits when cancel is clicked."""
        signal_received = []

        def on_closing(has_unsaved):
            signal_received.append(has_unsaved)

        dialog.dialog_closing.connect(on_closing)
        dialog._on_cancel_clicked()

        assert len(signal_received) == 1
        assert signal_received[0] is False  # No unsaved work initially


class TestVoiceDesignStudioAccessibility:
    """Tests for accessibility features."""

    def test_dialog_has_accessible_name(self, dialog):
        """Test dialog has accessible name set."""
        assert dialog.accessibleName() == "Voice Design Studio"

    def test_tab_widget_has_accessible_name(self, dialog):
        """Test tab widget has accessible name set."""
        assert dialog.tab_widget.accessibleName() == "Voice creation method"

    def test_cancel_button_has_accessible_name(self, dialog):
        """Test cancel button has accessible name set."""
        assert dialog.cancel_button.accessibleName() == "Cancel"

    def test_save_button_has_accessible_name(self, dialog):
        """Test save button has accessible name set."""
        assert dialog.save_button.accessibleName() == "Save Voice"


class TestVoiceDesignStudioUnsavedWorkTracking:
    """Tests for unsaved work tracking."""

    def test_initial_no_unsaved_work(self, dialog):
        """Test dialog starts with no unsaved work."""
        assert dialog._has_unsaved_work is False

    def test_set_has_unsaved_work(self, dialog):
        """Test set_has_unsaved_work updates the flag."""
        dialog.set_has_unsaved_work(True)
        assert dialog._has_unsaved_work is True

        dialog.set_has_unsaved_work(False)
        assert dialog._has_unsaved_work is False

    def test_closing_emits_unsaved_work_flag(self, dialog):
        """Test closing emits correct unsaved work flag."""
        signal_received = []

        def on_closing(has_unsaved):
            signal_received.append(has_unsaved)

        dialog.dialog_closing.connect(on_closing)

        # Set unsaved work and cancel
        dialog.set_has_unsaved_work(True)
        dialog._on_cancel_clicked()

        assert len(signal_received) == 1
        assert signal_received[0] is True


# Story 1.5 Tests

class TestSaveButtonState:
    """Tests for Save button state management (Story 1.5)."""

    def test_save_button_initially_disabled(self, dialog):
        """Test save button is disabled when no voice is ready."""
        assert not dialog.save_button.isEnabled()

    def test_save_button_enabled_when_save_ready(self, dialog):
        """Test save button enables when save_ready_changed emits True."""
        dialog._on_save_ready_changed(True)
        assert dialog.save_button.isEnabled()

    def test_save_button_disabled_when_save_not_ready(self, dialog):
        """Test save button disables when save_ready_changed emits False."""
        dialog._on_save_ready_changed(True)
        dialog._on_save_ready_changed(False)
        assert not dialog.save_button.isEnabled()

    def test_tab_switch_disables_save_on_sample_tab(self, dialog):
        """Test save button is disabled when switching to sample tab."""
        dialog._on_save_ready_changed(True)  # Enable on description tab
        dialog.tab_widget.setCurrentIndex(1)  # Switch to sample tab
        assert not dialog.save_button.isEnabled()

    def test_tab_switch_restores_save_state_on_description_tab(self, dialog, tmp_path):
        """Test save button state restored when returning to description tab."""
        # Setup: create fake embedding and set name
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK")
        audio_file = tmp_path / "preview.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 40)

        dialog.description_panel.set_generation_complete(audio_file, embedding_file)
        dialog.description_panel.set_voice_name("Test Voice")

        # Switch to sample tab then back
        dialog.tab_widget.setCurrentIndex(1)
        assert not dialog.save_button.isEnabled()

        dialog.tab_widget.setCurrentIndex(0)
        assert dialog.save_button.isEnabled()


class TestGetEmbeddingsSaveDir:
    """Tests for embeddings save directory (Story 1.5)."""

    def test_get_embeddings_save_dir_returns_path(self, dialog):
        """Test _get_embeddings_save_dir returns a Path."""
        from pathlib import Path
        save_dir = dialog._get_embeddings_save_dir()
        assert isinstance(save_dir, Path)

    def test_get_embeddings_save_dir_ends_with_embeddings(self, dialog):
        """Test save dir ends with voice_files/embeddings."""
        save_dir = dialog._get_embeddings_save_dir()
        assert save_dir.name == "embeddings"
        assert save_dir.parent.name == "voice_files"


class TestSaveFromDescription:
    """Tests for save from description functionality (Story 1.5)."""

    def test_voice_saved_signal_emits_on_save(self, dialog, tmp_path, monkeypatch):
        """Test voice_saved signal emits with voice name on successful save."""
        from PyQt6.QtWidgets import QMessageBox

        # Setup: create fake embedding and audio files
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK")
        audio_file = tmp_path / "preview.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 40)

        # NOTE: set_description triggers _on_content_changed which clears results
        # So we set description FIRST, then set paths directly
        dialog.description_panel.set_description("A test voice description")
        dialog.description_panel._generated_embedding_path = embedding_file
        dialog.description_panel._generated_audio_path = audio_file
        dialog.description_panel.set_voice_name("TestVoice")

        # Mock save directory
        save_dir = tmp_path / "save_location"
        monkeypatch.setattr(dialog, '_get_embeddings_save_dir', lambda: save_dir)

        # Mock QMessageBox to prevent blocking dialogs
        monkeypatch.setattr(QMessageBox, 'warning', lambda *args, **kwargs: None)
        monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.StandardButton.Yes)

        # Capture signal
        signal_received = []
        dialog.voice_saved.connect(lambda name: signal_received.append(name))

        # Trigger save
        dialog._save_from_description()

        assert len(signal_received) == 1
        assert signal_received[0] == "TestVoice"

    def test_save_creates_directory(self, dialog, tmp_path, monkeypatch):
        """Test save creates the voice directory."""
        from PyQt6.QtWidgets import QMessageBox

        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK")
        audio_file = tmp_path / "preview.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 40)

        # Set paths directly
        dialog.description_panel._generated_embedding_path = embedding_file
        dialog.description_panel._generated_audio_path = audio_file
        dialog.description_panel.set_voice_name("TestVoice")

        save_dir = tmp_path / "save_location"
        monkeypatch.setattr(dialog, '_get_embeddings_save_dir', lambda: save_dir)
        monkeypatch.setattr(QMessageBox, 'warning', lambda *args, **kwargs: None)

        dialog._save_from_description()

        voice_dir = save_dir / "TestVoice"
        assert voice_dir.exists()
        assert voice_dir.is_dir()

    def test_save_creates_embedding_file(self, dialog, tmp_path, monkeypatch):
        """Test save copies embedding.pt file."""
        from PyQt6.QtWidgets import QMessageBox

        embedding_file = tmp_path / "source_embedding.pt"
        embedding_file.write_bytes(b"PK_TEST_EMBEDDING")
        audio_file = tmp_path / "preview.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 40)

        # Set paths directly
        dialog.description_panel._generated_embedding_path = embedding_file
        dialog.description_panel._generated_audio_path = audio_file
        dialog.description_panel.set_voice_name("TestVoice")

        save_dir = tmp_path / "save_location"
        monkeypatch.setattr(dialog, '_get_embeddings_save_dir', lambda: save_dir)
        monkeypatch.setattr(QMessageBox, 'warning', lambda *args, **kwargs: None)

        dialog._save_from_description()

        dest_embedding = save_dir / "TestVoice" / "embedding.pt"
        assert dest_embedding.exists()
        assert dest_embedding.read_bytes() == b"PK_TEST_EMBEDDING"

    def test_save_creates_metadata_file(self, dialog, tmp_path, monkeypatch):
        """Test save creates metadata.json file."""
        import json
        from PyQt6.QtWidgets import QMessageBox

        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK")
        audio_file = tmp_path / "preview.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 40)

        # NOTE: set_description triggers _on_content_changed which clears results
        # So we set description FIRST, then set paths directly
        dialog.description_panel.set_description("A warm friendly voice")
        dialog.description_panel._generated_embedding_path = embedding_file
        dialog.description_panel._generated_audio_path = audio_file
        dialog.description_panel.set_voice_name("TestVoice")

        save_dir = tmp_path / "save_location"
        monkeypatch.setattr(dialog, '_get_embeddings_save_dir', lambda: save_dir)
        monkeypatch.setattr(QMessageBox, 'warning', lambda *args, **kwargs: None)

        dialog._save_from_description()

        metadata_path = save_dir / "TestVoice" / "metadata.json"
        assert metadata_path.exists()

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        assert metadata["name"] == "TestVoice"
        assert metadata["description"] == "A warm friendly voice"
        assert metadata["voice_type"] == "designed"
        assert metadata["emotion_capable"] is True
        assert "created_at" in metadata

    def test_save_clears_unsaved_work_flag(self, dialog, tmp_path, monkeypatch):
        """Test successful save clears unsaved work flag."""
        from PyQt6.QtWidgets import QMessageBox

        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK")
        audio_file = tmp_path / "preview.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 40)

        # Set paths directly
        dialog.description_panel._generated_embedding_path = embedding_file
        dialog.description_panel._generated_audio_path = audio_file
        dialog.description_panel.set_voice_name("TestVoice")
        dialog.set_has_unsaved_work(True)

        save_dir = tmp_path / "save_location"
        monkeypatch.setattr(dialog, '_get_embeddings_save_dir', lambda: save_dir)
        monkeypatch.setattr(QMessageBox, 'warning', lambda *args, **kwargs: None)

        dialog._save_from_description()

        assert dialog._has_unsaved_work is False


# Story 1.9 Tests

class TestRegenerateAllVariations:
    """Tests for Regenerate All functionality (Story 1.9)."""

    def test_regenerate_requested_signal_connected(self, dialog):
        """Test regenerate_requested signal is connected to dialog handler."""
        # Check signal is connected by verifying handler exists
        assert hasattr(dialog, '_on_regenerate_requested')
        assert callable(dialog._on_regenerate_requested)

    def test_regenerate_triggers_generate_flow(self, qapp, tts_service_stub):
        """Test regenerate calls the same flow as initial generate."""
        # tooling-4: the real generate flow is reached here, so the dialog
        # needs a service that passes the guards, and the async dispatch is
        # captured rather than scheduled.
        dialog = VoiceDesignStudioDialog(tts_service=tts_service_stub)
        dispatched = _capture_async_dispatch(dialog)

        # Setup description and preview text
        dialog.description_panel.set_description("A warm friendly voice")
        dialog.description_panel.set_preview_text("Hello world")

        # Track if _on_generate_requested was called
        generate_calls = []
        original_handler = dialog._on_generate_requested

        def mock_generate(desc, preview, language):
            generate_calls.append((desc, preview, language))
            original_handler(desc, preview, language)

        dialog._on_generate_requested = mock_generate

        # Trigger regenerate
        dialog._on_regenerate_requested()

        assert len(generate_calls) == 1
        assert generate_calls[0][:2] == ("A warm friendly voice", "Hello world")

        # The generate flow actually started (it got past both guards and
        # dispatched the first variant), rather than bouncing off a modal.
        assert dialog._is_generating is True
        assert dialog._last_description == "A warm friendly voice"
        assert dialog._last_preview_text == "Hello world"
        assert len(dispatched) == 1

        dialog._session_manager.cleanup()
        dialog.deleteLater()

    def test_regenerate_uses_current_description(self, dialog):
        """Test regenerate uses the current description text."""
        dialog.description_panel.set_description("New test description")

        generate_calls = []

        def mock_generate(desc, preview, language):
            generate_calls.append((desc, preview, language))

        dialog._on_generate_requested = mock_generate

        dialog._on_regenerate_requested()

        assert generate_calls[0][0] == "New test description"

    def test_regenerate_uses_current_preview_text(self, dialog):
        """Test regenerate uses the current preview text."""
        dialog.description_panel.set_description("Test")
        dialog.description_panel.set_preview_text("Custom preview")

        generate_calls = []

        def mock_generate(desc, preview, language):
            generate_calls.append((desc, preview, language))

        dialog._on_generate_requested = mock_generate

        dialog._on_regenerate_requested()

        assert generate_calls[0][1] == "Custom preview"


class TestGenerateGuardsOnTtsService:
    """The two service guards at the top of _on_generate_requested
    (tooling-4, AC #2).

    Each guard shows a modal QMessageBox.warning and returns without starting
    generation. Under pytest a real modal blocks forever, so the tests patch
    ``QMessageBox.warning`` and assert it was called with the guard's title.
    The guards themselves are production behaviour and are not changed.
    """

    @staticmethod
    def _capture_warning(monkeypatch):
        from PyQt6.QtWidgets import QMessageBox

        calls = []

        def fake_warning(parent, title, text, *args, **kwargs):
            calls.append((title, text))
            return QMessageBox.StandardButton.Ok

        monkeypatch.setattr(QMessageBox, "warning", fake_warning)
        return calls

    def test_generate_without_service_warns_instead_of_starting(
        self, dialog, monkeypatch
    ):
        """_tts_service is None -> "Service Unavailable", nothing starts."""
        assert dialog._tts_service is None
        warnings = self._capture_warning(monkeypatch)

        dialog._on_generate_requested("A voice", "Hello", "English")

        assert [title for title, _ in warnings] == ["Service Unavailable"]
        assert dialog._is_generating is False

    def test_regenerate_without_service_warns_instead_of_starting(
        self, dialog, monkeypatch
    ):
        """The exact shape that hung tooling-3: regenerate on a dialog built
        with no service routes into the guard and must end at the warning."""
        warnings = self._capture_warning(monkeypatch)
        dialog.description_panel.set_description("A voice")
        dialog.description_panel.set_preview_text("Hello")

        dialog._on_regenerate_requested()

        assert [title for title, _ in warnings] == ["Service Unavailable"]
        assert dialog._is_generating is False

    def test_generate_with_stopped_service_warns_instead_of_starting(
        self, qapp, tts_service_stub, monkeypatch
    ):
        """_tts_service present but not running -> "Service Not Ready"."""
        tts_service_stub.is_running.return_value = False
        dialog = VoiceDesignStudioDialog(tts_service=tts_service_stub)
        warnings = self._capture_warning(monkeypatch)

        dialog._on_generate_requested("A voice", "Hello", "English")

        assert [title for title, _ in warnings] == ["Service Not Ready"]
        assert dialog._is_generating is False

        dialog._session_manager.cleanup()
        dialog.deleteLater()


# Story 2.1 Tests

class TestSampleTab:
    """Tests for the 'From Sample' tab (Story 2.1)."""

    def test_sample_panel_exists(self, dialog):
        """Test sample panel exists."""
        assert hasattr(dialog, 'sample_panel')
        assert dialog.sample_panel is not None

    def test_sample_panel_is_sample_path_panel(self, dialog):
        """Test sample panel is SamplePathPanel instance."""
        from myvoice.ui.dialogs.voice_design_studio.sample_path_panel import SamplePathPanel
        assert isinstance(dialog.sample_panel, SamplePathPanel)

    def test_sample_tab_has_browse_button(self, dialog):
        """Test sample tab has browse button."""
        assert hasattr(dialog.sample_panel, 'browse_button')
        assert dialog.sample_panel.browse_button is not None

    def test_switch_to_sample_tab(self, dialog):
        """Test can switch to sample tab."""
        dialog.tab_widget.setCurrentIndex(1)
        assert dialog.tab_widget.currentIndex() == 1

    def test_save_button_disabled_on_sample_tab_initially(self, dialog):
        """Test save button is disabled on sample tab initially."""
        dialog.tab_widget.setCurrentIndex(1)
        assert not dialog.save_button.isEnabled()


class TestSampleTabSaveReady:
    """Tests for save button state on sample tab."""

    def test_save_button_enabled_when_sample_ready(self, dialog, tmp_path):
        """Test save button enables when sample loaded, extraction complete, and name entered."""
        from unittest.mock import patch

        dialog.tab_widget.setCurrentIndex(1)

        # Create a fake file
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        # Create a fake preview audio
        preview_audio = tmp_path / "preview.wav"
        preview_audio.write_bytes(b"RIFF" + b"\x00" * 40)

        # Load file (mock duration)
        with patch.object(dialog.sample_panel, '_get_audio_duration', return_value=5.0):
            dialog.sample_panel._load_audio_file(wav_file)

        # Story 2.3: Extraction must complete before save is available
        dialog.sample_panel.set_extraction_complete(preview_audio, auto_play=False)

        dialog.sample_panel.set_voice_name("Test Voice")

        assert dialog.save_button.isEnabled()

    def test_tab_switch_updates_save_button(self, dialog, tmp_path):
        """Test switching tabs updates save button state."""
        from unittest.mock import patch

        # Setup sample panel with file, extraction, and name
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        preview_audio = tmp_path / "preview.wav"
        preview_audio.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(dialog.sample_panel, '_get_audio_duration', return_value=5.0):
            dialog.sample_panel._load_audio_file(wav_file)

        # Story 2.3: Extraction must complete before save is available
        dialog.sample_panel.set_extraction_complete(preview_audio, auto_play=False)

        dialog.sample_panel.set_voice_name("Test Voice")

        # Switch to sample tab
        dialog.tab_widget.setCurrentIndex(1)
        assert dialog.save_button.isEnabled()

        # Switch back to description tab
        dialog.tab_widget.setCurrentIndex(0)
        assert not dialog.save_button.isEnabled()  # Description tab not ready


class TestSampleTabSignals:
    """Tests for sample tab signal handling."""

    def test_file_loaded_sets_unsaved_work(self, dialog, tmp_path):
        """Test loading file sets unsaved work flag."""
        from unittest.mock import patch

        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.object(dialog.sample_panel, '_get_audio_duration', return_value=5.0):
            dialog.sample_panel._load_audio_file(wav_file)

        assert dialog._has_unsaved_work is True


class TestSessionManagement:
    """Tests for session directory management (Story 4.1)."""

    def test_dialog_creates_session_manager(self, dialog):
        """Test dialog creates SessionManager on initialization."""
        from myvoice.utils.session_manager import SessionManager

        assert hasattr(dialog, '_session_manager')
        assert isinstance(dialog._session_manager, SessionManager)

    def test_dialog_has_session_dir_property(self, dialog):
        """Test dialog exposes session_dir property."""
        from pathlib import Path

        session_dir = dialog.session_dir
        assert isinstance(session_dir, Path)
        assert session_dir.exists()

    def test_session_dir_is_unique(self, qapp):
        """Test each dialog instance has unique session directory."""
        dialog1 = VoiceDesignStudioDialog()
        dialog2 = VoiceDesignStudioDialog()

        try:
            assert dialog1.session_dir != dialog2.session_dir
            assert dialog1._session_manager.session_id != dialog2._session_manager.session_id
        finally:
            dialog1.deleteLater()
            dialog2.deleteLater()

    def test_close_event_preserves_session(self, qapp):
        """Test closing dialog preserves session directory (Story 4.1 revised)."""
        from PyQt6.QtGui import QCloseEvent

        dialog = VoiceDesignStudioDialog()
        session_dir = dialog.session_dir

        # Create temp files in session
        (session_dir / "variant_1.pt").touch()
        (session_dir / "variant_1.wav").touch()

        # Verify session exists before close
        assert session_dir.exists()

        # Simulate close event
        event = QCloseEvent()
        dialog.closeEvent(event)

        # Story 4.1 REVISED: Session directory should be PRESERVED on close
        # This allows user to save additional variants later
        assert session_dir.exists(), "Session should be preserved on close"
        assert (session_dir / "variant_1.pt").exists(), "Variant files should be preserved"
        assert not dialog._session_manager.is_cleaned, "Session should not be marked cleaned"

        # Manual cleanup for test
        dialog._session_manager.cleanup()
        dialog.deleteLater()

    def test_cancel_button_preserves_session(self, qapp):
        """Test clicking Cancel preserves session directory (Story 4.1 revised)."""
        dialog = VoiceDesignStudioDialog()
        session_dir = dialog.session_dir

        # Create temp files
        (session_dir / "variant_1.pt").touch()

        assert session_dir.exists()

        # Click cancel
        dialog._on_cancel_clicked()

        # Story 4.1 REVISED: Session should be PRESERVED on cancel
        assert session_dir.exists(), "Session should be preserved on cancel"

        # Manual cleanup for test
        dialog._session_manager.cleanup()
        dialog.deleteLater()

    def test_session_files_preserved_until_regenerate(self, dialog, tmp_path):
        """Test temp files in session remain until Regenerate is clicked (Story 4.1/1.9)."""
        session_dir = dialog.session_dir

        # Create temp files in session (simulating generation)
        test_embedding = session_dir / "variant_1.pt"
        test_audio = session_dir / "variant_1.wav"
        test_embedding.touch()
        test_audio.touch()

        # Files should exist while dialog is open
        assert test_embedding.exists()
        assert test_audio.exists()

    def test_regenerate_clears_variant_files(self, qapp, tts_service_stub):
        """Test Regenerate clears variant files but preserves session dir (Story 1.9/4.1)."""
        # tooling-4: regenerate routes into the generate flow, which guards on
        # the TTS service with a modal; give it a service and capture the
        # async dispatch so the flow runs to its hand-off.
        dialog = VoiceDesignStudioDialog(tts_service=tts_service_stub)
        _capture_async_dispatch(dialog)
        session_dir = dialog.session_dir

        # Create various temp files
        (session_dir / "variant_1.pt").touch()
        (session_dir / "variant_1.wav").touch()
        (session_dir / "variant_2.pt").touch()
        (session_dir / "variant_2.wav").touch()

        # Trigger regenerate
        dialog._on_regenerate_requested()

        # Story 1.9: Variant files should be cleared
        assert not (session_dir / "variant_1.pt").exists()
        assert not (session_dir / "variant_1.wav").exists()
        assert not (session_dir / "variant_2.pt").exists()
        assert not (session_dir / "variant_2.wav").exists()

        # Story 4.1: Session directory should still exist for new variants
        assert session_dir.exists(), "Session directory should be preserved"

        # Manual cleanup for test
        dialog._session_manager.cleanup()
        dialog.deleteLater()

    def test_regenerate_preserves_non_variant_files(self, qapp, tts_service_stub):
        """Test Regenerate preserves non-variant files in session."""
        # tooling-4: see test_regenerate_clears_variant_files.
        dialog = VoiceDesignStudioDialog(tts_service=tts_service_stub)
        _capture_async_dispatch(dialog)
        session_dir = dialog.session_dir

        # Create variant and non-variant files
        (session_dir / "variant_1.pt").touch()
        (session_dir / "metadata.json").touch()
        (session_dir / "preview.wav").touch()

        # Trigger regenerate
        dialog._on_regenerate_requested()

        # Variant files cleared
        assert not (session_dir / "variant_1.pt").exists()

        # Non-variant files preserved
        assert (session_dir / "metadata.json").exists()
        assert (session_dir / "preview.wav").exists()

        # Manual cleanup for test
        dialog._session_manager.cleanup()
        dialog.deleteLater()
