"""
Tests for Model Switching Indicator in MainWindow

QA8: Add model switching indicator to UI
- Status bar shows model loading/ready messages
- TTS indicator updates during model switches
- Tooltip shows active model name
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from pathlib import Path

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QStatusBar
from PyQt6.QtCore import Qt


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestModelCallbackConnectionCode:
    """Tests verifying callback connection code exists in main_window.py."""

    def test_callback_connection_code_exists(self):
        """Test that model callback connection code exists in main_window.py."""
        main_window_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "ui" / "main_window.py"
        source = main_window_path.read_text(encoding='utf-8')

        # Verify callback connections are in set_tts_service
        assert "set_model_loading_callback" in source, \
            "Should connect model_loading_callback in set_tts_service"
        assert "set_model_ready_callback" in source, \
            "Should connect model_ready_callback in set_tts_service"
        assert "_on_model_loading" in source, \
            "Should have _on_model_loading handler method"
        assert "_on_model_ready" in source, \
            "Should have _on_model_ready handler method"

    def test_on_model_loading_handler_exists(self):
        """Test that _on_model_loading handler method exists."""
        main_window_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "ui" / "main_window.py"
        source = main_window_path.read_text(encoding='utf-8')

        # Verify handler signature
        assert "def _on_model_loading(self, message: str)" in source, \
            "_on_model_loading should accept message parameter"

    def test_on_model_ready_handler_exists(self):
        """Test that _on_model_ready handler method exists."""
        main_window_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "ui" / "main_window.py"
        source = main_window_path.read_text(encoding='utf-8')

        # Verify handler signature
        assert "def _on_model_ready(self, model_name: str)" in source, \
            "_on_model_ready should accept model_name parameter"


class TestModelLoadingHandlerBehavior:
    """Tests for _on_model_loading handler behavior."""

    def test_loading_updates_status_bar(self):
        """Test that loading handler updates status bar."""
        main_window_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "ui" / "main_window.py"
        source = main_window_path.read_text(encoding='utf-8')

        # Find _on_model_loading method
        in_method = False
        has_status_bar_update = False

        for line in source.split('\n'):
            if 'def _on_model_loading' in line:
                in_method = True
            if in_method:
                if 'status_bar.showMessage' in line:
                    has_status_bar_update = True
                if line.strip() and line[0] != ' ' and 'def ' in line and '_on_model_loading' not in line:
                    break

        assert has_status_bar_update, \
            "_on_model_loading should update status_bar"

    def test_loading_shows_emoji_indicator(self):
        """Test that loading handler shows loading emoji."""
        main_window_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "ui" / "main_window.py"
        source = main_window_path.read_text(encoding='utf-8')

        # Should show loading emoji in status bar
        assert "🔄" in source, \
            "Should show loading emoji (🔄) during model load"

    def test_loading_sets_tts_indicator_loading(self):
        """Test that loading handler sets TTS indicator to loading state."""
        main_window_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "ui" / "main_window.py"
        source = main_window_path.read_text(encoding='utf-8')

        # Find _on_model_loading method and check for set_service_loading
        in_method = False
        has_loading_indicator = False

        for line in source.split('\n'):
            if 'def _on_model_loading' in line:
                in_method = True
            if in_method:
                if 'set_service_loading' in line and 'True' in line:
                    has_loading_indicator = True
                if line.strip() and line[0] != ' ' and 'def ' in line and '_on_model_loading' not in line:
                    break

        assert has_loading_indicator, \
            "_on_model_loading should set TTS indicator to loading state"


class TestModelReadyHandlerBehavior:
    """Tests for _on_model_ready handler behavior."""

    def test_ready_updates_status_bar(self):
        """Test that ready handler updates status bar."""
        main_window_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "ui" / "main_window.py"
        source = main_window_path.read_text(encoding='utf-8')

        # Find _on_model_ready method
        in_method = False
        has_status_bar_update = False

        for line in source.split('\n'):
            if 'def _on_model_ready' in line:
                in_method = True
            if in_method:
                if 'status_bar.showMessage' in line:
                    has_status_bar_update = True
                if line.strip() and line[0] != ' ' and 'def ' in line and '_on_model_ready' not in line:
                    break

        assert has_status_bar_update, \
            "_on_model_ready should update status_bar"

    def test_ready_shows_checkmark(self):
        """Test that ready handler shows checkmark."""
        main_window_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "ui" / "main_window.py"
        source = main_window_path.read_text(encoding='utf-8')

        # Should show checkmark in ready message
        assert "✓" in source, \
            "Should show checkmark (✓) when model is ready"

    def test_ready_clears_loading_state(self):
        """Test that ready handler clears TTS loading indicator."""
        main_window_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "ui" / "main_window.py"
        source = main_window_path.read_text(encoding='utf-8')

        # Find _on_model_ready method and check for set_service_loading(False)
        in_method = False
        clears_loading = False

        for line in source.split('\n'):
            if 'def _on_model_ready' in line:
                in_method = True
            if in_method:
                if 'set_service_loading' in line and 'False' in line:
                    clears_loading = True
                if line.strip() and line[0] != ' ' and 'def ' in line and '_on_model_ready' not in line:
                    break

        assert clears_loading, \
            "_on_model_ready should clear TTS loading indicator"

    def test_ready_updates_tooltip_with_model_name(self):
        """Test that ready handler updates tooltip with active model name."""
        main_window_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "ui" / "main_window.py"
        source = main_window_path.read_text(encoding='utf-8')

        # Find _on_model_ready method and check for tooltip update
        in_method = False
        has_tooltip_update = False

        for line in source.split('\n'):
            if 'def _on_model_ready' in line:
                in_method = True
            if in_method:
                if 'setToolTip' in line and 'Active model' in line:
                    has_tooltip_update = True
                if line.strip() and line[0] != ' ' and 'def ' in line and '_on_model_ready' not in line:
                    break

        assert has_tooltip_update, \
            "_on_model_ready should update tooltip with 'Active model: {name}'"


class TestCallbackConnectionInSetTtsService:
    """Tests for callback connection in set_tts_service method."""

    def test_callbacks_connected_when_service_provided(self):
        """Test that callbacks are connected when tts_service is provided."""
        main_window_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "ui" / "main_window.py"
        source = main_window_path.read_text(encoding='utf-8')

        # Find set_tts_service method
        lines = source.split('\n')
        in_method = False
        has_conditional_connect = False

        for i, line in enumerate(lines):
            if 'def set_tts_service' in line:
                in_method = True
            if in_method:
                if 'if tts_service:' in line or 'if tts_service' in line:
                    # Check next few lines for callback connections
                    for j in range(i, min(i+10, len(lines))):
                        if 'set_model_loading_callback' in lines[j]:
                            has_conditional_connect = True
                            break
                if line.strip() and line[0] != ' ' and 'def ' in line and 'set_tts_service' not in line:
                    break

        assert has_conditional_connect, \
            "Callbacks should be connected conditionally when tts_service is not None"

    def test_callbacks_reference_handler_methods(self):
        """Test that callbacks reference the correct handler methods."""
        main_window_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "ui" / "main_window.py"
        source = main_window_path.read_text(encoding='utf-8')

        # Verify the connection pattern
        assert "set_model_loading_callback(self._on_model_loading)" in source, \
            "Loading callback should reference _on_model_loading"
        assert "set_model_ready_callback(self._on_model_ready)" in source, \
            "Ready callback should reference _on_model_ready"


class TestLoggingInHandlers:
    """Tests for logging in handler methods."""

    def test_loading_handler_logs(self):
        """Test that _on_model_loading logs the event."""
        main_window_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "ui" / "main_window.py"
        source = main_window_path.read_text(encoding='utf-8')

        # Find _on_model_loading and check for logging
        in_method = False
        has_logging = False

        for line in source.split('\n'):
            if 'def _on_model_loading' in line:
                in_method = True
            if in_method:
                if 'self.logger' in line and ('info' in line or 'debug' in line):
                    has_logging = True
                if line.strip() and line[0] != ' ' and 'def ' in line and '_on_model_loading' not in line:
                    break

        assert has_logging, \
            "_on_model_loading should log the loading event"

    def test_ready_handler_logs(self):
        """Test that _on_model_ready logs the event."""
        main_window_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "ui" / "main_window.py"
        source = main_window_path.read_text(encoding='utf-8')

        # Find _on_model_ready and check for logging
        in_method = False
        has_logging = False

        for line in source.split('\n'):
            if 'def _on_model_ready' in line:
                in_method = True
            if in_method:
                if 'self.logger' in line and ('info' in line or 'debug' in line):
                    has_logging = True
                if line.strip() and line[0] != ' ' and 'def ' in line and '_on_model_ready' not in line:
                    break

        assert has_logging, \
            "_on_model_ready should log the ready event"


class TestStatusBarMessageTimeout:
    """Tests for status bar message timeout behavior."""

    def test_ready_message_has_timeout(self):
        """Test that ready message includes timeout parameter."""
        main_window_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "ui" / "main_window.py"
        source = main_window_path.read_text(encoding='utf-8')

        # Find _on_model_ready's showMessage call
        in_method = False

        for line in source.split('\n'):
            if 'def _on_model_ready' in line:
                in_method = True
            if in_method:
                if 'status_bar.showMessage' in line:
                    # Should have timeout parameter (3000)
                    assert '3000' in line, \
                        "Ready message should have 3000ms timeout"
                    break
                if line.strip() and line[0] != ' ' and 'def ' in line and '_on_model_ready' not in line:
                    break
