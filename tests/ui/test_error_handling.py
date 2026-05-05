"""
Tests for Story 7.6: Error Messaging & Stability

Tests the error handling features including:
- ErrorMessageFormatter (NFR10)
- CriticalErrorDialog component
- GlobalExceptionHandler (NFR6)
- Error logging
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestErrorCode:
    """Tests for ErrorCode enum."""

    def test_error_codes_are_strings(self):
        """All error codes should be string values."""
        from myvoice.utils.error_handler import ErrorCode

        for code in ErrorCode:
            assert isinstance(code.value, str)
            assert len(code.value) > 0

    def test_tts_error_codes_exist(self):
        """TTS-related error codes should exist."""
        from myvoice.utils.error_handler import ErrorCode

        assert hasattr(ErrorCode, 'TTS_GENERATION_FAILED')
        assert hasattr(ErrorCode, 'TTS_MODEL_LOAD_FAILED')
        assert hasattr(ErrorCode, 'TTS_MODEL_MEMORY_ERROR')

    def test_audio_error_codes_exist(self):
        """Audio-related error codes should exist."""
        from myvoice.utils.error_handler import ErrorCode

        assert hasattr(ErrorCode, 'AUDIO_DEVICE_UNAVAILABLE')
        assert hasattr(ErrorCode, 'AUDIO_PLAYBACK_FAILED')

    def test_voice_error_codes_exist(self):
        """Voice-related error codes should exist."""
        from myvoice.utils.error_handler import ErrorCode

        assert hasattr(ErrorCode, 'VOICE_FILE_CORRUPTED')
        assert hasattr(ErrorCode, 'VOICE_NOT_FOUND')


class TestErrorMessageTemplates:
    """Tests for error message templates."""

    def test_templates_have_what_and_action(self):
        """All templates should have 'what' and 'action' fields."""
        from myvoice.utils.error_handler import ERROR_MESSAGE_TEMPLATES

        for code, template in ERROR_MESSAGE_TEMPLATES.items():
            assert 'what' in template, f"Template {code} missing 'what'"
            assert 'action' in template, f"Template {code} missing 'action'"

    def test_templates_have_severity(self):
        """All templates should have a severity level."""
        from myvoice.utils.error_handler import ERROR_MESSAGE_TEMPLATES

        for code, template in ERROR_MESSAGE_TEMPLATES.items():
            assert 'severity' in template, f"Template {code} missing 'severity'"


class TestErrorMessageFormatter:
    """Tests for ErrorMessageFormatter class."""

    @pytest.fixture
    def formatter(self):
        """Create an ErrorMessageFormatter instance."""
        from myvoice.utils.error_handler import ErrorMessageFormatter
        return ErrorMessageFormatter()

    def test_format_error_returns_string(self, formatter):
        """format_error should return a string."""
        from myvoice.utils.error_handler import ErrorCode

        result = formatter.format_error(ErrorCode.TTS_GENERATION_FAILED.value)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_error_has_what_and_action_pattern(self, formatter):
        """Formatted error should follow '[What happened]. [What to do]' pattern."""
        from myvoice.utils.error_handler import ErrorCode

        result = formatter.format_error(ErrorCode.TTS_GENERATION_FAILED.value)
        # Should have a period separating the two parts
        assert '. ' in result

    def test_format_error_substitutes_variables(self, formatter):
        """format_error should substitute placeholder variables."""
        from myvoice.utils.error_handler import ErrorCode

        result = formatter.format_error(
            ErrorCode.AUDIO_DEVICE_UNAVAILABLE.value,
            device="Test Speakers"
        )
        assert "Test Speakers" in result

    def test_format_error_handles_unknown_code(self, formatter):
        """format_error should handle unknown error codes gracefully."""
        result = formatter.format_error("UNKNOWN_ERROR_CODE")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_severity_returns_severity(self, formatter):
        """get_severity should return an ErrorSeverity enum value."""
        from myvoice.utils.error_handler import ErrorCode
        from myvoice.models.error import ErrorSeverity

        severity = formatter.get_severity(ErrorCode.TTS_MODEL_LOAD_FAILED.value)
        assert severity == ErrorSeverity.CRITICAL

    def test_create_error_returns_myvoice_error(self, formatter):
        """create_error should return a MyVoiceError instance."""
        from myvoice.utils.error_handler import ErrorCode
        from myvoice.models.error import MyVoiceError

        error = formatter.create_error(ErrorCode.TTS_GENERATION_FAILED.value)
        assert isinstance(error, MyVoiceError)

    def test_create_error_includes_technical_details(self, formatter):
        """create_error should include technical details when provided."""
        from myvoice.utils.error_handler import ErrorCode

        error = formatter.create_error(
            ErrorCode.TTS_GENERATION_FAILED.value,
            technical_details="Stack trace here"
        )
        assert error.technical_details == "Stack trace here"


class TestGlobalExceptionHandler:
    """Tests for GlobalExceptionHandler class."""

    @pytest.fixture
    def handler(self):
        """Create a GlobalExceptionHandler instance."""
        from myvoice.utils.error_handler import GlobalExceptionHandler
        return GlobalExceptionHandler()

    def test_install_sets_excepthook(self, handler):
        """install() should set sys.excepthook."""
        original = sys.excepthook

        handler.install()
        assert sys.excepthook != original

        handler.uninstall()
        assert sys.excepthook == original

    def test_uninstall_restores_original(self, handler):
        """uninstall() should restore the original excepthook."""
        original = sys.excepthook

        handler.install()
        handler.uninstall()

        assert sys.excepthook == original

    def test_set_error_callback(self, handler):
        """set_error_callback should store the callback."""
        callback = Mock()
        handler.set_error_callback(callback)
        assert handler._error_callback == callback

    def test_set_critical_callback(self, handler):
        """set_critical_callback should store the callback."""
        callback = Mock()
        handler.set_critical_callback(callback)
        assert handler._critical_callback == callback

    def test_handles_keyboard_interrupt_passthrough(self, handler):
        """KeyboardInterrupt should be passed through to original handler."""
        original_excepthook = Mock()
        handler._original_excepthook = original_excepthook

        # Simulate KeyboardInterrupt
        handler._handle_exception(KeyboardInterrupt, KeyboardInterrupt(), None)

        original_excepthook.assert_called_once()


class TestCriticalErrorDialog:
    """Tests for CriticalErrorDialog component."""

    @pytest.fixture
    def dialog(self, qapp):
        """Create a CriticalErrorDialog instance."""
        from myvoice.ui.components.critical_error_dialog import CriticalErrorDialog
        dialog = CriticalErrorDialog(
            title="Test Error",
            message="This is a test error message.",
            details="Technical details here.",
            allow_continue=True
        )
        yield dialog
        dialog.close()

    def test_dialog_creates_successfully(self, dialog):
        """CriticalErrorDialog should create without errors."""
        assert dialog is not None

    def test_dialog_is_modal(self, dialog):
        """CriticalErrorDialog should be modal."""
        assert dialog.isModal()

    def test_dialog_has_title(self, dialog):
        """Dialog should have the specified title."""
        assert dialog.windowTitle() == "Test Error"

    def test_dialog_with_details_has_copy_button(self, dialog, qapp):
        """Dialog with details should have a copy button."""
        # Find the copy button by text
        from PyQt6.QtWidgets import QPushButton
        copy_buttons = [
            child for child in dialog.findChildren(QPushButton)
            if "Copy" in child.text()
        ]
        assert len(copy_buttons) > 0

    def test_dialog_without_details_no_copy_button(self, qapp):
        """Dialog without details should not have a copy button."""
        from myvoice.ui.components.critical_error_dialog import CriticalErrorDialog
        from PyQt6.QtWidgets import QPushButton

        dialog = CriticalErrorDialog(
            title="Test Error",
            message="This is a test.",
            details=None,  # No details
            allow_continue=True
        )

        copy_buttons = [
            child for child in dialog.findChildren(QPushButton)
            if "Copy" in child.text()
        ]
        assert len(copy_buttons) == 0
        dialog.close()

    def test_dialog_allow_continue_has_continue_button(self, dialog):
        """Dialog with allow_continue=True should have Continue button."""
        from PyQt6.QtWidgets import QPushButton

        continue_buttons = [
            child for child in dialog.findChildren(QPushButton)
            if child.text() == "Continue"
        ]
        assert len(continue_buttons) == 1

    def test_dialog_disallow_continue_no_continue_button(self, qapp):
        """Dialog with allow_continue=False should not have Continue button."""
        from myvoice.ui.components.critical_error_dialog import CriticalErrorDialog
        from PyQt6.QtWidgets import QPushButton

        dialog = CriticalErrorDialog(
            title="Fatal Error",
            message="Cannot continue.",
            details=None,
            allow_continue=False
        )

        continue_buttons = [
            child for child in dialog.findChildren(QPushButton)
            if child.text() == "Continue"
        ]
        assert len(continue_buttons) == 0
        dialog.close()

    def test_dialog_has_exit_button(self, dialog):
        """Dialog should always have an Exit button."""
        from PyQt6.QtWidgets import QPushButton

        exit_buttons = [
            child for child in dialog.findChildren(QPushButton)
            if child.text() == "Exit"
        ]
        assert len(exit_buttons) == 1


class TestErrorNotificationWidget:
    """Tests for ErrorNotificationWidget component."""

    @pytest.fixture
    def notification(self, qapp):
        """Create an ErrorNotificationWidget instance."""
        from myvoice.ui.components.critical_error_dialog import ErrorNotificationWidget
        widget = ErrorNotificationWidget(message="Test notification")
        yield widget
        widget.deleteLater()

    def test_notification_creates_successfully(self, notification):
        """ErrorNotificationWidget should create without errors."""
        assert notification is not None

    def test_notification_has_dismiss_button(self, notification):
        """Notification should have a dismiss button."""
        from PyQt6.QtWidgets import QPushButton

        dismiss_buttons = [
            child for child in notification.findChildren(QPushButton)
            if "x" in child.text().lower() or child.toolTip() == "Dismiss"
        ]
        assert len(dismiss_buttons) > 0


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_error_formatter_returns_singleton(self):
        """get_error_formatter should return the same instance."""
        from myvoice.utils.error_handler import get_error_formatter

        formatter1 = get_error_formatter()
        formatter2 = get_error_formatter()
        assert formatter1 is formatter2

    def test_get_exception_handler_returns_singleton(self):
        """get_exception_handler should return the same instance."""
        from myvoice.utils.error_handler import get_exception_handler

        handler1 = get_exception_handler()
        handler2 = get_exception_handler()
        assert handler1 is handler2

    def test_format_error_function(self):
        """format_error convenience function should work."""
        from myvoice.utils.error_handler import format_error, ErrorCode

        result = format_error(ErrorCode.TTS_GENERATION_FAILED.value)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_create_error_function(self):
        """create_error convenience function should work."""
        from myvoice.utils.error_handler import create_error, ErrorCode
        from myvoice.models.error import MyVoiceError

        error = create_error(ErrorCode.TTS_GENERATION_FAILED.value)
        assert isinstance(error, MyVoiceError)
