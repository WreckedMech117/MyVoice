"""
Tests for QuickSpeakMenu

Story 5.2: Quick Speak Menu & Triggering
Tests the popup menu for rapid Quick Speak phrase selection.
"""

import pytest
from unittest.mock import Mock, MagicMock

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QPushButton
from PyQt6.QtCore import Qt

from myvoice.ui.components.quick_speak_menu import QuickSpeakMenu
from myvoice.models.quick_speak_entry import QuickSpeakEntry


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def mock_service():
    """Create a mock QuickSpeakService with no entries."""
    service = MagicMock()
    service.get_entries.return_value = []
    service.get_current_profile.return_value = "default"
    service.load_entries.return_value = None
    return service


@pytest.fixture
def mock_service_with_entries():
    """Create a mock QuickSpeakService with entries."""
    service = MagicMock()
    entries = [
        QuickSpeakEntry(id="1", text="Hello, how are you?", label="Greeting"),
        QuickSpeakEntry(id="2", text="Thank you very much", label="Thanks"),
        QuickSpeakEntry(id="3", text="I need help please", label="Help"),
    ]
    service.get_entries.return_value = entries
    service.get_current_profile.return_value = "default"
    service.load_entries.return_value = None
    return service


@pytest.fixture
def mock_service_long_entries():
    """Create a mock QuickSpeakService with long entries for truncation testing."""
    service = MagicMock()
    entries = [
        QuickSpeakEntry(
            id="1",
            text="This is a very long phrase that should definitely be truncated when displayed in the menu",
            label="Long"
        ),
        QuickSpeakEntry(id="2", text="Short", label="Short"),
    ]
    service.get_entries.return_value = entries
    service.get_current_profile.return_value = "default"
    service.load_entries.return_value = None
    return service


@pytest.fixture
def menu_empty(qapp, mock_service):
    """Create a QuickSpeakMenu with no entries."""
    menu = QuickSpeakMenu(quick_speak_service=mock_service)
    yield menu
    menu.deleteLater()


@pytest.fixture
def menu_with_entries(qapp, mock_service_with_entries):
    """Create a QuickSpeakMenu with entries."""
    menu = QuickSpeakMenu(quick_speak_service=mock_service_with_entries)
    yield menu
    menu.deleteLater()


@pytest.fixture
def menu_long_entries(qapp, mock_service_long_entries):
    """Create a QuickSpeakMenu with long entries."""
    menu = QuickSpeakMenu(quick_speak_service=mock_service_long_entries)
    yield menu
    menu.deleteLater()


class TestQuickSpeakMenuInit:
    """Tests for QuickSpeakMenu initialization."""

    def test_menu_creation(self, menu_empty):
        """Test menu creates successfully."""
        assert menu_empty is not None
        assert isinstance(menu_empty, QuickSpeakMenu)

    def test_menu_title(self, menu_empty):
        """Test menu has correct title."""
        assert menu_empty.title() == "Quick Speak"


class TestQuickSpeakMenuSignals:
    """Tests for QuickSpeakMenu signals."""

    def test_phrase_selected_signal_exists(self, menu_empty):
        """Test phrase_selected signal exists."""
        assert hasattr(menu_empty, 'phrase_selected')

    def test_open_settings_requested_signal_exists(self, menu_empty):
        """Test open_settings_requested signal exists."""
        assert hasattr(menu_empty, 'open_settings_requested')


class TestQuickSpeakMenuEmptyState:
    """Tests for empty state handling."""

    def test_empty_menu_has_configure_option(self, menu_empty):
        """Test empty menu has configure option."""
        menu_empty.refresh_entries()

        # Find actions
        actions = menu_empty.actions()
        action_texts = [a.text() for a in actions]

        assert "No phrases configured" in action_texts
        assert any("Configure" in text for text in action_texts)

    def test_empty_state_message_disabled(self, menu_empty):
        """Test empty state message is disabled (not selectable)."""
        menu_empty.refresh_entries()

        # Find the empty state action
        for action in menu_empty.actions():
            if "No phrases configured" in action.text():
                assert not action.isEnabled()
                break


class TestQuickSpeakMenuWithEntries:
    """Tests for menu with entries."""

    def test_entries_displayed(self, menu_with_entries):
        """Test entries are displayed in menu with their labels."""
        menu_with_entries.refresh_entries()

        actions = menu_with_entries.actions()
        action_texts = [a.text() for a in actions if a.isEnabled()]

        # Menu now displays labels instead of full text
        # Should have labels: "Greeting", "Thanks", "Help" + Configure option
        assert "Greeting" in action_texts

    def test_configure_option_present(self, menu_with_entries):
        """Test configure option is present at bottom."""
        menu_with_entries.refresh_entries()

        actions = menu_with_entries.actions()
        action_texts = [a.text() for a in actions]

        assert any("Configure" in text for text in action_texts)

    def test_entry_count(self, menu_with_entries):
        """Test correct number of entry actions created."""
        menu_with_entries.refresh_entries()

        # Count enabled actions (entries) excluding separator and configure
        enabled_entry_actions = [
            a for a in menu_with_entries.actions()
            if a.isEnabled() and "Configure" not in a.text() and not a.isSeparator()
        ]

        assert len(enabled_entry_actions) == 3


class TestQuickSpeakMenuTruncation:
    """Tests for text truncation."""

    def test_long_text_truncated(self, menu_long_entries):
        """Test entries with labels display the label, not truncated text."""
        menu_long_entries.refresh_entries()

        actions = menu_long_entries.actions()

        # Find the long entry action by its label
        long_action = None
        for action in actions:
            if action.text() == "Long":
                long_action = action
                break

        assert long_action is not None
        # When a label is provided, the menu displays the label (not truncated text)
        assert long_action.text() == "Long"
        # The full text is stored in the action's data
        assert long_action.data() is not None
        assert len(long_action.data()) > 50  # Full text is long

    def test_short_text_not_truncated(self, menu_long_entries):
        """Test short text is not truncated."""
        menu_long_entries.refresh_entries()

        actions = menu_long_entries.actions()

        # Find the short entry action
        for action in actions:
            if action.text() == "Short":
                assert not action.text().endswith("...")
                break

    def test_tooltip_contains_full_text(self, menu_long_entries):
        """Test tooltip contains full text for truncated entries."""
        menu_long_entries.refresh_entries()

        actions = menu_long_entries.actions()

        # Find the long entry action
        for action in actions:
            if action.data() and "very long phrase" in action.data():
                # Tooltip should have full text
                assert action.toolTip() == action.data()
                break


class TestQuickSpeakMenuSelection:
    """Tests for phrase selection."""

    def test_phrase_selected_signal_emitted(self, menu_with_entries):
        """Test phrase_selected signal is emitted on selection."""
        menu_with_entries.refresh_entries()

        signal_received = []
        menu_with_entries.phrase_selected.connect(lambda text: signal_received.append(text))

        # Trigger first entry action
        for action in menu_with_entries.actions():
            if action.data() and "Hello" in action.data():
                action.trigger()
                break

        assert len(signal_received) == 1
        assert signal_received[0] == "Hello, how are you?"

    def test_configure_emits_settings_signal(self, menu_with_entries):
        """Test configure option emits open_settings_requested signal."""
        menu_with_entries.refresh_entries()

        signal_received = []
        menu_with_entries.open_settings_requested.connect(lambda: signal_received.append(True))

        # Find and trigger configure action
        for action in menu_with_entries.actions():
            if "Configure" in action.text():
                action.trigger()
                break

        assert len(signal_received) == 1


class TestQuickSpeakMenuRefresh:
    """Tests for menu refresh functionality."""

    def test_refresh_clears_and_repopulates(self, menu_with_entries, mock_service_with_entries):
        """Test refresh clears old entries and repopulates."""
        menu_with_entries.refresh_entries()
        initial_count = len(menu_with_entries.actions())

        # Add another entry to mock
        new_entries = mock_service_with_entries.get_entries() + [
            QuickSpeakEntry(id="4", text="New entry", label="New")
        ]
        mock_service_with_entries.get_entries.return_value = new_entries

        # Refresh
        menu_with_entries.refresh_entries()

        # Should have one more action
        assert len(menu_with_entries.actions()) == initial_count + 1

    def test_refresh_reloads_from_service(self, menu_with_entries, mock_service_with_entries):
        """Test refresh calls load_entries on service."""
        menu_with_entries.refresh_entries()

        # Verify load_entries was called
        mock_service_with_entries.load_entries.assert_called()


class TestQuickSpeakMenuTruncateMethod:
    """Tests for _truncate_text method."""

    def test_truncate_short_text(self, menu_empty):
        """Test short text is not modified."""
        result = menu_empty._truncate_text("Short text")
        assert result == "Short text"

    def test_truncate_exact_length(self, menu_empty):
        """Test text at exact max length is not modified."""
        text = "X" * QuickSpeakMenu.MAX_DISPLAY_CHARS
        result = menu_empty._truncate_text(text)
        assert result == text
        assert not result.endswith("...")

    def test_truncate_long_text(self, menu_empty):
        """Test long text is truncated with ellipsis."""
        text = "X" * (QuickSpeakMenu.MAX_DISPLAY_CHARS + 10)
        result = menu_empty._truncate_text(text)
        assert len(result) == QuickSpeakMenu.MAX_DISPLAY_CHARS
        assert result.endswith("...")


class TestQuickSpeakMenuErrorHandling:
    """Tests for error handling."""

    def test_service_exception_shows_error_state(self, qapp):
        """Test exception from service shows error state."""
        mock_service = MagicMock()
        mock_service.load_entries.side_effect = Exception("Test error")

        menu = QuickSpeakMenu(quick_speak_service=mock_service)
        menu.refresh_entries()

        actions = menu.actions()
        action_texts = [a.text() for a in actions]

        assert any("Error" in text for text in action_texts)

        menu.deleteLater()
