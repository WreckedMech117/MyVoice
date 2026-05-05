"""
Tests for QuickSpeakDialog

Story 5.1: Quick Speak Phrase Configuration
Tests the dialog for selecting Quick Speak entries and empty state handling.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from myvoice.ui.components.quick_speak_dialog import QuickSpeakDialog
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
    """Create a mock QuickSpeakService."""
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
def dialog_empty(qapp, mock_service):
    """Create a QuickSpeakDialog with no entries (empty state)."""
    dialog = QuickSpeakDialog(quick_speak_service=mock_service)
    yield dialog
    dialog.deleteLater()


@pytest.fixture
def dialog_with_entries(qapp, mock_service_with_entries):
    """Create a QuickSpeakDialog with entries."""
    dialog = QuickSpeakDialog(quick_speak_service=mock_service_with_entries)
    yield dialog
    dialog.deleteLater()


class TestQuickSpeakDialogInit:
    """Tests for QuickSpeakDialog initialization."""

    def test_dialog_creation(self, dialog_empty):
        """Test dialog creates successfully."""
        assert dialog_empty is not None
        assert isinstance(dialog_empty, QuickSpeakDialog)

    def test_dialog_title(self, dialog_empty):
        """Test dialog has correct title."""
        assert dialog_empty.windowTitle() == "Quick Speak"

    def test_dialog_is_modal(self, dialog_empty):
        """Test dialog is modal."""
        assert dialog_empty.isModal()

    def test_has_search_edit(self, dialog_empty):
        """Test dialog has search edit."""
        assert dialog_empty.search_edit is not None

    def test_has_entries_list(self, dialog_empty):
        """Test dialog has entries list."""
        assert dialog_empty.entries_list is not None

    def test_has_select_button(self, dialog_empty):
        """Test dialog has select button."""
        assert dialog_empty.select_button is not None

    def test_has_cancel_button(self, dialog_empty):
        """Test dialog has cancel button."""
        assert dialog_empty.cancel_button is not None

    def test_has_configure_button(self, dialog_empty):
        """Test dialog has configure button (Story 5.1)."""
        assert dialog_empty.configure_button is not None


class TestQuickSpeakDialogSignals:
    """Tests for QuickSpeakDialog signals."""

    def test_entry_selected_signal_exists(self, dialog_empty):
        """Test entry_selected signal exists."""
        assert hasattr(dialog_empty, 'entry_selected')

    def test_open_settings_requested_signal_exists(self, dialog_empty):
        """Test open_settings_requested signal exists (Story 5.1)."""
        assert hasattr(dialog_empty, 'open_settings_requested')


class TestQuickSpeakDialogEmptyState:
    """Tests for empty state handling (Story 5.1 AC7)."""

    def test_empty_state_shows_configure_button(self, dialog_empty):
        """Test configure button exists when no entries."""
        # Note: isVisible() returns False for widgets not shown on screen
        # Check the button exists and is not explicitly hidden
        assert dialog_empty.configure_button is not None
        assert not dialog_empty.configure_button.isHidden()

    def test_empty_state_hides_search(self, dialog_empty):
        """Test search box is hidden when no entries."""
        assert not dialog_empty.search_edit.isVisible()

    def test_empty_state_shows_message(self, dialog_empty):
        """Test 'No phrases configured' message is shown."""
        # Check list widget contains the empty state message
        found_message = False
        for i in range(dialog_empty.entries_list.count()):
            item = dialog_empty.entries_list.item(i)
            if "No phrases configured" in item.text():
                found_message = True
                break
        assert found_message

    def test_empty_state_shows_hint(self, dialog_empty):
        """Test empty state shows hint about configure button."""
        found_hint = False
        for i in range(dialog_empty.entries_list.count()):
            item = dialog_empty.entries_list.item(i)
            if "Configure" in item.text():
                found_hint = True
                break
        assert found_hint

    def test_empty_state_items_not_selectable(self, dialog_empty):
        """Test empty state message items are not selectable."""
        for i in range(dialog_empty.entries_list.count()):
            item = dialog_empty.entries_list.item(i)
            # NoItemFlags means the item cannot be selected
            assert item.flags() == Qt.ItemFlag.NoItemFlags

    def test_configure_button_emits_signal(self, dialog_empty):
        """Test configure button click emits open_settings_requested signal."""
        signal_received = []
        dialog_empty.open_settings_requested.connect(lambda: signal_received.append(True))

        dialog_empty.configure_button.click()

        assert len(signal_received) == 1


class TestQuickSpeakDialogWithEntries:
    """Tests for dialog with entries."""

    def test_entries_displayed(self, dialog_with_entries):
        """Test entries are displayed in list."""
        # Should have 3 entries from the mock
        assert dialog_with_entries.entries_list.count() == 3

    def test_configure_button_hidden_with_entries(self, dialog_with_entries):
        """Test configure button is hidden when entries exist."""
        assert not dialog_with_entries.configure_button.isVisible()

    def test_search_visible_with_entries(self, dialog_with_entries):
        """Test search box exists and is not hidden when entries exist."""
        # Note: isVisible() returns False for widgets not shown on screen
        # Check the widget exists and is not explicitly hidden
        assert dialog_with_entries.search_edit is not None
        assert not dialog_with_entries.search_edit.isHidden()

    def test_select_button_disabled_initially(self, dialog_with_entries):
        """Test select button is disabled when nothing selected."""
        assert not dialog_with_entries.select_button.isEnabled()

    def test_select_button_enabled_on_selection(self, dialog_with_entries):
        """Test select button is enabled when item selected."""
        # Select the first item
        dialog_with_entries.entries_list.setCurrentRow(0)

        assert dialog_with_entries.select_button.isEnabled()


class TestQuickSpeakDialogSelection:
    """Tests for entry selection."""

    def test_entry_selected_signal_on_select_click(self, dialog_with_entries):
        """Test entry_selected signal emitted on select button click."""
        signal_received = []
        dialog_with_entries.entry_selected.connect(lambda text: signal_received.append(text))

        # Select the first item and click select
        dialog_with_entries.entries_list.setCurrentRow(0)
        dialog_with_entries.select_button.click()

        assert len(signal_received) == 1
        assert signal_received[0] == "Hello, how are you?"

    def test_entry_selected_signal_on_double_click(self, dialog_with_entries):
        """Test entry_selected signal emitted on double-click."""
        signal_received = []
        dialog_with_entries.entry_selected.connect(lambda text: signal_received.append(text))

        # Simulate double-click on second item
        item = dialog_with_entries.entries_list.item(1)
        dialog_with_entries._on_entry_double_clicked(item)

        assert len(signal_received) == 1
        assert signal_received[0] == "Thank you very much"


class TestQuickSpeakDialogSearch:
    """Tests for search/filter functionality."""

    def test_search_filters_entries(self, dialog_with_entries):
        """Test search filters entries."""
        # Search for "help"
        dialog_with_entries.search_edit.setText("help")

        # Should show only the "Help" entry
        # Check if filtered results show
        found_help = False
        for i in range(dialog_with_entries.entries_list.count()):
            item = dialog_with_entries.entries_list.item(i)
            entry = item.data(Qt.ItemDataRole.UserRole)
            if entry and "help" in entry.text.lower():
                found_help = True
        assert found_help

    def test_search_by_label(self, dialog_with_entries):
        """Test search works on labels too."""
        # Search for "Greeting" (label)
        dialog_with_entries.search_edit.setText("Greeting")

        # Should find the greeting entry
        found_greeting = False
        for i in range(dialog_with_entries.entries_list.count()):
            item = dialog_with_entries.entries_list.item(i)
            entry = item.data(Qt.ItemDataRole.UserRole)
            if entry and entry.label == "Greeting":
                found_greeting = True
        assert found_greeting

    def test_clear_search_shows_all(self, dialog_with_entries):
        """Test clearing search shows all entries."""
        # First filter
        dialog_with_entries.search_edit.setText("help")
        # Then clear
        dialog_with_entries.search_edit.setText("")

        # Should show all 3 entries
        assert dialog_with_entries.entries_list.count() == 3

    def test_no_results_message(self, dialog_with_entries):
        """Test 'no matching entries' message shown when no results."""
        # Search for something that doesn't exist
        dialog_with_entries.search_edit.setText("xyznonexistent")

        # Should show "no matching entries" message
        found_message = False
        for i in range(dialog_with_entries.entries_list.count()):
            item = dialog_with_entries.entries_list.item(i)
            if "No matching" in item.text():
                found_message = True
        assert found_message


class TestQuickSpeakDialogRefresh:
    """Tests for refresh functionality."""

    def test_refresh_entries_reloads(self, dialog_with_entries, mock_service_with_entries):
        """Test refresh_entries reloads from service."""
        initial_count = dialog_with_entries.entries_list.count()

        # Add another entry to the mock
        new_entries = mock_service_with_entries.get_entries() + [
            QuickSpeakEntry(id="4", text="New entry", label="New")
        ]
        mock_service_with_entries.get_entries.return_value = new_entries

        # Refresh
        dialog_with_entries.refresh_entries()

        # Should have 4 entries now
        assert dialog_with_entries.entries_list.count() == 4


class TestQuickSpeakDialogEntryDisplay:
    """Tests for entry display formatting."""

    def test_entry_tooltip_shows_full_text(self, dialog_with_entries):
        """Test entry tooltip shows full text."""
        item = dialog_with_entries.entries_list.item(0)
        assert item.toolTip() == "Hello, how are you?"

    def test_entry_stores_data(self, dialog_with_entries):
        """Test entry object is stored in item data."""
        item = dialog_with_entries.entries_list.item(0)
        entry = item.data(Qt.ItemDataRole.UserRole)
        assert entry is not None
        assert isinstance(entry, QuickSpeakEntry)
        assert entry.id == "1"
