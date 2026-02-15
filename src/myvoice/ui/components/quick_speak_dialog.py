"""
Quick Speak Selection Dialog

This module implements a popup dialog for selecting Quick Speak entries
to quickly populate the text field and trigger TTS generation.
"""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QLabel, QLineEdit, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from myvoice.models.quick_speak_entry import QuickSpeakEntry
from myvoice.services.quick_speak_service import QuickSpeakService


class QuickSpeakDialog(QDialog):
    """
    Dialog for selecting Quick Speak entries.

    Provides a simple list-based interface for selecting preset text
    entries that will be used for TTS generation.

    Story 5.1: Quick Speak Phrase Configuration
    - Shows empty state with "Configure in Settings" button when no entries
    - Search/filter functionality
    - Keyboard navigation

    Signals:
        entry_selected: Emitted when user selects an entry (str: text)
        open_settings_requested: Emitted when user wants to configure phrases
    """

    entry_selected = pyqtSignal(str)  # text content
    open_settings_requested = pyqtSignal()  # Request to open settings

    def __init__(self, quick_speak_service: QuickSpeakService, parent: Optional[QWidget] = None):
        """
        Initialize Quick Speak selection dialog.

        Args:
            quick_speak_service: QuickSpeakService instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.quick_speak_service = quick_speak_service

        self._setup_dialog()
        self._create_ui()
        self._load_entries()

        self.logger.debug("QuickSpeakDialog initialized")

    def _setup_dialog(self):
        """Configure dialog properties."""
        self.setWindowTitle("Quick Speak")
        self.setModal(True)
        self.resize(400, 300)

        # Set window flags
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.WindowTitleHint |
            Qt.WindowType.WindowCloseButtonHint
        )

    def _create_ui(self):
        """Create the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Title label
        title_label = QLabel("Select a Quick Speak entry:")
        title_font = QFont()
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # Search/filter box
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search entries...")
        self.search_edit.textChanged.connect(self._on_search_changed)
        layout.addWidget(self.search_edit)

        # Configure button (shown when no entries - Story 5.1 empty state)
        self.configure_button = QPushButton("Configure Phrases in Settings...")
        self.configure_button.clicked.connect(self._on_configure_clicked)
        self.configure_button.setVisible(False)  # Hidden by default
        layout.addWidget(self.configure_button)

        # List widget for entries
        self.entries_list = QListWidget()
        self.entries_list.setAlternatingRowColors(True)
        self.entries_list.itemDoubleClicked.connect(self._on_entry_double_clicked)
        self.entries_list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.entries_list)

        # Button row with tip
        button_layout = QHBoxLayout()

        # Tip label on the left
        tip_label = QLabel("Choice appended if an apostrophe ( ' ) is the last character in text box")
        tip_label.setProperty("class", "caption")
        tip_label.setWordWrap(True)
        tip_label.setStyleSheet("color: #94a3b8; font-size: 8pt;")
        button_layout.addWidget(tip_label, 1)  # Stretch factor 1 to take available space

        button_layout.addStretch()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        self.select_button = QPushButton("Select")
        self.select_button.setDefault(True)
        self.select_button.setEnabled(False)
        self.select_button.clicked.connect(self._on_select_clicked)

        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.select_button)

        layout.addLayout(button_layout)

    def _load_entries(self):
        """Load entries from service and populate list."""
        try:
            # Reload entries from current profile's CSV file
            self.quick_speak_service.load_entries()
            entries = self.quick_speak_service.get_entries()
            self._populate_list(entries)
            self.logger.debug(f"Loaded {len(entries)} Quick Speak entries from profile: {self.quick_speak_service.get_current_profile()}")

            # Show/hide empty state elements
            self._update_empty_state(len(entries) == 0)

        except Exception as e:
            self.logger.error(f"Error loading Quick Speak entries: {e}")
            self._update_empty_state(True)

    def _update_empty_state(self, is_empty: bool):
        """
        Update the empty state display.

        Story 5.1: Show "No phrases configured" with link to settings.

        Args:
            is_empty: True if no entries are available
        """
        if is_empty:
            # Show empty state message
            item = QListWidgetItem("No phrases configured")
            item.setFlags(Qt.ItemFlag.NoItemFlags)  # Make it non-selectable
            self.entries_list.addItem(item)

            hint_item = QListWidgetItem("Click 'Configure...' to add phrases in Settings")
            hint_item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.entries_list.addItem(hint_item)

        # Show/hide the configure button based on empty state
        self.configure_button.setVisible(is_empty)
        self.search_edit.setVisible(not is_empty)

    def _on_configure_clicked(self):
        """Handle configure button click - request to open settings."""
        self.logger.info("User requested to open Settings for Quick Speak configuration")
        self.open_settings_requested.emit()
        self.reject()  # Close this dialog

    def _populate_list(self, entries: list):
        """
        Populate list with entries.

        Args:
            entries: List of QuickSpeakEntry objects
        """
        self.entries_list.clear()

        for entry in entries:
            item = QListWidgetItem(entry.get_display_text())
            item.setData(Qt.ItemDataRole.UserRole, entry)  # Store entry object
            item.setToolTip(entry.text)  # Show full text on hover
            self.entries_list.addItem(item)

    def _on_search_changed(self, text: str):
        """
        Handle search text change to filter entries.

        Args:
            text: Search text
        """
        search_text = text.lower().strip()

        if not search_text:
            # Show all entries (reload to ensure we have current profile)
            self.quick_speak_service.load_entries()
            entries = self.quick_speak_service.get_entries()
            self._populate_list(entries)
            return

        # Filter entries (reload to ensure we have current profile)
        self.quick_speak_service.load_entries()
        all_entries = self.quick_speak_service.get_entries()
        filtered_entries = [
            entry for entry in all_entries
            if search_text in entry.text.lower() or
               (entry.label and search_text in entry.label.lower())
        ]

        self._populate_list(filtered_entries)

        if not filtered_entries:
            # Show no results message
            item = QListWidgetItem("No matching entries found")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.entries_list.addItem(item)

    def _on_selection_changed(self):
        """Handle list selection change."""
        has_selection = bool(self.entries_list.selectedItems())

        # Check if selected item is an actual entry (not a message item)
        if has_selection:
            selected_item = self.entries_list.currentItem()
            entry = selected_item.data(Qt.ItemDataRole.UserRole)
            has_selection = entry is not None

        self.select_button.setEnabled(has_selection)

    def _on_entry_double_clicked(self, item: QListWidgetItem):
        """
        Handle entry double-click.

        Args:
            item: Clicked list item
        """
        entry = item.data(Qt.ItemDataRole.UserRole)
        if entry:
            self._emit_selection(entry)

    def _on_select_clicked(self):
        """Handle select button click."""
        selected_item = self.entries_list.currentItem()
        if not selected_item:
            return

        entry = selected_item.data(Qt.ItemDataRole.UserRole)
        if entry:
            self._emit_selection(entry)

    def _emit_selection(self, entry: QuickSpeakEntry):
        """
        Emit entry selection and close dialog.

        Args:
            entry: Selected entry
        """
        self.logger.info(f"Quick Speak entry selected: {entry.get_display_text()}")
        self.entry_selected.emit(entry.text)
        self.accept()

    def refresh_entries(self):
        """Refresh entries from service."""
        self._load_entries()