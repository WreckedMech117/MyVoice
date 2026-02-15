"""
Quick Speak Popup Menu

Story 5.2: Quick Speak Menu & Triggering
This module implements a popup menu for rapid Quick Speak phrase selection
with minimal interactions (3 max: open, select, auto-generate).

FR34: Quick Speak button with popup menu
"""

import logging
from typing import Optional, List

from PyQt6.QtWidgets import QMenu, QWidget, QWidgetAction, QLabel
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QAction, QFont, QKeyEvent

from myvoice.models.quick_speak_entry import QuickSpeakEntry
from myvoice.services.quick_speak_service import QuickSpeakService


class QuickSpeakMenu(QMenu):
    """
    Popup menu for Quick Speak phrase selection.

    Story 5.2: Quick Speak Menu & Triggering
    - Shows all configured phrases in a popup menu
    - Single click triggers immediate TTS generation
    - Keyboard navigation with arrow keys + Enter
    - Escape closes menu
    - Long phrases truncated with tooltip for full text

    Signals:
        phrase_selected: Emitted when user selects a phrase (str: text)
        open_settings_requested: Emitted when user clicks configure option
    """

    # Maximum characters to display before truncation
    MAX_DISPLAY_CHARS = 50

    phrase_selected = pyqtSignal(str)  # phrase text
    open_settings_requested = pyqtSignal()

    def __init__(
        self,
        quick_speak_service: QuickSpeakService,
        parent: Optional[QWidget] = None
    ):
        """
        Initialize Quick Speak menu.

        Args:
            quick_speak_service: QuickSpeakService instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.quick_speak_service = quick_speak_service

        # Track entries for keyboard navigation
        self._entry_actions: List[QAction] = []

        self._setup_menu()
        self.logger.debug("QuickSpeakMenu initialized")

    def _setup_menu(self):
        """Configure menu properties."""
        self.setTitle("Quick Speak")

        # Enable keyboard navigation
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Style the menu for better visibility
        self.setStyleSheet("""
            QMenu {
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px 6px 20px;
                min-width: 200px;
            }
            QMenu::item:selected {
                background-color: #0d6efd;
                color: white;
            }
            QMenu::separator {
                height: 1px;
                background: #dee2e6;
                margin: 4px 8px;
            }
        """)

    def refresh_entries(self):
        """Refresh menu entries from service."""
        self.clear()
        self._entry_actions.clear()

        try:
            # Reload from current profile
            self.quick_speak_service.load_entries()
            entries = self.quick_speak_service.get_entries()

            if not entries:
                self._add_empty_state()
            else:
                self._populate_entries(entries)

            self.logger.debug(f"Refreshed menu with {len(entries)} entries")

        except Exception as e:
            self.logger.error(f"Error refreshing Quick Speak menu: {e}")
            self._add_error_state()

    def _populate_entries(self, entries: List[QuickSpeakEntry]):
        """
        Populate menu with phrase entries.

        Args:
            entries: List of QuickSpeakEntry objects
        """
        for entry in entries:
            action = self._create_entry_action(entry)
            self.addAction(action)
            self._entry_actions.append(action)

        # Add separator and configure option at bottom
        self.addSeparator()
        configure_action = QAction("Configure Phrases...", self)
        configure_action.triggered.connect(self._on_configure_clicked)
        self.addAction(configure_action)

    def _create_entry_action(self, entry: QuickSpeakEntry) -> QAction:
        """
        Create a menu action for a Quick Speak entry.

        Args:
            entry: QuickSpeakEntry object

        Returns:
            QAction configured for the entry
        """
        # Get display text (truncate if needed)
        display_text = self._truncate_text(entry.get_display_text())
        full_text = entry.text

        action = QAction(display_text, self)

        # Set tooltip with full text if truncated
        if len(entry.text) > self.MAX_DISPLAY_CHARS:
            action.setToolTip(full_text)

        # Store full text in action data
        action.setData(full_text)

        # Connect to selection handler
        action.triggered.connect(lambda checked, text=full_text: self._on_phrase_selected(text))

        return action

    def _truncate_text(self, text: str) -> str:
        """
        Truncate text if it exceeds maximum display characters.

        Args:
            text: Text to potentially truncate

        Returns:
            Truncated text with ellipsis if needed
        """
        if len(text) <= self.MAX_DISPLAY_CHARS:
            return text
        return text[:self.MAX_DISPLAY_CHARS - 3] + "..."

    def _add_empty_state(self):
        """Add empty state message when no entries configured."""
        # Add disabled label showing empty state
        empty_action = QAction("No phrases configured", self)
        empty_action.setEnabled(False)
        self.addAction(empty_action)

        # Add separator and configure option
        self.addSeparator()
        configure_action = QAction("Configure Phrases in Settings...", self)
        configure_action.triggered.connect(self._on_configure_clicked)
        self.addAction(configure_action)

    def _add_error_state(self):
        """Add error state message."""
        error_action = QAction("Error loading phrases", self)
        error_action.setEnabled(False)
        self.addAction(error_action)

        self.addSeparator()
        configure_action = QAction("Configure Phrases...", self)
        configure_action.triggered.connect(self._on_configure_clicked)
        self.addAction(configure_action)

    def _on_phrase_selected(self, text: str):
        """
        Handle phrase selection.

        Args:
            text: Full phrase text
        """
        self.logger.info(f"Quick Speak phrase selected: {text[:50]}...")
        self.phrase_selected.emit(text)

    def _on_configure_clicked(self):
        """Handle configure option click."""
        self.logger.info("User requested to open Settings for Quick Speak configuration")
        self.open_settings_requested.emit()

    def show_at_button(self, button: QWidget):
        """
        Show menu positioned below the specified button.

        Args:
            button: Widget to position menu relative to
        """
        self.refresh_entries()

        # Position below the button
        button_pos = button.mapToGlobal(QPoint(0, button.height()))
        self.popup(button_pos)

    def keyPressEvent(self, event: QKeyEvent):
        """
        Handle key events for keyboard navigation.

        Supports:
        - Arrow keys: Navigate entries
        - Enter/Return: Select current entry
        - Escape: Close menu

        Args:
            event: Key event
        """
        key = event.key()

        # Escape closes menu (handled by QMenu default)
        if key == Qt.Key.Key_Escape:
            self.close()
            return

        # Enter/Return selects current action
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            active_action = self.activeAction()
            if active_action and active_action.isEnabled():
                active_action.trigger()
                self.close()
            return

        # Let QMenu handle arrow key navigation
        super().keyPressEvent(event)
