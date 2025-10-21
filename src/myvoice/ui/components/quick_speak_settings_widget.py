"""
Quick Speak Settings Widget

This module implements the Quick Speak settings tab for managing
preset text entries with add/edit/delete functionality.
"""

import logging
from typing import Optional, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QInputDialog, QGroupBox, QComboBox, QLabel
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon

from myvoice.models.quick_speak_entry import QuickSpeakEntry
from myvoice.services.quick_speak_service import QuickSpeakService


class QuickSpeakSettingsWidget(QWidget):
    """
    Settings widget for managing Quick Speak entries.

    Provides table-based interface for adding, editing, and deleting
    preset text entries for quick TTS generation.

    Signals:
        entries_changed: Emitted when entries are modified
    """

    entries_changed = pyqtSignal()

    def __init__(self, quick_speak_service: QuickSpeakService, parent: Optional[QWidget] = None):
        """
        Initialize Quick Speak settings widget.

        Args:
            quick_speak_service: QuickSpeakService instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.quick_speak_service = quick_speak_service

        self._create_ui()
        self._load_entries()

        self.logger.debug("QuickSpeakSettingsWidget initialized")

    def _create_ui(self):
        """Create the Quick Speak settings UI."""
        layout = QVBoxLayout(self)

        # Profile management group
        profile_group = QGroupBox("Quick Speak Profile")
        profile_layout = QHBoxLayout(profile_group)

        profile_layout.addWidget(QLabel("Current Profile:"))

        self.profile_combo = QComboBox()
        self.profile_combo.currentTextChanged.connect(self._on_profile_changed)
        profile_layout.addWidget(self.profile_combo)

        self.new_profile_button = QPushButton("New Profile")
        self.new_profile_button.clicked.connect(self._on_new_profile)
        profile_layout.addWidget(self.new_profile_button)

        self.delete_profile_button = QPushButton("Delete Profile")
        self.delete_profile_button.clicked.connect(self._on_delete_profile)
        profile_layout.addWidget(self.delete_profile_button)

        profile_layout.addStretch()

        layout.addWidget(profile_group)

        # Quick Speak entries group
        entries_group = QGroupBox("Quick Speak Entries")
        entries_layout = QVBoxLayout(entries_group)

        # Table for entries
        self.entries_table = QTableWidget()
        self.entries_table.setColumnCount(2)
        self.entries_table.setHorizontalHeaderLabels(["Label", "Text"])

        # Configure table
        header = self.entries_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        self.entries_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.entries_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.entries_table.setAlternatingRowColors(True)

        # Double-click to edit
        self.entries_table.cellDoubleClicked.connect(self._on_edit_entry)

        entries_layout.addWidget(self.entries_table)

        # Button row
        button_layout = QHBoxLayout()

        self.add_button = QPushButton("Add Entry")
        self.add_button.clicked.connect(self._on_add_entry)

        self.edit_button = QPushButton("Edit Entry")
        self.edit_button.clicked.connect(self._on_edit_entry)
        self.edit_button.setEnabled(False)

        self.delete_button = QPushButton("Delete Entry")
        self.delete_button.clicked.connect(self._on_delete_entry)
        self.delete_button.setEnabled(False)

        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.edit_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addStretch()

        entries_layout.addLayout(button_layout)

        layout.addWidget(entries_group)

        # Connect selection change
        self.entries_table.itemSelectionChanged.connect(self._on_selection_changed)

    def _load_entries(self):
        """Load entries from service and populate table."""
        try:
            # Load available profiles
            profiles = self.quick_speak_service.get_profiles()
            current_profile = self.quick_speak_service.get_current_profile()

            # Update profile combo (block signals to prevent triggering profile change)
            self.profile_combo.blockSignals(True)
            self.profile_combo.clear()
            self.profile_combo.addItems(profiles)
            self.profile_combo.setCurrentText(current_profile)
            self.profile_combo.blockSignals(False)

            # Update delete button (can't delete default or current)
            can_delete = current_profile != "default" and len(profiles) > 1
            self.delete_profile_button.setEnabled(can_delete)

            # Load entries for current profile
            entries = self.quick_speak_service.get_entries()
            self._populate_table(entries)
            self.logger.debug(f"Loaded {len(entries)} Quick Speak entries from profile: {current_profile}")
        except Exception as e:
            self.logger.error(f"Error loading Quick Speak entries: {e}")
            QMessageBox.warning(
                self,
                "Load Error",
                f"Failed to load Quick Speak entries: {e}"
            )

    def _populate_table(self, entries: List[QuickSpeakEntry]):
        """
        Populate table with entries.

        Args:
            entries: List of QuickSpeakEntry objects
        """
        self.entries_table.setRowCount(0)

        for entry in entries:
            row = self.entries_table.rowCount()
            self.entries_table.insertRow(row)

            # Label column
            label_item = QTableWidgetItem(entry.label or "")
            label_item.setData(Qt.ItemDataRole.UserRole, entry.id)  # Store entry ID
            self.entries_table.setItem(row, 0, label_item)

            # Text column
            text_item = QTableWidgetItem(entry.text)
            self.entries_table.setItem(row, 1, text_item)

    def _on_selection_changed(self):
        """Handle table selection change."""
        has_selection = len(self.entries_table.selectedItems()) > 0
        self.edit_button.setEnabled(has_selection)
        self.delete_button.setEnabled(has_selection)

    def _on_add_entry(self):
        """Handle add entry button click."""
        try:
            # Get text from user
            text, ok = QInputDialog.getText(
                self,
                "Add Quick Speak Entry",
                "Enter text to speak:"
            )

            if not ok or not text.strip():
                return

            # Get optional label
            label, ok = QInputDialog.getText(
                self,
                "Add Quick Speak Entry",
                "Enter label (optional):"
            )

            if not ok:
                return

            # Add entry
            entry = self.quick_speak_service.add_entry(
                text=text.strip(),
                label=label.strip() if label else None
            )

            # Refresh table
            self._load_entries()
            self.entries_changed.emit()

            self.logger.info(f"Added Quick Speak entry: {entry.get_display_text()}")

        except Exception as e:
            self.logger.error(f"Error adding entry: {e}")
            QMessageBox.critical(
                self,
                "Add Error",
                f"Failed to add entry: {e}"
            )

    def _on_edit_entry(self):
        """Handle edit entry button click or double-click."""
        try:
            selected_row = self.entries_table.currentRow()
            if selected_row < 0:
                return

            # Get entry ID from table
            label_item = self.entries_table.item(selected_row, 0)
            entry_id = label_item.data(Qt.ItemDataRole.UserRole)

            # Get current entry
            entry = self.quick_speak_service.get_entry_by_id(entry_id)
            if not entry:
                self.logger.warning(f"Entry not found: {entry_id}")
                return

            # Get new text
            text, ok = QInputDialog.getText(
                self,
                "Edit Quick Speak Entry",
                "Enter text to speak:",
                text=entry.text
            )

            if not ok:
                return

            # Get new label
            label, ok = QInputDialog.getText(
                self,
                "Edit Quick Speak Entry",
                "Enter label (optional):",
                text=entry.label or ""
            )

            if not ok:
                return

            # Update entry
            success = self.quick_speak_service.update_entry(
                entry_id=entry_id,
                text=text.strip(),
                label=label.strip() if label else None
            )

            if success:
                # Refresh table
                self._load_entries()
                self.entries_changed.emit()
                self.logger.info(f"Updated Quick Speak entry: {entry_id}")
            else:
                QMessageBox.warning(
                    self,
                    "Update Error",
                    "Failed to update entry"
                )

        except Exception as e:
            self.logger.error(f"Error editing entry: {e}")
            QMessageBox.critical(
                self,
                "Edit Error",
                f"Failed to edit entry: {e}"
            )

    def _on_delete_entry(self):
        """Handle delete entry button click."""
        try:
            selected_row = self.entries_table.currentRow()
            if selected_row < 0:
                return

            # Get entry ID from table
            label_item = self.entries_table.item(selected_row, 0)
            entry_id = label_item.data(Qt.ItemDataRole.UserRole)

            # Get entry for display
            entry = self.quick_speak_service.get_entry_by_id(entry_id)
            if not entry:
                self.logger.warning(f"Entry not found: {entry_id}")
                return

            # Confirm deletion
            reply = QMessageBox.question(
                self,
                "Delete Entry",
                f"Delete Quick Speak entry:\n\n{entry.get_display_text()}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

            # Delete entry
            success = self.quick_speak_service.delete_entry(entry_id)

            if success:
                # Refresh table
                self._load_entries()
                self.entries_changed.emit()
                self.logger.info(f"Deleted Quick Speak entry: {entry_id}")
            else:
                QMessageBox.warning(
                    self,
                    "Delete Error",
                    "Failed to delete entry"
                )

        except Exception as e:
            self.logger.error(f"Error deleting entry: {e}")
            QMessageBox.critical(
                self,
                "Delete Error",
                f"Failed to delete entry: {e}"
            )

    def refresh_entries(self):
        """Refresh entries from service."""
        self._load_entries()

    def _on_profile_changed(self, profile_name: str):
        """Handle profile selection change."""
        if not profile_name:
            return

        try:
            current = self.quick_speak_service.get_current_profile()
            if profile_name == current:
                return

            success = self.quick_speak_service.switch_profile(profile_name)
            if success:
                self._load_entries()
                self.entries_changed.emit()
                self.logger.info(f"Switched to profile: {profile_name}")
            else:
                QMessageBox.warning(
                    self,
                    "Profile Switch Error",
                    f"Failed to switch to profile: {profile_name}"
                )
                # Revert combo box
                self.profile_combo.blockSignals(True)
                self.profile_combo.setCurrentText(current)
                self.profile_combo.blockSignals(False)

        except Exception as e:
            self.logger.error(f"Error switching profile: {e}")
            QMessageBox.critical(
                self,
                "Profile Error",
                f"Error switching profile: {e}"
            )

    def _on_new_profile(self):
        """Handle new profile button click."""
        try:
            profile_name, ok = QInputDialog.getText(
                self,
                "New Quick Speak Profile",
                "Enter profile name:"
            )

            if not ok or not profile_name.strip():
                return

            profile_name = profile_name.strip()
            success = self.quick_speak_service.create_profile(profile_name)

            if success:
                # Switch to the new profile
                self.quick_speak_service.switch_profile(profile_name)
                self._load_entries()
                self.entries_changed.emit()
                self.logger.info(f"Created and switched to new profile: {profile_name}")
            else:
                QMessageBox.warning(
                    self,
                    "Create Profile Error",
                    f"Failed to create profile '{profile_name}'. It may already exist."
                )

        except Exception as e:
            self.logger.error(f"Error creating profile: {e}")
            QMessageBox.critical(
                self,
                "Profile Error",
                f"Error creating profile: {e}"
            )

    def _on_delete_profile(self):
        """Handle delete profile button click."""
        try:
            current_profile = self.quick_speak_service.get_current_profile()
            profiles = self.quick_speak_service.get_profiles()

            # Can't delete default or if it's the only profile
            if current_profile == "default":
                QMessageBox.warning(
                    self,
                    "Cannot Delete",
                    "Cannot delete the default profile."
                )
                return

            if len(profiles) <= 1:
                QMessageBox.warning(
                    self,
                    "Cannot Delete",
                    "Cannot delete the only profile."
                )
                return

            # Confirm deletion
            reply = QMessageBox.question(
                self,
                "Delete Profile",
                f"Are you sure you want to delete profile '{current_profile}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

            # Switch to default before deleting
            self.quick_speak_service.switch_profile("default")

            # Delete the profile
            success = self.quick_speak_service.delete_profile(current_profile)

            if success:
                self._load_entries()
                self.entries_changed.emit()
                self.logger.info(f"Deleted profile: {current_profile}")
            else:
                QMessageBox.warning(
                    self,
                    "Delete Error",
                    f"Failed to delete profile: {current_profile}"
                )

        except Exception as e:
            self.logger.error(f"Error deleting profile: {e}")
            QMessageBox.critical(
                self,
                "Profile Error",
                f"Error deleting profile: {e}"
            )