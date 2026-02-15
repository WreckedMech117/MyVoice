"""
Voice Library Widget Component

This module implements the Voice Library management widget for viewing,
selecting, and managing voice profiles in the MyVoice application.

Story 4.4: Voice Library Management
- FR19: User can delete user-created voices
- FR46: Voice library persistence across restarts
"""

import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, List, TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QMessageBox, QFrame,
    QAbstractItemView, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QIcon, QAction

if TYPE_CHECKING:
    from myvoice.services.voice_profile_service import VoiceProfileManager

from myvoice.models.voice_profile import VoiceProfile, VoiceType


class VoiceListItem(QListWidgetItem):
    """
    Custom list widget item for voice profiles.

    Stores the VoiceProfile reference and provides formatted display.
    """

    def __init__(self, profile: VoiceProfile, parent: Optional[QListWidget] = None):
        super().__init__(parent)
        self.profile = profile

        # Set display text with icon
        self.setText(self._format_display())

        # Set tooltip with more details
        self.setToolTip(self._format_tooltip())

        # Store voice type for sorting
        self.setData(Qt.ItemDataRole.UserRole, profile.voice_type.sort_order)
        self.setData(Qt.ItemDataRole.UserRole + 1, profile.name)

    def _format_display(self) -> str:
        """Format the display text for the list item."""
        icon = self.profile.voice_type.icon
        name = self.profile.name

        if self.profile.duration:
            duration = f"({self.profile.duration:.1f}s)"
        else:
            duration = ""

        return f"{icon} {name} {duration}".strip()

    def _format_tooltip(self) -> str:
        """Format tooltip with detailed info."""
        lines = [
            f"Name: {self.profile.name}",
            f"Type: {self.profile.voice_type.display_name}",
        ]

        if self.profile.duration:
            lines.append(f"Duration: {self.profile.duration:.1f} seconds")

        if self.profile.emotion_capable:
            lines.append("Supports emotion control")
        else:
            lines.append("No emotion control")

        return "\n".join(lines)


class VoiceLibraryWidget(QWidget):
    """
    Voice library management widget.

    Provides a list view of all voice profiles with:
    - Voice type icons (📦 bundled, 🎭 designed, 🎤 cloned)
    - Delete button for user-created voices
    - Confirmation dialog for deletion
    - Empty state with create voice buttons
    - Voice selection

    Story 4.4 Acceptance Criteria:
    - All voices shown with type icons
    - Bundled voices (📦) cannot be deleted
    - User-created voices (🎭/🎤) show delete button
    - Deletion requires confirmation
    - If deleted voice was selected, switch to bundled
    - "Create your first voice" empty state

    Signals:
        voice_selected: Emitted when a voice is selected (VoiceProfile)
        voice_deleted: Emitted when a voice is deleted (voice_name)
        design_voice_requested: Emitted when user wants to design a voice
        clone_voice_requested: Emitted when user wants to clone a voice
        refresh_requested: Emitted when voice list should be refreshed
    """

    voice_selected = pyqtSignal(object)  # VoiceProfile
    voice_deleted = pyqtSignal(str)  # voice_name
    design_voice_requested = pyqtSignal()  # Opens Voice Design Studio
    refresh_requested = pyqtSignal()

    def __init__(
        self,
        voice_manager: Optional['VoiceProfileManager'] = None,
        parent: Optional[QWidget] = None
    ):
        """
        Initialize the Voice Library Widget.

        Args:
            voice_manager: VoiceProfileManager instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.voice_manager = voice_manager
        self._profiles: Dict[str, VoiceProfile] = {}
        self._current_selection: Optional[str] = None
        self._is_updating = False

        self._setup_ui()
        self._setup_connections()

        if self.voice_manager:
            self.refresh_voices()

        self.logger.debug("VoiceLibraryWidget initialized")

    def _setup_ui(self):
        """Create the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Header with title and action buttons
        header_layout = QHBoxLayout()

        title_label = QLabel("Voice Library")
        title_font = QFont()
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setMinimumWidth(80)
        self.refresh_button.clicked.connect(self._on_refresh_clicked)
        header_layout.addWidget(self.refresh_button)

        layout.addLayout(header_layout)

        # Voice list
        self.voice_list = QListWidget()
        self.voice_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.voice_list.setMinimumHeight(150)
        self.voice_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.voice_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.voice_list.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.voice_list)

        # Selected voice info and delete button
        self.action_frame = QFrame()
        action_layout = QHBoxLayout(self.action_frame)
        action_layout.setContentsMargins(0, 4, 0, 4)

        self.selected_label = QLabel("No voice selected")
        self.selected_label.setProperty("class", "caption")
        action_layout.addWidget(self.selected_label, 1)

        self.delete_button = QPushButton("Delete")
        self.delete_button.setMinimumWidth(80)
        self.delete_button.setEnabled(False)
        self.delete_button.setToolTip("Delete selected voice")
        self.delete_button.clicked.connect(self._on_delete_clicked)
        action_layout.addWidget(self.delete_button)

        layout.addWidget(self.action_frame)

        # Empty state widget (hidden by default)
        self.empty_state = QWidget()
        empty_layout = QVBoxLayout(self.empty_state)
        empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        empty_label = QLabel("Create your first voice")
        empty_font = QFont()
        empty_font.setPointSize(11)
        empty_label.setFont(empty_font)
        empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(empty_label)

        empty_hint = QLabel("Create a voice from description or audio sample")
        empty_hint.setProperty("class", "caption")
        empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(empty_hint)

        # Create voice button
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.design_button = QPushButton("Design Voice")
        self.design_button.setMinimumWidth(150)
        self.design_button.setEnabled(False)  # Disabled until TTS service available
        self.design_button.setToolTip("TTS service initializing...")
        self.design_button.clicked.connect(self._on_design_clicked)
        button_layout.addWidget(self.design_button)

        empty_layout.addLayout(button_layout)

        self.empty_state.setVisible(False)
        layout.addWidget(self.empty_state)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        # Bottom action buttons (always visible)
        bottom_layout = QHBoxLayout()

        self.bottom_design_button = QPushButton("Design Voice")
        self.bottom_design_button.setMinimumWidth(130)
        self.bottom_design_button.setEnabled(False)  # Disabled until TTS service available
        self.bottom_design_button.setToolTip("TTS service initializing...")
        self.bottom_design_button.clicked.connect(self._on_design_clicked)
        bottom_layout.addWidget(self.bottom_design_button)

        bottom_layout.addStretch()

        layout.addLayout(bottom_layout)

    def _setup_connections(self):
        """Set up signal connections."""
        pass

    def set_voice_manager(self, voice_manager: 'VoiceProfileManager'):
        """
        Set the voice manager and refresh the list.

        Args:
            voice_manager: VoiceProfileManager instance
        """
        self.voice_manager = voice_manager
        self.refresh_voices()

    def set_tts_available(self, available: bool):
        """
        Enable or disable voice creation buttons based on TTS service availability.

        Args:
            available: True if TTS service is ready, False otherwise
        """
        tooltip = "" if available else "TTS service initializing..."

        # Empty state buttons
        self.design_button.setEnabled(available)
        self.design_button.setToolTip(tooltip)

        # Bottom action buttons
        self.bottom_design_button.setEnabled(available)
        self.bottom_design_button.setToolTip(tooltip)

        self.logger.info(f"Voice creation buttons {'ENABLED' if available else 'disabled'} - TTS available: {available}")

    def refresh_voices(self):
        """Refresh the voice list from the voice manager."""
        if not self.voice_manager:
            self._show_empty_state()
            return

        try:
            self._is_updating = True

            # Clear current list
            self.voice_list.clear()
            self._profiles.clear()

            # Get all profiles
            profiles = self.voice_manager.get_valid_profiles()

            if not profiles:
                self._show_empty_state()
                return

            # Hide empty state, show list
            self.empty_state.setVisible(False)
            self.voice_list.setVisible(True)
            self.action_frame.setVisible(True)

            # Sort profiles: bundled first, then designed, then cloned
            sorted_profiles = sorted(
                profiles.items(),
                key=lambda x: (x[1].voice_type.sort_order, x[0].lower())
            )

            # Add profiles to list
            for name, profile in sorted_profiles:
                item = VoiceListItem(profile, self.voice_list)
                self._profiles[name] = profile

            # Restore selection or select first bundled voice
            self._restore_selection()

            self.logger.info(f"Refreshed voice library: {len(profiles)} voices")

        except Exception as e:
            self.logger.exception(f"Error refreshing voices: {e}")
            self._show_empty_state()
        finally:
            self._is_updating = False

    def _show_empty_state(self):
        """Show the empty state (no user-created voices)."""
        self.voice_list.clear()
        self._profiles.clear()

        # Check if we have bundled voices
        if self.voice_manager:
            profiles = self.voice_manager.get_valid_profiles()
            bundled_count = sum(
                1 for p in profiles.values()
                if p.voice_type == VoiceType.BUNDLED
            )
            user_count = len(profiles) - bundled_count

            if user_count == 0 and bundled_count > 0:
                # Have bundled but no user voices - show list with empty state message
                self.empty_state.setVisible(True)
                self.voice_list.setVisible(True)
                self.action_frame.setVisible(True)

                # Still show bundled voices in list
                sorted_profiles = sorted(
                    profiles.items(),
                    key=lambda x: (x[1].voice_type.sort_order, x[0].lower())
                )
                for name, profile in sorted_profiles:
                    item = VoiceListItem(profile, self.voice_list)
                    self._profiles[name] = profile
            else:
                # No voices at all
                self.empty_state.setVisible(True)
                self.voice_list.setVisible(False)
                self.action_frame.setVisible(False)
        else:
            self.empty_state.setVisible(True)
            self.voice_list.setVisible(False)
            self.action_frame.setVisible(False)

    def _restore_selection(self):
        """Restore previous selection or select first bundled voice."""
        if not self.voice_manager:
            return

        # Try to select currently active profile
        active_profile = self.voice_manager.get_active_profile()
        if active_profile:
            self._select_voice_by_name(active_profile.name)
            return

        # Otherwise select first bundled voice
        for i in range(self.voice_list.count()):
            item = self.voice_list.item(i)
            if isinstance(item, VoiceListItem):
                if item.profile.voice_type == VoiceType.BUNDLED:
                    self.voice_list.setCurrentItem(item)
                    return

        # If no bundled, select first item
        if self.voice_list.count() > 0:
            self.voice_list.setCurrentRow(0)

    def _select_voice_by_name(self, name: str):
        """Select a voice by name."""
        for i in range(self.voice_list.count()):
            item = self.voice_list.item(i)
            if isinstance(item, VoiceListItem):
                if item.profile.name == name:
                    self.voice_list.setCurrentItem(item)
                    return

    def _on_selection_changed(self):
        """Handle voice selection change."""
        if self._is_updating:
            return

        selected_items = self.voice_list.selectedItems()
        if not selected_items:
            self._current_selection = None
            self.selected_label.setText("No voice selected")
            self.delete_button.setEnabled(False)
            return

        item = selected_items[0]
        if not isinstance(item, VoiceListItem):
            return

        profile = item.profile
        self._current_selection = profile.name

        # Update selected label
        type_name = profile.voice_type.display_name
        self.selected_label.setText(f"Selected: {profile.name} ({type_name})")

        # Enable delete only for user-created voices (not bundled)
        can_delete = profile.voice_type in (VoiceType.DESIGNED, VoiceType.CLONED, VoiceType.EMBEDDING)
        self.delete_button.setEnabled(can_delete)

        if not can_delete:
            self.delete_button.setToolTip("Bundled voices cannot be deleted")
        else:
            self.delete_button.setToolTip("Delete this voice")

        # Emit selection signal
        self.voice_selected.emit(profile)

        self.logger.debug(f"Voice selected: {profile.name}")

    def _show_context_menu(self, position):
        """Show context menu for voice item."""
        item = self.voice_list.itemAt(position)
        if not item or not isinstance(item, VoiceListItem):
            return

        profile = item.profile
        menu = QMenu(self)

        # Select action
        select_action = QAction(f"Select '{profile.name}'", self)
        select_action.triggered.connect(lambda: self._select_voice_by_name(profile.name))
        menu.addAction(select_action)

        menu.addSeparator()

        # Delete action (only for user-created voices - not bundled)
        if profile.voice_type in (VoiceType.DESIGNED, VoiceType.CLONED, VoiceType.EMBEDDING):
            delete_action = QAction(f"Delete '{profile.name}'", self)
            delete_action.triggered.connect(lambda: self._delete_voice(profile))
            menu.addAction(delete_action)
        else:
            # Show disabled delete for bundled
            delete_action = QAction("Delete (bundled voices cannot be deleted)", self)
            delete_action.setEnabled(False)
            menu.addAction(delete_action)

        menu.exec(self.voice_list.mapToGlobal(position))

    def _on_delete_clicked(self):
        """Handle delete button click."""
        if not self._current_selection:
            return

        profile = self._profiles.get(self._current_selection)
        if not profile:
            return

        self._delete_voice(profile)

    def _delete_voice(self, profile: VoiceProfile):
        """
        Delete a voice profile with confirmation.

        Args:
            profile: VoiceProfile to delete
        """
        # Check if deletable
        if profile.voice_type == VoiceType.BUNDLED:
            QMessageBox.warning(
                self,
                "Cannot Delete",
                "Bundled voices cannot be deleted.",
                QMessageBox.StandardButton.Ok
            )
            return

        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Delete Voice",
            f"Delete '{profile.name}' from your voice library?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
            QMessageBox.StandardButton.Cancel
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Check if this is the currently selected voice
        was_selected = (self._current_selection == profile.name)

        try:
            # Delete the voice files/directory
            # EMBEDDING voices use checkpoint_path (embedding directory)
            # Other voices use file_path (audio file)
            if profile.voice_type == VoiceType.EMBEDDING:
                # Embedding voice - delete the embedding directory
                embedding_dir = profile.checkpoint_path
                if embedding_dir and embedding_dir.exists() and embedding_dir.is_dir():
                    shutil.rmtree(embedding_dir)
                    self.logger.info(f"Deleted embedding directory: {embedding_dir}")
                else:
                    self.logger.warning(f"Embedding directory not found: {embedding_dir}")
            else:
                # Regular voice - delete directory or file
                voice_dir = profile.file_path.parent
                if voice_dir.exists() and voice_dir.is_dir():
                    shutil.rmtree(voice_dir)
                    self.logger.info(f"Deleted voice directory: {voice_dir}")
                elif profile.file_path.exists():
                    # Single file voice
                    profile.file_path.unlink()
                    self.logger.info(f"Deleted voice file: {profile.file_path}")

            # Remove from profiles dict
            if profile.name in self._profiles:
                del self._profiles[profile.name]

            # Emit deleted signal
            self.voice_deleted.emit(profile.name)

            # Refresh the list
            self.refresh_voices()

            # If the deleted voice was selected, select a bundled voice
            if was_selected:
                self._select_first_bundled_voice()

            self.logger.info(f"Voice deleted: {profile.name}")

        except Exception as e:
            self.logger.exception(f"Error deleting voice: {e}")
            QMessageBox.critical(
                self,
                "Delete Failed",
                f"Failed to delete voice: {str(e)}",
                QMessageBox.StandardButton.Ok
            )

    def _select_first_bundled_voice(self):
        """Select the first bundled voice."""
        for i in range(self.voice_list.count()):
            item = self.voice_list.item(i)
            if isinstance(item, VoiceListItem):
                if item.profile.voice_type == VoiceType.BUNDLED:
                    self.voice_list.setCurrentItem(item)
                    return

    def _on_refresh_clicked(self):
        """Handle refresh button click."""
        self.refresh_requested.emit()
        self.refresh_voices()

    def _on_design_clicked(self):
        """Handle design voice button click - opens Voice Design Studio."""
        self.design_voice_requested.emit()

    def get_selected_profile(self) -> Optional[VoiceProfile]:
        """Get the currently selected voice profile."""
        if self._current_selection:
            return self._profiles.get(self._current_selection)
        return None

    def get_selected_profile_name(self) -> Optional[str]:
        """Get the name of the currently selected voice profile."""
        return self._current_selection

    def select_voice(self, voice_name: str):
        """
        Select a voice by name.

        Args:
            voice_name: Name of the voice to select
        """
        self._select_voice_by_name(voice_name)

    def get_user_voice_count(self) -> int:
        """Get the count of user-created voices."""
        return sum(
            1 for p in self._profiles.values()
            if p.voice_type in (VoiceType.DESIGNED, VoiceType.CLONED)
        )

    def get_bundled_voice_count(self) -> int:
        """Get the count of bundled voices."""
        return sum(
            1 for p in self._profiles.values()
            if p.voice_type == VoiceType.BUNDLED
        )
