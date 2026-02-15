"""
Voice Selector Component

This module provides a UI component for selecting voice profiles from available
voice cloning samples with dropdown selection and refresh functionality.
"""

import logging
from typing import Optional, Dict, List
from PyQt6.QtWidgets import QWidget, QComboBox, QHBoxLayout, QPushButton, QLabel, QToolButton, QMenu
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QAction

from myvoice.models.voice_profile import VoiceProfile, TranscriptionStatus, VoiceType
from myvoice.services.voice_profile_service import VoiceProfileManager


class VoiceSelector(QWidget):
    """
    Voice profile selector component with dropdown and refresh functionality.

    This widget provides a QComboBox for selecting voice profiles managed by
    the VoiceProfileManager service. It includes automatic population, selection
    handling, display formatting, and refresh mechanism for rescanning profiles.

    Signals:
        voice_selected: Emitted when a voice profile is selected (profile_name: str)
        refresh_requested: Emitted when user requests a refresh
        transcription_edit_requested: Emitted when user requests transcription editing (profile_name: str)
    """

    voice_selected = pyqtSignal(str)  # profile_name
    refresh_requested = pyqtSignal()
    transcription_edit_requested = pyqtSignal(str)  # profile_name

    def __init__(self, voice_manager: Optional[VoiceProfileManager] = None, parent: Optional[QWidget] = None):
        """
        Initialize the voice selector component.

        Args:
            voice_manager: VoiceProfileManager service instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.voice_manager = voice_manager
        self.logger = logging.getLogger(self.__class__.__name__)

        # UI components
        self._voice_combo: Optional[QComboBox] = None
        self._refresh_button: Optional[QPushButton] = None
        self._transcription_button: Optional[QToolButton] = None
        self._voice_label: Optional[QLabel] = None

        # State tracking
        self._profiles: Dict[str, VoiceProfile] = {}
        self._current_selection: Optional[str] = None
        self._is_updating = False

        # Auto-refresh timer
        self._auto_refresh_timer = QTimer()
        self._auto_refresh_timer.timeout.connect(self._auto_refresh_profiles)

        # Create UI
        self._create_ui()

        # Initial population if voice manager is available
        if self.voice_manager:
            self.populate_voices()

        self.logger.debug("VoiceSelector component initialized")

    def _create_ui(self):
        """Create the voice selector UI components."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Voice selection dropdown
        self._voice_combo = QComboBox()
        self._voice_combo.setMinimumWidth(200)
        self._voice_combo.setMaximumWidth(300)
        self._voice_combo.setToolTip("Select a voice profile for speech generation")

        # Set font and styling
        font = QFont()
        font.setPointSize(9)
        self._voice_combo.setFont(font)

        # Connect selection signal
        self._voice_combo.currentTextChanged.connect(self.on_voice_selected)

        # Refresh button with Qt standard icon
        self._refresh_button = QPushButton()
        refresh_icon = self.style().standardIcon(self.style().StandardPixmap.SP_BrowserReload)
        self._refresh_button.setIcon(refresh_icon)
        self._refresh_button.setFixedSize(24, 24)
        self._refresh_button.setObjectName("refresh_button")
        self._refresh_button.setToolTip("Refresh voice profiles from directory")
        self._refresh_button.clicked.connect(self._on_refresh_clicked)

        # Transcription button with dropdown menu and Qt standard icon
        self._transcription_button = QToolButton()
        transcription_icon = self.style().standardIcon(self.style().StandardPixmap.SP_FileIcon)
        self._transcription_button.setIcon(transcription_icon)
        self._transcription_button.setFixedSize(24, 24)
        self._transcription_button.setObjectName("transcription_button")
        self._transcription_button.setToolTip("Transcription options")
        self._transcription_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._setup_transcription_menu()

        # Voice label with count
        self._voice_label = QLabel("Voice:")

        # Add widgets to layout
        layout.addWidget(self._voice_label)
        layout.addWidget(self._voice_combo)
        layout.addStretch()  # This pushes the buttons to the right
        layout.addWidget(self._refresh_button)
        layout.addWidget(self._transcription_button)

        # Set initial state
        self._update_voice_label()
        self._update_transcription_button_state()

    def set_voice_manager(self, voice_manager: VoiceProfileManager):
        """
        Set the voice profile manager and populate voices.

        Args:
            voice_manager: VoiceProfileManager service instance
        """
        self.voice_manager = voice_manager
        self.populate_voices()
        self.logger.debug("VoiceManager set and profiles populated")

    def populate_voices(self):
        """
        Populate the voice dropdown with available profiles from the voice manager.
        """
        if not self.voice_manager:
            self.logger.warning("No voice manager available for populating voices")
            return

        try:
            self._is_updating = True

            # Clear existing items
            self._voice_combo.clear()
            self._profiles.clear()

            # Get valid profiles from voice manager
            profiles = self.voice_manager.get_valid_profiles()

            if not profiles:
                self._voice_combo.addItem("No voice profiles available")
                self._voice_combo.setEnabled(False)
                self.logger.info("No valid voice profiles found")
            else:
                self._voice_combo.setEnabled(True)

                # Story 4.1: Sort profiles by voice type (bundled first), then by name
                # Group order: BUNDLED (0) → DESIGNED (1) → CLONED (2)
                sorted_profiles = sorted(
                    profiles.items(),
                    key=lambda x: (x[1].voice_type.sort_order, x[0].lower())
                )

                # Add each profile to dropdown with formatted display (includes icon)
                for profile_name, profile in sorted_profiles:
                    display_text = self._format_profile_display(profile)
                    self._voice_combo.addItem(display_text)
                    self._profiles[display_text] = profile

                # Try to restore previous selection or set to active profile
                self._restore_selection()

                self.logger.info(f"Populated {len(profiles)} voice profiles (grouped by type)")

            self._update_voice_label()
            self._update_transcription_button_state()

        except Exception as e:
            self.logger.exception(f"Error populating voices: {e}")
            self._voice_combo.clear()
            self._voice_combo.addItem("Error loading profiles")
            self._voice_combo.setEnabled(False)
            self._update_transcription_button_state()
        finally:
            self._is_updating = False

    def _format_profile_display(self, profile: VoiceProfile) -> str:
        """
        Format a voice profile for display in the dropdown.

        Story 4.1: Voice Library & Selection (FR11)
        - Displays icon based on voice type: 📦 bundled, 🎭 designed, 🎤 cloned
        - Shows name and duration/type indicator

        Args:
            profile: VoiceProfile to format

        Returns:
            str: Formatted display text with icon
        """
        # Get voice type icon (Story 4.1: FR11)
        icon = profile.voice_type.icon

        # Format based on voice type
        # Bundled voices show "bundled" since they have no audio file
        # DESIGNED voices (future) may show description or "designed"
        # CLONED voices show duration
        if profile.voice_type == VoiceType.BUNDLED:
            info_str = "bundled"
        elif profile.duration:
            info_str = f"{profile.duration:.1f}s"
        else:
            info_str = profile.voice_type.display_name.lower()

        return f"{icon} {profile.name} ({info_str})"

    def _restore_selection(self):
        """Restore the previously selected profile or set to active profile."""
        if not self.voice_manager:
            return

        # Try to set to currently active profile
        active_profile = self.voice_manager.get_active_profile()
        if active_profile:
            target_name = active_profile.name
            for i in range(self._voice_combo.count()):
                item_text = self._voice_combo.itemText(i)
                # Match profile name in display text (handles icon prefix: "📦 Ryan (unknown)")
                if f" {target_name} (" in item_text or item_text.startswith(f"{target_name} ("):
                    self._voice_combo.setCurrentIndex(i)
                    self._current_selection = target_name
                    self.logger.debug(f"Restored selection to active profile: {target_name}")
                    return

        # If no active profile but we have items, select the first one
        if self._voice_combo.count() > 0:
            self._voice_combo.setCurrentIndex(0)
            # Extract profile name from first item (handles icon prefix)
            first_item = self._voice_combo.itemText(0)
            profile = self._profiles.get(first_item)
            if profile:
                self._current_selection = profile.name
            elif " (" in first_item:
                # Fallback: extract name between icon and duration
                # Format: "📦 Name (duration)" -> extract "Name"
                parts = first_item.split(" (")[0]  # "📦 Name"
                name_parts = parts.split(" ", 1)  # ["📦", "Name"] or ["Name"]
                self._current_selection = name_parts[-1] if name_parts else None
            else:
                self._current_selection = None
        else:
            self._current_selection = None

    def on_voice_selected(self, display_text: str):
        """
        Handle voice profile selection from dropdown.

        Args:
            display_text: Display text from the combo box
        """
        if self._is_updating:
            return

        try:
            # Skip default selection item
            if display_text.startswith("Select a voice") or display_text.startswith("No voice") or display_text.startswith("Error"):
                self._current_selection = None
                return

            # Find the profile by display text
            if display_text in self._profiles:
                profile = self._profiles[display_text]
                self._current_selection = profile.name

                # Emit signal for external handlers
                # The signal will be handled by app.py which properly manages async operations
                self.voice_selected.emit(profile.name)

                self.logger.info(f"Voice profile selected: {profile.name}")

            else:
                self.logger.warning(f"Profile not found for display text: {display_text}")
                self._current_selection = None

            # Update transcription button state after selection change
            self._update_transcription_button_state()

        except Exception as e:
            self.logger.exception(f"Error handling voice selection: {e}")
            self._current_selection = None
            self._update_transcription_button_state()

    def _on_refresh_clicked(self):
        """Handle refresh button click."""
        try:
            self.logger.info("Refresh button clicked")

            # Disable refresh button temporarily
            self._refresh_button.setEnabled(False)

            # Emit refresh signal for external handlers (app will handle async refresh)
            self.refresh_requested.emit()

            # Re-enable button after a short delay (actual refresh happens via app)
            QTimer.singleShot(2000, lambda: self._refresh_button.setEnabled(True))

        except Exception as e:
            self.logger.exception(f"Error handling refresh click: {e}")
            self._refresh_button.setEnabled(True)


    def _auto_refresh_profiles(self):
        """Auto-refresh profiles periodically."""
        if self.voice_manager and not self._is_updating:
            self.populate_voices()

    def _update_voice_label(self):
        """Update the voice label with current profile count."""
        try:
            profile_count = len(self._profiles)
            if profile_count == 0:
                self._voice_label.setText("Voices (0):")
            elif profile_count == 1:
                self._voice_label.setText("Voices (1):")
            else:
                self._voice_label.setText(f"Voices ({profile_count}):")

        except Exception as e:
            self.logger.error(f"Error updating voice label: {e}")
            self._voice_label.setText("Voices (?):")

    def get_selected_profile(self) -> Optional[VoiceProfile]:
        """
        Get the currently selected voice profile.

        Returns:
            Optional[VoiceProfile]: Selected profile or None if no selection
        """
        if self._current_selection and self.voice_manager:
            profiles = self.voice_manager.get_valid_profiles()
            return profiles.get(self._current_selection)
        return None

    def get_selected_profile_name(self) -> Optional[str]:
        """
        Get the name of the currently selected voice profile.

        Returns:
            Optional[str]: Selected profile name or None if no selection
        """
        return self._current_selection

    def set_auto_refresh(self, enabled: bool, interval_ms: int = 30000):
        """
        Enable or disable auto-refresh of voice profiles.

        Args:
            enabled: Whether to enable auto-refresh
            interval_ms: Refresh interval in milliseconds (default: 30 seconds)
        """
        if enabled:
            self._auto_refresh_timer.start(interval_ms)
            self.logger.debug(f"Auto-refresh enabled with {interval_ms}ms interval")
        else:
            self._auto_refresh_timer.stop()
            self.logger.debug("Auto-refresh disabled")

    def clear_selection(self):
        """Clear the current voice selection."""
        self._is_updating = True
        try:
            self._voice_combo.setCurrentIndex(0)  # Set to default "Select..." item
            self._current_selection = None
            self.logger.debug("Voice selection cleared")
        finally:
            self._is_updating = False

    def set_selected_voice(self, voice_name: str):
        """
        Set the selected voice by name.

        Args:
            voice_name: Name of the voice profile to select
        """
        if not voice_name or not self._voice_combo:
            return

        self._is_updating = True
        try:
            # Find the voice in the combo box by matching the display text
            # Handles icon prefix: "📦 Ryan (unknown)" for bundled voices
            for i in range(self._voice_combo.count()):
                item_text = self._voice_combo.itemText(i)
                # Match with or without icon prefix
                if f" {voice_name} (" in item_text or item_text.startswith(f"{voice_name} ("):
                    self._voice_combo.setCurrentIndex(i)
                    self._current_selection = voice_name
                    self.logger.debug(f"Set selected voice to: {voice_name}")
                    return

            self.logger.warning(f"Voice profile not found in selector: {voice_name}")
        finally:
            self._is_updating = False

    def set_enabled(self, enabled: bool):
        """
        Enable or disable the voice selector.

        Args:
            enabled: Whether to enable the component
        """
        self._voice_combo.setEnabled(enabled)
        self._refresh_button.setEnabled(enabled)

    def _setup_transcription_menu(self):
        """Setup the transcription dropdown menu."""
        try:
            menu = QMenu(self)

            # Edit transcription action
            edit_action = QAction("📝 Edit Transcription", self)
            edit_action.setToolTip("Edit transcription text for selected voice in Notepad")
            edit_action.triggered.connect(self._on_edit_transcription)
            menu.addAction(edit_action)

            menu.addSeparator()

            # Generate transcription action
            generate_action = QAction("🎤 Generate New", self)
            generate_action.setToolTip("Generate new transcription using Whisper")
            generate_action.triggered.connect(self._on_generate_transcription)
            menu.addAction(generate_action)

            # Refresh status action
            refresh_status_action = QAction("🔄 Refresh Status", self)
            refresh_status_action.setToolTip("Refresh transcription status from saved files")
            refresh_status_action.triggered.connect(self._on_refresh_transcription_status)
            menu.addAction(refresh_status_action)

            self._transcription_button.setMenu(menu)

        except Exception as e:
            self.logger.error(f"Error setting up transcription menu: {e}")

    def _update_transcription_button_state(self):
        """Update transcription button state based on current selection."""
        try:
            has_selection = self._current_selection is not None
            self._transcription_button.setEnabled(has_selection)

            # Update tooltip with current transcription status
            if has_selection and self._current_selection in self._get_profile_map():
                profile = self._get_profile_map()[self._current_selection]
                status_text = profile.transcription_status.value.title()

                if profile.transcription:
                    word_count = len(profile.transcription.split())
                    tooltip = f"Transcription: {status_text} ({word_count} words)"
                else:
                    tooltip = f"Transcription: {status_text}"

                self._transcription_button.setToolTip(tooltip)
            else:
                self._transcription_button.setToolTip("Select a voice to manage transcription")

        except Exception as e:
            self.logger.error(f"Error updating transcription button state: {e}")

    def _get_profile_map(self) -> Dict[str, VoiceProfile]:
        """Get mapping of profile names to VoiceProfile objects."""
        if not self.voice_manager:
            return {}
        return self.voice_manager.get_valid_profiles()

    def _on_edit_transcription(self):
        """Handle edit transcription menu action by opening the transcription file in Notepad."""
        try:
            if not self._current_selection:
                self.logger.warning("No voice selected for transcription editing")
                return

            # Get the voice profile to find the transcription file
            if not self.voice_manager:
                self.logger.warning("Voice manager not available")
                return

            profiles = self.voice_manager.get_valid_profiles()
            if self._current_selection not in profiles:
                self.logger.warning(f"Voice profile not found: {self._current_selection}")
                return

            voice_profile = profiles[self._current_selection]
            transcription_file = voice_profile.file_path.with_suffix('.txt')

            # Create the transcription file if it doesn't exist
            if not transcription_file.exists():
                try:
                    transcription_file.write_text("", encoding='utf-8')
                    self.logger.info(f"Created empty transcription file: {transcription_file}")
                except Exception as e:
                    self.logger.error(f"Failed to create transcription file: {e}")
                    return

            # Open the transcription file in Notepad
            import subprocess
            try:
                subprocess.run(['notepad.exe', str(transcription_file)], check=True)
                self.logger.info(f"Opened transcription file in Notepad: {transcription_file}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to open Notepad: {e}")
            except FileNotFoundError:
                self.logger.error("Notepad not found on system")

        except Exception as e:
            self.logger.exception(f"Error handling edit transcription: {e}")


    def _on_generate_transcription(self):
        """Handle generate transcription menu action."""
        try:
            if not self._current_selection:
                return

            # Emit the edit signal - the dialog can handle generation
            self.transcription_edit_requested.emit(self._current_selection)

        except Exception as e:
            self.logger.exception(f"Error handling generate transcription: {e}")

    def _on_refresh_transcription_status(self):
        """Handle refresh transcription status menu action."""
        try:
            # Simply refresh the UI - async operations should be handled by parent/app
            # For now, just re-populate the voices from the current state
            self.populate_voices()

        except Exception as e:
            self.logger.exception(f"Error refreshing transcription status: {e}")

    def cleanup(self):
        """Clean up resources when widget is destroyed."""
        if self._auto_refresh_timer:
            self._auto_refresh_timer.stop()
        self.logger.debug("VoiceSelector cleaned up")