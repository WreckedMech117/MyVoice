"""
Settings Dialog Component

This module implements the settings dialog with monitor device selection
and other application configuration options.
"""

import logging
import subprocess
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QComboBox, QLabel, QCheckBox, QSlider, QLineEdit,
    QSpinBox, QTabWidget, QWidget, QMessageBox, QProgressBar, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

from myvoice.models.app_settings import AppSettings
from myvoice.models.audio_device import AudioDevice
from myvoice.models.validation import ValidationResult
from myvoice.ui.styles.theme_manager import get_theme_manager
from myvoice.ui.components.quick_speak_settings_widget import QuickSpeakSettingsWidget
from myvoice.ui.components.voice_selector import VoiceSelector
from myvoice.ui.components.voice_library_widget import VoiceLibraryWidget
from myvoice.ui.dialogs.voice_design_studio import VoiceDesignStudioDialog
from myvoice.services.quick_speak_service import QuickSpeakService
from myvoice.services.voice_profile_service import VoiceProfileManager


class SettingsDialog(QDialog):
    """
    Settings dialog for MyVoice application configuration.

    Provides interface for:
    - Monitor device selection
    - Audio settings configuration
    - TTS service settings
    - UI preferences

    Signals:
        settings_changed: Emitted when settings are modified and applied
        device_refresh_requested: Emitted when user requests device list refresh
        device_test_requested: Emitted when user wants to test a device
    """

    settings_changed = pyqtSignal(AppSettings)
    device_refresh_requested = pyqtSignal()
    device_test_requested = pyqtSignal(str)  # device_id
    virtual_device_test_requested = pyqtSignal(str)  # device_id
    voice_directory_changed = pyqtSignal(str)  # new_directory_path
    voice_changed = pyqtSignal(str)  # voice_profile_name
    voice_refresh_requested = pyqtSignal()  # Voice profile refresh requested
    voice_transcription_requested = pyqtSignal(str)  # voice_profile_name - Transcription generation requested
    quick_speak_entries_changed = pyqtSignal()  # Quick Speak entries or profile changed
    custom_emotion_applied = pyqtSignal(str)  # Custom emotion text applied (Story 3.4)
    transparency_preview_requested = pyqtSignal(float)  # Story 7.3: Real-time transparency preview (opacity 0.0-1.0)
    voice_created = pyqtSignal(object)  # VoiceProfile - emitted when voice is created (design/clone)
    voice_deleted = pyqtSignal(str)  # voice_name - emitted when voice is deleted

    def __init__(self, settings: AppSettings, parent: Optional[QWidget] = None, quick_speak_service: Optional[QuickSpeakService] = None, audio_client=None):
        """
        Initialize the settings dialog.

        Args:
            settings: Current application settings
            parent: Parent widget
            quick_speak_service: Optional shared QuickSpeakService instance (recommended)
            audio_client: Optional WindowsAudioClient instance for smart device matching
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Store settings
        self.original_settings = settings
        self.current_settings = AppSettings.from_dict(settings.to_dict())

        # Audio devices cache
        self.audio_devices: List[AudioDevice] = []
        self.output_devices: List[AudioDevice] = []
        self.virtual_input_devices: List[AudioDevice] = []

        # Audio client for smart device matching
        self.audio_client = audio_client

        # Voice manager (will be set by parent)
        self.voice_manager: Optional[VoiceProfileManager] = None
        self.selected_voice_name: Optional[str] = None

        # TTS service (will be set by parent for voice creation dialogs)
        self.tts_service = None

        # Whisper service for transcription (QA3 fix)
        self.whisper_service = None

        # UI state
        self.device_refresh_in_progress = False

        # Theme manager
        self.theme_manager = get_theme_manager()

        # Quick Speak service - use shared instance if provided, otherwise create new one
        if quick_speak_service is not None:
            self.quick_speak_service = quick_speak_service
            self.logger.debug("Using shared QuickSpeakService instance")
        else:
            self.quick_speak_service = QuickSpeakService(settings.config_directory)
            self.quick_speak_service.load_entries()
            self.logger.debug("Created new QuickSpeakService instance")

        # Setup dialog
        self._setup_dialog()
        self._create_ui()
        self._load_current_settings()

        self.logger.debug("SettingsDialog initialized")

    def _setup_dialog(self):
        """Configure dialog properties."""
        self.setWindowTitle("MyVoice Settings")
        self.setModal(True)
        self.resize(500, 400)

        # Set window flags for proper dialog behavior
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.WindowTitleHint |
            Qt.WindowType.WindowCloseButtonHint
        )

    def _create_ui(self):
        """Create the settings dialog UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Create tab widget for organized settings
        self.tab_widget = QTabWidget()

        # Audio settings tab
        self._create_audio_tab()

        # Voices tab (Story 4.2: Voice Design Creation Workflow)
        self._create_voices_tab()

        # TTS settings tab
        self._create_tts_tab()

        # UI settings tab
        self._create_ui_tab()

        # Quick Speak settings tab
        self._create_quick_speak_tab()

        layout.addWidget(self.tab_widget)

        # Dialog buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.test_button = QPushButton("Test Monitor")
        self.test_button.setEnabled(False)
        self.test_button.clicked.connect(self._on_test_device)

        self.virtual_test_button = QPushButton("Test Virtual")
        self.virtual_test_button.setEnabled(False)
        self.virtual_test_button.clicked.connect(self._on_test_virtual_device)

        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self._on_reset_defaults)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        self.ok_button = QPushButton("OK")
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self._on_ok_clicked)

        button_layout.addWidget(self.test_button)
        button_layout.addWidget(self.virtual_test_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.ok_button)

        layout.addLayout(button_layout)

    def _create_audio_tab(self):
        """Create the audio settings tab."""
        audio_tab = QWidget()
        layout = QVBoxLayout(audio_tab)

        # Monitor Device Group
        monitor_group = QGroupBox("Monitor Audio Device")
        monitor_layout = QVBoxLayout(monitor_group)

        # Device selection row
        device_layout = QHBoxLayout()

        device_layout.addWidget(QLabel("Output Device:"))

        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(250)
        self.device_combo.currentTextChanged.connect(self._on_device_selection_changed)
        device_layout.addWidget(self.device_combo, 1)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setMaximumWidth(120)
        self.refresh_button.setObjectName("settings_refresh_button")
        self.refresh_button.clicked.connect(self._on_refresh_devices)
        device_layout.addWidget(self.refresh_button)

        monitor_layout.addLayout(device_layout)

        # Device validation feedback
        self.device_status_label = QLabel("")
        self.device_status_label.setWordWrap(True)
        self.device_status_label.setProperty("class", "status-label")
        monitor_layout.addWidget(self.device_status_label)

        # Progress bar for device operations
        self.device_progress = QProgressBar()
        self.device_progress.setVisible(False)
        self.device_progress.setRange(0, 0)  # Indeterminate progress
        monitor_layout.addWidget(self.device_progress)

        layout.addWidget(monitor_group)

        # Virtual Microphone Device Group
        virtual_mic_group = QGroupBox("Virtual Microphone Device")
        virtual_mic_layout = QVBoxLayout(virtual_mic_group)

        # Virtual device selection row
        virtual_device_layout = QHBoxLayout()

        virtual_device_layout.addWidget(QLabel("Virtual Device:"))

        self.virtual_device_combo = QComboBox()
        self.virtual_device_combo.setMinimumWidth(250)
        self.virtual_device_combo.currentTextChanged.connect(self._on_virtual_device_selection_changed)
        virtual_device_layout.addWidget(self.virtual_device_combo, 1)

        self.virtual_refresh_button = QPushButton("Refresh")
        self.virtual_refresh_button.setMaximumWidth(120)
        self.virtual_refresh_button.setObjectName("settings_refresh_button")
        self.virtual_refresh_button.clicked.connect(self._on_refresh_virtual_devices)
        virtual_device_layout.addWidget(self.virtual_refresh_button)

        virtual_mic_layout.addLayout(virtual_device_layout)

        # Virtual device validation feedback
        self.virtual_device_status_label = QLabel("")
        self.virtual_device_status_label.setWordWrap(True)
        self.virtual_device_status_label.setProperty("class", "status-label")
        virtual_mic_layout.addWidget(self.virtual_device_status_label)

        # Progress bar for virtual device operations
        self.virtual_device_progress = QProgressBar()
        self.virtual_device_progress.setVisible(False)
        self.virtual_device_progress.setRange(0, 0)  # Indeterminate progress
        virtual_mic_layout.addWidget(self.virtual_device_progress)

        layout.addWidget(virtual_mic_group)

        # VB-Cable Credit
        vb_cable_group = QGroupBox("Virtual Audio Cable")
        vb_cable_layout = QVBoxLayout(vb_cable_group)

        vb_credit_label = QLabel('MyVoice would not be possible without <a href="https://vb-audio.com/Cable/index.htm" style="color: #5DADE2;">VB-Cable</a>.')
        vb_credit_label.setOpenExternalLinks(True)
        vb_credit_label.setWordWrap(True)
        vb_cable_layout.addWidget(vb_credit_label)

        vb_donation_label = QLabel("VB-CABLE is a donationware, all participations are welcome.")
        vb_donation_label.setWordWrap(True)
        vb_donation_label.setStyleSheet("color: #888; font-style: italic;")
        vb_cable_layout.addWidget(vb_donation_label)

        layout.addWidget(vb_cable_group)

        layout.addStretch()
        self.tab_widget.addTab(audio_tab, "Audio")

    def _create_voices_tab(self):
        """
        Create the Voices tab for voice creation and management.

        Story 4.2: Voice Design Creation Workflow
        - Enables users to create new custom voices using VoiceDesign model
        - Shows voice library with bundled (📦), designed (🎭), and cloned (🎤) voices
        - Provides buttons to design from description or clone from audio
        - Allows selection of active voice and deletion of user-created voices
        """
        voices_tab = QWidget()
        layout = QVBoxLayout(voices_tab)

        # Voice Library Widget (provides full voice management UI)
        self.voice_library_widget = VoiceLibraryWidget(voice_manager=None)

        # Connect voice library signals
        self.voice_library_widget.voice_selected.connect(self._on_library_voice_selected)
        self.voice_library_widget.voice_deleted.connect(self._on_library_voice_deleted)
        self.voice_library_widget.design_voice_requested.connect(self._on_design_voice_requested)
        self.voice_library_widget.refresh_requested.connect(self._on_voice_refresh_requested)

        layout.addWidget(self.voice_library_widget)

        self.tab_widget.addTab(voices_tab, "Voices")

    def _on_library_voice_selected(self, profile):
        """
        Handle voice selection from voice library widget.

        Args:
            profile: VoiceProfile that was selected
        """
        if profile:
            self.selected_voice_name = profile.name
            # Also update the voice selector in TTS tab if it exists
            if hasattr(self, 'voice_selector') and self.voice_selector:
                self.voice_selector.set_selected_voice(profile.name)
            self.logger.debug(f"Voice selected from library: {profile.name}")

    def _on_library_voice_deleted(self, voice_name: str):
        """
        Handle voice deletion from voice library widget.

        Args:
            voice_name: Name of the deleted voice
        """
        self.logger.info(f"Voice deleted from library: {voice_name}")
        # Emit signal for parent to handle
        self.voice_deleted.emit(voice_name)
        # Refresh voice selector in TTS tab
        if hasattr(self, 'voice_selector') and self.voice_selector:
            self.voice_selector.populate_voices()

    def _on_design_voice_requested(self):
        """Handle request to open Voice Design Studio."""
        self.logger.info("Voice Design Studio requested from library")
        self._open_voice_design_studio()

    def _open_voice_design_studio(self):
        """
        Open the Voice Design Studio dialog.

        Voice Design Studio provides a unified interface for voice creation:
        - From Description: Generate voices from text descriptions with 5 variations
        - From Sample: Extract voice from audio samples with emotion support

        Both paths produce identical .pt embedding files with full emotion support.
        """
        try:
            # QA4: Request whisper service initialization if not available
            # This triggers on-demand initialization in app.py so transcription works
            if not self.whisper_service:
                self.logger.debug("Whisper service not available, requesting initialization")
                parent = self.parent()
                if parent and hasattr(parent, 'whisper_init_requested'):
                    parent.whisper_init_requested.emit()
                    self.logger.debug("Emitted whisper_init_requested signal")

            dialog = VoiceDesignStudioDialog(
                tts_service=self.tts_service,
                whisper_service=self.whisper_service,
                parent=self
            )

            # Connect voice saved signal
            dialog.voice_saved.connect(self._on_voice_saved_from_studio)

            # Show dialog
            dialog.exec()

            # QA4-2: Trigger async rescan AFTER dialog closes
            # This ensures voice manager rescans disk before UI refresh
            # The signal triggers force_rescan() which then calls refresh_voice_list()
            self.voice_refresh_requested.emit()

        except Exception as e:
            self.logger.exception(f"Error opening Voice Design Studio: {e}")
            QMessageBox.warning(
                self,
                "Error",
                f"Could not open Voice Design Studio: {str(e)}"
            )

    def _on_voice_saved_from_studio(self, voice_name: str):
        """
        Handle voice saved from Voice Design Studio.

        Args:
            voice_name: Name of the saved voice
        """
        # QA3-3: Just log the save - refresh happens after dialog closes
        self.logger.info(f"Voice saved from studio: {voice_name}")

    def _on_voice_created_from_dialog(self, profile):
        """
        Handle voice creation from design or clone dialog.

        Args:
            profile: VoiceProfile that was created
        """
        try:
            self.logger.info(f"Voice created: {profile.name} (type={profile.voice_type.value})")

            # Refresh the voice library
            if hasattr(self, 'voice_library_widget') and self.voice_library_widget:
                self.voice_library_widget.refresh_voices()

            # Refresh the voice selector in TTS tab
            if hasattr(self, 'voice_selector') and self.voice_selector:
                self.voice_selector.populate_voices()

            # Select the newly created voice
            self.selected_voice_name = profile.name

            # Emit signal for parent to handle
            self.voice_created.emit(profile)

        except Exception as e:
            self.logger.exception(f"Error handling voice creation: {e}")

    def _create_tts_tab(self):
        """Create the TTS settings tab."""
        tts_tab = QWidget()
        layout = QVBoxLayout(tts_tab)

        # Voice Settings Group
        voice_group = QGroupBox("Voice Profile Settings")
        voice_layout = QFormLayout(voice_group)

        # Voice selector
        self.voice_selector = VoiceSelector(voice_manager=None)  # Will be set later
        self.voice_selector.voice_selected.connect(self._on_voice_selection_changed)
        self.voice_selector.refresh_requested.connect(self._on_voice_refresh_requested)
        self.voice_selector.transcription_edit_requested.connect(self._on_transcription_edit_requested)
        voice_layout.addRow("Active Voice Profile:", self.voice_selector)

        # Voice files directory selection
        directory_layout = QHBoxLayout()
        self.voice_directory_edit = QLineEdit()
        self.voice_directory_edit.setPlaceholderText("Select voice files directory...")
        self.voice_directory_edit.setReadOnly(True)
        self.browse_directory_button = QPushButton("Change Folder...")
        self.browse_directory_button.clicked.connect(self._on_browse_directory)
        self.open_folder_button = QPushButton("Open Folder")
        self.open_folder_button.clicked.connect(self._on_open_folder)

        directory_layout.addWidget(self.voice_directory_edit, 1)
        directory_layout.addWidget(self.open_folder_button)
        directory_layout.addWidget(self.browse_directory_button)
        voice_layout.addRow("Voice Files Directory:", directory_layout)

        # Directory status and info
        self.directory_status_label = QLabel()
        self.directory_status_label.setWordWrap(True)
        self.directory_status_label.setProperty("class", "status-label")
        voice_layout.addRow("", self.directory_status_label)

        self.refresh_interval_spin = QSpinBox()
        self.refresh_interval_spin.setRange(10, 300)
        self.refresh_interval_spin.setSuffix(" seconds")
        voice_layout.addRow("Auto-refresh Interval:", self.refresh_interval_spin)

        layout.addWidget(voice_group)

        # Custom Emotion Group (Story 3.4: FR7)
        emotion_group = QGroupBox("Custom Emotion")
        emotion_layout = QVBoxLayout(emotion_group)

        # Description label
        emotion_desc = QLabel("Create custom emotions beyond the 5 presets. Select a dynamic expression or type your own.")
        emotion_desc.setWordWrap(True)
        emotion_desc.setProperty("class", "description-label")
        emotion_layout.addWidget(emotion_desc)

        # Note about bundled voices compatibility
        emotion_note = QLabel("Note: Custom emotions will only work with bundled voices until the Qwen3-TTS model is updated (mid-2026).")
        emotion_note.setWordWrap(True)
        emotion_note.setProperty("class", "note-label")
        emotion_note.setStyleSheet("color: #888; font-style: italic; margin-top: 4px;")
        emotion_layout.addWidget(emotion_note)

        # Dynamic expressions dropdown
        expressions_layout = QHBoxLayout()
        expressions_layout.addWidget(QLabel("Expression:"))
        self.emotion_preset_combo = QComboBox()
        self.emotion_preset_combo.setMinimumWidth(200)
        self.emotion_preset_combo.addItem("(Select a preset...)")
        # Presets will be loaded from settings
        self.emotion_preset_combo.currentTextChanged.connect(self._on_emotion_preset_selected)
        expressions_layout.addWidget(self.emotion_preset_combo, 1)
        emotion_layout.addLayout(expressions_layout)

        # Freeform text field
        custom_text_layout = QHBoxLayout()
        custom_text_layout.addWidget(QLabel("Custom Text:"))
        self.custom_emotion_edit = QLineEdit()
        self.custom_emotion_edit.setPlaceholderText("e.g., 'Speak with rising excitement and energy'")
        self.custom_emotion_edit.textChanged.connect(self._on_custom_emotion_text_changed)
        custom_text_layout.addWidget(self.custom_emotion_edit, 1)
        emotion_layout.addLayout(custom_text_layout)

        # Apply button
        apply_layout = QHBoxLayout()
        apply_layout.addStretch()
        self.apply_emotion_button = QPushButton("Apply to Main Window")
        self.apply_emotion_button.setEnabled(False)  # Disabled until text is entered
        self.apply_emotion_button.setMaximumWidth(180)
        self.apply_emotion_button.clicked.connect(self._on_apply_custom_emotion)
        apply_layout.addWidget(self.apply_emotion_button)
        emotion_layout.addLayout(apply_layout)

        # Status label for feedback
        self.emotion_status_label = QLabel("")
        self.emotion_status_label.setWordWrap(True)
        self.emotion_status_label.setProperty("class", "status-label")
        emotion_layout.addWidget(self.emotion_status_label)

        layout.addWidget(emotion_group)

        layout.addStretch()
        self.tab_widget.addTab(tts_tab, "TTS")

    def _create_ui_tab(self):
        """Create the UI settings tab."""
        ui_tab = QWidget()
        layout = QVBoxLayout(ui_tab)

        # Theme Settings Group
        theme_group = QGroupBox("Appearance")
        theme_layout = QFormLayout(theme_group)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light"])
        theme_layout.addRow("Theme:", self.theme_combo)

        self.always_on_top_checkbox = QCheckBox("Keep window always on top")
        theme_layout.addRow(self.always_on_top_checkbox)

        # Story 7.3: Window Transparency (FR41)
        # Slider range: 20% to 100% (minimum 20% enforced - window never invisible)
        transparency_layout = QHBoxLayout()
        self.transparency_slider = QSlider(Qt.Orientation.Horizontal)
        self.transparency_slider.setMinimum(20)  # 20% minimum - never fully invisible
        self.transparency_slider.setMaximum(100)  # 100% fully opaque
        self.transparency_slider.setValue(100)  # Default: fully opaque
        self.transparency_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.transparency_slider.setTickInterval(10)
        self.transparency_slider.valueChanged.connect(self._on_transparency_changed)

        self.transparency_value_label = QLabel("100%")
        self.transparency_value_label.setMinimumWidth(40)

        transparency_layout.addWidget(self.transparency_slider)
        transparency_layout.addWidget(self.transparency_value_label)
        theme_layout.addRow("Window Transparency:", transparency_layout)

        layout.addWidget(theme_group)

        # Advanced Settings Group
        advanced_group = QGroupBox("Advanced")
        advanced_layout = QFormLayout(advanced_group)

        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        advanced_layout.addRow("Log Level:", self.log_level_combo)

        layout.addWidget(advanced_group)

        layout.addStretch()
        self.tab_widget.addTab(ui_tab, "Interface")

    def _create_quick_speak_tab(self):
        """Create the Quick Speak settings tab."""
        self.quick_speak_widget = QuickSpeakSettingsWidget(self.quick_speak_service)

        # Forward entries_changed signal to parent
        self.quick_speak_widget.entries_changed.connect(self.quick_speak_entries_changed.emit)

        self.tab_widget.addTab(self.quick_speak_widget, "Quick Speak")

    def _mark_settings_modified(self):
        """Mark settings as modified (for future save tracking)."""
        # Currently settings are applied on OK, but this could be used
        # for "Apply" button or real-time saving
        pass

    def _load_current_settings(self):
        """Load current settings into UI controls."""
        try:
            # Audio settings - monitoring checkbox removed, was only for debugging

            # Voice settings
            self.refresh_interval_spin.setValue(self.current_settings.auto_refresh_interval)

            # Voice directory settings
            self.voice_directory_edit.setText(self.current_settings.voice_files_directory)
            if self.current_settings.voice_files_directory:
                self._validate_voice_directory(self.current_settings.voice_files_directory)

            # UI settings
            theme_index = self.theme_combo.findText(self.current_settings.ui_theme)
            if theme_index >= 0:
                self.theme_combo.setCurrentIndex(theme_index)

            self.always_on_top_checkbox.setChecked(self.current_settings.always_on_top)

            # Story 7.3: Window Transparency (FR41)
            # Convert opacity (0.0-1.0) to percentage (20-100)
            transparency_percent = int(self.current_settings.window_transparency * 100)
            # Enforce minimum 20%
            transparency_percent = max(20, min(100, transparency_percent))
            self.transparency_slider.setValue(transparency_percent)
            self.transparency_value_label.setText(f"{transparency_percent}%")

            log_level_index = self.log_level_combo.findText(self.current_settings.log_level)
            if log_level_index >= 0:
                self.log_level_combo.setCurrentIndex(log_level_index)

            # Custom emotion settings (Story 3.4)
            self._load_custom_emotion_presets()
            if self.current_settings.custom_emotion_text:
                self.custom_emotion_edit.setText(self.current_settings.custom_emotion_text)

            self.logger.debug("Loaded current settings into UI")

        except Exception as e:
            self.logger.error(f"Error loading settings into UI: {e}")

    def _save_current_settings(self):
        """Save UI values to current settings."""
        try:
            # Audio settings - monitoring and volume settings removed

            # Get selected device ID and metadata
            current_device_id = self.device_combo.currentData()
            self.current_settings.monitor_device_id = current_device_id

            # Capture device metadata for persistence
            if current_device_id:
                selected_device = None
                for device in self.output_devices:
                    if device.device_id == current_device_id:
                        selected_device = device
                        break

                if selected_device:
                    self.current_settings.monitor_device_name = selected_device.name
                    self.current_settings.monitor_device_host_api = getattr(selected_device, 'host_api_name', None)
                    self.logger.debug(f"Saved monitor device metadata: {selected_device.name} ({self.current_settings.monitor_device_host_api})")
            else:
                # Clear metadata if no device selected (system default)
                self.current_settings.monitor_device_name = None
                self.current_settings.monitor_device_host_api = None

            # Get selected virtual device ID and metadata
            current_virtual_device_id = self.virtual_device_combo.currentData()
            self.current_settings.virtual_microphone_device_id = current_virtual_device_id

            # Capture virtual device metadata for persistence
            if current_virtual_device_id:
                selected_virtual_device = None
                for device in self.virtual_input_devices:
                    if device.device_id == current_virtual_device_id:
                        selected_virtual_device = device
                        break

                if selected_virtual_device:
                    self.current_settings.virtual_microphone_device_name = selected_virtual_device.name
                    self.current_settings.virtual_microphone_device_host_api = getattr(selected_virtual_device, 'host_api_name', None)
                    self.logger.debug(f"Saved virtual device metadata: {selected_virtual_device.name} ({self.current_settings.virtual_microphone_device_host_api})")
            else:
                # Clear metadata if no virtual device selected
                self.current_settings.virtual_microphone_device_name = None
                self.current_settings.virtual_microphone_device_host_api = None

            # Voice settings
            self.current_settings.auto_refresh_interval = self.refresh_interval_spin.value()

            # Voice directory settings
            voice_directory = self.voice_directory_edit.text().strip()
            if voice_directory:
                self.current_settings.voice_files_directory = voice_directory

            # UI settings
            self.current_settings.ui_theme = self.theme_combo.currentText()
            self.current_settings.always_on_top = self.always_on_top_checkbox.isChecked()

            # Story 7.3: Window Transparency (FR41)
            # Convert percentage (20-100) to opacity (0.2-1.0)
            self.current_settings.window_transparency = self.transparency_slider.value() / 100.0

            self.current_settings.log_level = self.log_level_combo.currentText()

            # Custom emotion settings (Story 3.4)
            custom_text = self.custom_emotion_edit.text().strip()
            self.current_settings.custom_emotion_text = custom_text if custom_text else None

            self.logger.debug("Saved UI values to current settings")

        except Exception as e:
            self.logger.error(f"Error saving settings from UI: {e}")
            raise

    def update_device_list(self, devices: List[AudioDevice]):
        """
        Update the device dropdown with available devices.

        Args:
            devices: List of audio devices to populate dropdown
        """
        try:
            self.output_devices = [d for d in devices if d.device_type.name in ['OUTPUT', 'VIRTUAL']]

            # Store current selection metadata for smart matching
            current_device_id = self.current_settings.monitor_device_id
            current_device_name = self.current_settings.monitor_device_name
            current_host_api = self.current_settings.monitor_device_host_api

            # Clear and repopulate combo box
            self.device_combo.clear()

            # Add default option
            self.device_combo.addItem("(System Default)", None)

            # Add available devices
            for device in self.output_devices:
                # Only add available devices to the dropdown
                if not device.is_available:
                    continue

                display_name = f"{device.name}"
                if device.is_default:
                    display_name += " (Default)"
                if device.device_type.name == 'VIRTUAL':
                    display_name += " [Virtual]"

                self.device_combo.addItem(display_name, device.device_id)

            # Restore selection using smart device matching
            if current_device_id or current_device_name:
                matched_device = None

                # Try smart matching if audio_client is available
                if self.audio_client and hasattr(self.audio_client, 'find_device_by_metadata'):
                    matched_device = self.audio_client.find_device_by_metadata(
                        device_id=current_device_id,
                        device_name=current_device_name,
                        host_api_name=current_host_api
                    )

                # Find the matched device in combo box
                if matched_device:
                    index = self.device_combo.findData(matched_device.device_id)
                    if index >= 0:
                        self.device_combo.setCurrentIndex(index)
                        self._validate_selected_device()
                        self.logger.debug(f"Restored monitor device using smart matching: {matched_device.name}")
                    else:
                        # Fallback to exact ID match
                        index = self.device_combo.findData(current_device_id)
                        if index >= 0:
                            self.device_combo.setCurrentIndex(index)
                            self._validate_selected_device()
                        else:
                            self._show_device_status("Previously selected device not found", "warning")
                else:
                    # Fallback: Try exact device_id match
                    if current_device_id:
                        index = self.device_combo.findData(current_device_id)
                        if index >= 0:
                            self.device_combo.setCurrentIndex(index)
                            self._validate_selected_device()
                        else:
                            self._show_device_status("Previously selected device not found", "warning")

            self.logger.info(f"Updated device list with {len(self.output_devices)} devices")

        except Exception as e:
            self.logger.error(f"Error updating device list: {e}")
            self._show_device_status("Error updating device list", "error")

    def update_virtual_device_list(self, virtual_devices: List[AudioDevice]):
        """
        Update the virtual device dropdown with available virtual input devices.

        Args:
            virtual_devices: List of virtual input devices to populate dropdown
        """
        try:
            self.virtual_input_devices = virtual_devices

            # Store current selection metadata for smart matching
            current_virtual_device_id = self.current_settings.virtual_microphone_device_id
            current_virtual_device_name = self.current_settings.virtual_microphone_device_name
            current_virtual_host_api = self.current_settings.virtual_microphone_device_host_api

            # Clear and repopulate combo box
            self.virtual_device_combo.clear()

            # Add default option
            self.virtual_device_combo.addItem("(None - Disable dual routing)", None)

            if not virtual_devices:
                self.virtual_device_combo.addItem("(No virtual devices found)", None)
                self.virtual_device_combo.setEnabled(False)
                self._show_virtual_device_status("No virtual microphone devices found. Install VB-Cable or Voicemeeter for dual routing.", "warning")
            else:
                self.virtual_device_combo.setEnabled(True)

                # Add available virtual devices
                for device in virtual_devices:
                    # Only add available virtual devices to the dropdown
                    if not device.is_available:
                        continue

                    display_name = f"{device.name}"
                    if device.is_default:
                        display_name += " (Default)"

                    # Add device type information
                    if hasattr(device, 'driver_name') and device.driver_name:
                        display_name += f" [{device.driver_name}]"

                    self.virtual_device_combo.addItem(display_name, device.device_id)

                # Restore selection using smart device matching
                if current_virtual_device_id or current_virtual_device_name:
                    matched_device = None

                    # Try smart matching if audio_client is available
                    if self.audio_client and hasattr(self.audio_client, 'find_device_by_metadata'):
                        matched_device = self.audio_client.find_device_by_metadata(
                            device_id=current_virtual_device_id,
                            device_name=current_virtual_device_name,
                            host_api_name=current_virtual_host_api
                        )

                    # Find the matched device in combo box
                    if matched_device:
                        index = self.virtual_device_combo.findData(matched_device.device_id)
                        if index >= 0:
                            self.virtual_device_combo.setCurrentIndex(index)
                            self._validate_selected_virtual_device()
                            self.logger.debug(f"Restored virtual device using smart matching: {matched_device.name}")
                        else:
                            # Fallback to exact ID match
                            index = self.virtual_device_combo.findData(current_virtual_device_id)
                            if index >= 0:
                                self.virtual_device_combo.setCurrentIndex(index)
                                self._validate_selected_virtual_device()
                            else:
                                self._show_virtual_device_status("Previously selected virtual device not found", "warning")
                    else:
                        # Fallback: Try exact device_id match
                        if current_virtual_device_id:
                            index = self.virtual_device_combo.findData(current_virtual_device_id)
                            if index >= 0:
                                self.virtual_device_combo.setCurrentIndex(index)
                                self._validate_selected_virtual_device()
                            else:
                                self._show_virtual_device_status("Previously selected virtual device not found", "warning")
                        else:
                            self._show_virtual_device_status("Select a virtual device to enable dual routing to communication apps", "info")
                else:
                    self._show_virtual_device_status("Select a virtual device to enable dual routing to communication apps", "info")

            self.logger.info(f"Updated virtual device list with {len(virtual_devices)} devices")

        except Exception as e:
            self.logger.error(f"Error updating virtual device list: {e}")
            self._show_virtual_device_status("Error updating virtual device list", "error")

    def set_voice_manager(self, voice_manager: VoiceProfileManager):
        """
        Set the voice profile manager and populate voice selector.

        Args:
            voice_manager: VoiceProfileManager instance
        """
        self.voice_manager = voice_manager

        # Update voice selector in TTS tab
        self.voice_selector.voice_manager = voice_manager
        self.voice_selector.populate_voices()

        # Update voice library widget in Voices tab
        if hasattr(self, 'voice_library_widget') and self.voice_library_widget:
            self.voice_library_widget.set_voice_manager(voice_manager)

        # Set current active voice
        if voice_manager:
            active_profile = voice_manager.get_active_profile()
            if active_profile:
                self.voice_selector.set_selected_voice(active_profile.name)
                self.selected_voice_name = active_profile.name
                # Also select in voice library
                if hasattr(self, 'voice_library_widget') and self.voice_library_widget:
                    self.voice_library_widget.select_voice(active_profile.name)

    def set_tts_service(self, tts_service):
        """
        Set the TTS service for voice creation dialogs.

        Args:
            tts_service: QwenTTSService instance
        """
        self.tts_service = tts_service
        tts_available = tts_service is not None

        # Enable voice creation buttons based on TTS availability
        if hasattr(self, 'voice_library_widget') and self.voice_library_widget:
            self.voice_library_widget.set_tts_available(tts_available)
            self.logger.info(f"Voice library buttons {'enabled' if tts_available else 'disabled'} (TTS {'available' if tts_available else 'not available'})")
        else:
            self.logger.warning("voice_library_widget not available when setting TTS service")

        self.logger.debug(f"TTS service set for settings dialog: {'available' if tts_available else 'None'}")

    def set_whisper_service(self, whisper_service):
        """
        Set the Whisper service for transcription in Voice Design Studio.

        Args:
            whisper_service: WhisperService instance for transcription
        """
        self.whisper_service = whisper_service
        self.logger.debug(f"Whisper service set for settings dialog: {'available' if whisper_service else 'None'}")

    def _on_voice_selection_changed(self, voice_name: str):
        """
        Handle voice selection change in the dialog.

        Args:
            voice_name: Name of the selected voice profile
        """
        self.selected_voice_name = voice_name
        self.logger.debug(f"Voice selection changed to: {voice_name}")

    def _on_voice_refresh_requested(self):
        """Handle voice refresh request from voice selector - emit signal for app to handle."""
        self.logger.info("Voice refresh requested from settings dialog")
        # Emit signal for app to handle async refresh
        self.voice_refresh_requested.emit()

    def _on_transcription_edit_requested(self, voice_name: str):
        """
        Handle transcription edit/generation request from voice selector - emit signal for app to handle.

        Args:
            voice_name: Name of the voice profile to transcribe
        """
        self.logger.info(f"Transcription requested for voice: {voice_name}")
        # Emit signal for app to handle async transcription
        self.voice_transcription_requested.emit(voice_name)

    def refresh_voice_list(self):
        """Refresh the voice selector and library after external updates."""
        if self.voice_selector:
            self.voice_selector.populate_voices()
            self.logger.debug("Voice selector refreshed")

        if hasattr(self, 'voice_library_widget') and self.voice_library_widget:
            self.voice_library_widget.refresh_voices()
            self.logger.debug("Voice library widget refreshed")

    def _on_device_selection_changed(self):
        """Handle device selection change."""
        self._validate_selected_device()
        self._validate_device_conflicts()
        # Always enable test button - default device selection is valid
        self.test_button.setEnabled(True)

    def _validate_selected_device(self):
        """Validate the currently selected device."""
        try:
            device_id = self.device_combo.currentData()

            if device_id is None:
                self._show_device_status("Using system default audio device", "info")
                return

            # Find device info
            selected_device = None
            for device in self.output_devices:
                if device.device_id == device_id:
                    selected_device = device
                    break

            if selected_device:
                if selected_device.is_available:
                    status_msg = f"Device available: {selected_device.name}"
                    if hasattr(selected_device, 'driver_name') and selected_device.driver_name:
                        status_msg += f" ({selected_device.driver_name})"
                    self._show_device_status(status_msg, "success")
                else:
                    self._show_device_status("Device not currently available", "warning")
            else:
                self._show_device_status("Device not found", "error")

        except Exception as e:
            self.logger.error(f"Error validating device: {e}")
            self._show_device_status("Error validating device", "error")

    def _on_virtual_device_selection_changed(self):
        """Handle virtual device selection change."""
        self._validate_selected_virtual_device()
        self._validate_device_conflicts()
        self.virtual_test_button.setEnabled(self.virtual_device_combo.currentData() is not None)

    def _validate_selected_virtual_device(self):
        """Validate the currently selected virtual device."""
        try:
            device_id = self.virtual_device_combo.currentData()

            if device_id is None:
                self._show_virtual_device_status("Dual routing disabled. TTS will only output to monitor speakers.", "info")
                return

            # Find device info
            selected_device = None
            for device in self.virtual_input_devices:
                if device.device_id == device_id:
                    selected_device = device
                    break

            if selected_device:
                if selected_device.is_available:
                    status_msg = f"Virtual device available: {selected_device.name}"
                    if hasattr(selected_device, 'driver_name') and selected_device.driver_name:
                        status_msg += f" ({selected_device.driver_name})"
                    status_msg += " - Dual routing enabled."
                    self._show_virtual_device_status(status_msg, "success")
                else:
                    self._show_virtual_device_status("Virtual device not currently available", "warning")
            else:
                self._show_virtual_device_status("Virtual device not found", "error")

        except Exception as e:
            self.logger.error(f"Error validating virtual device: {e}")
            self._show_virtual_device_status("Error validating virtual device", "error")

    def _show_device_status(self, message: str, status_type: str = "info"):
        """
        Show device validation feedback to user.

        Args:
            message: Status message to display
            status_type: Type of status (info, success, warning, error)
        """
        colors = {
            "info": "#666",
            "success": "#28a745",
            "warning": "#ffc107",
            "error": "#dc3545"
        }

        # Set CSS class based on status type for theme-based styling
        self.device_status_label.setText(message)
        self.device_status_label.setProperty("status", status_type)

    def _show_virtual_device_status(self, message: str, status_type: str = "info"):
        """
        Show virtual device validation feedback to user.

        Args:
            message: Status message to display
            status_type: Type of status (info, success, warning, error)
        """
        colors = {
            "info": "#666",
            "success": "#28a745",
            "warning": "#ffc107",
            "error": "#dc3545"
        }

        # Set CSS class based on status type for theme-based styling
        self.virtual_device_status_label.setText(message)
        self.virtual_device_status_label.setProperty("status", status_type)

    def _validate_device_conflicts(self):
        """
        Validate device selections for conflicts.

        Checks if the same device is selected for both monitor output and virtual microphone,
        which could cause audio feedback loops.
        """
        try:
            monitor_device_id = self.device_combo.currentData()
            virtual_device_id = self.virtual_device_combo.currentData()

            # Clear any existing conflict warnings first
            if hasattr(self, 'conflict_status_label'):
                self.conflict_status_label.setText("")
                self.conflict_status_label.hide()

            # Skip validation if either device is not selected or is None
            if not monitor_device_id or not virtual_device_id:
                return

            # Check for device conflicts
            if monitor_device_id == virtual_device_id:
                self._show_conflict_warning(
                    "⚠️ Warning: Same device selected for monitor output and virtual microphone. "
                    "This may cause audio feedback loops or routing conflicts."
                )
                return

            # Check for potential virtual device conflicts with system defaults
            if monitor_device_id == "default" and virtual_device_id:
                # Find if virtual device might conflict with system default
                selected_virtual_device = None
                for device in getattr(self, 'virtual_input_devices', []):
                    if device.device_id == virtual_device_id:
                        selected_virtual_device = device
                        break

                if selected_virtual_device and hasattr(selected_virtual_device, 'driver_name'):
                    if selected_virtual_device.driver_name and 'cable' in selected_virtual_device.driver_name.lower():
                        self._show_conflict_warning(
                            "ℹ️ Note: Using system default output with virtual cable. "
                            "Ensure your default audio device differs from the virtual cable endpoint."
                        )

        except Exception as e:
            self.logger.error(f"Error validating device conflicts: {e}")

    def _show_conflict_warning(self, message: str):
        """
        Show device conflict warning to user.

        Args:
            message: Conflict warning message to display
        """
        # Create conflict status label if it doesn't exist
        if not hasattr(self, 'conflict_status_label'):
            # Find the audio tab to add the conflict warning
            audio_tab = self.tab_widget.widget(0)  # Audio tab is first
            layout = audio_tab.layout()

            # Create conflict warning area
            self.conflict_status_label = QLabel()
            self.conflict_status_label.setWordWrap(True)
            self.conflict_status_label.setProperty("class", "conflict-warning")
            self.conflict_status_label.setContentsMargins(5, 5, 5, 5)

            # Insert before the stretch at the bottom
            layout.insertWidget(layout.count() - 1, self.conflict_status_label)

        self.conflict_status_label.setText(message)
        self.conflict_status_label.show()

    def _on_refresh_devices(self):
        """Handle device refresh button click."""
        if self.device_refresh_in_progress:
            return

        self.device_refresh_in_progress = True
        self.refresh_button.setEnabled(False)
        self.device_progress.setVisible(True)
        self._show_device_status("Refreshing device list...", "info")

        # Emit signal for parent to handle
        self.device_refresh_requested.emit()

        # Re-enable button after delay
        QTimer.singleShot(2000, self._refresh_complete)

    def _on_refresh_virtual_devices(self):
        """Handle virtual device refresh button click."""
        if self.device_refresh_in_progress:
            return

        self.device_refresh_in_progress = True
        self.virtual_refresh_button.setEnabled(False)
        self.virtual_device_progress.setVisible(True)
        self._show_virtual_device_status("Refreshing virtual device list...", "info")

        # Emit signal for parent to handle
        self.device_refresh_requested.emit()

        # Re-enable button after delay
        QTimer.singleShot(2000, self._virtual_refresh_complete)

    def _virtual_refresh_complete(self):
        """Complete virtual device refresh operation."""
        self.device_refresh_in_progress = False
        self.virtual_refresh_button.setEnabled(True)
        self.virtual_device_progress.setVisible(False)

    def _refresh_complete(self):
        """Complete device refresh operation."""
        self.device_refresh_in_progress = False
        self.refresh_button.setEnabled(True)
        self.device_progress.setVisible(False)

    def showEvent(self, event):
        """
        Handle dialog show event.

        Story 2.5 AC4: New devices detected when settings opened
        Automatically refresh the device list when settings dialog is opened.
        """
        super().showEvent(event)

        # Auto-refresh devices when dialog is shown
        if not self.device_refresh_in_progress:
            self.logger.info("Settings dialog opened - triggering device refresh")
            # Use a small delay to ensure UI is ready
            QTimer.singleShot(100, self._auto_refresh_devices_on_show)

    def _auto_refresh_devices_on_show(self):
        """Auto-refresh devices when dialog is shown (Story 2.5)."""
        if self.device_refresh_in_progress:
            return

        self.device_refresh_in_progress = True
        self._show_device_status("Detecting audio devices...", "info")
        self._show_virtual_device_status("Detecting virtual devices...", "info")

        # Emit signal for parent to handle device refresh
        self.device_refresh_requested.emit()

        # Reset state after brief delay
        QTimer.singleShot(1500, self._auto_refresh_complete)

    def _auto_refresh_complete(self):
        """Complete auto-refresh on dialog show."""
        self.device_refresh_in_progress = False
        self._show_device_status("Device list updated", "success")
        self._show_virtual_device_status("Virtual device list updated", "success")

        # Clear status after delay
        QTimer.singleShot(2000, lambda: self._show_device_status("", ""))
        QTimer.singleShot(2000, lambda: self._show_virtual_device_status("", ""))

    def _on_test_device(self):
        """Handle test device button click."""
        device_id = self.device_combo.currentData()
        # Emit "default" string for system default, or actual device_id
        # None (system default) is a valid selection and should be testable
        test_id = device_id if device_id is not None else "default"
        self.device_test_requested.emit(test_id)
        self._show_device_status("Testing device...", "info")

    def _on_test_virtual_device(self):
        """Handle virtual device test button click."""
        device_id = self.virtual_device_combo.currentData()
        if device_id:
            self.virtual_device_test_requested.emit(device_id)
            self._show_virtual_device_status("Testing virtual device...", "info")

    def _on_reset_defaults(self):
        """
        Handle reset to defaults button click (Story 6.4: FR48).

        Resets:
        - Devices to system default
        - Voice to default bundled (Sarira-F)
        - Position centered (cleared window_geometry)
        - always_on_top true
        - transparency 100%
        - Quick Speak cleared (optional - with confirmation)

        Does NOT delete user voices.
        """
        # Show confirmation dialog with details
        msg = QMessageBox(self)
        msg.setWindowTitle("Reset Settings")
        msg.setText("Are you sure you want to reset all settings to their default values?")
        msg.setInformativeText(
            "This will reset:\n"
            "• Audio devices to system default\n"
            "• Voice to default bundled voice\n"
            "• Window position (will be centered)\n"
            "• Always on top: enabled\n"
            "• Transparency: 100%\n\n"
            "Note: Your voice files will NOT be deleted."
        )
        msg.setIcon(QMessageBox.Icon.Question)

        # Add checkbox for Quick Speak reset
        checkbox = QCheckBox("Also clear Quick Speak phrases")
        msg.setCheckBox(checkbox)

        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.setDefaultButton(QMessageBox.StandardButton.No)

        if msg.exec() == QMessageBox.StandardButton.Yes:
            try:
                # Reset AppSettings to defaults
                self.current_settings.reset_to_defaults()

                # Story 6.4: Set voice to default bundled
                self.current_settings.selected_voice_profile = "Sarira-F"

                # Story 6.4: Clear Quick Speak if checkbox was checked
                if checkbox.isChecked():
                    self._reset_quick_speak_entries()

                # Reload settings into UI
                self._load_current_settings()

                # Reset device combos to system default
                self.device_combo.setCurrentIndex(0)  # "(System Default)"
                self.virtual_device_combo.setCurrentIndex(0)  # "(None)"

                # Reset voice selector to default bundled
                if self.voice_selector:
                    self.voice_selector.set_selected_voice("Sarira-F")
                    self.selected_voice_name = "Sarira-F"

                # Show success message
                self._show_device_status("Settings reset to defaults", "success")

                self.logger.info("Settings reset to defaults (Story 6.4)")

            except Exception as e:
                self.logger.error(f"Error resetting settings: {e}")
                QMessageBox.warning(
                    self,
                    "Reset Failed",
                    f"Failed to reset some settings:\n{str(e)}"
                )

    def _reset_quick_speak_entries(self):
        """
        Reset Quick Speak entries to defaults (Story 6.4).

        Creates a fresh default profile with sample entries.
        """
        try:
            if not self.quick_speak_service:
                return

            # Create a fresh default profile
            self.quick_speak_service._create_default_profile()
            self.quick_speak_service.load_entries()

            # Refresh the Quick Speak widget if it exists
            if hasattr(self, 'quick_speak_widget') and self.quick_speak_widget:
                self.quick_speak_widget.refresh_entries()

            self.logger.info("Quick Speak entries reset to defaults")

        except Exception as e:
            self.logger.error(f"Error resetting Quick Speak entries: {e}")

    def _on_ok_clicked(self):
        """Handle OK button click."""
        try:
            # Save current UI values
            self._save_current_settings()

            # Validate device conflicts before applying settings
            conflict_errors = self._validate_settings_conflicts()
            if conflict_errors:
                error_msg = "Device configuration conflicts detected:\n\n"
                for error in conflict_errors:
                    error_msg += f"• {error}\n"
                error_msg += "\nPlease resolve these conflicts before applying settings."

                QMessageBox.warning(self, "Device Conflicts Detected", error_msg)
                return

            # Validate settings using AppSettings validation
            validation = self.current_settings.validate()
            if not validation.is_valid:
                error_msg = "Settings validation failed:\n\n"
                for issue in validation.issues:
                    error_msg += f"• {issue.message}\n"

                QMessageBox.warning(self, "Invalid Settings", error_msg)
                return

            # Additional validation warnings (non-blocking)
            warnings = []
            if validation.warnings:
                for warning in validation.warnings:
                    warnings.append(warning.message)

            # Show warnings if any (but don't prevent application)
            if warnings:
                warning_msg = "Settings applied with warnings:\n\n"
                for warning in warnings:
                    warning_msg += f"• {warning}\n"
                warning_msg += "\nSettings have been applied, but please review these warnings."

                QMessageBox.information(self, "Settings Applied with Warnings", warning_msg)

            # Apply theme if it changed
            self._apply_theme_if_changed()

            # Emit voice change if voice selection changed
            if self.selected_voice_name:
                self.voice_changed.emit(self.selected_voice_name)

            # Emit settings changed signal
            self.settings_changed.emit(self.current_settings)

            # Accept dialog
            self.accept()

        except Exception as e:
            self.logger.error(f"Error applying settings: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to apply settings:\n{str(e)}"
            )

    def _validate_settings_conflicts(self) -> List[str]:
        """
        Validate settings for device conflicts.

        Returns:
            List[str]: List of conflict error messages, empty if no conflicts
        """
        conflicts = []

        try:
            monitor_device_id = self.device_combo.currentData()
            virtual_device_id = self.virtual_device_combo.currentData()

            # Check for direct device ID conflicts
            if monitor_device_id and virtual_device_id and monitor_device_id == virtual_device_id:
                conflicts.append(
                    "Same device selected for both monitor output and virtual microphone. "
                    "This will cause audio feedback loops."
                )

            # Check for device availability conflicts
            if monitor_device_id and monitor_device_id != "default":
                selected_device = None
                for device in getattr(self, 'output_devices', []):
                    if device.device_id == monitor_device_id:
                        selected_device = device
                        break

                if selected_device and not selected_device.is_available:
                    conflicts.append(
                        f"Selected monitor device '{selected_device.name}' is not currently available."
                    )

            if virtual_device_id:
                selected_virtual_device = None
                for device in getattr(self, 'virtual_input_devices', []):
                    if device.device_id == virtual_device_id:
                        selected_virtual_device = device
                        break

                if selected_virtual_device and not selected_virtual_device.is_available:
                    conflicts.append(
                        f"Selected virtual device '{selected_virtual_device.name}' is not currently available."
                    )

        except Exception as e:
            self.logger.error(f"Error validating settings conflicts: {e}")
            conflicts.append(f"Error during conflict validation: {str(e)}")

        return conflicts

    def _on_browse_directory(self):
        """Handle browse directory button click."""
        try:
            current_directory = self.voice_directory_edit.text()
            if current_directory and Path(current_directory).exists():
                start_directory = current_directory
            else:
                start_directory = str(Path.cwd())

            directory = QFileDialog.getExistingDirectory(
                self,
                "Select Voice Files Directory",
                start_directory,
                QFileDialog.Option.ShowDirsOnly
            )

            if directory:
                self.voice_directory_edit.setText(directory)
                self._validate_voice_directory(directory)
                # Emit signal for directory change handling
                self.voice_directory_changed.emit(directory)

        except Exception as e:
            self.logger.error(f"Error browsing directory: {e}")
            self._show_directory_status("Error browsing directory", "error")

    def _on_open_folder(self):
        """Open the voice files directory in Windows Explorer."""
        try:
            directory = self.voice_directory_edit.text()
            if not directory:
                self._show_directory_status("No directory selected", "warning")
                return

            directory_path = Path(directory)
            if not directory_path.exists():
                self._show_directory_status("Directory does not exist", "error")
                return

            # Open the folder in Windows Explorer
            if os.name == 'nt':  # Windows
                os.startfile(directory_path)
            else:  # macOS/Linux fallback
                subprocess.run(['xdg-open', str(directory_path)])

            self.logger.debug(f"Opened folder: {directory}")

        except Exception as e:
            self.logger.error(f"Error opening folder: {e}")
            self._show_directory_status("Error opening folder", "error")

    def _validate_voice_directory(self, directory_path: str):
        """
        Validate the selected voice files directory.

        Args:
            directory_path: Path to the directory to validate
        """
        try:
            from pathlib import Path

            directory = Path(directory_path)

            # Check if directory exists
            if not directory.exists():
                self._show_directory_status("Directory does not exist", "error")
                return False

            # Check if it's actually a directory
            if not directory.is_dir():
                self._show_directory_status("Path is not a directory", "error")
                return False

            # Check if directory is readable
            try:
                list(directory.iterdir())
            except PermissionError:
                self._show_directory_status("Directory is not readable", "error")
                return False

            # Check if directory is writable
            try:
                test_file = directory / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
                is_writable = True
            except (PermissionError, OSError):
                is_writable = False

            # Scan for voice files
            voice_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
            voice_files = []

            try:
                for file_path in directory.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in voice_extensions:
                        voice_files.append(file_path)
            except Exception as e:
                self.logger.warning(f"Error scanning directory: {e}")

            # Show validation results
            if not voice_files:
                if is_writable:
                    self._show_directory_status(
                        f"Directory is valid but empty. You can add voice files (.wav, .mp3, .flac, .ogg, .m4a) to this directory.",
                        "warning"
                    )
                else:
                    self._show_directory_status(
                        f"Directory is valid but empty and read-only. Consider choosing a writable directory.",
                        "warning"
                    )
            else:
                status_parts = [f"✓ Found {len(voice_files)} voice file(s)"]
                if is_writable:
                    status_parts.append("writable")
                else:
                    status_parts.append("read-only")

                self._show_directory_status(" • ".join(status_parts), "success")

            return True

        except Exception as e:
            self.logger.error(f"Error validating voice directory: {e}")
            self._show_directory_status(f"Validation error: {str(e)}", "error")
            return False

    def _show_directory_status(self, message: str, status_type: str = "info"):
        """
        Show directory validation status to user.

        Args:
            message: Status message to display
            status_type: Type of status (info, success, warning, error)
        """
        colors = {
            "info": "#666",
            "success": "#28a745",
            "warning": "#ffc107",
            "error": "#dc3545"
        }

        # Set CSS class based on status type for theme-based styling
        self.directory_status_label.setText(message)
        self.directory_status_label.setProperty("status", status_type)

    def _apply_theme_if_changed(self):
        """Apply theme if it has changed from the original setting."""
        try:
            current_theme = self.current_settings.ui_theme
            original_theme = self.original_settings.ui_theme

            if current_theme != original_theme:
                self.logger.info(f"Applying theme change from '{original_theme}' to '{current_theme}'")

                # Clear cache and apply theme application-wide to ensure latest changes
                self.theme_manager.clear_cache()
                success = self.theme_manager.apply_theme(current_theme)

                if success:
                    self.logger.info(f"Successfully applied theme: {current_theme}")
                else:
                    self.logger.warning(f"Failed to apply theme: {current_theme}")
                    QMessageBox.warning(
                        self,
                        "Theme Application Failed",
                        f"Failed to apply theme '{current_theme}'. The application will continue with the current theme."
                    )
            else:
                self.logger.debug("Theme unchanged, no application needed")

        except Exception as e:
            self.logger.error(f"Error applying theme: {e}")
            QMessageBox.warning(
                self,
                "Theme Error",
                f"An error occurred while applying the theme: {str(e)}"
            )

    def get_current_settings(self) -> AppSettings:
        """
        Get the current settings from the dialog.

        Returns:
            AppSettings: Current settings with UI values applied
        """
        self._save_current_settings()
        return self.current_settings

    # =========================================================================
    # Custom Emotion Methods (Story 3.4: FR7)
    # =========================================================================

    def _load_custom_emotion_presets(self):
        """Load custom emotion presets from settings into the dropdown."""
        try:
            self.emotion_preset_combo.clear()
            self.emotion_preset_combo.addItem("(Select a preset...)")

            # Load presets from settings
            presets = self.current_settings.custom_emotion_presets
            for preset in presets:
                self.emotion_preset_combo.addItem(preset)

            self.logger.debug(f"Loaded {len(presets)} custom emotion presets")

        except Exception as e:
            self.logger.error(f"Error loading custom emotion presets: {e}")

    def _on_emotion_preset_selected(self, text: str):
        """
        Handle selection of a preset expression from the dropdown.

        Args:
            text: The selected preset text
        """
        try:
            # Ignore placeholder selection
            if text.startswith("(Select"):
                return

            # Populate the text field with the selected preset
            self.custom_emotion_edit.setText(f"Speak with {text.lower()}")
            self.logger.debug(f"Selected emotion preset: {text}")

        except Exception as e:
            self.logger.error(f"Error handling emotion preset selection: {e}")

    def _on_custom_emotion_text_changed(self, text: str):
        """
        Handle changes to the custom emotion text field.

        Args:
            text: The current text in the field
        """
        try:
            # Enable/disable apply button based on text content
            has_text = bool(text.strip())
            self.apply_emotion_button.setEnabled(has_text)

            # Clear status when text changes
            self.emotion_status_label.setText("")

        except Exception as e:
            self.logger.error(f"Error handling custom emotion text change: {e}")

    def _on_apply_custom_emotion(self):
        """Handle Apply to Main Window button click."""
        try:
            custom_text = self.custom_emotion_edit.text().strip()

            if not custom_text:
                self._show_emotion_status("Please enter custom emotion text", "warning")
                return

            # Save to settings
            self.current_settings.custom_emotion_text = custom_text

            # Emit signal to notify main window
            self.custom_emotion_applied.emit(custom_text)

            # Show success feedback
            self._show_emotion_status("Custom emotion applied to main window", "success")

            self.logger.info(f"Applied custom emotion: {custom_text}")

            # Close the dialog after applying
            self.accept()

        except Exception as e:
            self.logger.error(f"Error applying custom emotion: {e}")
            self._show_emotion_status("Error applying custom emotion", "error")

    def _show_emotion_status(self, message: str, status_type: str = "info"):
        """
        Show emotion configuration feedback to user.

        Args:
            message: Status message to display
            status_type: Type of status (info, success, warning, error)
        """
        self.emotion_status_label.setText(message)
        self.emotion_status_label.setProperty("status", status_type)

    def navigate_to_custom_emotion(self):
        """Navigate to the Custom Emotion section in Settings (called from main window)."""
        try:
            # Switch to TTS tab (index 2 - after Audio, Voices)
            self.tab_widget.setCurrentIndex(2)
            # Focus the custom emotion text field
            self.custom_emotion_edit.setFocus()
            self.logger.debug("Navigated to Custom Emotion section")
        except Exception as e:
            self.logger.error(f"Error navigating to custom emotion section: {e}")

    def navigate_to_quick_speak(self):
        """
        Navigate to the Quick Speak tab in Settings (Story 5.2).

        Called from main window when user clicks configure in Quick Speak menu.
        """
        try:
            # Switch to Quick Speak tab (index 4 - after Audio, Voices, TTS, Interface)
            self.tab_widget.setCurrentIndex(4)
            self.logger.debug("Navigated to Quick Speak tab")
        except Exception as e:
            self.logger.error(f"Error navigating to Quick Speak tab: {e}")

    def navigate_to_voices(self):
        """
        Navigate to the Voices tab in Settings.

        Called when user wants to manage voices or create new voices.
        """
        try:
            # Switch to Voices tab (index 1 - after Audio)
            self.tab_widget.setCurrentIndex(1)
            self.logger.debug("Navigated to Voices tab")
        except Exception as e:
            self.logger.error(f"Error navigating to Voices tab: {e}")

    # =========================================================================
    # Window Transparency Methods (Story 7.3: FR41)
    # =========================================================================

    def _on_transparency_changed(self, value: int):
        """
        Handle transparency slider value change (Story 7.3).

        Provides real-time preview of window transparency as slider moves.
        Minimum 20% enforced by slider minimum value.

        Args:
            value: Slider value (20-100 representing percentage)
        """
        try:
            # Update value label
            self.transparency_value_label.setText(f"{value}%")

            # Convert percentage to opacity (0.0 - 1.0)
            opacity = value / 100.0

            # Emit signal for real-time preview on main window
            self.transparency_preview_requested.emit(opacity)

            self.logger.debug(f"Transparency changed to: {value}% (opacity: {opacity})")

        except Exception as e:
            self.logger.error(f"Error handling transparency change: {e}")