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
    tts_health_check_requested = pyqtSignal()  # TTS health check requested
    voice_changed = pyqtSignal(str)  # voice_profile_name
    voice_refresh_requested = pyqtSignal()  # Voice profile refresh requested
    voice_transcription_requested = pyqtSignal(str)  # voice_profile_name - Transcription generation requested
    quick_speak_entries_changed = pyqtSignal()  # Quick Speak entries or profile changed

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

        layout.addStretch()
        self.tab_widget.addTab(audio_tab, "Audio")

    def _create_tts_tab(self):
        """Create the TTS settings tab."""
        tts_tab = QWidget()
        layout = QVBoxLayout(tts_tab)

        # TTS Service Group
        service_group = QGroupBox("TTS Service Configuration")
        service_layout = QFormLayout(service_group)

        self.tts_url_edit = QLineEdit()
        self.tts_url_edit.setPlaceholderText("http://localhost:9880")
        service_layout.addRow("Service URL:", self.tts_url_edit)

        self.tts_timeout_spin = QSpinBox()
        self.tts_timeout_spin.setRange(5, 300)
        self.tts_timeout_spin.setSuffix(" seconds")
        service_layout.addRow("Request Timeout:", self.tts_timeout_spin)

        # TTS Health Check Button
        health_check_layout = QHBoxLayout()
        self.tts_health_check_button = QPushButton("Check TTS Status")
        self.tts_health_check_button.setMaximumWidth(150)
        self.tts_health_check_button.clicked.connect(self._on_check_tts_health)
        health_check_layout.addWidget(self.tts_health_check_button)
        health_check_layout.addStretch()
        service_layout.addRow("Connection:", health_check_layout)

        # TTS Health Check Status Label
        self.tts_health_status_label = QLabel("")
        self.tts_health_status_label.setWordWrap(True)
        self.tts_health_status_label.setProperty("class", "status-label")
        service_layout.addRow("", self.tts_health_status_label)

        layout.addWidget(service_group)

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

        self.max_duration_spin = QSpinBox()
        self.max_duration_spin.setRange(1, 120)
        self.max_duration_spin.setSuffix(" seconds")
        voice_layout.addRow("Max Voice Duration:", self.max_duration_spin)

        self.refresh_interval_spin = QSpinBox()
        self.refresh_interval_spin.setRange(10, 300)
        self.refresh_interval_spin.setSuffix(" seconds")
        voice_layout.addRow("Auto-refresh Interval:", self.refresh_interval_spin)

        layout.addWidget(voice_group)

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

    def _load_current_settings(self):
        """Load current settings into UI controls."""
        try:
            # Audio settings - monitoring checkbox removed, was only for debugging

            # TTS settings
            self.tts_url_edit.setText(self.current_settings.tts_service_url)
            self.tts_timeout_spin.setValue(self.current_settings.tts_service_timeout)
            self.max_duration_spin.setValue(int(self.current_settings.max_voice_duration))
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

            log_level_index = self.log_level_combo.findText(self.current_settings.log_level)
            if log_level_index >= 0:
                self.log_level_combo.setCurrentIndex(log_level_index)

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

            # TTS settings
            self.current_settings.tts_service_url = self.tts_url_edit.text().strip()
            self.current_settings.tts_service_timeout = self.tts_timeout_spin.value()
            self.current_settings.max_voice_duration = float(self.max_duration_spin.value())
            self.current_settings.auto_refresh_interval = self.refresh_interval_spin.value()

            # Voice directory settings
            voice_directory = self.voice_directory_edit.text().strip()
            if voice_directory:
                self.current_settings.voice_files_directory = voice_directory

            # UI settings
            self.current_settings.ui_theme = self.theme_combo.currentText()
            self.current_settings.always_on_top = self.always_on_top_checkbox.isChecked()
            self.current_settings.log_level = self.log_level_combo.currentText()

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
        self.voice_selector.voice_manager = voice_manager
        self.voice_selector.populate_voices()

        # Set current active voice
        if voice_manager:
            active_profile = voice_manager.get_active_profile()
            if active_profile:
                self.voice_selector.set_selected_voice(active_profile.name)
                self.selected_voice_name = active_profile.name

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
        """Refresh the voice selector list after external updates."""
        if self.voice_selector:
            self.voice_selector.populate_voices()
            self.logger.debug("Voice selector refreshed")

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
        """Handle reset to defaults button click."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to reset all settings to their default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Reset to defaults
            self.current_settings.reset_to_defaults()
            self._load_current_settings()
            self._show_device_status("Settings reset to defaults", "info")

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

    def _on_check_tts_health(self):
        """Handle TTS health check button click."""
        try:
            import requests

            # Get the TTS URL from the input field
            tts_url = self.tts_url_edit.text().strip() or "http://127.0.0.1:9880"

            # Update UI to show checking
            self.tts_health_check_button.setEnabled(False)
            self.tts_health_status_label.setText("Checking TTS connection...")
            self.tts_health_status_label.setProperty("status", "info")

            # Try to connect to the TTS service
            # GPT-SoVITS doesn't have a specific health endpoint, but any response means it's running
            try:
                response = requests.get(tts_url, timeout=2)
                # Any response (200, 404, etc.) indicates the service is running
                self._show_tts_health_status("✓ TTS service is available and responding", "success")
                # Emit signal to update main window TTS ready status
                self.tts_health_check_requested.emit()
            except requests.exceptions.Timeout:
                self._show_tts_health_status("✗ TTS service connection timeout", "error")
            except requests.exceptions.ConnectionError:
                self._show_tts_health_status("✗ TTS service not available (connection refused)", "error")
            except Exception as e:
                self._show_tts_health_status(f"✗ Error checking TTS: {str(e)}", "error")

        except ImportError:
            self._show_tts_health_status("✗ Requests library not available", "error")
        except Exception as e:
            self.logger.error(f"Error during TTS health check: {e}")
            self._show_tts_health_status(f"✗ Unexpected error: {str(e)}", "error")
        finally:
            # Re-enable button after check
            QTimer.singleShot(1000, lambda: self.tts_health_check_button.setEnabled(True))

    def _show_tts_health_status(self, message: str, status_type: str = "info"):
        """
        Show TTS health check status to user.

        Args:
            message: Status message to display
            status_type: Type of status (info, success, warning, error)
        """
        self.tts_health_status_label.setText(message)
        self.tts_health_status_label.setProperty("status", status_type)

    def update_tts_health_status(self, is_available: bool):
        """
        Update TTS health status from external source (e.g., main window).

        Args:
            is_available: Whether TTS service is currently available
        """
        if is_available:
            self._show_tts_health_status("✓ TTS service is available", "success")
        else:
            self._show_tts_health_status("✗ TTS service not available", "error")

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