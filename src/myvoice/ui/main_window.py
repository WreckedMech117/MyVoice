"""
MyVoice Main Window

This module implements the primary MainWindow component with a compact design
for desktop accessibility. Features always-on-top behavior and minimal footprint.
"""

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QLineEdit, QComboBox,
    QSizePolicy, QStatusBar, QSystemTrayIcon, QMenu
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QTimer, QEvent
from PyQt6.QtGui import QIcon, QFont, QCloseEvent, QKeyEvent, QShowEvent

# QAccessibleEvent may not be available in all PyQt6 versions
try:
    from PyQt6.QtGui import QAccessibleEvent, QAccessible
    ACCESSIBILITY_AVAILABLE = True
except ImportError:
    ACCESSIBILITY_AVAILABLE = False

from myvoice.ui.styles.theme_manager import get_theme_manager
from myvoice.ui.components.service_status_indicator import ServiceStatusBar, ServiceStatusIndicator
from myvoice.ui.components.settings_dialog import SettingsDialog
from myvoice.ui.components.virtual_mic_setup_dialog import VirtualMicSetupDialog
from myvoice.ui.dialogs.voice_design_studio import VoiceDesignStudioDialog
from myvoice.ui.components.custom_title_bar import CustomTitleBar
from myvoice.ui.components.resize_grip import SideGrip, CornerGrip
from myvoice.ui.components.quick_speak_dialog import QuickSpeakDialog
from myvoice.ui.components.quick_speak_menu import QuickSpeakMenu
from myvoice.ui.components.emotion_button_group import EmotionButtonGroup, EmotionPreset
from myvoice.models.ui_state import UIState, ServiceStatusInfo, ServiceHealthStatus
from myvoice.models.app_settings import AppSettings
from myvoice.services.quick_speak_service import QuickSpeakService


class MainWindow(QMainWindow):
    """
    Main application window with compact design and always-on-top behavior.

    This window provides the primary interface for the MyVoice TTS application,
    featuring a minimal footprint suitable for desktop accessibility.

    Signals:
        text_generate_requested: Emitted when user requests TTS generation
        voice_changed: Emitted when user changes voice selection
        settings_requested: Emitted when user opens settings
    """

    # Signals for application communication
    text_generate_requested = pyqtSignal(str)  # text to generate
    voice_changed = pyqtSignal(str)  # voice profile name
    settings_requested = pyqtSignal()
    service_status_clicked = pyqtSignal(str)  # service name
    settings_changed = pyqtSignal(AppSettings)  # new settings
    audio_device_refresh_requested = pyqtSignal()
    audio_device_test_requested = pyqtSignal(str)  # device_id
    whisper_init_requested = pyqtSignal()  # QA4: Request whisper service initialization
    virtual_device_test_requested = pyqtSignal(str)  # device_id
    voice_directory_changed = pyqtSignal(str)  # new_directory_path
    voice_refresh_requested = pyqtSignal()  # voice profile refresh
    voice_transcription_requested = pyqtSignal(str)  # voice_profile_name - transcription requested
    replay_last_requested = pyqtSignal()  # Story 2.4: Replay last generated audio (FR28)

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the main window with compact design.

        Args:
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Theme manager
        self.theme_manager = get_theme_manager()

        # UI State management
        self.ui_state = UIState()

        # Voice manager reference (will be set by application)
        self.voice_manager = None

        # Audio manager and settings references (will be set by application)
        self.audio_coordinator = None
        self.app_settings = None

        # Settings dialog
        self.settings_dialog = None

        # Voice Design Studio dialog (Story 1.2)
        self.voice_design_studio_dialog = None

        # Quick Speak service, menu, and dialog
        self.quick_speak_service = None
        self.quick_speak_menu = None  # Story 5.2: Popup menu for quick access
        self.quick_speak_dialog = None  # Full dialog for empty state navigation

        # TTS service availability tracking
        self._tts_available = False

        # TTS service reference (will be set by application for voice creation dialogs)
        self.tts_service = None

        # Whisper service reference for transcription (QA3 fix)
        self.whisper_service = None

        # Flag to track when theme needs reapplication after window flag changes
        self._needs_theme_reapplication = False

        # Story 7.2: System tray integration (FR38, FR39)
        self._force_quit = False  # Flag to distinguish real exit from minimize to tray
        self.tray_icon = None  # Will be created after window setup

        # Window configuration
        self._setup_window_properties()

        # Create UI components
        self._create_ui()

        # Apply styling (will be updated when app_settings is set)
        self._apply_default_theme()

        # Setup window behavior
        self._setup_window_behavior()

        # Story 7.2: Setup system tray (FR38, FR39)
        self._setup_system_tray()

        self.logger.debug("MainWindow initialized successfully")

    def _setup_window_properties(self):
        """Configure basic window properties for compact design."""
        # Set window title and icon
        self.setWindowTitle("MyVoice")

        # Set window icon for taskbar
        icon_path = Path(__file__).parent.parent.parent / "icon" / "MyVoice.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Set frameless window flags
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Window
        )

        # Set compact window size (account for custom title bar ~28px)
        self.setMinimumSize(QSize(320, 200))  # Minimum to fit all controls
        self.setMaximumSize(QSize(600, 300))  # Maximum window size
        self.resize(QSize(400, 220))          # Default size with room for buttons

        # Make window resizable
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        self.logger.debug("Window properties configured")

    def _create_ui(self):
        """Create and layout the main UI components."""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Add custom title bar as first element
        self.title_bar = CustomTitleBar(self)
        main_layout.addWidget(self.title_bar)

        # Create content container with original margins
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        # QA Round 2 Item #7: Reduced vertical margins for compact overlay
        content_layout.setContentsMargins(8, 4, 8, 4)
        content_layout.setSpacing(4)

        # Text input area with inline action buttons (moved to top)
        text_input_layout = QHBoxLayout()
        text_input_layout.setSpacing(4)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter text to convert to speech...")
        self.text_input.setMinimumHeight(60)  # Approximately 2 lines minimum
        self.text_input.setMaximumHeight(100)  # Allow growth to accommodate button column
        self.text_input.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.text_input.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_input.installEventFilter(self)  # Install event filter for Enter key handling

        # Action buttons layout (vertical stack)
        action_buttons_layout = QVBoxLayout()
        action_buttons_layout.setSpacing(4)
        action_buttons_layout.setContentsMargins(0, 0, 0, 0)

        # Quick Speak button as small icon button
        self.quick_speak_button = QPushButton()
        quick_speak_icon = self.style().standardIcon(self.style().StandardPixmap.SP_FileDialogListView)
        self.quick_speak_button.setIcon(quick_speak_icon)
        self.quick_speak_button.setFixedSize(24, 24)
        self.quick_speak_button.setObjectName("quick_speak_button")
        self.quick_speak_button.setToolTip("Quick Speak")
        self.quick_speak_button.clicked.connect(self._on_quick_speak_clicked)

        # Generate button as small icon button
        self.generate_button = QPushButton()
        generate_icon = self.style().standardIcon(self.style().StandardPixmap.SP_MediaPlay)
        self.generate_button.setIcon(generate_icon)
        self.generate_button.setFixedSize(24, 24)
        self.generate_button.setObjectName("generate_button")
        self.generate_button.setToolTip("Generate speech (Enter)")
        self.generate_button.clicked.connect(self._on_generate_clicked)

        # Story 2.4: Replay Last button (FR28, FR29)
        self.replay_button = QPushButton()
        replay_icon = self.style().standardIcon(self.style().StandardPixmap.SP_BrowserReload)
        self.replay_button.setIcon(replay_icon)
        self.replay_button.setFixedSize(24, 24)
        self.replay_button.setObjectName("replay_button")
        self.replay_button.setToolTip("No audio to replay")  # Session-only cache: disabled on fresh start
        self.replay_button.clicked.connect(self._on_replay_last_clicked)
        self.replay_button.setEnabled(False)  # Disabled until audio is generated

        # Clear button as small icon button
        self.clear_button = QPushButton()
        clear_icon = self.style().standardIcon(self.style().StandardPixmap.SP_LineEditClearButton)
        self.clear_button.setIcon(clear_icon)
        self.clear_button.setFixedSize(24, 24)
        self.clear_button.setObjectName("clear_button")
        self.clear_button.setToolTip("Clear text")
        self.clear_button.clicked.connect(self._on_clear_clicked)

        action_buttons_layout.addWidget(self.quick_speak_button)
        action_buttons_layout.addWidget(self.generate_button)
        action_buttons_layout.addWidget(self.replay_button)  # Story 2.4: Replay Last
        action_buttons_layout.addWidget(self.clear_button)
        action_buttons_layout.addStretch()  # Push buttons to top

        text_input_layout.addWidget(self.text_input, 1)
        text_input_layout.addLayout(action_buttons_layout)

        content_layout.addLayout(text_input_layout)

        # QA Round 2 Item #7: Combined emotion + voice + settings row (compact layout)
        emotion_voice_layout = QHBoxLayout()
        emotion_voice_layout.setSpacing(8)
        emotion_voice_layout.setContentsMargins(0, 2, 0, 2)  # Reduced vertical margins

        # Emotion Button Group (Story 3.1: FR6, FR10)
        # 5 emoji buttons: 😐 😄 😢 😠 😏 (28x28px, 4px spacing)
        # Neutral selected by default, accent border on selected
        self.emotion_button_group = EmotionButtonGroup(self)
        self.emotion_button_group.emotion_changed.connect(self._on_emotion_changed)
        self.emotion_button_group.custom_emotion_requested.connect(self._on_custom_emotion_requested)
        emotion_voice_layout.addWidget(self.emotion_button_group)

        # Current voice display label (moved to right of emotions)
        self.current_voice_label = QLabel("Voice: (None)")
        self.current_voice_label.setObjectName("voice_label")
        emotion_voice_layout.addWidget(self.current_voice_label, 1)

        # Settings button with Qt standard icon
        self.settings_button = QPushButton()
        settings_icon = self.style().standardIcon(self.style().StandardPixmap.SP_FileDialogDetailedView)
        self.settings_button.setIcon(settings_icon)
        self.settings_button.setFixedSize(QSize(24, 24))
        self.settings_button.setObjectName("settings_button")
        self.settings_button.setToolTip("Open Settings (Ctrl+S)")
        self.settings_button.clicked.connect(self._on_settings_clicked)
        emotion_voice_layout.addWidget(self.settings_button)

        content_layout.addLayout(emotion_voice_layout)

        # Add content widget to main layout
        main_layout.addWidget(content_widget)

        # Status bar with service indicators
        self.status_bar = QStatusBar()
        self.status_bar.showMessage(self._get_ready_message())

        # Add service status indicators to the right side
        # Story 7.4: Use emoji mode for accessibility (FR42, FR43)
        self.service_status_bar = ServiceStatusBar(use_emoji=True)
        self.service_status_bar.service_status_clicked.connect(self._on_service_status_clicked)
        self.status_bar.addPermanentWidget(self.service_status_bar)

        # Story 7.4: Add TTS and Mic indicators by default (FR42, FR43)
        self._setup_default_status_indicators()

        self.setStatusBar(self.status_bar)

        # Create resize grips for frameless window
        self._create_resize_grips()

        # Story 7.5: Setup accessibility features (NFR18, NFR19, NFR20)
        self._setup_accessibility()

        self.logger.debug("UI components created")

    def _create_resize_grips(self):
        """Create and initialize all resize grips for frameless window."""
        # Create corner grips
        self.corner_grips = [
            CornerGrip(self, CornerGrip.CORNER_TOP_LEFT),
            CornerGrip(self, CornerGrip.CORNER_TOP_RIGHT),
            CornerGrip(self, CornerGrip.CORNER_BOTTOM_RIGHT),
            CornerGrip(self, CornerGrip.CORNER_BOTTOM_LEFT),
        ]

        # Create edge grips
        self.edge_grips = {
            'left': SideGrip(self, SideGrip.EDGE_LEFT),
            'top': SideGrip(self, SideGrip.EDGE_TOP),
            'right': SideGrip(self, SideGrip.EDGE_RIGHT),
            'bottom': SideGrip(self, SideGrip.EDGE_BOTTOM),
        }

        # Position grips initially
        self._position_grips()

        # Raise grips to ensure they're on top
        for grip in self.corner_grips:
            grip.raise_()
        for grip in self.edge_grips.values():
            grip.raise_()

        self.logger.debug("Resize grips created and positioned")

    def _setup_accessibility(self):
        """
        Setup accessibility features for keyboard navigation and screen readers.

        Story 7.5: NFR18 (keyboard navigation), NFR19 (visual clarity), NFR20 (focus indicators)
        """
        # =======================================================================
        # Task 1: Set Accessible Names for Screen Readers
        # =======================================================================

        # Text input
        self.text_input.setAccessibleName("Text to speak")
        self.text_input.setAccessibleDescription("Enter the text you want to convert to speech")

        # Action buttons
        self.quick_speak_button.setAccessibleName("Quick Speak")
        self.quick_speak_button.setAccessibleDescription("Open quick phrases menu")

        self.generate_button.setAccessibleName("Generate speech")
        self.generate_button.setAccessibleDescription("Generate speech from entered text")

        self.replay_button.setAccessibleName("Replay last audio")
        self.replay_button.setAccessibleDescription("Replay the last generated speech")

        self.clear_button.setAccessibleName("Clear text")
        self.clear_button.setAccessibleDescription("Clear the text input field")

        # Voice label (read-only, informational)
        self.current_voice_label.setAccessibleName("Current voice")

        # Settings button
        self.settings_button.setAccessibleName("Settings")
        self.settings_button.setAccessibleDescription("Open application settings")

        # Status bar
        self.status_bar.setAccessibleName("Status")
        self.status_bar.setAccessibleDescription("Application status messages")

        # =======================================================================
        # Task 2: Set Focus Policies for Keyboard Navigation
        # =======================================================================

        # Text input - strong focus (both keyboard and mouse)
        self.text_input.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Buttons - strong focus
        self.quick_speak_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.generate_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.replay_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.clear_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.settings_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Voice label - no focus (informational only)
        self.current_voice_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # =======================================================================
        # Task 3: Set Tab Order for Logical Navigation
        # =======================================================================

        # Tab order: Text → Quick Speak → Generate → Replay → Clear → Settings → Emotions
        QWidget.setTabOrder(self.text_input, self.quick_speak_button)
        QWidget.setTabOrder(self.quick_speak_button, self.generate_button)
        QWidget.setTabOrder(self.generate_button, self.replay_button)
        QWidget.setTabOrder(self.replay_button, self.clear_button)
        QWidget.setTabOrder(self.clear_button, self.settings_button)

        # Connect settings button to emotion buttons
        # Get first emotion button from the group
        emotion_buttons = self.emotion_button_group.get_emotion_buttons()
        if emotion_buttons:
            QWidget.setTabOrder(self.settings_button, emotion_buttons[0])

            # Chain emotion buttons together
            for i in range(len(emotion_buttons) - 1):
                QWidget.setTabOrder(emotion_buttons[i], emotion_buttons[i + 1])

            # Connect last emotion button back to text input for circular navigation
            # (Actually, tab should naturally cycle, so we don't need this)

        self.logger.debug("Accessibility features configured (NFR18, NFR19, NFR20)")

    def announce_status(self, message: str, priority: str = "polite"):
        """
        Announce a status message for screen readers via live regions.

        Story 7.5: NFR18 - Accessible status announcements for screen readers.

        Args:
            message: The message to announce
            priority: "polite" (wait for user to finish) or "assertive" (interrupt)
        """
        try:
            # Update status bar message (visible to users)
            self.status_bar.showMessage(message)

            # Notify accessibility clients of the change
            # This triggers screen readers to announce the new status
            if ACCESSIBILITY_AVAILABLE:
                event = QAccessibleEvent(self.status_bar, QAccessible.Event.ValueChanged)
                QAccessible.updateAccessibility(event)

            self.logger.debug(f"Announced status: {message}")
        except Exception as e:
            self.logger.warning(f"Failed to announce status: {e}")

    def announce_generation_complete(self):
        """Announce that speech generation is complete (Story 7.5)."""
        self.announce_status("Generation complete", priority="polite")

    def announce_error(self, error_message: str):
        """Announce an error for immediate screen reader notification (Story 7.5)."""
        self.announce_status(f"Error: {error_message}", priority="assertive")

    def resizeEvent(self, event):
        """Handle window resize to reposition grips."""
        super().resizeEvent(event)
        if hasattr(self, 'corner_grips') and hasattr(self, 'edge_grips'):
            self._position_grips()

    def _position_grips(self):
        """Position all resize grips along window edges and corners."""
        rect = self.rect()
        grip_size = 6
        title_bar_height = 28

        # Position corner grips
        self.corner_grips[0].move(0, 0)  # Top-left
        self.corner_grips[1].move(rect.width() - grip_size, 0)  # Top-right
        self.corner_grips[2].move(rect.width() - grip_size, rect.height() - grip_size)  # Bottom-right
        self.corner_grips[3].move(0, rect.height() - grip_size)  # Bottom-left

        # Position edge grips
        # Left edge
        self.edge_grips['left'].setGeometry(
            0,
            grip_size,
            grip_size,
            rect.height() - 2 * grip_size
        )

        # Top edge (avoid title bar for easier dragging)
        self.edge_grips['top'].setGeometry(
            grip_size,
            0,
            rect.width() - 2 * grip_size,
            grip_size
        )

        # Right edge
        self.edge_grips['right'].setGeometry(
            rect.width() - grip_size,
            grip_size,
            grip_size,
            rect.height() - 2 * grip_size
        )

        # Bottom edge
        self.edge_grips['bottom'].setGeometry(
            grip_size,
            rect.height() - grip_size,
            rect.width() - 2 * grip_size,
            grip_size
        )

    def changeEvent(self, event):
        """Handle window state changes."""
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            # Update title bar maximize button state
            if hasattr(self, 'title_bar'):
                is_maximized = self.windowState() & Qt.WindowState.WindowMaximized
                self.title_bar.update_maximize_state(bool(is_maximized))

    def _apply_default_theme(self):
        """Apply default theme to the window using the theme manager."""
        try:
            # Set window class for CSS targeting
            self.setProperty("class", "compact")

            # Apply dark theme by default (compact variant)
            success = self.theme_manager.apply_theme("dark", self)

            if success:
                self.logger.debug("Successfully applied default dark theme")

                # Connect theme manager signals
                self.theme_manager.theme_changed.connect(self._on_theme_changed)
                self.theme_manager.theme_load_failed.connect(self._on_theme_load_failed)
            else:
                self.logger.warning("Failed to apply dark theme, using compact fallback")
                # Fallback to compact theme if dark theme fails
                fallback_success = self.theme_manager.apply_theme("light", self)
                if not fallback_success:
                    self._apply_fallback_style()

        except Exception as e:
            self.logger.exception(f"Error applying default theme: {e}")
            self._apply_fallback_style()

    def apply_theme_from_settings(self):
        """Apply theme based on current app settings."""
        if not self.app_settings:
            self.logger.debug("No app settings available, keeping current theme")
            return

        try:
            theme_name = self.app_settings.ui_theme
            self.logger.info(f"Applying theme from settings: {theme_name}")

            # Set window class for CSS targeting
            self.setProperty("class", "compact")

            success = self.theme_manager.apply_theme(theme_name, self)

            if success:
                self.logger.info(f"THEME DEBUG: Successfully applied theme from settings: {theme_name}")
            else:
                self.logger.warning(f"THEME DEBUG: Failed to apply theme '{theme_name}', falling back to dark theme")
                # Try to fallback to dark theme
                fallback_success = self.theme_manager.apply_theme("dark", self)
                if not fallback_success:
                    self.logger.warning("Dark theme fallback failed, trying light theme")
                    light_fallback = self.theme_manager.apply_theme("light", self)
                    if not light_fallback:
                        self.logger.error("All theme fallbacks failed, using minimal styling")
                        self._apply_fallback_style()

        except Exception as e:
            self.logger.exception(f"Error applying theme from settings: {e}")

    def showEvent(self, event: QShowEvent):
        """Override showEvent to handle theme reapplication after window flag changes."""
        super().showEvent(event)

        # If we need to reapply theme after window recreation
        if self._needs_theme_reapplication:
            self._needs_theme_reapplication = False
            # Use a very short timer to ensure the window is fully shown
            QTimer.singleShot(10, self._force_theme_reapplication)
            self.logger.debug("Scheduled theme reapplication after showEvent")

    def _force_theme_reapplication(self):
        """
        Force theme reapplication after window recreation.

        This method ensures the theme is always applied, even if app_settings
        is temporarily unavailable during window flag changes.
        """
        try:
            # Clear any existing stylesheet first to force a clean state
            self.setStyleSheet("")

            # Set window class for CSS targeting
            self.setProperty("class", "compact")

            # Force immediate processing of property changes
            self.style().polish(self)

            # Try to apply theme from settings first
            if self.app_settings:
                theme_name = self.app_settings.ui_theme
                success = self.theme_manager.apply_theme(theme_name, self)
                if success:
                    self.logger.debug(f"Successfully reapplied theme: {theme_name}")
                    self._force_style_refresh()
                    return

            # Fallback to dark theme if settings not available or failed
            self.logger.debug("Falling back to dark theme for window recreation")
            success = self.theme_manager.apply_theme("dark", self)

            if not success:
                # Last resort: apply fallback style
                self.logger.warning("Theme manager failed, applying fallback style")
                self._apply_fallback_style()
                self._force_style_refresh()
            else:
                self.logger.debug("Successfully applied dark theme fallback")
                self._force_style_refresh()

        except Exception as e:
            self.logger.error(f"Error in force theme reapplication: {e}")
            # Ultimate fallback - force any theme to avoid white window
            try:
                self.logger.debug("Attempting ultimate fallback theme application")
                self._apply_fallback_style()
                self._force_style_refresh()
            except Exception as fallback_error:
                self.logger.error(f"Even fallback style failed: {fallback_error}")
                # Last ditch effort - apply minimal dark background directly
                self.setStyleSheet("QMainWindow { background-color: #2b2b2b; color: white; }")
                self._force_style_refresh()

    def _force_style_refresh(self):
        """
        Force a complete style refresh and repaint of the window.

        This simulates what happens when minimizing/restoring the window.
        """
        try:
            # Force style recalculation for the window and all children
            self.style().polish(self)

            # Force immediate update and repaint
            self.update()
            self.repaint()

            # Also refresh all child widgets
            for child in self.findChildren(QWidget):
                child.style().polish(child)
                child.update()
                child.repaint()

            self.logger.debug("Forced complete style refresh and repaint")

        except Exception as e:
            self.logger.error(f"Error in force style refresh: {e}")

    def _on_voice_changed_from_settings(self, voice_name: str):
        """
        Handle voice change from settings dialog.

        Args:
            voice_name: Name of the selected voice profile
        """
        try:
            # Update voice label
            self.update_voice_label(voice_name)

            # Emit voice change signal to app
            self.voice_changed.emit(voice_name)

            self.logger.debug(f"Voice changed from settings: {voice_name}")
        except Exception as e:
            self.logger.error(f"Error handling voice change from settings: {e}")

    def _on_voice_created_from_settings(self, profile):
        """
        Handle voice creation from settings dialog (design or clone).

        Story 4.2: Voice Design Creation Workflow

        Args:
            profile: VoiceProfile that was created
        """
        try:
            self.logger.info(f"Voice created from settings: {profile.name} (type={profile.voice_type.value})")

            # Update the voice label to the newly created voice
            self.update_voice_label(profile.name)

            # Emit voice change to notify app to update active voice
            self.voice_changed.emit(profile.name)

            # Request voice list refresh to pick up the new voice
            self.voice_refresh_requested.emit()

            self.status_bar.showMessage(f"Voice '{profile.name}' created successfully", 5000)

        except Exception as e:
            self.logger.error(f"Error handling voice creation from settings: {e}")

    def _on_voice_deleted_from_settings(self, voice_name: str):
        """
        Handle voice deletion from settings dialog.

        Story 4.4: Voice Library Management (FR19)

        Args:
            voice_name: Name of the deleted voice
        """
        try:
            self.logger.info(f"Voice deleted from settings: {voice_name}")

            # Request voice list refresh
            self.voice_refresh_requested.emit()

            self.status_bar.showMessage(f"Voice '{voice_name}' deleted", 3000)

        except Exception as e:
            self.logger.error(f"Error handling voice deletion from settings: {e}")

    def _on_settings_changed(self, new_settings: AppSettings):
        """
        Handle settings change notifications from the settings dialog.

        Args:
            new_settings: New settings to apply
        """
        self.app_settings = new_settings

        # Apply theme from updated settings
        self.apply_theme_from_settings()

        # Apply always-on-top setting
        self.set_always_on_top(new_settings.always_on_top)

        # Story 7.3: Apply window transparency (FR41)
        self.set_window_transparency(new_settings.window_transparency)

        self.logger.debug("Settings changed and applied to MainWindow")

    def _apply_fallback_style(self):
        """Apply minimal fallback styling when theme manager fails."""
        self.logger.warning("THEME DEBUG: Applying fallback style - this indicates theme loading failed!")
        fallback_style = """
            QMainWindow {
                background-color: #f8f9fa;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 8pt;
                color: #212529;
            }

            QLabel {
                color: #495057;
                font-weight: 500;
                padding: 1px 0px;
            }

            QPushButton {
                background-color: #0d6efd;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: 500;
                font-size: 8pt;
                min-height: 16px;
            }

            QPushButton:hover {
                background-color: #0b5ed7;
            }

            QPushButton:pressed {
                background-color: #0a58ca;
            }

            QPushButton:disabled {
                background-color: #6c757d;
                color: #adb5bd;
                border: 1px solid #495057;
            }

            QPushButton#generate_button {
                background-color: #28a745;
                border-radius: 12px;
            }

            QPushButton#generate_button:hover {
                background-color: #218838;
            }

            QPushButton#generate_button:pressed {
                background-color: #1e7e34;
            }

            QPushButton#generate_button:disabled {
                background-color: #6c757d;
            }

            /* Story 2.4: Replay Last button styling */
            QPushButton#replay_button {
                background-color: #17a2b8;
                border-radius: 12px;
            }

            QPushButton#replay_button:hover {
                background-color: #138496;
            }

            QPushButton#replay_button:pressed {
                background-color: #117a8b;
            }

            QPushButton#replay_button:disabled {
                background-color: #6c757d;
            }

            QPushButton#clear_button {
                background-color: #6c757d;
                border-radius: 12px;
            }

            QPushButton#clear_button:hover {
                background-color: #5a6268;
            }

            QPushButton#clear_button:pressed {
                background-color: #545b62;
            }

            QComboBox {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 4px 8px;
                background-color: white;
                font-size: 8pt;
            }

            QLineEdit {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 6px 8px;
                background-color: white;
                font-size: 8pt;
                min-height: 20px;
            }

            QTextEdit {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 6px 8px;
                background-color: white;
                font-size: 8pt;
            }

            QStatusBar {
                background-color: #f1f3f4;
                border-top: 1px solid #dee2e6;
                font-size: 7pt;
                color: #6c757d;
                padding: 2px 6px;
            }
        """

        self.setStyleSheet(fallback_style)
        self.logger.debug("Applied fallback compact stylesheet")

    def _on_theme_changed(self, theme_name: str):
        """
        Handle theme change notifications.

        Args:
            theme_name: Name of the new theme
        """
        self.logger.debug(f"Theme changed to: {theme_name}")

    def _on_theme_load_failed(self, theme_name: str, error_message: str):
        """
        Handle theme load failure notifications.

        Args:
            theme_name: Name of the theme that failed to load
            error_message: Error message
        """
        self.logger.error(f"Failed to load theme '{theme_name}': {error_message}")
        self._apply_fallback_style()

    def switch_theme(self, theme_name: str) -> bool:
        """
        Switch to a different theme.

        Args:
            theme_name: Name of the theme to switch to

        Returns:
            True if theme switch successful, False otherwise
        """
        return self.theme_manager.switch_theme(theme_name, self)

    def _setup_window_behavior(self):
        """Setup additional window behavior and event handling."""
        # Set focus policy for better keyboard navigation
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Auto-focus on text input when window is shown
        self.text_input.setFocus()

        # Setup auto-save timer for text content (optional)
        self._auto_save_timer = QTimer()
        self._auto_save_timer.timeout.connect(self._auto_save_text)
        self._auto_save_timer.start(30000)  # Auto-save every 30 seconds

        self.logger.debug("Window behavior configured")

    def _setup_system_tray(self):
        """
        Setup system tray icon and context menu (Story 7.2: FR38, FR39).

        Creates a system tray icon with:
        - Left-click: Restore window
        - Right-click: Context menu with Restore, Settings, Exit
        - Tooltip: "MyVoice - Click to restore"
        """
        # Check if system tray is available
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self.logger.warning("System tray not available on this platform")
            return

        # Create tray icon
        icon_path = Path(__file__).parent.parent.parent / "icon" / "MyVoice.png"
        if icon_path.exists():
            tray_icon = QIcon(str(icon_path))
        else:
            # Fallback to application icon
            tray_icon = self.windowIcon()
            self.logger.warning(f"Tray icon not found at {icon_path}, using window icon")

        self.tray_icon = QSystemTrayIcon(tray_icon, self)
        self.tray_icon.setToolTip("MyVoice - Click to restore")

        # Create context menu
        tray_menu = QMenu()

        # Restore action
        restore_action = tray_menu.addAction("Restore")
        restore_action.triggered.connect(self._restore_from_tray)

        # Settings action
        settings_action = tray_menu.addAction("Settings")
        settings_action.triggered.connect(self._on_settings_clicked)

        # Voice Design Studio action (Story 1.2: FR1)
        voice_design_action = tray_menu.addAction("Voice Design Studio")
        voice_design_action.triggered.connect(self._on_voice_design_studio_clicked)

        tray_menu.addSeparator()

        # Exit action
        exit_action = tray_menu.addAction("Exit")
        exit_action.triggered.connect(self._quit_application)

        self.tray_icon.setContextMenu(tray_menu)

        # Connect activated signal (handles clicks)
        self.tray_icon.activated.connect(self._on_tray_icon_activated)

        self.logger.debug("System tray configured")

    def _on_tray_icon_activated(self, reason: QSystemTrayIcon.ActivationReason):
        """
        Handle tray icon activation (click).

        Args:
            reason: The activation reason (click, double-click, etc.)
        """
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            # Left-click: restore window
            self._restore_from_tray()
        elif reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            # Double-click: also restore window
            self._restore_from_tray()

    def _minimize_to_tray(self):
        """
        Minimize window to system tray (Story 7.2: FR38).

        Hides the window and shows the tray icon. On first use,
        shows a notification explaining the behavior.
        """
        if not self.tray_icon:
            # Tray not available, just minimize normally
            self.showMinimized()
            return

        # Hide the window
        self.hide()

        # Show tray icon
        self.tray_icon.show()

        # Show one-time notification (Story 7.2)
        if self.app_settings and not self.app_settings.tray_notification_shown:
            self.tray_icon.showMessage(
                "MyVoice",
                "MyVoice is still running in the system tray. Click the icon to restore.",
                QSystemTrayIcon.MessageIcon.Information,
                3000  # 3 seconds
            )
            # Mark notification as shown
            self.app_settings.tray_notification_shown = True
            # Emit settings changed to persist this
            self.settings_changed.emit(self.app_settings)

        self.logger.debug("Window minimized to system tray")

    def _restore_from_tray(self):
        """
        Restore window from system tray (Story 7.2: FR39).

        Shows the window, brings it to front, and focuses the text input.
        """
        # Show the window
        self.show()
        self.showNormal()

        # Restore always-on-top if enabled
        if self.app_settings and self.app_settings.always_on_top:
            self.set_always_on_top(True)

        # Bring to front and activate
        self.raise_()
        self.activateWindow()

        # Focus text input
        self.text_input.setFocus()

        # Hide tray icon (optional - keep visible if desired)
        # self.tray_icon.hide()

        self.logger.debug("Window restored from system tray")

    def _quit_application(self):
        """
        Fully quit the application (Story 7.2).

        Sets force quit flag and closes the window, which triggers
        proper cleanup in closeEvent.
        """
        self._force_quit = True

        # Hide tray icon
        if self.tray_icon:
            self.tray_icon.hide()

        # Close the window (triggers closeEvent with _force_quit=True)
        self.close()

        self.logger.info("Application quit requested from tray menu")

    def _on_generate_clicked(self):
        """Handle generate button click with enhanced visual feedback."""
        text = self.text_input.toPlainText().strip()

        if not text:
            self.status_bar.showMessage("Please enter text to generate speech", 3000)
            self.text_input.setFocus()
            return

        # Start TTS generation with visual feedback
        self._start_tts_generation(text)

        # Clear the text input after starting generation
        self.text_input.clear()
        self.text_input.setFocus()

    def _on_replay_last_clicked(self):
        """
        Handle replay last button click (Story 2.4: FR28, FR29).

        Emits replay_last_requested signal to replay the last generated audio
        through both monitor and virtual mic.
        """
        self.logger.info("Replay last audio requested")
        self.status_bar.showMessage("Replaying last audio...", 2000)
        self.replay_last_requested.emit()

    def _start_tts_generation(self, text: str):
        """
        Start TTS generation with visual feedback and stub implementation.

        Args:
            text: Text to convert to speech
        """
        self.logger.info(f"Starting TTS generation for: {text[:50]}...")

        # Set generating state with visual feedback
        self.set_generation_status("Generating speech...", is_generating=True)

        # Emit signal for actual TTS processing (when service is connected)
        self.text_generate_requested.emit(text)

        # TTS Handler Stub - Simulate processing time for demo/testing
        # In production, this will be replaced by actual TTS service integration
        QTimer.singleShot(2000, lambda: self._complete_tts_generation_stub(text))

    def _complete_tts_generation_stub(self, text: str):
        """
        Complete TTS generation stub - simulates successful generation.
        This method will be replaced when actual TTS service is integrated.

        Args:
            text: Text that was processed
        """
        self.logger.info(f"TTS generation completed (stub) for: {text[:30]}...")

        # Reset button state
        self.set_generation_status("Speech generated successfully!", is_generating=False)

        # Clear status message after 3 seconds and show appropriate ready message
        QTimer.singleShot(3000, lambda: self.status_bar.showMessage(self._get_ready_message()))

    def eventFilter(self, obj, event):
        """Handle events for text input, specifically Enter key for generation."""
        if obj is self.text_input and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
                # Ctrl+Enter allows normal newline behavior
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    return False  # Allow normal processing
                else:
                    # Plain Enter triggers generation
                    self._on_generate_clicked()
                    return True  # Consume the event
        return super().eventFilter(obj, event)

    def _on_clear_clicked(self):
        """
        Handle clear button click with smart apostrophe detection.

        - If text ends with apostrophe: Clear all text
        - If text contains apostrophe (not at end): Clear everything after first apostrophe
        - Otherwise: Clear all text
        """
        try:
            current_text = self.text_input.toPlainText()

            if not current_text:
                return  # Nothing to clear

            # Check if last character is apostrophe
            if current_text.endswith("'"):
                # Clear all text
                self.text_input.clear()
                self.logger.debug("Clear button: Cleared all text (ends with apostrophe)")
            elif "'" in current_text:
                # Clear everything after first apostrophe (keep text before apostrophe + apostrophe)
                apostrophe_index = current_text.index("'")
                cleared_text = current_text[:apostrophe_index + 1]
                self.text_input.setPlainText(cleared_text)
                self.logger.debug(f"Clear button: Cleared after apostrophe - kept '{cleared_text}'")
            else:
                # No apostrophe, clear all text
                self.text_input.clear()
                self.logger.debug("Clear button: Cleared all text (no apostrophe)")

        except Exception as e:
            self.logger.error(f"Error in clear button handler: {e}")
            # Fallback to simple clear
            self.text_input.clear()

    def _on_enter_pressed(self):
        """Handle Enter key press in text input - deprecated, using eventFilter now."""
        self._on_generate_clicked()

    def _auto_save_text(self):
        """Auto-save text content (placeholder for future implementation)."""
        # TODO: Implement auto-save functionality when configuration service is ready
        pass

    def _on_emotion_changed(self, preset: EmotionPreset):
        """
        Handle emotion button group selection changes.

        Args:
            preset: The selected emotion preset
        """
        try:
            self.logger.debug(f"Emotion changed to: {preset.display_name}")
            # Status bar feedback
            self.status_bar.showMessage(f"Emotion: {preset.display_name}", 2000)

        except Exception as e:
            self.logger.error(f"Error handling emotion change: {e}")

    def _on_custom_emotion_requested(self):
        """Handle Custom... button click - opens Settings to Custom Emotion section (Story 3.4)."""
        try:
            self.logger.debug("Custom emotion requested - opening settings")
            self._on_settings_clicked()
            # Navigate to Custom Emotion section in settings
            if self.settings_dialog:
                self.settings_dialog.navigate_to_custom_emotion()
        except Exception as e:
            self.logger.error(f"Error handling custom emotion request: {e}")

    def _on_custom_emotion_applied(self, custom_text: str):
        """
        Handle custom emotion applied from settings dialog (Story 3.4).

        Args:
            custom_text: The custom emotion instruction text
        """
        try:
            self.logger.info(f"Custom emotion applied: {custom_text}")
            # Set custom emotion in the emotion button group
            self.emotion_button_group.set_custom_emotion(custom_text)
            # Update status bar
            self.status_bar.showMessage("Custom emotion applied", 2000)
        except Exception as e:
            self.logger.error(f"Error applying custom emotion: {e}")

    def get_emotion_preset(self) -> EmotionPreset:
        """
        Get the currently selected emotion preset.

        Returns:
            EmotionPreset: Current emotion preset
        """
        return self.emotion_button_group.get_current_preset()

    def get_emotion_instruct(self) -> str:
        """
        Get the TTS instruct parameter for the current emotion.

        Returns:
            str: Instruct string for TTS generation
        """
        return self.emotion_button_group.get_current_instruct()

    def set_emotion_preset(self, preset: EmotionPreset):
        """
        Set the selected emotion preset.

        Args:
            preset: The emotion preset to select
        """
        self.emotion_button_group.set_preset(preset)

    def set_emotion_by_index(self, index: int) -> bool:
        """
        Set the emotion by index (0=Neutral, 1=Happy, 2=Sad, 3=Angry, 4=Flirtatious).

        Args:
            index: Emotion index (0-4)

        Returns:
            bool: True if index was valid, False otherwise
        """
        return self.emotion_button_group.set_preset_by_index(index)

    def set_emotion_enabled(self, enabled: bool):
        """
        Enable or disable emotion selection.

        Used when switching to cloned voices which don't support emotion control.

        Args:
            enabled: Whether emotion buttons should be enabled
        """
        self.emotion_button_group.set_emotion_enabled(enabled)

    def update_voice_emotions(self, available_emotions: list, voice_name: str = ""):
        """
        Update emotion buttons based on available emotions for the voice.

        Emotion Variants: EMBEDDING voices have specific emotions available.
        This enables only the buttons for available emotions.

        Args:
            available_emotions: List of emotion IDs available for this voice
                               (e.g., ["neutral", "happy", "sad"])
            voice_name: Voice name for tooltip on disabled buttons
        """
        self.emotion_button_group.update_available_emotions(available_emotions, voice_name)

    def set_voice_manager(self, voice_manager):
        """
        Set the voice profile manager (now passed to settings dialog).

        Args:
            voice_manager: VoiceProfileManager instance
        """
        self.voice_manager = voice_manager

        # Update voice label with current active voice
        try:
            if voice_manager:
                active_profile = voice_manager.get_active_profile()
                if active_profile:
                    self.update_voice_label(active_profile.name)
        except Exception as e:
            self.logger.debug(f"Could not update voice label during voice manager setup: {e}")

        self.logger.debug("Voice manager set for MainWindow")

    def set_tts_service(self, tts_service):
        """
        Set the TTS service for voice creation dialogs.

        Args:
            tts_service: QwenTTSService instance
        """
        self.tts_service = tts_service
        self.logger.info(f"TTS service set for MainWindow: {'available' if tts_service else 'None'}")

        # Connect model loading/ready callbacks for UI feedback
        if tts_service:
            tts_service.set_model_loading_callback(self._on_model_loading)
            tts_service.set_model_ready_callback(self._on_model_ready)
            self.logger.debug("Connected model loading/ready callbacks")

        # Update settings dialog if it exists (propagate TTS availability)
        if self.settings_dialog:
            self.settings_dialog.set_tts_service(tts_service)
            self.logger.info("Propagated TTS service to existing settings dialog")
        else:
            self.logger.debug("Settings dialog not yet created, TTS service will be set when opened")

    def set_whisper_service(self, whisper_service):
        """
        Set the Whisper service for transcription in Voice Design Studio.

        Args:
            whisper_service: WhisperService instance for transcription
        """
        self.whisper_service = whisper_service
        self.logger.info(f"Whisper service set for MainWindow: {'available' if whisper_service else 'None'}")

        # Update settings dialog if it exists (propagate Whisper availability)
        if self.settings_dialog:
            self.settings_dialog.set_whisper_service(whisper_service)
            self.logger.info("Propagated Whisper service to existing settings dialog")
        else:
            self.logger.debug("Settings dialog not yet created, Whisper service will be set when opened")

    def update_voice_list(self, voices: list[str]):
        """
        Update the voice list (now handled by settings dialog).
        This method is kept for compatibility.

        Args:
            voices: List of available voice names
        """
        # Voice selection is now in settings dialog, this is a no-op
        self.logger.debug(f"Voice list update requested with {len(voices)} voices (handled by settings dialog)")

    def update_voice_label(self, voice_name: str):
        """
        Update the voice display label.

        Args:
            voice_name: Name of the currently selected voice
        """
        self.current_voice_label.setText(f"Voice: {voice_name}")

    def _get_ready_message(self) -> str:
        """
        Get the appropriate ready message based on TTS availability.

        Returns:
            str: Ready message reflecting current TTS status
        """
        if self._tts_available:
            return "Ready"
        else:
            return "TTS Unavailable"

    def set_generation_status(self, status: str, is_generating: bool = False):
        """
        Update the generation status display.

        Args:
            status: Status message to display
            is_generating: Whether generation is in progress
        """
        self.status_bar.showMessage(status)
        self.generate_button.setEnabled(not is_generating)

        # Story 7.4: Enable/disable pulsing animation during generation
        self.set_generation_pulsing(is_generating)

        if is_generating:
            # Change icon to indicate loading/processing for icon button
            loading_icon = self.style().standardIcon(self.style().StandardPixmap.SP_BrowserReload)
            self.generate_button.setIcon(loading_icon)
            self.generate_button.setToolTip("Generating speech...")
        else:
            # Restore original play icon
            generate_icon = self.style().standardIcon(self.style().StandardPixmap.SP_MediaPlay)
            self.generate_button.setIcon(generate_icon)
            self.generate_button.setToolTip("Generate speech (Enter)")

    def set_replay_enabled(self, enabled: bool):
        """
        Enable or disable the replay last button (Story 2.4: FR29).

        Called when audio is generated to enable replay, or on fresh start
        to keep it disabled (session-only cache).

        Args:
            enabled: Whether replay should be enabled
        """
        self.replay_button.setEnabled(enabled)
        if enabled:
            self.replay_button.setToolTip("Replay last audio (Ctrl+R)")
        else:
            self.replay_button.setToolTip("No audio to replay")

    def get_current_settings(self) -> dict:
        """
        Get current UI settings.

        Returns:
            Dictionary with current UI state
        """
        current_preset = self.emotion_button_group.get_current_preset()
        return {
            'voice': self.current_voice_label.text().replace("Voice: ", ""),
            'emotion_id': current_preset.id,
            'emotion_name': current_preset.display_name,
            'emotion_instruct': current_preset.instruct,
            'text': self.text_input.toPlainText()
        }

    # =========================================================================
    # Story 7.4: Status Indicators (FR42, FR43, FR44)
    # =========================================================================

    def _setup_default_status_indicators(self):
        """
        Setup default TTS and Mic status indicators (Story 7.4).

        Creates indicators for:
        - TTS: Shows model loading/ready/error state (FR42)
        - Mic: Shows virtual microphone connection state (FR43)
        """
        # Add TTS indicator
        self.service_status_bar.add_service("TTS")

        # Add Mic indicator
        self.service_status_bar.add_service("Mic")

        self.logger.debug("Default status indicators (TTS, Mic) created")

    def _on_service_status_clicked(self, service_name: str):
        """
        Handle click on a service status indicator (Story 7.4: FR44).

        Args:
            service_name: Name of the service that was clicked
        """
        self.logger.debug(f"Service status clicked: {service_name}")

        # Forward the signal
        self.service_status_clicked.emit(service_name)

        # Story 7.4 (FR44): Show virtual mic setup guidance when Mic warning is clicked
        if service_name.lower() == "mic":
            # Check if mic is in warning state (not detected)
            mic_indicator = self.service_status_bar.get_indicator("Mic")
            if mic_indicator:
                status_info = mic_indicator.get_current_status()
                if status_info and status_info.health_status == ServiceHealthStatus.WARNING:
                    self._show_virtual_mic_setup_dialog()

    def _show_virtual_mic_setup_dialog(self):
        """
        Show the virtual microphone setup guidance dialog (Story 7.4: FR44).
        """
        try:
            dialog = VirtualMicSetupDialog(self)
            dialog.exec()
            self.logger.info("Virtual mic setup dialog shown")
        except Exception as e:
            self.logger.error(f"Error showing virtual mic setup dialog: {e}")

    def set_tts_loading(self, is_loading: bool):
        """
        Set TTS loading state indicator (Story 7.4: FR42).

        Shows yellow/🟡 indicator while model is loading.

        Args:
            is_loading: Whether TTS model is currently loading
        """
        self.service_status_bar.set_service_loading("TTS", is_loading)

        # Also update tooltip
        if is_loading:
            indicator = self.service_status_bar.get_indicator("TTS")
            if indicator:
                indicator.setToolTip("Loading voice model...")

        self.logger.debug(f"TTS loading state: {is_loading}")

    def _on_model_loading(self, message: str):
        """
        Handle model loading callback from TTS service.

        Shows model switching indicator in status bar and TTS indicator.

        Args:
            message: Loading message (e.g., "Loading CustomVoice...")
        """
        # Update status bar with model loading message
        self.status_bar.showMessage(f"🔄 {message}")

        # Update TTS indicator to loading state
        self.service_status_bar.set_service_loading("TTS", True)
        indicator = self.service_status_bar.get_indicator("TTS")
        if indicator:
            indicator.setToolTip(message)

        self.logger.info(f"Model loading: {message}")

    def _on_model_ready(self, model_name: str):
        """
        Handle model ready callback from TTS service.

        Updates status bar and TTS indicator to show active model.

        Args:
            model_name: Name of the loaded model
        """
        # Update status bar with ready message
        self.status_bar.showMessage(f"✓ {model_name} ready", 3000)

        # Update TTS indicator to ready state
        self.service_status_bar.set_service_loading("TTS", False)
        indicator = self.service_status_bar.get_indicator("TTS")
        if indicator:
            indicator.setToolTip(f"Active model: {model_name}")

        self.logger.info(f"Model ready: {model_name}")

    def set_generation_pulsing(self, enabled: bool):
        """
        Set generation progress pulsing animation (Story 7.4).

        Shows pulsing indicator during TTS generation.

        Args:
            enabled: Whether generation is in progress
        """
        self.service_status_bar.set_service_pulsing("TTS", enabled)
        self.logger.debug(f"TTS pulsing: {enabled}")

    def update_virtual_mic_status(self, is_connected: bool, device_name: Optional[str] = None):
        """
        Update virtual microphone status indicator (Story 7.4: FR43).

        Args:
            is_connected: Whether a virtual microphone is connected
            device_name: Name of the connected device (if any)
        """
        from myvoice.models.service_enums import ServiceStatus

        if is_connected:
            status_info = ServiceStatusInfo(
                service_name="Mic",
                status=ServiceStatus.RUNNING,
                health_status=ServiceHealthStatus.HEALTHY,
                error_message=None
            )
            tooltip = f"Virtual microphone connected: {device_name}" if device_name else "Virtual microphone connected"
        else:
            status_info = ServiceStatusInfo(
                service_name="Mic",
                status=ServiceStatus.RUNNING,
                health_status=ServiceHealthStatus.WARNING,
                error_message="Virtual microphone not detected"
            )
            tooltip = "Virtual microphone not detected. Click for setup help."

        self.service_status_bar.update_service_status("Mic", status_info)

        # Update tooltip
        indicator = self.service_status_bar.get_indicator("Mic")
        if indicator:
            indicator.setToolTip(tooltip)

        self.logger.debug(f"Virtual mic status: connected={is_connected}, device={device_name}")

    def add_service_monitoring(self, service_name: str):
        """
        Add a service to the status monitoring display.

        Args:
            service_name: Name of the service to monitor
        """
        self.service_status_bar.add_service(service_name)
        self.logger.debug(f"Added service monitoring for {service_name}")

    def update_service_status(self, service_name: str, status_info: ServiceStatusInfo):
        """
        Update the status of a monitored service.

        Args:
            service_name: Name of the service
            status_info: Updated status information
        """
        # Update UI state
        self.ui_state.update_service_status(
            service_name,
            status_info.status,
            status_info.health_status,
            status_info.error_message
        )

        # Update status bar indicator
        self.service_status_bar.update_service_status(service_name, status_info)

        # Track TTS availability specifically
        if service_name.lower() == "tts":
            old_tts_available = self._tts_available
            self._tts_available = (status_info.health_status == ServiceHealthStatus.HEALTHY)

            # If TTS availability changed, always update the ready message
            if old_tts_available != self._tts_available:
                self.status_bar.showMessage(self._get_ready_message())

        # Update main status message based on overall health
        overall_health = self.service_status_bar.get_overall_health()
        if overall_health == ServiceHealthStatus.ERROR:
            self.status_bar.showMessage("Service issues detected", 5000)
        elif overall_health == ServiceHealthStatus.WARNING:
            self.status_bar.showMessage("Service warnings", 3000)

        self.logger.debug(f"Updated service status for {service_name}: {status_info.status_display}")

    def show_service_notification(self, title: str, message: str, severity: str = "info"):
        """
        Show a service-related notification to the user.

        Args:
            title: Notification title
            message: Notification message
            severity: Message severity (info, warning, error)
        """
        # For now, show in status bar (could be enhanced with popup notifications)
        if severity == "error":
            self.status_bar.showMessage(f"Error: {message}", 10000)
        elif severity == "warning":
            self.status_bar.showMessage(f"Warning: {message}", 5000)
        else:
            self.status_bar.showMessage(message, 3000)

        self.logger.info(f"Service notification: {title} - {message}")

    def get_ui_state(self) -> UIState:
        """
        Get the current UI state.

        Returns:
            UIState: Current UI state object
        """
        # Update current values
        self.ui_state.last_text_input = self.text_input.toPlainText()
        self.ui_state.selected_voice = self.current_voice_label.text().replace("Voice: ", "")
        # Store emotion preset index for persistence
        current_preset = self.emotion_button_group.get_current_preset()
        preset_list = list(EmotionPreset)
        self.ui_state.emotion_position = preset_list.index(current_preset) if current_preset in preset_list else 0
        self.ui_state.window_visible = self.isVisible()

        return self.ui_state

    def keyPressEvent(self, event: QKeyEvent):
        """
        Handle key press events.

        Handles:
        - ESC: Clear text input
        - Ctrl+R: Replay last audio (Story 2.4: FR28)
        - F1-F5: Emotion keyboard shortcuts (Story 3.5: FR6)
          - F1: Neutral, F2: Happy, F3: Sad, F4: Angry, F5: Flirtatious
          - Window-scoped (works even when text input focused)
          - No effect when emotion disabled (cloned voice)
          - No effect in dialogs (only main window)

        Args:
            event: Key event
        """
        key = event.key()

        # Clear text input when ESC is pressed
        if key == Qt.Key.Key_Escape:
            self.text_input.clear()
            self.logger.debug("Text input cleared via ESC key")
            return

        # Story 2.4: Ctrl+R to replay last audio
        if key == Qt.Key.Key_R and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if self.replay_button.isEnabled():
                self._on_replay_last_clicked()
                self.logger.debug("Replay triggered via Ctrl+R")
            else:
                self.status_bar.showMessage("No audio to replay", 2000)
                self.logger.debug("Ctrl+R ignored: no cached audio")
            return

        # Story 6.1: Ctrl+S to open Settings (FR40)
        if key == Qt.Key.Key_S and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._on_settings_clicked()
            self.logger.debug("Settings opened via Ctrl+S")
            return

        # Story 3.5: F1-F5 Emotion Keyboard Shortcuts
        # Maps F1-F5 to emotion indices 0-4
        emotion_key_map = {
            Qt.Key.Key_F1: 0,  # Neutral
            Qt.Key.Key_F2: 1,  # Happy
            Qt.Key.Key_F3: 2,  # Sad
            Qt.Key.Key_F4: 3,  # Angry
            Qt.Key.Key_F5: 4,  # Flirtatious
        }

        if key in emotion_key_map:
            # Check if emotion control is enabled (not disabled for cloned voices)
            if self.emotion_button_group.is_emotion_enabled():
                emotion_index = emotion_key_map[key]
                success = self.set_emotion_by_index(emotion_index)
                if success:
                    preset = self.emotion_button_group.get_current_preset()
                    self.logger.debug(f"Emotion shortcut {event.text()}: {preset.display_name}")
            else:
                # Emotion disabled (cloned voice) - shortcut has no effect (AC8)
                self.logger.debug(f"Emotion shortcut {event.text()} ignored: emotion control disabled")
            return

        # Pass other key events to parent
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent):
        """
        Handle window close event (Story 7.2: FR38).

        If minimize_to_tray is enabled and _force_quit is False,
        minimizes to system tray instead of closing.
        Otherwise, performs proper cleanup and closes.

        Args:
            event: Close event
        """
        # Story 7.2: Check if we should minimize to tray instead of closing
        should_minimize_to_tray = (
            not self._force_quit and
            self.tray_icon is not None and
            self.app_settings is not None and
            self.app_settings.minimize_to_tray
        )

        if should_minimize_to_tray:
            # Minimize to tray instead of closing
            event.ignore()
            self._minimize_to_tray()
            self.logger.debug("Close event ignored - minimized to tray")
            return

        # QA4-3: Show confirmation dialog unless force quit (from tray Exit)
        if not self._force_quit:
            from PyQt6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                "Close MyVoice",
                "Are you sure you wish to close MyVoice?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                self.logger.debug("Close cancelled by user")
                return

        # Actual close - perform cleanup
        self.logger.info("MainWindow closing")

        try:
            # Hide tray icon if exists
            if self.tray_icon:
                self.tray_icon.hide()

            # Stop auto-save timer
            if hasattr(self, '_auto_save_timer'):
                self._auto_save_timer.stop()

            # Cleanup voice selector
            if hasattr(self, 'voice_selector'):
                self.voice_selector.cleanup()

            # Cleanup service status bar timers
            if hasattr(self, 'service_status_bar'):
                self.service_status_bar.cleanup_all_services()

            # Cleanup background processes (launcher)
            self._cleanup_background_processes()

            # TODO: Save window geometry and state when configuration service is ready

            # Accept the close event
            event.accept()

        except Exception as e:
            self.logger.exception(f"Error during window close: {e}")
            event.accept()  # Close anyway to prevent hanging

    def _cleanup_background_processes(self):
        """
        Terminate background processes spawned by launcher.

        This terminates any tracked processes using PID files created during startup.
        """
        import os
        import subprocess
        import tempfile

        temp_dir = tempfile.gettempdir()

        # PID files to check
        pid_files = [
            os.path.join(temp_dir, "myvoice_splash.pid"),
        ]

        for pid_file in pid_files:
            try:
                if os.path.exists(pid_file):
                    with open(pid_file, 'r') as f:
                        pid = f.read().strip()

                    if pid and pid.isdigit():
                        self.logger.info(f"Terminating process with PID: {pid}")
                        # Use taskkill with /T to terminate process tree, /F for force
                        subprocess.run(
                            ['taskkill', '/PID', pid, '/T', '/F'],
                            capture_output=True,
                            timeout=5
                        )

                    # Remove PID file
                    os.remove(pid_file)
                    self.logger.debug(f"Cleaned up PID file: {pid_file}")

            except Exception as e:
                # Log but don't fail - cleanup is best-effort
                self.logger.warning(f"Failed to cleanup process from {pid_file}: {e}")

        self.logger.info("Background process cleanup completed")

    def show_and_raise(self):
        """Show the window and bring it to front."""
        self.show()
        self.raise_()
        self.activateWindow()
        self.text_input.setFocus()

    # Settings Management
    def set_audio_coordinator(self, audio_coordinator):
        """
        Set the audio coordinator for dual-service architecture.

        Args:
            audio_coordinator: AudioCoordinator instance
        """
        self.audio_coordinator = audio_coordinator
        self.logger.debug("Audio coordinator set for MainWindow")

    def set_app_settings(self, app_settings: AppSettings):
        """
        Set the application settings instance.

        Args:
            app_settings: AppSettings instance
        """
        self.app_settings = app_settings
        self.logger.debug("App settings set for MainWindow")

        # Initialize Quick Speak service
        self.quick_speak_service = QuickSpeakService(app_settings.config_directory)
        self.quick_speak_service.load_entries()

        # Apply theme from settings
        self.apply_theme_from_settings()

        # Apply always-on-top setting
        self.set_always_on_top(app_settings.always_on_top)

        # Story 7.3: Apply window transparency on startup (FR41)
        self.set_window_transparency(app_settings.window_transparency)

    def set_always_on_top(self, always_on_top: bool):
        """
        Set the always-on-top behavior of the window.

        Args:
            always_on_top: Whether to keep window always on top
        """
        try:
            # Base frameless window flags
            base_flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window

            if always_on_top:
                # Add the always on top flag
                new_flags = base_flags | Qt.WindowType.WindowStaysOnTopHint
            else:
                # Keep just the base frameless flags
                new_flags = base_flags

            current_flags = self.windowFlags()

            # Only update if flags actually changed
            if new_flags != current_flags:
                # Store current geometry before changing flags
                current_geometry = self.geometry()

                # Pre-apply theme before flag change to ensure it's in memory
                self._force_theme_reapplication()

                # Set flag to indicate theme needs reapplication
                self._needs_theme_reapplication = True

                # Hide window before flag change to minimize visual glitches
                self.hide()

                self.setWindowFlags(new_flags)
                self.show()  # Required to apply flag changes

                # Restore geometry
                self.setGeometry(current_geometry)

                # Apply theme immediately after show
                self._force_theme_reapplication()

                # Schedule multiple theme applications to ensure consistency
                QTimer.singleShot(1, self._force_theme_reapplication)
                QTimer.singleShot(10, self._force_theme_reapplication)
                QTimer.singleShot(50, self._force_theme_reapplication)
                QTimer.singleShot(100, self._force_theme_reapplication)

                # Ensure window is properly activated and focused
                self.activateWindow()
                self.raise_()

                self.logger.debug(f"Set always-on-top to: {always_on_top} with aggressive theme reapplication")

        except Exception as e:
            self.logger.error(f"Error setting always-on-top: {e}")

    def set_window_transparency(self, opacity: float):
        """
        Set the window transparency (Story 7.3: FR41).

        Args:
            opacity: Window opacity value (0.2-1.0)
                     0.2 = 20% opaque (80% transparent)
                     1.0 = 100% opaque (fully visible)
        """
        try:
            # Enforce minimum 20% opacity (maximum 80% transparency)
            # This ensures the window is never fully invisible
            clamped_opacity = max(0.2, min(1.0, opacity))

            self.setWindowOpacity(clamped_opacity)

            self.logger.debug(f"Set window transparency to: {clamped_opacity * 100:.0f}%")

        except Exception as e:
            self.logger.error(f"Error setting window transparency: {e}")

    def _on_transparency_preview(self, opacity: float):
        """
        Handle real-time transparency preview from settings dialog (Story 7.3).

        Called when user moves the transparency slider to provide immediate
        visual feedback without needing to apply settings.

        Args:
            opacity: Window opacity value (0.2-1.0)
        """
        self.set_window_transparency(opacity)

    def _on_settings_clicked(self):
        """Handle settings button click."""
        try:
            if not self.app_settings:
                self.status_bar.showMessage("Settings not available", 3000)
                return

            # Create settings dialog if not exists
            if not self.settings_dialog:
                # Get audio_client from audio_coordinator for smart device matching
                audio_client = None
                if self.audio_coordinator and hasattr(self.audio_coordinator, 'monitor_service'):
                    if hasattr(self.audio_coordinator.monitor_service, 'windows_audio_client'):
                        audio_client = self.audio_coordinator.monitor_service.windows_audio_client

                self.settings_dialog = SettingsDialog(
                    self.app_settings,
                    self,
                    self.quick_speak_service,
                    audio_client=audio_client
                )

                # Connect signals
                self.settings_dialog.settings_changed.connect(self._on_settings_changed)
                self.settings_dialog.settings_changed.connect(self.settings_changed.emit)
                self.settings_dialog.device_refresh_requested.connect(self._on_device_refresh_requested)
                self.settings_dialog.device_test_requested.connect(self.audio_device_test_requested.emit)
                self.settings_dialog.virtual_device_test_requested.connect(self.virtual_device_test_requested.emit)
                self.settings_dialog.voice_directory_changed.connect(self._on_voice_directory_changed)
                self.settings_dialog.voice_refresh_requested.connect(self._on_voice_refresh_requested)
                self.settings_dialog.voice_transcription_requested.connect(self._on_voice_transcription_requested)
                self.settings_dialog.quick_speak_entries_changed.connect(self._on_quick_speak_entries_changed)
                self.settings_dialog.voice_changed.connect(self._on_voice_changed_from_settings)
                # Custom emotion signal (Story 3.4)
                self.settings_dialog.custom_emotion_applied.connect(self._on_custom_emotion_applied)
                # Transparency preview signal (Story 7.3)
                self.settings_dialog.transparency_preview_requested.connect(self._on_transparency_preview)
                # Voice creation/deletion signals (Story 4.2: Voice Design Creation Workflow)
                self.settings_dialog.voice_created.connect(self._on_voice_created_from_settings)
                self.settings_dialog.voice_deleted.connect(self._on_voice_deleted_from_settings)
            else:
                # Update with current settings
                self.settings_dialog.current_settings = AppSettings.from_dict(self.app_settings.to_dict())
                self.settings_dialog._load_current_settings()

            # Set voice manager on settings dialog
            if self.voice_manager:
                self.settings_dialog.set_voice_manager(self.voice_manager)

            # Set TTS service on settings dialog for voice creation dialogs
            # Always call to ensure buttons are in correct state (enabled if TTS ready, disabled if not)
            self.settings_dialog.set_tts_service(self.tts_service)
            self.logger.debug(f"Set TTS service on settings dialog: tts_service={'available' if self.tts_service else 'None'}")

            # Set Whisper service on settings dialog for transcription (QA3 fix)
            if hasattr(self, 'whisper_service') and self.whisper_service:
                self.settings_dialog.set_whisper_service(self.whisper_service)
                self.logger.debug(f"Set Whisper service on settings dialog")

            # Populate device list if audio manager is available
            # CRITICAL: This must happen BEFORE showing the dialog so dropdowns are populated
            if self.audio_coordinator:
                self.logger.info("Populating device lists in settings dialog")
                self._populate_device_list()
            else:
                self.logger.warning("Audio coordinator not available, cannot populate device lists")

            # QA Round 2 Item #5: Always refresh voice list before showing dialog
            # This ensures newly saved voices from VDS are visible
            self.settings_dialog.refresh_voice_list()

            # Show dialog (Story 6.1: FR40)
            self.settings_dialog.exec()

            # Story 6.1: Return focus to text input after settings closes
            self.text_input.setFocus()
            self.logger.debug("Focus returned to text input after settings closed")

        except Exception as e:
            self.logger.error(f"Error opening settings dialog: {e}")
            self.status_bar.showMessage("Error opening settings", 3000)

    def _on_voice_design_studio_clicked(self):
        """
        Handle Voice Design Studio menu action (Story 1.2: FR1).

        Opens the Voice Design Studio dialog for creating voices
        from descriptions or audio samples with acoustic embeddings.
        """
        try:
            self.logger.debug("Opening Voice Design Studio")

            # Create dialog if not exists
            if not self.voice_design_studio_dialog:
                self.voice_design_studio_dialog = VoiceDesignStudioDialog(self)

                # Connect signals
                self.voice_design_studio_dialog.voice_saved.connect(
                    self._on_voice_saved_from_design_studio
                )
                self.voice_design_studio_dialog.dialog_closing.connect(
                    self._on_voice_design_studio_closing
                )

            # Show dialog
            self.voice_design_studio_dialog.exec()

            # QA3-3: Refresh voice list AFTER dialog closes (not on save)
            # This ensures save operation completes before refresh
            self.voice_refresh_requested.emit()

            # Return focus to text input after dialog closes
            self.text_input.setFocus()
            self.logger.debug("Focus returned to text input after Voice Design Studio closed")

        except Exception as e:
            self.logger.error(f"Error opening Voice Design Studio: {e}")
            self.status_bar.showMessage("Error opening Voice Design Studio", 3000)

    def _on_voice_saved_from_design_studio(self, voice_name: str):
        """
        Handle voice saved signal from Voice Design Studio.

        Args:
            voice_name: Name of the saved voice
        """
        self.logger.info(f"Voice saved from Design Studio: {voice_name}")
        self.status_bar.showMessage(f"Voice '{voice_name}' saved successfully", 5000)
        # QA3-3: Don't refresh here - refresh happens after dialog closes

    def _on_voice_design_studio_closing(self, has_unsaved_work: bool):
        """
        Handle Voice Design Studio closing signal.

        Args:
            has_unsaved_work: True if there is unsaved work being discarded
        """
        if has_unsaved_work:
            self.logger.debug("Voice Design Studio closed with unsaved work")
        else:
            self.logger.debug("Voice Design Studio closed")

    def _on_device_refresh_requested(self):
        """Handle device refresh request from settings dialog."""
        try:
            self.audio_device_refresh_requested.emit()

            # Refresh device list after a short delay
            QTimer.singleShot(1000, self._populate_device_list)

        except Exception as e:
            self.logger.error(f"Error refreshing device list: {e}")

    def _populate_device_list(self):
        """Populate device list in settings dialog using device refresh mechanism."""
        try:
            if not self.audio_coordinator:
                self.logger.error("Cannot populate device list: audio_coordinator is None")
                return

            if not self.settings_dialog:
                self.logger.error("Cannot populate device list: settings_dialog is None")
                return

            self.logger.info("Triggering device refresh for settings dialog")

            # Instead of trying to enumerate ourselves (which causes event loop issues),
            # trigger the existing device refresh mechanism which works correctly
            self.audio_device_refresh_requested.emit()

            self.logger.info("Device refresh signal emitted successfully")

        except Exception as e:
            self.logger.error(f"Error populating device list: {e}", exc_info=True)

    def update_settings(self, new_settings: AppSettings):
        """
        Update the current application settings.

        Args:
            new_settings: New settings to apply
        """
        self.app_settings = new_settings

        # Update settings dialog if open
        if self.settings_dialog:
            self.settings_dialog.current_settings = AppSettings.from_dict(new_settings.to_dict())
            self.settings_dialog._load_current_settings()

        self.logger.debug("Updated application settings")

    def _on_voice_directory_changed(self, directory_path: str):
        """
        Handle voice directory change from settings dialog.

        Args:
            directory_path: New voice files directory path
        """
        try:
            self.logger.info(f"Voice directory changed to: {directory_path}")

            # Emit signal for app controller to handle
            self.voice_directory_changed.emit(directory_path)

            # Show feedback in status bar
            self.status_bar.showMessage(f"Voice directory updated: {directory_path}", 3000)

        except Exception as e:
            self.logger.error(f"Error handling voice directory change: {e}")

    def _on_voice_refresh_requested(self):
        """Handle voice refresh request from settings dialog."""
        try:
            self.logger.info("Voice refresh requested from settings")

            # Emit signal for app controller to handle
            self.voice_refresh_requested.emit()

            # Show feedback in status bar
            self.status_bar.showMessage("Refreshing voice profiles...", 2000)

        except Exception as e:
            self.logger.error(f"Error handling voice refresh request: {e}")

    def _on_voice_transcription_requested(self, voice_name: str):
        """
        Handle transcription request from settings dialog.

        Args:
            voice_name: Name of the voice profile to transcribe
        """
        try:
            self.logger.info(f"Transcription requested for voice: {voice_name}")

            # Emit signal for app controller to handle
            self.voice_transcription_requested.emit(voice_name)

            # Show feedback in status bar
            self.status_bar.showMessage(f"Queueing transcription for {voice_name}...", 2000)

        except Exception as e:
            self.logger.error(f"Error handling transcription request: {e}")

    def _on_quick_speak_clicked(self):
        """
        Handle Quick Speak button click.

        Story 5.2: Shows popup menu for quick phrase selection.
        - 3 interactions max: open menu, select phrase, auto-generate
        - Keyboard navigation with arrow keys + Enter
        - Escape closes menu
        """
        try:
            if not self.quick_speak_service:
                self.status_bar.showMessage("Quick Speak not available", 3000)
                return

            # Create menu if not exists
            if not self.quick_speak_menu:
                self.quick_speak_menu = QuickSpeakMenu(self.quick_speak_service, self)
                self.quick_speak_menu.phrase_selected.connect(self._on_quick_speak_entry_selected)
                self.quick_speak_menu.open_settings_requested.connect(self._on_quick_speak_settings_requested)

            # Show menu at button position
            self.quick_speak_menu.show_at_button(self.quick_speak_button)

        except Exception as e:
            self.logger.error(f"Error opening Quick Speak menu: {e}")
            self.status_bar.showMessage("Error opening Quick Speak", 3000)

    def _on_quick_speak_settings_requested(self):
        """
        Handle request to open settings from Quick Speak menu.

        Story 5.2: Navigate to Quick Speak settings when user clicks configure.
        """
        try:
            self.logger.info("Quick Speak settings requested from menu")
            self._on_settings_clicked()

            # Navigate to Quick Speak tab if settings dialog is open
            if self.settings_dialog:
                self.settings_dialog.navigate_to_quick_speak()

        except Exception as e:
            self.logger.error(f"Error opening Quick Speak settings: {e}")

    def _on_quick_speak_entry_selected(self, text: str):
        """
        Handle Quick Speak entry selection with append mode.

        If the current text ends with an apostrophe "'", the apostrophe is kept and
        the selected text is appended. Otherwise, the text is replaced.

        Args:
            text: Selected text to speak
        """
        try:
            self.logger.info(f"Quick Speak entry selected: {text[:50]}...")

            # Get current text in input field
            current_text = self.text_input.toPlainText()

            # Check if current text ends with apostrophe for append mode
            if current_text.endswith("'"):
                # Append mode: Keep apostrophe and append selected text with space
                final_text = f"{current_text} {text}"
                self.logger.info(f"Quick Speak append mode: '{current_text}' + '{text}' = '{final_text}'")
                self.status_bar.showMessage("Quick Speak appended and generating...", 3000)
            else:
                # Replace mode: Normal behavior
                final_text = text
                self.logger.info(f"Quick Speak replace mode: '{text}'")
                self.status_bar.showMessage("Quick Speak entry loaded and generating...", 3000)

            # Set final text in input field
            self.text_input.setPlainText(final_text)

            # Trigger TTS generation
            self._start_tts_generation(final_text)

        except Exception as e:
            self.logger.error(f"Error handling Quick Speak selection: {e}")
            self.status_bar.showMessage("Error processing Quick Speak entry", 3000)

    def _on_quick_speak_entries_changed(self):
        """
        Handle Quick Speak entries or profile change.

        Invalidate cached menu and dialog so they refresh with new profile data.
        """
        try:
            # Invalidate cached menu to force refresh on next open
            if self.quick_speak_menu:
                self.quick_speak_menu.deleteLater()
                self.quick_speak_menu = None
                self.logger.debug("Quick Speak menu invalidated due to entries/profile change")

            # Also invalidate dialog if it exists
            if self.quick_speak_dialog:
                self.quick_speak_dialog.deleteLater()
                self.quick_speak_dialog = None
                self.logger.debug("Quick Speak dialog invalidated due to entries/profile change")
        except Exception as e:
            self.logger.error(f"Error handling Quick Speak entries change: {e}")