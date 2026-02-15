"""
Model Loading Indicator Widget

This module provides a UI component for displaying model loading status
to the user. Shows "Loading voice model..." feedback during model switches.

Story 4.5: Model Lazy Loading for Voice Types
- Loading feedback: "Loading voice model..."
- Progress indication during model load
- Hides automatically when model is ready
"""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QProgressBar, QFrame
)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont


class ModelLoadingIndicator(QWidget):
    """
    Widget for displaying model loading status.

    Shows a loading message and progress bar when a TTS model is being loaded.
    Automatically hides when loading completes.

    Story 4.5 Features:
    - Shows "Loading voice model..." during model load
    - Progress bar for visual feedback
    - Smooth fade in/out animations
    - Auto-hide on completion
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the Model Loading Indicator.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        self._is_loading = False
        self._auto_hide_timer = QTimer(self)
        self._auto_hide_timer.setSingleShot(True)
        self._auto_hide_timer.timeout.connect(self._do_hide)

        self._setup_ui()

        # Initially hidden
        self.setVisible(False)

        self.logger.debug("ModelLoadingIndicator initialized")

    def _setup_ui(self):
        """Create the widget UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Container frame with styling
        self.setStyleSheet("""
            ModelLoadingIndicator {
                background-color: #e3f2fd;
                border: 1px solid #90caf9;
                border-radius: 4px;
            }
        """)

        # Loading icon/emoji
        self.icon_label = QLabel("⏳")
        self.icon_label.setFixedWidth(20)
        layout.addWidget(self.icon_label)

        # Loading message
        self.message_label = QLabel("Loading voice model...")
        font = QFont()
        font.setPointSize(9)
        self.message_label.setFont(font)
        layout.addWidget(self.message_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate by default
        self.progress_bar.setFixedWidth(100)
        self.progress_bar.setMaximumHeight(8)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        # Stretch to push content left
        layout.addStretch()

        # Set fixed height
        self.setFixedHeight(32)

    @pyqtSlot(str, str)
    def on_loading_started(self, model_type: str, message: str):
        """
        Handle model loading started.

        Args:
            model_type: Name of model being loaded
            message: Loading message
        """
        self._is_loading = True
        self._auto_hide_timer.stop()

        self.message_label.setText(message)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.icon_label.setText("⏳")

        self.setVisible(True)
        self.logger.debug(f"Loading started: {message}")

    @pyqtSlot(float, str)
    def on_loading_progress(self, percent: float, message: str):
        """
        Handle loading progress update.

        Args:
            percent: Progress percentage (0-100)
            message: Progress message
        """
        if percent > 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(int(percent))
        else:
            self.progress_bar.setRange(0, 0)  # Indeterminate

        if message:
            self.message_label.setText(message)

    @pyqtSlot(bool, str)
    def on_loading_complete(self, success: bool, message: str):
        """
        Handle loading completion.

        Args:
            success: Whether loading succeeded
            message: Completion message or error
        """
        self._is_loading = False

        if success:
            self.icon_label.setText("✓")
            self.message_label.setText(message or "Model ready")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)

            # Auto-hide after delay
            self._auto_hide_timer.start(2000)  # 2 seconds
        else:
            self.icon_label.setText("✗")
            self.message_label.setText(f"Failed: {message}")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)

            # Keep visible longer for error
            self._auto_hide_timer.start(5000)  # 5 seconds

        self.logger.debug(f"Loading complete: success={success}, message={message}")

    @pyqtSlot(str)
    def on_model_ready(self, model_type: str):
        """
        Handle model ready signal.

        Args:
            model_type: Name of ready model
        """
        if not self._is_loading:
            # If not actively loading, just update and hide
            self.message_label.setText(f"{model_type} ready")
            self.icon_label.setText("✓")
            self._auto_hide_timer.start(1500)

    def _do_hide(self):
        """Hide the indicator."""
        self.setVisible(False)

    def show_loading(self, message: str = "Loading voice model..."):
        """
        Manually show loading state.

        Args:
            message: Loading message to display
        """
        self.on_loading_started("", message)

    def show_ready(self, message: str = "Model ready"):
        """
        Manually show ready state.

        Args:
            message: Ready message to display
        """
        self.on_loading_complete(True, message)

    def show_error(self, message: str):
        """
        Manually show error state.

        Args:
            message: Error message to display
        """
        self.on_loading_complete(False, message)

    def hide_indicator(self):
        """Manually hide the indicator immediately."""
        self._auto_hide_timer.stop()
        self.setVisible(False)

    def is_loading(self) -> bool:
        """Check if currently showing loading state."""
        return self._is_loading


class ModelLoadingOverlay(QWidget):
    """
    Full overlay for blocking UI during critical model operations.

    Used when user must wait for model loading to complete before
    proceeding (e.g., first use of a voice type).
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the overlay.

        Args:
            parent: Parent widget to overlay
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        self._setup_ui()

        # Initially hidden
        self.setVisible(False)

    def _setup_ui(self):
        """Create the overlay UI."""
        # Semi-transparent background
        self.setStyleSheet("""
            ModelLoadingOverlay {
                background-color: rgba(0, 0, 0, 0.5);
            }
        """)

        layout = QHBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Central loading card
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                padding: 20px;
            }
        """)
        card.setFixedSize(300, 120)

        card_layout = QHBoxLayout(card)
        card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Loading content
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)

        self.icon_label = QLabel("⏳")
        icon_font = QFont()
        icon_font.setPointSize(24)
        self.icon_label.setFont(icon_font)
        content_layout.addWidget(self.icon_label)

        text_widget = QWidget()
        text_layout = QHBoxLayout(text_widget)
        text_layout.setContentsMargins(12, 0, 0, 0)

        self.message_label = QLabel("Loading voice model...")
        message_font = QFont()
        message_font.setPointSize(11)
        self.message_label.setFont(message_font)
        text_layout.addWidget(self.message_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setMaximumHeight(6)
        self.progress_bar.setTextVisible(False)
        text_layout.addWidget(self.progress_bar)

        content_layout.addWidget(text_widget)
        card_layout.addWidget(content_widget)
        layout.addWidget(card)

    @pyqtSlot(str, str)
    def on_loading_started(self, model_type: str, message: str):
        """Handle loading started."""
        self.message_label.setText(message)
        self.progress_bar.setRange(0, 0)
        self.icon_label.setText("⏳")
        self.setVisible(True)

    @pyqtSlot(bool, str)
    def on_loading_complete(self, success: bool, message: str):
        """Handle loading complete."""
        self.setVisible(False)

    def show_loading(self, message: str = "Loading voice model..."):
        """Show the overlay with a message."""
        self.message_label.setText(message)
        self.setVisible(True)

    def hide_overlay(self):
        """Hide the overlay."""
        self.setVisible(False)
