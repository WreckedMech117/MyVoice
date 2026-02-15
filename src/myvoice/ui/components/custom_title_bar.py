"""
Custom Title Bar Component

This module implements a custom title bar for frameless windows with
window control buttons and draggable functionality.
"""

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt, QPoint, QSize
from PyQt6.QtGui import QMouseEvent, QPixmap, QIcon


class CustomTitleBar(QWidget):
    """
    Custom title bar widget for frameless windows.

    Provides window control buttons (minimize, maximize/restore, close),
    draggable title area, and compact design suitable for desktop applications.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the custom title bar.

        Args:
            parent: Parent widget (typically the main window)
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Track dragging state
        self._drag_position = QPoint()
        self._is_dragging = False

        # Reference to parent window
        self._parent_window = parent

        # Track maximize state
        self._is_maximized = False

        # Set object name for QSS styling
        self.setObjectName("custom_title_bar")

        # Set fixed height for compact design
        self.setFixedHeight(28)

        # Create UI components
        self._create_ui()

        self.logger.debug("CustomTitleBar initialized")

    def _create_ui(self):
        """Build the title bar layout with title and control buttons."""
        # Main horizontal layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(4)

        # App icon (if available)
        icon_label = QLabel()
        icon_path = Path(__file__).parent.parent.parent.parent / "icon" / "MyVoice.png"
        if icon_path.exists():
            pixmap = QPixmap(str(icon_path))
            # Scale to 20x20 for title bar
            scaled_pixmap = pixmap.scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            icon_label.setPixmap(scaled_pixmap)
            icon_label.setFixedSize(20, 20)
            layout.addWidget(icon_label)

        # App title/icon area
        self.title_label = QLabel("MyVoice")
        self.title_label.setObjectName("title_label")
        layout.addWidget(self.title_label)

        # Spacer to push buttons to the right
        layout.addStretch()

        # Window control buttons
        self._setup_buttons(layout)

    def _setup_buttons(self, layout: QHBoxLayout):
        """
        Create and configure window control buttons.

        Args:
            layout: Layout to add buttons to
        """
        # Minimize button
        self.minimize_button = QPushButton("−")
        self.minimize_button.setObjectName("minimize_button")
        self.minimize_button.setFixedSize(32, 24)
        self.minimize_button.clicked.connect(self._on_minimize_clicked)
        self.minimize_button.setToolTip("Minimize")
        layout.addWidget(self.minimize_button)

        # Maximize/Restore button
        self.maximize_button = QPushButton("□")
        self.maximize_button.setObjectName("maximize_button")
        self.maximize_button.setFixedSize(32, 24)
        self.maximize_button.clicked.connect(self._on_maximize_clicked)
        self.maximize_button.setToolTip("Maximize")
        layout.addWidget(self.maximize_button)

        # Close button
        self.close_button = QPushButton("×")
        self.close_button.setObjectName("close_button")
        self.close_button.setFixedSize(32, 24)
        self.close_button.clicked.connect(self._on_close_clicked)
        self.close_button.setToolTip("Close")
        layout.addWidget(self.close_button)

    def mousePressEvent(self, event: QMouseEvent):
        """
        Handle mouse press to initiate window dragging.

        Args:
            event: Mouse event
        """
        if event.button() == Qt.MouseButton.LeftButton:
            # Store the position where dragging started
            self._drag_position = event.globalPosition().toPoint() - self._parent_window.frameGeometry().topLeft()
            self._is_dragging = True
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """
        Handle mouse move to drag the window.

        Args:
            event: Mouse event
        """
        if self._is_dragging and event.buttons() == Qt.MouseButton.LeftButton:
            # Move the window to follow the mouse
            new_pos = event.globalPosition().toPoint() - self._drag_position
            self._parent_window.move(new_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """
        Handle mouse release to stop dragging.

        Args:
            event: Mouse event
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """
        Handle double-click to toggle maximize/restore.

        Args:
            event: Mouse event
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_maximize_clicked()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def _on_minimize_clicked(self):
        """
        Handle minimize button click.

        Story 7.2: If parent window has system tray enabled,
        minimizes to tray instead of taskbar.
        """
        if self._parent_window:
            # Story 7.2: Check if parent supports minimize to tray
            if hasattr(self._parent_window, '_minimize_to_tray'):
                self._parent_window._minimize_to_tray()
                self.logger.debug("Window minimized to tray")
            else:
                self._parent_window.showMinimized()
                self.logger.debug("Window minimized")

    def _on_maximize_clicked(self):
        """Handle maximize/restore button click."""
        if self._parent_window:
            if self._is_maximized:
                self._parent_window.showNormal()
                self.maximize_button.setText("□")
                self.maximize_button.setToolTip("Maximize")
                self._is_maximized = False
                self.logger.debug("Window restored")
            else:
                self._parent_window.showMaximized()
                self.maximize_button.setText("❐")
                self.maximize_button.setToolTip("Restore")
                self._is_maximized = True
                self.logger.debug("Window maximized")

    def _on_close_clicked(self):
        """Handle close button click."""
        if self._parent_window:
            self._parent_window.close()
            self.logger.debug("Window close requested")

    def update_maximize_state(self, is_maximized: bool):
        """
        Update the maximize button state externally.

        Args:
            is_maximized: Whether the window is currently maximized
        """
        self._is_maximized = is_maximized
        if is_maximized:
            self.maximize_button.setText("❐")
            self.maximize_button.setToolTip("Restore")
        else:
            self.maximize_button.setText("□")
            self.maximize_button.setToolTip("Maximize")