"""
Device Notification Widget

This module provides a UI component for displaying audio device change
notifications to users in a non-intrusive way.

Story 2.5: Audio Device Resilience
- FR30: Handle audio device changes gracefully
- Shows warnings on device disconnect
- Shows info on device reconnect
"""

import logging
from typing import Optional, List
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QGraphicsOpacityEffect
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QColor

from myvoice.models.device_notification import (
    DeviceNotification,
    NotificationType,
    NotificationSeverity,
)


class DeviceNotificationItem(QFrame):
    """Individual device notification item widget."""

    # Signal emitted when close button is clicked
    close_requested = pyqtSignal(object)  # DeviceNotificationItem

    def __init__(
        self,
        notification: DeviceNotification,
        parent: Optional[QWidget] = None
    ):
        """
        Initialize device notification item.

        Args:
            notification: Device notification to display
            parent: Parent widget
        """
        super().__init__(parent)
        self.notification = notification
        self.logger = logging.getLogger(self.__class__.__name__)

        # Auto-dismiss timer
        self._auto_dismiss_timer = QTimer()
        self._auto_dismiss_timer.timeout.connect(self._on_auto_dismiss)

        # Setup UI
        self._create_ui()
        self._apply_styling()

        # Start auto-dismiss timer if configured
        if notification.auto_dismiss_seconds:
            self._auto_dismiss_timer.start(notification.auto_dismiss_seconds * 1000)

    def _create_ui(self):
        """Create the notification item UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # Icon based on notification type
        icon_label = QLabel(self._get_icon())
        icon_label.setFont(QFont("Segoe UI Emoji", 14))
        layout.addWidget(icon_label)

        # Content area
        content_layout = QVBoxLayout()
        content_layout.setSpacing(2)

        # Title
        title_label = QLabel(self.notification.title)
        title_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        title_label.setWordWrap(True)
        content_layout.addWidget(title_label)

        # Message
        message_label = QLabel(self.notification.message)
        message_label.setFont(QFont("Segoe UI", 9))
        message_label.setWordWrap(True)
        message_label.setStyleSheet("color: #888;")
        content_layout.addWidget(message_label)

        # Suggested action (if any)
        if self.notification.suggested_action:
            action_label = QLabel(self.notification.suggested_action)
            action_label.setFont(QFont("Segoe UI", 8, QFont.Weight.Normal, italic=True))
            action_label.setStyleSheet("color: #666;")
            action_label.setWordWrap(True)
            content_layout.addWidget(action_label)

        layout.addLayout(content_layout, 1)

        # Close button
        close_button = QPushButton("x")
        close_button.setFixedSize(20, 20)
        close_button.setStyleSheet("""
            QPushButton {
                border: none;
                background: transparent;
                color: #888;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #fff;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
            }
        """)
        close_button.clicked.connect(lambda: self.close_requested.emit(self))
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignTop)

    def _get_icon(self) -> str:
        """Get icon based on notification type."""
        icons = {
            NotificationType.DEVICE_ADDED: "🔌",
            NotificationType.DEVICE_REMOVED: "⚠️",
            NotificationType.DEVICE_CHANGED: "🔄",
            NotificationType.DEVICE_FALLBACK: "↩️",
            NotificationType.DEVICE_RESTORED: "✅",
            NotificationType.DEVICE_LIST_REFRESHED: "🔄",
        }
        return icons.get(self.notification.notification_type, "🔊")

    def _apply_styling(self):
        """Apply styling based on notification severity."""
        # Color scheme based on severity
        colors = {
            NotificationSeverity.INFO: ("#2d2d2d", "#1a5f7a", "#5ab4dc"),
            NotificationSeverity.WARNING: ("#3d3020", "#7a5f1a", "#dcb45a"),
            NotificationSeverity.ERROR: ("#3d2020", "#7a1a1a", "#dc5a5a"),
            NotificationSeverity.CRITICAL: ("#4d1a1a", "#9a1a1a", "#ff5a5a"),
        }

        bg_color, border_color, accent_color = colors.get(
            self.notification.severity,
            colors[NotificationSeverity.INFO]
        )

        self.setStyleSheet(f"""
            DeviceNotificationItem {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-left: 3px solid {accent_color};
                border-radius: 4px;
            }}
        """)

    def _on_auto_dismiss(self):
        """Handle auto-dismiss timeout."""
        self._auto_dismiss_timer.stop()
        self.close_requested.emit(self)


class DeviceNotificationWidget(QWidget):
    """
    Container widget for displaying device notifications.

    Story 2.5: Shows notifications for:
    - Device disconnected (warning)
    - Device reconnected (info)
    - Fallback to another device (warning)
    - Device list refreshed (info)
    """

    # Maximum number of visible notifications
    MAX_NOTIFICATIONS = 3

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize device notification widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Notification items
        self._notification_items: List[DeviceNotificationItem] = []

        # Setup UI
        self._create_ui()

    def _create_ui(self):
        """Create the notification container UI."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(8)
        self.layout.addStretch()

        # Set fixed width for notifications
        self.setFixedWidth(350)

        # Start hidden if no notifications
        self.setVisible(False)

    def add_notification(self, notification: DeviceNotification) -> None:
        """
        Add a new device notification.

        Args:
            notification: DeviceNotification to display
        """
        self.logger.info(f"Adding device notification: {notification.title}")

        # Create notification item
        item = DeviceNotificationItem(notification, self)
        item.close_requested.connect(self._on_item_close_requested)

        # Add to layout (before the stretch)
        self.layout.insertWidget(self.layout.count() - 1, item)
        self._notification_items.append(item)

        # Show widget
        self.setVisible(True)

        # Remove oldest notifications if over limit
        while len(self._notification_items) > self.MAX_NOTIFICATIONS:
            self._remove_oldest_notification()

        # Animate in
        self._animate_in(item)

    def _animate_in(self, item: DeviceNotificationItem) -> None:
        """Animate notification item appearing."""
        try:
            effect = QGraphicsOpacityEffect(item)
            item.setGraphicsEffect(effect)

            animation = QPropertyAnimation(effect, b"opacity")
            animation.setDuration(200)
            animation.setStartValue(0.0)
            animation.setEndValue(1.0)
            animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            animation.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

            # Store animation reference to prevent garbage collection
            item._fade_animation = animation
        except Exception as e:
            self.logger.debug(f"Animation error: {e}")

    def _animate_out(self, item: DeviceNotificationItem, callback=None) -> None:
        """Animate notification item disappearing."""
        try:
            effect = item.graphicsEffect()
            if not effect or not isinstance(effect, QGraphicsOpacityEffect):
                effect = QGraphicsOpacityEffect(item)
                item.setGraphicsEffect(effect)

            animation = QPropertyAnimation(effect, b"opacity")
            animation.setDuration(200)
            animation.setStartValue(1.0)
            animation.setEndValue(0.0)
            animation.setEasingCurve(QEasingCurve.Type.InCubic)

            if callback:
                animation.finished.connect(callback)

            animation.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

            # Store animation reference
            item._fade_animation = animation
        except Exception as e:
            self.logger.debug(f"Animation error: {e}")
            if callback:
                callback()

    def _on_item_close_requested(self, item: DeviceNotificationItem) -> None:
        """Handle notification item close request."""
        self._remove_notification(item)

    def _remove_notification(self, item: DeviceNotificationItem) -> None:
        """Remove a notification item with animation."""
        if item not in self._notification_items:
            return

        def do_remove():
            if item in self._notification_items:
                self._notification_items.remove(item)
            self.layout.removeWidget(item)
            item.deleteLater()

            # Hide if no more notifications
            if not self._notification_items:
                self.setVisible(False)

        self._animate_out(item, do_remove)

    def _remove_oldest_notification(self) -> None:
        """Remove the oldest notification."""
        if self._notification_items:
            self._remove_notification(self._notification_items[0])

    def clear_all(self) -> None:
        """Clear all notifications."""
        for item in list(self._notification_items):
            self._remove_notification(item)
