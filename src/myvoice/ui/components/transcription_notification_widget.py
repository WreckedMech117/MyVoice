"""
Transcription Notification Widget

This module provides a UI component for displaying transcription notifications
and progress updates to users in a non-intrusive way.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QFrame, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect
from PyQt6.QtGui import QFont, QPalette, QColor

from myvoice.services.background_transcription_manager import (
    TranscriptionNotification, NotificationType
)


class NotificationItem(QFrame):
    """Individual notification item widget."""

    # Signal emitted when notification is clicked
    notification_clicked = pyqtSignal(object)  # TranscriptionNotification

    # Signal emitted when close button is clicked
    close_requested = pyqtSignal(object)  # NotificationItem

    def __init__(self, notification: TranscriptionNotification, parent: Optional[QWidget] = None):
        """
        Initialize notification item.

        Args:
            notification: Transcription notification to display
            parent: Parent widget
        """
        super().__init__(parent)
        self.notification = notification
        self.logger = logging.getLogger(self.__class__.__name__)

        # Auto-hide timer
        self._auto_hide_timer = QTimer()
        self._auto_hide_timer.timeout.connect(self._on_auto_hide)

        # Setup UI
        self._create_ui()
        self._apply_styling()

        # Start auto-hide timer for completed/failed notifications
        if notification.type in [NotificationType.COMPLETED, NotificationType.FAILED]:
            self._auto_hide_timer.start(5000)  # 5 seconds

    def _create_ui(self):
        """Create the notification item UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # Icon/Status indicator
        self._status_label = QLabel()
        self._status_label.setFixedSize(16, 16)
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._update_status_icon()

        # Content area
        content_layout = QVBoxLayout()
        content_layout.setSpacing(2)

        # Title
        self._title_label = QLabel()
        title_font = QFont()
        title_font.setWeight(QFont.Weight.Bold)
        self._title_label.setFont(title_font)
        self._title_label.setText(self._get_title_text())

        # Message
        self._message_label = QLabel()
        self._message_label.setWordWrap(True)
        self._message_label.setText(self.notification.message)

        # Progress bar (for progress notifications)
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        if (self.notification.type == NotificationType.PROGRESS and
            self.notification.progress_percent is not None):
            self._progress_bar.setVisible(True)
            self._progress_bar.setValue(int(self.notification.progress_percent))

        # Timestamp
        self._time_label = QLabel()
        time_font = QFont()
        time_font.setPointSize(8)
        self._time_label.setFont(time_font)
        self._time_label.setText(self._format_timestamp())

        content_layout.addWidget(self._title_label)
        content_layout.addWidget(self._message_label)
        content_layout.addWidget(self._progress_bar)
        content_layout.addWidget(self._time_label)

        # Close button
        self._close_button = QPushButton("×")
        self._close_button.setFixedSize(20, 20)
        self._close_button.clicked.connect(self._on_close_clicked)

        # Add to main layout
        layout.addWidget(self._status_label)
        layout.addLayout(content_layout, 1)
        layout.addWidget(self._close_button)

        # Make the entire item clickable
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _apply_styling(self):
        """Apply styling based on notification type."""
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setLineWidth(1)

        # Set CSS class based on notification type
        self.setProperty("class", "notification-item")
        if self.notification.type == NotificationType.COMPLETED:
            self.setProperty("status", "completed")
        elif self.notification.type == NotificationType.FAILED:
            self.setProperty("status", "failed")
        elif self.notification.type == NotificationType.PROGRESS:
            self.setProperty("status", "progress")
        else:  # STARTED, BATCH_COMPLETED
            self.setProperty("status", "started")

    def _update_status_icon(self):
        """Update the status icon based on notification type."""
        icons = {
            NotificationType.STARTED: "🚀",
            NotificationType.PROGRESS: "⏳",
            NotificationType.COMPLETED: "✅",
            NotificationType.FAILED: "❌",
            NotificationType.BATCH_COMPLETED: "📦"
        }

        icon = icons.get(self.notification.type, "ℹ️")
        self._status_label.setText(icon)

    def _get_title_text(self) -> str:
        """Get the title text for the notification."""
        profile_name = self.notification.voice_profile.name

        if self.notification.type == NotificationType.STARTED:
            return f"Transcription Started"
        elif self.notification.type == NotificationType.PROGRESS:
            return f"Transcribing {profile_name}"
        elif self.notification.type == NotificationType.COMPLETED:
            return f"Transcription Complete"
        elif self.notification.type == NotificationType.FAILED:
            return f"Transcription Failed"
        elif self.notification.type == NotificationType.BATCH_COMPLETED:
            if self.notification.batch_info:
                total = self.notification.batch_info.get('total_profiles', 0)
                return f"Batch Complete ({total} files)"
            return "Batch Complete"
        else:
            return "Transcription Update"

    def _format_timestamp(self) -> str:
        """Format the notification timestamp."""
        now = datetime.now()
        diff = now - self.notification.timestamp

        if diff < timedelta(minutes=1):
            return "Just now"
        elif diff < timedelta(hours=1):
            minutes = int(diff.total_seconds() / 60)
            return f"{minutes}m ago"
        elif diff < timedelta(days=1):
            hours = int(diff.total_seconds() / 3600)
            return f"{hours}h ago"
        else:
            return self.notification.timestamp.strftime("%m/%d %H:%M")

    def mousePressEvent(self, event):
        """Handle mouse press event to emit notification clicked signal."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.notification_clicked.emit(self.notification)
        super().mousePressEvent(event)

    def _on_close_clicked(self):
        """Handle close button click."""
        self.close_requested.emit(self)

    def _on_auto_hide(self):
        """Handle auto-hide timer timeout."""
        self.close_requested.emit(self)

    def stop_auto_hide(self):
        """Stop the auto-hide timer."""
        if self._auto_hide_timer.isActive():
            self._auto_hide_timer.stop()

    def update_progress(self, progress_percent: float):
        """Update progress bar if this is a progress notification."""
        if self.notification.type == NotificationType.PROGRESS:
            if not self._progress_bar.isVisible():
                self._progress_bar.setVisible(True)
            self._progress_bar.setValue(int(progress_percent))


class TranscriptionNotificationWidget(QWidget):
    """
    Widget for displaying transcription notifications and progress.

    This widget shows a list of recent transcription notifications
    in a non-intrusive panel that can be toggled by the user.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the transcription notification widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Notification storage
        self._notifications: List[TranscriptionNotification] = []
        self._notification_items: List[NotificationItem] = []
        self._max_notifications = 10

        # Setup UI
        self._create_ui()
        self._apply_styling()

        # Auto-cleanup timer
        self._cleanup_timer = QTimer()
        self._cleanup_timer.timeout.connect(self._cleanup_old_notifications)
        self._cleanup_timer.start(60000)  # Clean up every minute

    def _create_ui(self):
        """Create the notification widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Header
        header_layout = QHBoxLayout()

        self._title_label = QLabel("Transcription Activity")
        title_font = QFont()
        title_font.setWeight(QFont.Weight.Bold)
        title_font.setPointSize(10)
        self._title_label.setFont(title_font)

        self._clear_button = QPushButton("Clear All")
        self._clear_button.clicked.connect(self.clear_all_notifications)
        self._clear_button.setMaximumWidth(80)

        header_layout.addWidget(self._title_label)
        header_layout.addStretch()
        header_layout.addWidget(self._clear_button)

        # Scrollable notification area
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarNever)
        self._scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Container for notifications
        self._notification_container = QWidget()
        self._notification_layout = QVBoxLayout(self._notification_container)
        self._notification_layout.setSpacing(4)
        self._notification_layout.addStretch()

        self._scroll_area.setWidget(self._notification_container)

        # Empty state label
        self._empty_label = QLabel("No recent transcription activity")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setProperty("class", "caption")

        layout.addLayout(header_layout)
        layout.addWidget(self._scroll_area)
        layout.addWidget(self._empty_label)

        # Initially show empty state
        self._update_empty_state()

    def _apply_styling(self):
        """Apply styling to the widget."""
        self.setProperty("class", "transcription-notification")

    def add_notification(self, notification: TranscriptionNotification):
        """
        Add a new transcription notification.

        Args:
            notification: Transcription notification to add
        """
        try:
            # Add to storage
            self._notifications.insert(0, notification)  # Add to beginning

            # Create notification item
            item = NotificationItem(notification)
            item.notification_clicked.connect(self._on_notification_clicked)
            item.close_requested.connect(self._on_notification_close)

            # Add to UI (insert at beginning)
            self._notification_layout.insertWidget(0, item)
            self._notification_items.insert(0, item)

            # Limit number of notifications
            while len(self._notification_items) > self._max_notifications:
                old_item = self._notification_items.pop()
                old_notification = self._notifications.pop()
                old_item.deleteLater()

            self._update_empty_state()

            # Auto-scroll to top to show new notification
            QTimer.singleShot(100, lambda: self._scroll_area.verticalScrollBar().setValue(0))

            self.logger.debug(f"Added notification: {notification.type.value} for {notification.voice_profile.name}")

        except Exception as e:
            self.logger.error(f"Error adding notification: {e}")

    def clear_all_notifications(self):
        """Clear all notifications."""
        try:
            # Clear UI items
            for item in self._notification_items:
                item.stop_auto_hide()
                item.deleteLater()

            # Clear storage
            self._notification_items.clear()
            self._notifications.clear()

            self._update_empty_state()
            self.logger.debug("Cleared all notifications")

        except Exception as e:
            self.logger.error(f"Error clearing notifications: {e}")

    def update_progress_notification(self, voice_profile_name: str, progress_percent: float):
        """
        Update progress for an existing progress notification.

        Args:
            voice_profile_name: Name of the voice profile
            progress_percent: Progress percentage (0-100)
        """
        try:
            for item in self._notification_items:
                if (item.notification.type == NotificationType.PROGRESS and
                    item.notification.voice_profile.name == voice_profile_name):
                    item.update_progress(progress_percent)
                    break

        except Exception as e:
            self.logger.error(f"Error updating progress notification: {e}")

    def get_notification_count(self) -> int:
        """Get the current number of notifications."""
        return len(self._notifications)

    def _update_empty_state(self):
        """Update the visibility of the empty state label."""
        has_notifications = len(self._notification_items) > 0
        self._empty_label.setVisible(not has_notifications)
        self._scroll_area.setVisible(has_notifications)

    def _on_notification_clicked(self, notification: TranscriptionNotification):
        """
        Handle notification item click.

        Args:
            notification: Clicked notification
        """
        try:
            self.logger.debug(f"Notification clicked: {notification.type.value} for {notification.voice_profile.name}")
            # Could emit a signal here for external handling
            # For now, just log the event

        except Exception as e:
            self.logger.error(f"Error handling notification click: {e}")

    def _on_notification_close(self, item: NotificationItem):
        """
        Handle notification close request.

        Args:
            item: Notification item to close
        """
        try:
            if item in self._notification_items:
                index = self._notification_items.index(item)

                # Remove from lists
                self._notification_items.pop(index)
                self._notifications.pop(index)

                # Remove from UI
                item.stop_auto_hide()
                item.deleteLater()

                self._update_empty_state()

        except Exception as e:
            self.logger.error(f"Error closing notification: {e}")

    def _cleanup_old_notifications(self):
        """Clean up old notifications that have exceeded their lifetime."""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(hours=1)  # Remove notifications older than 1 hour

            items_to_remove = []

            for i, notification in enumerate(self._notifications):
                if notification.timestamp < cutoff_time:
                    items_to_remove.append(i)

            # Remove from end to start to maintain indices
            for i in reversed(items_to_remove):
                item = self._notification_items.pop(i)
                self._notifications.pop(i)
                item.stop_auto_hide()
                item.deleteLater()

            if items_to_remove:
                self._update_empty_state()
                self.logger.debug(f"Cleaned up {len(items_to_remove)} old notifications")

        except Exception as e:
            self.logger.error(f"Error during notification cleanup: {e}")

    def closeEvent(self, event):
        """Handle widget close event."""
        # Stop timers
        if self._cleanup_timer.isActive():
            self._cleanup_timer.stop()

        # Stop auto-hide timers on all items
        for item in self._notification_items:
            item.stop_auto_hide()

        super().closeEvent(event)