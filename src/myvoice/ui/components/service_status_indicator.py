"""
Service Status Indicator Component

This module provides a compact UI component for displaying service health status
with visual indicators and tooltips.

Story 7.4: Enhanced with emoji-based indicators (🟢, 🟡, 🔴, ⚠️) for accessibility
and pulsing animation support during generation.
"""

import logging
from typing import Optional
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QToolTip
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QPixmap, QPainter, QColor, QBrush

from myvoice.models.ui_state import ServiceStatusInfo, ServiceHealthStatus


# Story 7.4: Emoji indicators for accessibility (FR42, FR43)
STATUS_EMOJI = {
    ServiceHealthStatus.HEALTHY: "🟢",
    ServiceHealthStatus.WARNING: "⚠️",
    ServiceHealthStatus.ERROR: "🔴",
    ServiceHealthStatus.UNKNOWN: "⚪",
}

# Loading state uses yellow circle
LOADING_EMOJI = "🟡"


class ServiceStatusIndicator(QWidget):
    """
    Compact service status indicator with visual health status.

    This widget displays service status using color-coded indicators
    and provides detailed information via tooltips.

    Signals:
        status_clicked: Emitted when user clicks on status indicator
    """

    status_clicked = pyqtSignal(str)  # service_name

    def __init__(self, service_name: str, parent: Optional[QWidget] = None, use_emoji: bool = True):
        """
        Initialize the service status indicator.

        Args:
            service_name: Name of the service to monitor
            parent: Parent widget
            use_emoji: If True, use emoji indicators (Story 7.4). If False, use colored dots.
        """
        super().__init__(parent)
        self.service_name = service_name
        self.logger = logging.getLogger(self.__class__.__name__)

        # Current status
        self._status_info: Optional[ServiceStatusInfo] = None

        # Story 7.4: Emoji mode vs dot mode
        self._use_emoji = use_emoji

        # Story 7.4: Loading state (separate from health status for TTS model loading)
        self._is_loading = False

        # Story 7.4: Pulsing state for generation progress
        self._is_pulsing = False
        self._pulse_timer: Optional[QTimer] = None
        self._pulse_opacity = 1.0

        # UI components
        self._status_dot: Optional[QLabel] = None
        self._status_label: Optional[QLabel] = None

        # Create UI
        self._create_ui()

        # Update timer for tooltip refresh
        self._tooltip_timer = QTimer()
        self._tooltip_timer.timeout.connect(self._update_tooltip)
        self._tooltip_timer.start(5000)  # Update every 5 seconds

        self.logger.debug(f"ServiceStatusIndicator created for {service_name}")

    def _create_ui(self):
        """Create the status indicator UI components."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # Story 7.4: Status indicator - emoji or colored dot
        self._status_dot = QLabel()
        if self._use_emoji:
            # Emoji mode: Use text label with emoji
            self._status_dot.setFixedSize(16, 16)
            font = QFont()
            font.setPointSize(10)
            self._status_dot.setFont(font)
            self._status_dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._update_status_emoji(ServiceHealthStatus.UNKNOWN)
        else:
            # Dot mode: Use colored circle pixmap
            self._status_dot.setFixedSize(12, 12)
            self._status_dot.setScaledContents(True)
            self._update_status_dot(ServiceHealthStatus.UNKNOWN)

        # Service name label
        self._status_label = QLabel(self.service_name)
        font = QFont()
        font.setPointSize(7)
        self._status_label.setFont(font)
        self._status_label.setProperty("class", "caption")

        # Make the entire widget clickable
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout.addWidget(self._status_dot)
        layout.addWidget(self._status_label)
        layout.addStretch()

        # Set initial tooltip
        self.setToolTip(f"{self.service_name}: Status unknown")

    def _create_status_dot(self, color: str) -> QPixmap:
        """
        Create a colored dot pixmap for status display.

        Args:
            color: Hex color string (e.g., "#28a745")

        Returns:
            QPixmap: Colored dot pixmap
        """
        pixmap = QPixmap(12, 12)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw colored circle
        brush = QBrush(QColor(color))
        painter.setBrush(brush)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(1, 1, 10, 10)

        painter.end()
        return pixmap

    def _update_status_dot(self, health_status: ServiceHealthStatus):
        """
        Update the status dot color based on health status.

        Args:
            health_status: Current health status
        """
        # Story 7.4: Check loading state first
        if self._is_loading:
            color = "#ffc107"  # Yellow for loading
        elif health_status == ServiceHealthStatus.HEALTHY:
            color = "#28a745"  # Green
        elif health_status == ServiceHealthStatus.WARNING:
            color = "#ffc107"  # Yellow
        elif health_status == ServiceHealthStatus.ERROR:
            color = "#dc3545"  # Red
        else:
            color = "#6c757d"  # Gray

        pixmap = self._create_status_dot(color)
        self._status_dot.setPixmap(pixmap)

    def _update_status_emoji(self, health_status: ServiceHealthStatus):
        """
        Update the status indicator with emoji based on health status.

        Story 7.4: Emoji indicators for accessibility (FR42, FR43).

        Args:
            health_status: Current health status
        """
        # Loading state takes priority
        if self._is_loading:
            emoji = LOADING_EMOJI  # 🟡
        else:
            emoji = STATUS_EMOJI.get(health_status, STATUS_EMOJI[ServiceHealthStatus.UNKNOWN])

        self._status_dot.setText(emoji)

    def update_status(self, status_info: ServiceStatusInfo):
        """
        Update the status indicator with new status information.

        Args:
            status_info: Updated service status information
        """
        self._status_info = status_info

        # Update visual indicator (emoji or dot mode)
        if self._use_emoji:
            self._update_status_emoji(status_info.health_status)
        else:
            self._update_status_dot(status_info.health_status)

        # Update label color based on status
        if status_info.is_healthy:
            self._status_label.setProperty("status", "success")
        else:
            self._status_label.setProperty("status", "error")

        # Update tooltip
        self._update_tooltip()

        self.logger.debug(f"Status updated for {self.service_name}: {status_info.status_display}")

    def set_loading(self, is_loading: bool):
        """
        Set the loading state (Story 7.4: FR42 - TTS model loading indicator).

        When loading, the indicator shows yellow/🟡 regardless of health status.

        Args:
            is_loading: Whether the service is in loading state
        """
        self._is_loading = is_loading

        # Update visual immediately
        if self._status_info:
            if self._use_emoji:
                self._update_status_emoji(self._status_info.health_status)
            else:
                self._update_status_dot(self._status_info.health_status)
        else:
            if self._use_emoji:
                self._update_status_emoji(ServiceHealthStatus.UNKNOWN)
            else:
                self._update_status_dot(ServiceHealthStatus.UNKNOWN)

        self.logger.debug(f"Loading state for {self.service_name}: {is_loading}")

    def set_pulsing(self, enabled: bool):
        """
        Enable or disable pulsing animation (Story 7.4: Generation progress).

        Pulsing provides visual feedback that generation is in progress.

        Args:
            enabled: Whether to enable pulsing animation
        """
        if enabled == self._is_pulsing:
            return

        self._is_pulsing = enabled

        if enabled:
            # Start pulsing animation
            if not self._pulse_timer:
                self._pulse_timer = QTimer()
                self._pulse_timer.timeout.connect(self._animate_pulse)
            self._pulse_timer.start(500)  # Pulse every 500ms
            self.logger.debug(f"Started pulsing for {self.service_name}")
        else:
            # Stop pulsing animation
            if self._pulse_timer:
                self._pulse_timer.stop()
            self._pulse_opacity = 1.0
            self._status_dot.setStyleSheet("")  # Reset opacity
            self.logger.debug(f"Stopped pulsing for {self.service_name}")

    def _animate_pulse(self):
        """Animate the pulse effect by toggling opacity."""
        # Toggle between full opacity and reduced opacity
        if self._pulse_opacity == 1.0:
            self._pulse_opacity = 0.4
        else:
            self._pulse_opacity = 1.0

        # Apply opacity via stylesheet
        self._status_dot.setStyleSheet(f"opacity: {self._pulse_opacity};")

    def _update_tooltip(self):
        """Update the tooltip with current status information."""
        if not self._status_info:
            self.setToolTip(f"{self.service_name}: Status unknown")
            return

        tooltip_lines = [
            f"<b>{self.service_name}</b>",
            f"Status: {self._status_info.status_display}",
        ]

        if self._status_info.last_check:
            age_seconds = (self._status_info.last_check.timestamp() -
                          self._status_info.last_check.timestamp()) if self._status_info.last_check else 0
            tooltip_lines.append(f"Last check: {abs(age_seconds):.0f}s ago")

        if self._status_info.error_message:
            tooltip_lines.append(f"Error: {self._status_info.error_message}")

        if self._status_info.uptime_seconds is not None:
            uptime_str = self._format_uptime(self._status_info.uptime_seconds)
            tooltip_lines.append(f"Uptime: {uptime_str}")

        tooltip_html = "<br>".join(tooltip_lines)
        self.setToolTip(tooltip_html)

    def _format_uptime(self, seconds: float) -> str:
        """
        Format uptime seconds into human-readable string.

        Args:
            seconds: Uptime in seconds

        Returns:
            str: Formatted uptime string
        """
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}h"
        else:
            return f"{seconds/86400:.1f}d"

    def mousePressEvent(self, event):
        """Handle mouse click events."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.status_clicked.emit(self.service_name)
        super().mousePressEvent(event)

    def get_current_status(self) -> Optional[ServiceStatusInfo]:
        """Get the current status information."""
        return self._status_info

    def cleanup(self):
        """Clean up resources when widget is destroyed."""
        if self._tooltip_timer:
            self._tooltip_timer.stop()
        # Story 7.4: Clean up pulse timer
        if self._pulse_timer:
            self._pulse_timer.stop()
        self.logger.debug(f"ServiceStatusIndicator cleaned up for {self.service_name}")

    def is_loading(self) -> bool:
        """Check if indicator is in loading state."""
        return self._is_loading

    def is_pulsing(self) -> bool:
        """Check if indicator is pulsing."""
        return self._is_pulsing


class ServiceStatusBar(QWidget):
    """
    Status bar that displays multiple service status indicators.

    This widget manages multiple ServiceStatusIndicator widgets
    and provides a compact view of overall system health.

    Story 7.4: Enhanced with emoji mode support and pulsing animation.
    """

    service_status_clicked = pyqtSignal(str)  # service_name

    def __init__(self, parent: Optional[QWidget] = None, use_emoji: bool = True):
        """
        Initialize the service status bar.

        Args:
            parent: Parent widget
            use_emoji: If True, use emoji indicators (Story 7.4). If False, use colored dots.
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Story 7.4: Emoji mode
        self._use_emoji = use_emoji

        # Service indicators
        self._indicators: dict[str, ServiceStatusIndicator] = {}

        # Create UI
        self._create_ui()

    def _create_ui(self):
        """Create the status bar UI."""
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(4, 2, 4, 2)
        self._layout.setSpacing(8)

        # Add stretch to right-align indicators
        self._layout.addStretch()

        # Set CSS class for theme-based styling
        self.setProperty("class", "service-status-bar")

        self.setMaximumHeight(20)

    def add_service(self, service_name: str) -> ServiceStatusIndicator:
        """
        Add a service status indicator.

        Args:
            service_name: Name of the service to add

        Returns:
            ServiceStatusIndicator: Created indicator widget
        """
        if service_name in self._indicators:
            self.logger.warning(f"Service {service_name} already exists in status bar")
            return self._indicators[service_name]

        # Story 7.4: Use emoji mode from bar settings
        indicator = ServiceStatusIndicator(service_name, use_emoji=self._use_emoji)
        indicator.status_clicked.connect(self.service_status_clicked.emit)

        self._indicators[service_name] = indicator
        self._layout.addWidget(indicator)

        self.logger.debug(f"Added service indicator for {service_name}")
        return indicator

    def set_service_loading(self, service_name: str, is_loading: bool):
        """
        Set loading state for a service indicator (Story 7.4: FR42).

        Args:
            service_name: Name of the service
            is_loading: Whether the service is in loading state
        """
        if service_name in self._indicators:
            self._indicators[service_name].set_loading(is_loading)

    def set_service_pulsing(self, service_name: str, enabled: bool):
        """
        Set pulsing state for a service indicator (Story 7.4: Generation progress).

        Args:
            service_name: Name of the service
            enabled: Whether to enable pulsing animation
        """
        if service_name in self._indicators:
            self._indicators[service_name].set_pulsing(enabled)

    def get_indicator(self, service_name: str) -> Optional[ServiceStatusIndicator]:
        """
        Get the indicator widget for a specific service.

        Args:
            service_name: Name of the service

        Returns:
            ServiceStatusIndicator or None if not found
        """
        return self._indicators.get(service_name)

    def remove_service(self, service_name: str) -> bool:
        """
        Remove a service status indicator.

        Args:
            service_name: Name of the service to remove

        Returns:
            bool: True if service was removed, False if not found
        """
        if service_name not in self._indicators:
            return False

        indicator = self._indicators[service_name]
        indicator.cleanup()
        indicator.setParent(None)
        del self._indicators[service_name]

        self.logger.debug(f"Removed service indicator for {service_name}")
        return True

    def update_service_status(self, service_name: str, status_info: ServiceStatusInfo):
        """
        Update status for a specific service.

        Args:
            service_name: Name of the service
            status_info: Updated status information
        """
        if service_name not in self._indicators:
            self.add_service(service_name)

        self._indicators[service_name].update_status(status_info)

    def get_service_count(self) -> int:
        """Get the number of services being monitored."""
        return len(self._indicators)

    def get_service_names(self) -> list[str]:
        """Get list of all monitored service names."""
        return list(self._indicators.keys())

    def get_overall_health(self) -> ServiceHealthStatus:
        """
        Get overall health status across all services.

        Returns:
            ServiceHealthStatus: Overall health status
        """
        if not self._indicators:
            return ServiceHealthStatus.UNKNOWN

        statuses = []
        for indicator in self._indicators.values():
            status_info = indicator.get_current_status()
            if status_info:
                statuses.append(status_info.health_status)

        if not statuses:
            return ServiceHealthStatus.UNKNOWN
        elif ServiceHealthStatus.ERROR in statuses:
            return ServiceHealthStatus.ERROR
        elif ServiceHealthStatus.WARNING in statuses:
            return ServiceHealthStatus.WARNING
        elif ServiceHealthStatus.HEALTHY in statuses:
            return ServiceHealthStatus.HEALTHY
        else:
            return ServiceHealthStatus.UNKNOWN

    def cleanup_all_services(self):
        """Clean up all service indicators and their timers."""
        for service_name in list(self._indicators.keys()):
            self.remove_service(service_name)
        self.logger.debug("All service indicators cleaned up")