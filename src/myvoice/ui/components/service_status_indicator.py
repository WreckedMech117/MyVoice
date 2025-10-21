"""
Service Status Indicator Component

This module provides a compact UI component for displaying service health status
with visual indicators and tooltips.
"""

import logging
from typing import Optional
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QToolTip
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QPainter, QColor, QBrush

from myvoice.models.ui_state import ServiceStatusInfo, ServiceHealthStatus


class ServiceStatusIndicator(QWidget):
    """
    Compact service status indicator with visual health status.

    This widget displays service status using color-coded indicators
    and provides detailed information via tooltips.

    Signals:
        status_clicked: Emitted when user clicks on status indicator
    """

    status_clicked = pyqtSignal(str)  # service_name

    def __init__(self, service_name: str, parent: Optional[QWidget] = None):
        """
        Initialize the service status indicator.

        Args:
            service_name: Name of the service to monitor
            parent: Parent widget
        """
        super().__init__(parent)
        self.service_name = service_name
        self.logger = logging.getLogger(self.__class__.__name__)

        # Current status
        self._status_info: Optional[ServiceStatusInfo] = None

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

        # Status dot (colored circle)
        self._status_dot = QLabel()
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
        if health_status == ServiceHealthStatus.HEALTHY:
            color = "#28a745"  # Green
        elif health_status == ServiceHealthStatus.WARNING:
            color = "#ffc107"  # Yellow
        elif health_status == ServiceHealthStatus.ERROR:
            color = "#dc3545"  # Red
        else:
            color = "#6c757d"  # Gray

        pixmap = self._create_status_dot(color)
        self._status_dot.setPixmap(pixmap)

    def update_status(self, status_info: ServiceStatusInfo):
        """
        Update the status indicator with new status information.

        Args:
            status_info: Updated service status information
        """
        self._status_info = status_info

        # Update visual indicator
        self._update_status_dot(status_info.health_status)

        # Update label color based on status
        if status_info.is_healthy:
            self._status_label.setProperty("status", "success")
        else:
            self._status_label.setProperty("status", "error")

        # Update tooltip
        self._update_tooltip()

        self.logger.debug(f"Status updated for {self.service_name}: {status_info.status_display}")

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
        self.logger.debug(f"ServiceStatusIndicator cleaned up for {self.service_name}")


class ServiceStatusBar(QWidget):
    """
    Status bar that displays multiple service status indicators.

    This widget manages multiple ServiceStatusIndicator widgets
    and provides a compact view of overall system health.
    """

    service_status_clicked = pyqtSignal(str)  # service_name

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the service status bar.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

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

        indicator = ServiceStatusIndicator(service_name)
        indicator.status_clicked.connect(self.service_status_clicked.emit)

        self._indicators[service_name] = indicator
        self._layout.addWidget(indicator)

        self.logger.debug(f"Added service indicator for {service_name}")
        return indicator

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