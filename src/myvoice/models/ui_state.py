"""
UI State Models

This module contains models for managing application UI state, including
service status, user notifications, and interface states.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from myvoice.models.service_enums import ServiceStatus
from myvoice.models.error import MyVoiceError, ErrorSeverity


class ServiceHealthStatus(Enum):
    """Service health status for UI display."""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class ServiceStatusInfo:
    """Information about a service's status for UI display."""
    service_name: str
    status: ServiceStatus
    health_status: ServiceHealthStatus
    last_check: Optional[datetime] = None
    error_message: Optional[str] = None
    uptime_seconds: Optional[float] = None

    @property
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self.health_status == ServiceHealthStatus.HEALTHY

    @property
    def status_display(self) -> str:
        """Get human-readable status for display."""
        if self.status == ServiceStatus.RUNNING and self.is_healthy:
            return "Online"
        elif self.status == ServiceStatus.RUNNING and self.health_status == ServiceHealthStatus.WARNING:
            return "Running (Issues)"
        elif self.status == ServiceStatus.STARTING:
            return "Starting..."
        elif self.status == ServiceStatus.STOPPING:
            return "Stopping..."
        elif self.status == ServiceStatus.ERROR:
            return "Error"
        else:
            return "Offline"

    @property
    def status_color(self) -> str:
        """Get color for UI display."""
        if self.health_status == ServiceHealthStatus.HEALTHY:
            return "#28a745"  # Green
        elif self.health_status == ServiceHealthStatus.WARNING:
            return "#ffc107"  # Yellow
        elif self.health_status == ServiceHealthStatus.ERROR:
            return "#dc3545"  # Red
        else:
            return "#6c757d"  # Gray


@dataclass
class NotificationMessage:
    """User notification message."""
    id: str
    title: str
    message: str
    severity: ErrorSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    is_dismissible: bool = True
    auto_dismiss_seconds: Optional[int] = None
    action_text: Optional[str] = None
    action_callback: Optional[str] = None  # Name of callback function

    @property
    def age_seconds(self) -> float:
        """Get age of notification in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()

    @property
    def should_auto_dismiss(self) -> bool:
        """Check if notification should auto-dismiss."""
        if not self.auto_dismiss_seconds:
            return False
        return self.age_seconds >= self.auto_dismiss_seconds


@dataclass
class UIState:
    """
    Application UI state container.

    This class manages the overall state of the user interface,
    including service status, notifications, and user preferences.
    """

    # Service status tracking
    service_status: Dict[str, ServiceStatusInfo] = field(default_factory=dict)

    # User notifications
    notifications: List[NotificationMessage] = field(default_factory=list)

    # UI preferences
    always_on_top: bool = True
    compact_mode: bool = True
    theme_name: str = "compact"

    # Generation state
    is_generating: bool = False
    current_operation: Optional[str] = None
    progress_percentage: Optional[float] = None

    # Window state
    window_visible: bool = True
    last_text_input: str = ""
    selected_voice: str = "Default"
    speech_speed: float = 1.0

    # Emotion state (Story 3.3: FR9 - persist emotion selection)
    current_emotion: str = "neutral"  # Emotion preset ID (neutral, happy, sad, angry, flirtatious)
    emotion_enabled: bool = True  # Whether emotion controls are enabled (FR8a)

    def update_service_status(
        self,
        service_name: str,
        status: ServiceStatus,
        health_status: ServiceHealthStatus,
        error_message: Optional[str] = None
    ):
        """
        Update service status information.

        Args:
            service_name: Name of the service
            status: Service status enum
            health_status: Health status enum
            error_message: Optional error message
        """
        if service_name not in self.service_status:
            self.service_status[service_name] = ServiceStatusInfo(
                service_name=service_name,
                status=status,
                health_status=health_status
            )
        else:
            service_info = self.service_status[service_name]
            service_info.status = status
            service_info.health_status = health_status
            service_info.error_message = error_message
            service_info.last_check = datetime.now()

    def get_service_status(self, service_name: str) -> Optional[ServiceStatusInfo]:
        """Get status information for a specific service."""
        return self.service_status.get(service_name)

    def add_notification(
        self,
        title: str,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.INFO,
        auto_dismiss_seconds: Optional[int] = None,
        action_text: Optional[str] = None,
        action_callback: Optional[str] = None
    ) -> str:
        """
        Add a user notification.

        Args:
            title: Notification title
            message: Notification message
            severity: Message severity level
            auto_dismiss_seconds: Auto-dismiss time (None for manual dismiss)
            action_text: Text for action button
            action_callback: Name of callback function for action

        Returns:
            str: Notification ID
        """
        import uuid
        notification_id = str(uuid.uuid4())

        notification = NotificationMessage(
            id=notification_id,
            title=title,
            message=message,
            severity=severity,
            auto_dismiss_seconds=auto_dismiss_seconds,
            action_text=action_text,
            action_callback=action_callback
        )

        self.notifications.append(notification)
        return notification_id

    def dismiss_notification(self, notification_id: str) -> bool:
        """
        Dismiss a notification by ID.

        Args:
            notification_id: ID of notification to dismiss

        Returns:
            bool: True if notification was found and dismissed
        """
        for i, notification in enumerate(self.notifications):
            if notification.id == notification_id:
                del self.notifications[i]
                return True
        return False

    def clear_expired_notifications(self):
        """Remove notifications that should auto-dismiss."""
        self.notifications = [
            n for n in self.notifications
            if not n.should_auto_dismiss
        ]

    def get_critical_notifications(self) -> List[NotificationMessage]:
        """Get notifications that require immediate attention."""
        return [
            n for n in self.notifications
            if n.severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]
        ]

    def has_service_errors(self) -> bool:
        """Check if any services have errors."""
        return any(
            service.health_status == ServiceHealthStatus.ERROR
            for service in self.service_status.values()
        )

    def get_overall_service_health(self) -> ServiceHealthStatus:
        """Get overall health status across all services."""
        if not self.service_status:
            return ServiceHealthStatus.UNKNOWN

        health_statuses = [service.health_status for service in self.service_status.values()]

        if ServiceHealthStatus.ERROR in health_statuses:
            return ServiceHealthStatus.ERROR
        elif ServiceHealthStatus.WARNING in health_statuses:
            return ServiceHealthStatus.WARNING
        elif ServiceHealthStatus.HEALTHY in health_statuses:
            return ServiceHealthStatus.HEALTHY
        else:
            return ServiceHealthStatus.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        """Convert UI state to dictionary for serialization."""
        return {
            "service_status": {
                name: {
                    "service_name": info.service_name,
                    "status": info.status.value,
                    "health_status": info.health_status.value,
                    "last_check": info.last_check.isoformat() if info.last_check else None,
                    "error_message": info.error_message,
                    "uptime_seconds": info.uptime_seconds
                }
                for name, info in self.service_status.items()
            },
            "notifications_count": len(self.notifications),
            "critical_notifications_count": len(self.get_critical_notifications()),
            "always_on_top": self.always_on_top,
            "compact_mode": self.compact_mode,
            "theme_name": self.theme_name,
            "is_generating": self.is_generating,
            "current_operation": self.current_operation,
            "window_visible": self.window_visible,
            "selected_voice": self.selected_voice,
            "speech_speed": self.speech_speed,
            "current_emotion": self.current_emotion,  # Story 3.3: FR9
            "emotion_enabled": self.emotion_enabled,  # Story 3.3: FR8a
        }