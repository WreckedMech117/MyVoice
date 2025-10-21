"""
Device Notification Models

This module contains models for handling device change notifications and user feedback
for audio device changes in the MyVoice application.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from datetime import datetime

from myvoice.models.audio_device import AudioDevice
from myvoice.models.validation import ValidationResult, ValidationIssue, ValidationStatus


class NotificationSeverity(Enum):
    """Severity levels for device notifications."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationType(Enum):
    """Types of device notifications."""
    DEVICE_ADDED = "device_added"
    DEVICE_REMOVED = "device_removed"
    DEVICE_CHANGED = "device_changed"
    DEVICE_FALLBACK = "device_fallback"
    DEVICE_RESTORED = "device_restored"
    DEVICE_LIST_REFRESHED = "device_list_refreshed"


@dataclass
class DeviceNotification:
    """
    Notification for device changes that require user awareness.

    This class represents notifications about audio device changes that should
    be communicated to the user through the UI notification system.

    Attributes:
        notification_id: Unique identifier for this notification
        notification_type: Type of device change notification
        severity: Severity level of the notification
        title: Short title for the notification
        message: Detailed message for the user
        device: Audio device involved in the change (if applicable)
        suggested_action: Recommended action for the user
        auto_dismiss_seconds: Auto-dismiss after X seconds (None = manual dismiss)
        timestamp: When the notification was created
        data: Additional data for the notification
    """
    notification_id: str
    notification_type: NotificationType
    severity: NotificationSeverity
    title: str
    message: str
    device: Optional[AudioDevice] = None
    suggested_action: Optional[str] = None
    auto_dismiss_seconds: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize and validate the device notification."""
        validation_result = self.validate()
        if not validation_result.is_valid:
            logging.warning(f"Invalid device notification: {validation_result.summary}")

    def validate(self) -> ValidationResult:
        """
        Validate the device notification for required fields and constraints.

        Returns:
            ValidationResult: Detailed validation result with issues and warnings
        """
        issues = []
        warnings = []

        try:
            # Notification ID validation
            if not self.notification_id or not self.notification_id.strip():
                issues.append(ValidationIssue(
                    field="notification_id",
                    message="Notification ID cannot be empty",
                    code="EMPTY_NOTIFICATION_ID",
                    severity=ValidationStatus.INVALID
                ))

            # Title validation
            if not self.title or not self.title.strip():
                issues.append(ValidationIssue(
                    field="title",
                    message="Notification title cannot be empty",
                    code="EMPTY_TITLE",
                    severity=ValidationStatus.INVALID
                ))

            # Message validation
            if not self.message or not self.message.strip():
                issues.append(ValidationIssue(
                    field="message",
                    message="Notification message cannot be empty",
                    code="EMPTY_MESSAGE",
                    severity=ValidationStatus.INVALID
                ))

            # Enum validation
            if not isinstance(self.notification_type, NotificationType):
                issues.append(ValidationIssue(
                    field="notification_type",
                    message="Invalid notification type",
                    code="INVALID_NOTIFICATION_TYPE",
                    severity=ValidationStatus.INVALID
                ))

            if not isinstance(self.severity, NotificationSeverity):
                issues.append(ValidationIssue(
                    field="severity",
                    message="Invalid notification severity",
                    code="INVALID_SEVERITY",
                    severity=ValidationStatus.INVALID
                ))

            # Auto-dismiss validation
            if self.auto_dismiss_seconds is not None:
                if not isinstance(self.auto_dismiss_seconds, int) or self.auto_dismiss_seconds <= 0:
                    warnings.append(ValidationIssue(
                        field="auto_dismiss_seconds",
                        message="Auto-dismiss seconds should be a positive integer",
                        code="INVALID_AUTO_DISMISS",
                        severity=ValidationStatus.WARNING
                    ))

            # Device validation (if provided)
            if self.device is not None:
                if not isinstance(self.device, AudioDevice):
                    issues.append(ValidationIssue(
                        field="device",
                        message="Device must be an AudioDevice instance",
                        code="INVALID_DEVICE",
                        severity=ValidationStatus.INVALID
                    ))

            # Determine overall status
            if issues:
                status = ValidationStatus.INVALID
                is_valid = False
            elif warnings:
                status = ValidationStatus.WARNING
                is_valid = True
            else:
                status = ValidationStatus.VALID
                is_valid = True

            return ValidationResult(
                is_valid=is_valid,
                status=status,
                issues=issues,
                warnings=warnings
            )

        except Exception as e:
            logging.exception(f"Error during device notification validation: {e}")
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.INVALID,
                issues=[ValidationIssue(
                    field="general",
                    message=f"Validation error: {str(e)}",
                    code="VALIDATION_ERROR",
                    severity=ValidationStatus.INVALID
                )],
                warnings=[],
                summary="Validation failed due to internal error"
            )

    def get_notification_info(self) -> Dict[str, Any]:
        """
        Get comprehensive notification information.

        Returns:
            dict: Notification information including all attributes and metadata
        """
        return {
            'notification_id': self.notification_id,
            'notification_type': self.notification_type.value,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'device': {
                'device_id': self.device.device_id,
                'name': self.device.name,
                'device_type': self.device.device_type.value,
                'is_available': self.device.is_available
            } if self.device else None,
            'suggested_action': self.suggested_action,
            'auto_dismiss_seconds': self.auto_dismiss_seconds,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'data': self.data,
            'is_valid': self.validate().is_valid
        }

    def __str__(self) -> str:
        """String representation of the device notification."""
        device_name = self.device.name if self.device else "System"
        return f"[{self.severity.value.upper()}] {self.title}: {device_name}"

    def __repr__(self) -> str:
        """Developer representation of the device notification."""
        return (f"DeviceNotification(id='{self.notification_id}', "
                f"type={self.notification_type}, severity={self.severity}, "
                f"title='{self.title}')")

    @classmethod
    def create_device_removed_notification(cls, device: AudioDevice) -> 'DeviceNotification':
        """
        Factory method to create a device removed notification.

        Args:
            device: AudioDevice that was removed

        Returns:
            DeviceNotification: Notification for device removal
        """
        import uuid

        return cls(
            notification_id=str(uuid.uuid4()),
            notification_type=NotificationType.DEVICE_REMOVED,
            severity=NotificationSeverity.WARNING,
            title="Audio Device Disconnected",
            message=f"'{device.name}' has been disconnected and is no longer available.",
            device=device,
            suggested_action="Select a different audio device in settings",
            auto_dismiss_seconds=10,
            data={'device_id': device.device_id, 'device_name': device.name}
        )

    @classmethod
    def create_device_added_notification(cls, device: AudioDevice) -> 'DeviceNotification':
        """
        Factory method to create a device added notification.

        Args:
            device: AudioDevice that was added

        Returns:
            DeviceNotification: Notification for device addition
        """
        import uuid

        return cls(
            notification_id=str(uuid.uuid4()),
            notification_type=NotificationType.DEVICE_ADDED,
            severity=NotificationSeverity.INFO,
            title="New Audio Device Available",
            message=f"'{device.name}' is now available for audio output.",
            device=device,
            suggested_action="You can select this device in audio settings",
            auto_dismiss_seconds=8,
            data={'device_id': device.device_id, 'device_name': device.name}
        )

    @classmethod
    def create_device_fallback_notification(cls, failed_device: AudioDevice,
                                          fallback_device: AudioDevice) -> 'DeviceNotification':
        """
        Factory method to create a device fallback notification.

        Args:
            failed_device: AudioDevice that failed
            fallback_device: AudioDevice used as fallback

        Returns:
            DeviceNotification: Notification for device fallback
        """
        import uuid

        return cls(
            notification_id=str(uuid.uuid4()),
            notification_type=NotificationType.DEVICE_FALLBACK,
            severity=NotificationSeverity.WARNING,
            title="Audio Device Switched",
            message=f"'{failed_device.name}' is unavailable. Switched to '{fallback_device.name}'.",
            device=fallback_device,
            suggested_action="Check device connections and select preferred device in settings",
            auto_dismiss_seconds=12,
            data={
                'failed_device_id': failed_device.device_id,
                'failed_device_name': failed_device.name,
                'fallback_device_id': fallback_device.device_id,
                'fallback_device_name': fallback_device.name
            }
        )

    @classmethod
    def create_device_restored_notification(cls, device: AudioDevice) -> 'DeviceNotification':
        """
        Factory method to create a device restored notification.

        Args:
            device: AudioDevice that was restored

        Returns:
            DeviceNotification: Notification for device restoration
        """
        import uuid

        return cls(
            notification_id=str(uuid.uuid4()),
            notification_type=NotificationType.DEVICE_RESTORED,
            severity=NotificationSeverity.INFO,
            title="Audio Device Restored",
            message=f"'{device.name}' is available again and has been restored as your audio device.",
            device=device,
            suggested_action=None,
            auto_dismiss_seconds=6,
            data={'device_id': device.device_id, 'device_name': device.name}
        )

    @classmethod
    def create_device_list_refresh_notification(cls, added_count: int,
                                              removed_count: int) -> 'DeviceNotification':
        """
        Factory method to create a device list refresh notification.

        Args:
            added_count: Number of devices added
            removed_count: Number of devices removed

        Returns:
            DeviceNotification: Notification for device list refresh
        """
        import uuid

        if added_count > 0 and removed_count > 0:
            message = f"Audio device list updated: {added_count} added, {removed_count} removed."
        elif added_count > 0:
            message = f"Audio device list updated: {added_count} new devices available."
        elif removed_count > 0:
            message = f"Audio device list updated: {removed_count} devices removed."
        else:
            message = "Audio device list refreshed."

        return cls(
            notification_id=str(uuid.uuid4()),
            notification_type=NotificationType.DEVICE_LIST_REFRESHED,
            severity=NotificationSeverity.INFO,
            title="Audio Devices Updated",
            message=message,
            device=None,
            suggested_action="Check audio settings for device changes" if (added_count + removed_count) > 0 else None,
            auto_dismiss_seconds=5,
            data={'added_count': added_count, 'removed_count': removed_count}
        )


@dataclass
class DeviceChangeHandler:
    """
    Handler for managing device change detection and notification logic.

    This class provides the business logic for handling device changes,
    including fallback strategies and notification generation.

    Attributes:
        logger: Logger instance for this handler
        preferred_device: Currently preferred audio device
        fallback_device: Current fallback audio device
        notification_callbacks: List of callback functions for notifications
    """
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("DeviceChangeHandler"))
    preferred_device: Optional[AudioDevice] = None
    fallback_device: Optional[AudioDevice] = None
    notification_callbacks: List[Callable[[DeviceNotification], None]] = field(default_factory=list)

    def add_notification_callback(self, callback: Callable[[DeviceNotification], None]) -> None:
        """
        Add a callback for device notifications.

        Args:
            callback: Function to call when notifications are generated
        """
        if callback not in self.notification_callbacks:
            self.notification_callbacks.append(callback)
            callback_name = getattr(callback, '__name__', str(callback))
            self.logger.debug(f"Added notification callback: {callback_name}")

    def remove_notification_callback(self, callback: Callable[[DeviceNotification], None]) -> None:
        """
        Remove a notification callback.

        Args:
            callback: Callback function to remove
        """
        if callback in self.notification_callbacks:
            self.notification_callbacks.remove(callback)
            callback_name = getattr(callback, '__name__', str(callback))
            self.logger.debug(f"Removed notification callback: {callback_name}")

    def _emit_notification(self, notification: DeviceNotification) -> None:
        """
        Emit a notification to all registered callbacks.

        Args:
            notification: DeviceNotification to emit
        """
        self.logger.info(f"Emitting notification: {notification.title}")

        for callback in self.notification_callbacks:
            try:
                callback(notification)
            except Exception as e:
                self.logger.error(f"Error in notification callback {callback.__name__}: {e}")

    def handle_device_removed(self, removed_device: AudioDevice) -> Optional[AudioDevice]:
        """
        Handle when a device is removed/disconnected.

        Args:
            removed_device: AudioDevice that was removed

        Returns:
            Optional[AudioDevice]: Fallback device if available, None otherwise
        """
        self.logger.warning(f"Device removed: {removed_device.name}")

        # Create notification for removed device
        notification = DeviceNotification.create_device_removed_notification(removed_device)
        self._emit_notification(notification)

        # If this was our preferred device, we need fallback logic
        if self.preferred_device and self.preferred_device.device_id == removed_device.device_id:
            self.logger.warning("Preferred device was removed, attempting fallback")

            # Use fallback device if available
            if self.fallback_device and self.fallback_device.is_available:
                self.logger.info(f"Falling back to: {self.fallback_device.name}")

                # Create fallback notification
                fallback_notification = DeviceNotification.create_device_fallback_notification(
                    removed_device, self.fallback_device
                )
                self._emit_notification(fallback_notification)

                # Update preferred device to fallback
                self.preferred_device = self.fallback_device
                return self.fallback_device
            else:
                self.logger.error("No fallback device available")
                self.preferred_device = None
                return None

        return None

    def handle_device_added(self, added_device: AudioDevice) -> None:
        """
        Handle when a new device is added/connected.

        Args:
            added_device: AudioDevice that was added
        """
        self.logger.debug(f"Device added: {added_device.name}")

        # Create notification for added device
        notification = DeviceNotification.create_device_added_notification(added_device)
        self._emit_notification(notification)

        # If we don't have a preferred device, this could be a candidate
        if not self.preferred_device:
            self.logger.info("No preferred device set, considering new device as fallback")
            self.fallback_device = added_device

    def handle_device_changed(self, changed_device: AudioDevice) -> None:
        """
        Handle when a device's properties change.

        Args:
            changed_device: AudioDevice that changed
        """
        self.logger.debug(f"Device changed: {changed_device.name}")

        # Update our references if this device is preferred or fallback
        if self.preferred_device and self.preferred_device.device_id == changed_device.device_id:
            self.preferred_device = changed_device

        if self.fallback_device and self.fallback_device.device_id == changed_device.device_id:
            self.fallback_device = changed_device

    def handle_device_list_refresh(self, added_devices: List[AudioDevice],
                                 removed_devices: List[AudioDevice]) -> None:
        """
        Handle when the device list is refreshed.

        Args:
            added_devices: List of newly added devices
            removed_devices: List of removed devices
        """
        self.logger.info(f"Device list refreshed: +{len(added_devices)}, -{len(removed_devices)}")

        # Create refresh notification
        notification = DeviceNotification.create_device_list_refresh_notification(
            len(added_devices), len(removed_devices)
        )
        self._emit_notification(notification)

        # Handle each removed device
        for device in removed_devices:
            self.handle_device_removed(device)

        # Handle each added device
        for device in added_devices:
            self.handle_device_added(device)

    def set_preferred_device(self, device: AudioDevice) -> None:
        """
        Set the preferred audio device.

        Args:
            device: AudioDevice to set as preferred
        """
        self.logger.info(f"Setting preferred device: {device.name}")
        self.preferred_device = device

    def set_fallback_device(self, device: AudioDevice) -> None:
        """
        Set the fallback audio device.

        Args:
            device: AudioDevice to set as fallback
        """
        self.logger.info(f"Setting fallback device: {device.name}")
        self.fallback_device = device

    def get_current_device(self) -> Optional[AudioDevice]:
        """
        Get the currently active device (preferred or fallback).

        Returns:
            Optional[AudioDevice]: Current active device
        """
        return self.preferred_device or self.fallback_device