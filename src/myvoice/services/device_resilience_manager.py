"""
Device Resilience Manager

This module provides the DeviceResilienceManager service that coordinates
audio device monitoring, change handling, and user notifications for
graceful device disconnect/reconnect handling.

Story 2.5: Audio Device Resilience
- FR30: Handle audio device changes gracefully
- NFR8: Audio device disconnect handling
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from myvoice.models.audio_device import AudioDevice, DeviceType
from myvoice.models.device_notification import (
    DeviceNotification,
    DeviceChangeHandler,
    NotificationType,
    NotificationSeverity,
)
from myvoice.models.error import MyVoiceError, ErrorSeverity
from myvoice.services.integrations.windows_audio_client import (
    WindowsAudioClient,
    DeviceChangeEvent,
)
from myvoice.services.core.base_service import BaseService, ServiceStatus


class DeviceRole(Enum):
    """Role of an audio device in the application."""
    MONITOR = "monitor"
    VIRTUAL_MIC = "virtual_mic"


@dataclass
class DeviceState:
    """Tracks the state of a device by role."""
    role: DeviceRole
    device: Optional[AudioDevice] = None
    is_connected: bool = False
    last_seen: Optional[datetime] = None
    fallback_device: Optional[AudioDevice] = None


@dataclass
class DeviceResilienceConfig:
    """Configuration for device resilience behavior."""
    # Monitoring settings
    enable_monitoring: bool = True
    poll_interval_seconds: float = 2.0

    # Recovery settings
    auto_recovery_enabled: bool = True
    recovery_delay_seconds: float = 1.0

    # Notification settings
    show_disconnect_warning: bool = True
    show_reconnect_notification: bool = True
    show_fallback_notification: bool = True

    # Fallback behavior
    fallback_to_default_on_disconnect: bool = True


class DeviceResilienceManager(BaseService):
    """
    Device Resilience Manager Service

    Coordinates audio device monitoring, change handling, and user notifications
    for graceful device disconnect/reconnect handling.

    Story 2.5 Requirements:
    - Monitor speaker disconnect → graceful stop, warning shown
    - Device reconnect → auto-recovery, no manual reconfiguration
    - Virtual mic disconnect → monitor still works, warning shown
    - New devices detected when settings opened
    - Graceful fallback with notification when selected device unavailable

    Key Features:
    - Centralized device monitoring for all audio services
    - Automatic device recovery on reconnection
    - User-friendly notifications for device changes
    - Independent handling of monitor and virtual mic devices
    """

    def __init__(self, config: Optional[DeviceResilienceConfig] = None):
        """
        Initialize the Device Resilience Manager.

        Args:
            config: Configuration for resilience behavior
        """
        super().__init__("DeviceResilienceManager")

        self.config = config or DeviceResilienceConfig()
        self.logger = logging.getLogger(__name__)

        # Windows Audio Client for device monitoring
        self._audio_client: Optional[WindowsAudioClient] = None

        # Device state tracking
        self._device_states: Dict[DeviceRole, DeviceState] = {
            DeviceRole.MONITOR: DeviceState(role=DeviceRole.MONITOR),
            DeviceRole.VIRTUAL_MIC: DeviceState(role=DeviceRole.VIRTUAL_MIC),
        }

        # Device change handler for business logic
        self._change_handler = DeviceChangeHandler()

        # Callbacks for external notification
        self._notification_callbacks: List[Callable[[DeviceNotification], None]] = []
        self._device_change_callbacks: List[Callable[[DeviceRole, DeviceChangeEvent], None]] = []
        self._recovery_callbacks: List[Callable[[DeviceRole, AudioDevice], None]] = []

        # Pending recovery tracking (device_id -> timestamp)
        self._pending_recoveries: Dict[str, datetime] = {}

        # Service state
        self._is_initialized = False
        self._is_monitoring = False

        self.logger.info("DeviceResilienceManager initialized")

    async def start(self) -> bool:
        """Start the device resilience manager."""
        return await self.initialize()

    async def stop(self) -> bool:
        """Stop the device resilience manager."""
        return await self.shutdown()

    async def health_check(self) -> tuple[bool, Optional[MyVoiceError]]:
        """Check health of the device resilience manager."""
        try:
            if not self._is_initialized:
                return False, MyVoiceError(
                    severity=ErrorSeverity.WARNING,
                    code="SERVICE_NOT_INITIALIZED",
                    user_message="Device resilience manager is not initialized"
                )

            if not self._audio_client:
                return False, MyVoiceError(
                    severity=ErrorSeverity.WARNING,
                    code="AUDIO_CLIENT_UNAVAILABLE",
                    user_message="Audio client is not available"
                )

            return True, None

        except Exception as e:
            return False, MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="HEALTH_CHECK_FAILED",
                user_message="Device resilience health check failed",
                technical_details=str(e)
            )

    async def initialize(self) -> bool:
        """Initialize the device resilience manager."""
        try:
            self.logger.info("Initializing DeviceResilienceManager")

            # Initialize Windows Audio Client with monitoring enabled
            self._audio_client = WindowsAudioClient(
                enable_monitoring=self.config.enable_monitoring
            )

            # Register our device change callback
            self._audio_client.add_device_change_callback(self._handle_device_change_event)

            # Set up the change handler notification callback
            self._change_handler.add_notification_callback(self._emit_notification)

            # Start device monitoring if enabled
            if self.config.enable_monitoring:
                self._audio_client.start_device_monitoring(
                    poll_interval=self.config.poll_interval_seconds
                )
                self._is_monitoring = True
                self.logger.info(f"Device monitoring started (interval: {self.config.poll_interval_seconds}s)")

            self._is_initialized = True
            self.status = ServiceStatus.RUNNING
            self.logger.info("DeviceResilienceManager initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize DeviceResilienceManager: {e}")
            self.status = ServiceStatus.ERROR
            return False

    async def shutdown(self) -> bool:
        """Shutdown the device resilience manager gracefully."""
        try:
            self.logger.info("Shutting down DeviceResilienceManager")

            # Stop monitoring
            if self._audio_client:
                self._audio_client.stop_device_monitoring()
                self._audio_client.close()
                self._audio_client = None

            self._is_monitoring = False
            self._is_initialized = False
            self.status = ServiceStatus.STOPPED

            self.logger.info("DeviceResilienceManager shutdown complete")
            return True

        except Exception as e:
            self.logger.error(f"Error during DeviceResilienceManager shutdown: {e}")
            return False

    # =========================================================================
    # Device Registration and Tracking
    # =========================================================================

    def register_device(
        self,
        role: DeviceRole,
        device: AudioDevice,
        fallback_device: Optional[AudioDevice] = None
    ) -> None:
        """
        Register a device for monitoring.

        Args:
            role: Role of the device (MONITOR or VIRTUAL_MIC)
            device: AudioDevice to monitor
            fallback_device: Optional fallback device if primary fails
        """
        state = self._device_states[role]
        state.device = device
        state.is_connected = True
        state.last_seen = datetime.now()
        state.fallback_device = fallback_device

        # Also register with the change handler
        if role == DeviceRole.MONITOR:
            self._change_handler.set_preferred_device(device)
            if fallback_device:
                self._change_handler.set_fallback_device(fallback_device)

        self.logger.info(f"Registered {role.value} device: {device.name}")

    def unregister_device(self, role: DeviceRole) -> None:
        """
        Unregister a device from monitoring.

        Args:
            role: Role of the device to unregister
        """
        state = self._device_states[role]
        device_name = state.device.name if state.device else "unknown"

        state.device = None
        state.is_connected = False
        state.last_seen = None
        state.fallback_device = None

        self.logger.info(f"Unregistered {role.value} device: {device_name}")

    def get_device_state(self, role: DeviceRole) -> DeviceState:
        """Get the current state of a device by role."""
        return self._device_states[role]

    # =========================================================================
    # Device Change Event Handling
    # =========================================================================

    def _handle_device_change_event(self, event: DeviceChangeEvent) -> None:
        """
        Handle device change events from WindowsAudioClient.

        Args:
            event: Device change event
        """
        self.logger.info(f"Device change event: {event.event_type} - {event.device_name}")

        try:
            # Check if this event affects any of our registered devices
            for role, state in self._device_states.items():
                if state.device is None:
                    continue

                # Check if the event matches our registered device
                if self._event_matches_device(event, state.device):
                    if event.event_type == "removed":
                        self._handle_device_removed(role, state, event)
                    elif event.event_type == "added":
                        self._handle_device_added(role, state, event)
                    elif event.event_type == "changed":
                        self._handle_device_changed(role, state, event)

                    # Notify external listeners
                    for callback in self._device_change_callbacks:
                        try:
                            callback(role, event)
                        except Exception as e:
                            self.logger.error(f"Error in device change callback: {e}")

            # Also check for reconnection of previously disconnected devices
            self._check_for_device_recovery(event)

        except Exception as e:
            self.logger.error(f"Error handling device change event: {e}")

    def _event_matches_device(self, event: DeviceChangeEvent, device: AudioDevice) -> bool:
        """Check if an event matches a registered device."""
        # Match by device name (most reliable)
        if device.name and event.device_name:
            # Use contains match for flexibility
            if device.name.lower() in event.device_name.lower():
                return True
            if event.device_name.lower() in device.name.lower():
                return True

        # Match by device index
        if event.device_index is not None and device.device_id:
            device_index = self._extract_device_index(device.device_id)
            if device_index == event.device_index:
                return True

        return False

    def _extract_device_index(self, device_id: str) -> Optional[int]:
        """Extract device index from device_id string."""
        if device_id.startswith('pyaudio_'):
            try:
                return int(device_id.split('_')[1])
            except (IndexError, ValueError):
                return None
        try:
            return int(device_id)
        except ValueError:
            return None

    def _handle_device_removed(
        self,
        role: DeviceRole,
        state: DeviceState,
        event: DeviceChangeEvent
    ) -> None:
        """
        Handle device removal event.

        Story 2.5 AC: Monitor speaker disconnect → graceful stop, warning shown
        Story 2.5 AC: Virtual mic disconnect → monitor still works, warning shown
        """
        self.logger.warning(f"{role.value} device removed: {event.device_name}")

        # Update state
        state.is_connected = False
        state.last_seen = datetime.now()

        # Store for potential recovery
        if state.device:
            self._pending_recoveries[state.device.device_id] = datetime.now()

        # Create and emit warning notification
        if self.config.show_disconnect_warning and state.device:
            notification = DeviceNotification.create_device_removed_notification(state.device)
            self._emit_notification(notification)

        # Attempt fallback for monitor device
        if role == DeviceRole.MONITOR and self.config.fallback_to_default_on_disconnect:
            self._attempt_fallback(role, state, event)

    def _handle_device_added(
        self,
        role: DeviceRole,
        state: DeviceState,
        event: DeviceChangeEvent
    ) -> None:
        """
        Handle device added event.

        Story 2.5 AC: Device reconnect → auto-recovery, no manual reconfiguration
        """
        self.logger.info(f"{role.value} device added: {event.device_name}")

        # Check if this is a recovery of our registered device
        if state.device and not state.is_connected:
            # Device reconnected - trigger auto-recovery
            if self.config.auto_recovery_enabled:
                self._trigger_recovery(role, state, event)

    def _handle_device_changed(
        self,
        role: DeviceRole,
        state: DeviceState,
        event: DeviceChangeEvent
    ) -> None:
        """Handle device properties changed event."""
        self.logger.debug(f"{role.value} device changed: {event.device_name}")
        state.last_seen = datetime.now()

    def _attempt_fallback(
        self,
        role: DeviceRole,
        state: DeviceState,
        event: DeviceChangeEvent
    ) -> None:
        """
        Attempt to fall back to a backup device.

        Story 2.5 AC: Graceful fallback with notification when selected device unavailable
        """
        self.logger.info(f"Attempting fallback for {role.value}")

        if not state.fallback_device:
            # Try to get default device
            if self._audio_client and role == DeviceRole.MONITOR:
                fallback = self._audio_client.get_default_output_device()
                if fallback:
                    state.fallback_device = fallback

        if state.fallback_device:
            self.logger.info(f"Falling back to: {state.fallback_device.name}")

            # Emit fallback notification
            if self.config.show_fallback_notification and state.device:
                notification = DeviceNotification.create_device_fallback_notification(
                    state.device,
                    state.fallback_device
                )
                self._emit_notification(notification)

            # Notify recovery callbacks to switch to fallback
            for callback in self._recovery_callbacks:
                try:
                    callback(role, state.fallback_device)
                except Exception as e:
                    self.logger.error(f"Error in fallback callback: {e}")
        else:
            self.logger.warning(f"No fallback device available for {role.value}")

    def _trigger_recovery(
        self,
        role: DeviceRole,
        state: DeviceState,
        event: DeviceChangeEvent
    ) -> None:
        """
        Trigger auto-recovery for a reconnected device.

        Story 2.5 AC: Device reconnect → auto-recovery, no manual reconfiguration
        """
        self.logger.info(f"Triggering auto-recovery for {role.value}: {event.device_name}")

        # Update state
        state.is_connected = True
        state.last_seen = datetime.now()

        # Remove from pending recoveries
        if state.device and state.device.device_id in self._pending_recoveries:
            del self._pending_recoveries[state.device.device_id]

        # Emit restored notification
        if self.config.show_reconnect_notification and state.device:
            notification = DeviceNotification.create_device_restored_notification(state.device)
            self._emit_notification(notification)

        # Notify recovery callbacks
        if state.device:
            for callback in self._recovery_callbacks:
                try:
                    callback(role, state.device)
                except Exception as e:
                    self.logger.error(f"Error in recovery callback: {e}")

    def _check_for_device_recovery(self, event: DeviceChangeEvent) -> None:
        """Check if an added device should trigger recovery for a pending device."""
        if event.event_type != "added":
            return

        # Check all roles for pending recovery
        for role, state in self._device_states.items():
            if state.device and not state.is_connected:
                # Check if the added device matches our disconnected device
                if self._event_matches_device(event, state.device):
                    self._trigger_recovery(role, state, event)

    # =========================================================================
    # Device Refresh (for Settings Dialog)
    # =========================================================================

    def refresh_devices(self) -> bool:
        """
        Refresh the audio device list.

        Story 2.5 AC: New devices detected when settings opened

        Returns:
            bool: True if refresh was successful
        """
        if not self._audio_client:
            self.logger.warning("Cannot refresh devices: audio client not available")
            return False

        try:
            self.logger.info("Refreshing audio device list")
            success = self._audio_client.refresh_audio_devices()

            if success:
                self.logger.info("Audio device list refreshed successfully")
            else:
                self.logger.warning("Audio device refresh failed")

            return success

        except Exception as e:
            self.logger.error(f"Error refreshing devices: {e}")
            return False

    def enumerate_devices(self, refresh: bool = False) -> List[AudioDevice]:
        """
        Enumerate all available audio devices.

        Args:
            refresh: If True, force refresh the device cache

        Returns:
            List of available audio devices
        """
        if not self._audio_client:
            return []

        try:
            return self._audio_client.enumerate_audio_devices(refresh_cache=refresh)
        except Exception as e:
            self.logger.error(f"Error enumerating devices: {e}")
            return []

    def find_device_by_metadata(
        self,
        device_id: Optional[str],
        device_name: Optional[str],
        host_api_name: Optional[str] = None
    ) -> Optional[AudioDevice]:
        """
        Find a device using smart matching.

        Args:
            device_id: PyAudio device ID
            device_name: Device name for fallback matching
            host_api_name: Host API name for precise matching

        Returns:
            Matched device or None
        """
        if not self._audio_client:
            return None

        return self._audio_client.find_device_by_metadata(
            device_id, device_name, host_api_name
        )

    # =========================================================================
    # Callback Management
    # =========================================================================

    def add_notification_callback(
        self,
        callback: Callable[[DeviceNotification], None]
    ) -> None:
        """
        Add callback for device notifications.

        Args:
            callback: Function to call when notifications are generated
        """
        if callback not in self._notification_callbacks:
            self._notification_callbacks.append(callback)
            self.logger.debug("Added notification callback")

    def remove_notification_callback(
        self,
        callback: Callable[[DeviceNotification], None]
    ) -> None:
        """Remove a notification callback."""
        if callback in self._notification_callbacks:
            self._notification_callbacks.remove(callback)
            self.logger.debug("Removed notification callback")

    def add_device_change_callback(
        self,
        callback: Callable[[DeviceRole, DeviceChangeEvent], None]
    ) -> None:
        """
        Add callback for device change events.

        Args:
            callback: Function to call when device changes occur
        """
        if callback not in self._device_change_callbacks:
            self._device_change_callbacks.append(callback)

    def add_recovery_callback(
        self,
        callback: Callable[[DeviceRole, AudioDevice], None]
    ) -> None:
        """
        Add callback for device recovery events.

        This callback is triggered when:
        - A device is reconnected and auto-recovery is triggered
        - A fallback device is selected after primary disconnect

        Args:
            callback: Function to call with (role, device) when recovery occurs
        """
        if callback not in self._recovery_callbacks:
            self._recovery_callbacks.append(callback)
            self.logger.debug("Added recovery callback")

    def _emit_notification(self, notification: DeviceNotification) -> None:
        """Emit a notification to all registered callbacks."""
        self.logger.info(f"Emitting notification: {notification.title}")

        for callback in self._notification_callbacks:
            try:
                callback(notification)
            except Exception as e:
                self.logger.error(f"Error in notification callback: {e}")

    # =========================================================================
    # Status and Health
    # =========================================================================

    async def get_health(self) -> Dict[str, Any]:
        """Get health status of the device resilience manager."""
        return {
            "service_name": self.service_name,
            "status": self.status.value,
            "initialized": self._is_initialized,
            "monitoring_active": self._is_monitoring,
            "monitor_device": {
                "name": self._device_states[DeviceRole.MONITOR].device.name
                    if self._device_states[DeviceRole.MONITOR].device else None,
                "connected": self._device_states[DeviceRole.MONITOR].is_connected,
            },
            "virtual_mic_device": {
                "name": self._device_states[DeviceRole.VIRTUAL_MIC].device.name
                    if self._device_states[DeviceRole.VIRTUAL_MIC].device else None,
                "connected": self._device_states[DeviceRole.VIRTUAL_MIC].is_connected,
            },
            "pending_recoveries": len(self._pending_recoveries),
        }
