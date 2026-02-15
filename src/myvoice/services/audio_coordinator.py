"""
Audio Coordinator Service

This module provides the AudioCoordinator that manages both MonitorAudioService
and VirtualMicrophoneService for coordinated dual-stream audio routing without
resource conflicts.

This replaces the problematic play_synchronized_dual_stream approach with proper
service separation as specified in the dual-service architecture design.

Story 2.5: Audio Device Resilience
- Integrated DeviceResilienceManager for graceful device disconnect/reconnect handling
- Auto-recovery when devices reconnect
- User notifications for device changes
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from datetime import datetime

from myvoice.models.audio_device import AudioDevice
from myvoice.models.error import MyVoiceError, ErrorSeverity
from myvoice.models.app_settings import AppSettings
from myvoice.models.device_notification import DeviceNotification
from myvoice.services.monitor_audio_service import MonitorAudioService, MonitorPlaybackTask
from myvoice.services.virtual_microphone_service import VirtualMicrophoneService, VirtualPlaybackTask
from myvoice.services.device_resilience_manager import (
    DeviceResilienceManager,
    DeviceResilienceConfig,
    DeviceRole,
)
from myvoice.services.core.base_service import BaseService, ServiceStatus


@dataclass
class DualStreamResult:
    """Result from coordinated dual-stream playback."""
    monitor_task: Optional[MonitorPlaybackTask]
    virtual_task: Optional[VirtualPlaybackTask]
    coordination_id: str
    start_time: datetime
    success: bool = True
    error_message: Optional[str] = None

    @property
    def both_successful(self) -> bool:
        """Check if both streams were started successfully."""
        # Tasks run in background - we just need to verify they started, not completed
        monitor_ok = self.monitor_task and self.monitor_task.status.value in ['playing', 'completed']
        virtual_ok = self.virtual_task and self.virtual_task.status.value in ['playing', 'completed']
        return monitor_ok and virtual_ok

    @property
    def any_successful(self) -> bool:
        """Check if at least one stream was started successfully."""
        # Tasks run in background - we just need to verify they started, not completed
        # Status values: 'pending', 'playing', 'completed', 'failed', 'stopped'
        monitor_ok = self.monitor_task and self.monitor_task.status.value in ['playing', 'completed']
        virtual_ok = self.virtual_task and self.virtual_task.status.value in ['playing', 'completed']
        return monitor_ok or virtual_ok


@dataclass
class AudioCoordinatorConfig:
    """Configuration for audio coordination behavior."""
    # Coordination settings
    max_start_delay_ms: int = 50  # Maximum delay between service starts
    coordination_timeout: float = 30.0  # Timeout for coordinated operations

    # Error handling
    allow_partial_failure: bool = True  # Allow one service to fail
    retry_failed_service: bool = True   # Retry failed service once

    # Monitoring
    enable_health_monitoring: bool = True
    health_check_interval: float = 60.0


class AudioCoordinator(BaseService):
    """
    Audio Coordinator Service

    Coordinates audio playback between MonitorAudioService (Audio Service 1) and
    VirtualMicrophoneService (Audio Service 2) to provide seamless dual-stream
    routing without resource conflicts.

    Key Features:
    - Independent service management (no resource sharing)
    - Coordinated parallel playback execution
    - Graceful fallback handling (monitor-only or virtual-only)
    - Comprehensive error handling and recovery
    - Health monitoring for both services
    """

    def __init__(self, app_settings: Optional[AppSettings] = None):
        """
        Initialize the Audio Coordinator.

        Args:
            app_settings: Application settings for device preferences
        """
        super().__init__("AudioCoordinator")

        self.app_settings = app_settings
        self.config = AudioCoordinatorConfig()
        self.logger = logging.getLogger(__name__)

        # Service instances
        self.monitor_service: Optional[MonitorAudioService] = None
        self.virtual_service: Optional[VirtualMicrophoneService] = None

        # Story 2.5: Device Resilience Manager for graceful disconnect/reconnect
        self.resilience_manager: Optional[DeviceResilienceManager] = None

        # Coordination tracking
        self._active_coordinations: Dict[str, DualStreamResult] = {}
        self._coordination_counter = 0

        # Service health tracking
        self._last_health_check: Optional[datetime] = None
        self._services_healthy = True

        # Callbacks (Story 2.1: playback_complete signal)
        self._playback_complete_callback: Optional[Callable[[str], None]] = None

        # Story 2.5: Device notification callbacks
        self._device_notification_callbacks: List[Callable[[DeviceNotification], None]] = []

        self.logger.info("AudioCoordinator initialized")

    async def start(self) -> bool:
        """Start the audio coordinator and both services."""
        return await self.initialize()

    async def stop(self) -> bool:
        """Stop the audio coordinator and both services."""
        return await self.shutdown()

    async def health_check(self) -> tuple[bool, Optional[MyVoiceError]]:
        """Check health of the coordinator and both services."""
        try:
            # Check coordinator state
            if not self._is_initialized:
                return False, MyVoiceError(
                    severity=ErrorSeverity.WARNING,
                    code="COORDINATOR_NOT_INITIALIZED",
                    user_message="Audio coordinator is not initialized"
                )

            # Check both services
            monitor_healthy = True
            virtual_healthy = True

            if self.monitor_service:
                monitor_health = await self.monitor_service.health_check()
                monitor_healthy = monitor_health[0]

            if self.virtual_service:
                virtual_health = await self.virtual_service.health_check()
                virtual_healthy = virtual_health[0]

            # Coordinator is healthy if at least one service is healthy
            overall_healthy = monitor_healthy or virtual_healthy

            if not overall_healthy:
                return False, MyVoiceError(
                    severity=ErrorSeverity.ERROR,
                    code="ALL_SERVICES_UNHEALTHY",
                    user_message="Both audio services are unhealthy"
                )

            return True, None

        except Exception as e:
            return False, MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="HEALTH_CHECK_FAILED",
                user_message="Coordinator health check failed",
                technical_details=str(e)
            )

    async def initialize(self) -> bool:
        """Initialize the audio coordinator and both services."""
        try:
            self.logger.info("Initializing AudioCoordinator with dual services")

            # Story 2.5: Initialize Device Resilience Manager first
            resilience_config = DeviceResilienceConfig(
                enable_monitoring=True,
                poll_interval_seconds=2.0,
                auto_recovery_enabled=True,
                show_disconnect_warning=True,
                show_reconnect_notification=True,
                show_fallback_notification=True,
                fallback_to_default_on_disconnect=True,
            )
            self.resilience_manager = DeviceResilienceManager(resilience_config)
            resilience_success = await self.resilience_manager.initialize()

            if resilience_success:
                self.logger.info("DeviceResilienceManager initialized successfully")
                # Register callbacks for device recovery and notifications
                self.resilience_manager.add_recovery_callback(self._handle_device_recovery)
                self.resilience_manager.add_notification_callback(self._handle_device_notification)
            else:
                self.logger.warning("⚠ DeviceResilienceManager initialization failed - continuing without resilience")

            # Initialize MonitorAudioService (Audio Service 1)
            self.monitor_service = MonitorAudioService(self.app_settings)
            monitor_success = await self.monitor_service.initialize()

            if monitor_success:
                self.logger.info("MonitorAudioService initialized successfully")
                # Story 2.5: Register monitor device with resilience manager
                if self.resilience_manager and self.monitor_service.current_monitor_device:
                    self.resilience_manager.register_device(
                        DeviceRole.MONITOR,
                        self.monitor_service.current_monitor_device
                    )
            else:
                self.logger.warning("⚠ MonitorAudioService initialization failed")

            # Initialize VirtualMicrophoneService (Audio Service 2)
            self.virtual_service = VirtualMicrophoneService(self.app_settings)
            virtual_success = await self.virtual_service.initialize()

            if virtual_success:
                self.logger.info("VirtualMicrophoneService initialized successfully")
                # Story 2.5: Register virtual mic device with resilience manager
                virtual_device = getattr(self.virtual_service, 'current_virtual_device', None)
                if self.resilience_manager and virtual_device:
                    self.resilience_manager.register_device(
                        DeviceRole.VIRTUAL_MIC,
                        virtual_device
                    )
            else:
                self.logger.warning("⚠ VirtualMicrophoneService initialization failed")

            # Coordinator succeeds if at least one service succeeds
            if monitor_success or virtual_success:
                self._is_initialized = True
                self.status = ServiceStatus.RUNNING
                self.logger.info(f"AudioCoordinator initialized - Monitor: {'OK' if monitor_success else 'FAILED'}, Virtual: {'OK' if virtual_success else 'FAILED'}")
                return True
            else:
                self.logger.error("AudioCoordinator initialization failed - both services failed")
                self.status = ServiceStatus.ERROR
                return False

        except Exception as e:
            self.logger.error(f"Failed to initialize AudioCoordinator: {e}")
            self.status = ServiceStatus.ERROR
            return False

    async def shutdown(self) -> bool:
        """Shutdown the audio coordinator and both services gracefully."""
        try:
            self.logger.info("Shutting down AudioCoordinator")

            # Stop all active coordinations
            for coordination_id in list(self._active_coordinations.keys()):
                await self.stop_coordination(coordination_id)

            # Shutdown both services in parallel
            shutdown_tasks = []

            if self.monitor_service:
                shutdown_tasks.append(self.monitor_service.shutdown())

            if self.virtual_service:
                shutdown_tasks.append(self.virtual_service.shutdown())

            if shutdown_tasks:
                results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)

                # Log results
                for i, result in enumerate(results):
                    service_name = "Monitor" if i == 0 else "Virtual"
                    if isinstance(result, Exception):
                        self.logger.warning(f"{service_name} service shutdown error: {result}")
                    elif result:
                        self.logger.info(f"{service_name} service shutdown successful")
                    else:
                        self.logger.warning(f"⚠ {service_name} service shutdown failed")

            # Story 2.5: Shutdown resilience manager
            if self.resilience_manager:
                await self.resilience_manager.shutdown()
                self.resilience_manager = None

            self._is_initialized = False
            self.status = ServiceStatus.STOPPED
            self.logger.info("AudioCoordinator shutdown complete")
            return True

        except Exception as e:
            self.logger.error(f"Error during AudioCoordinator shutdown: {e}")
            return False

    async def play_dual_stream(self,
                             audio_data: bytes,
                             monitor_device: Optional[AudioDevice] = None,
                             virtual_device: Optional[AudioDevice] = None,
                             volume: float = 1.0) -> DualStreamResult:
        """
        Execute coordinated dual-stream playback to both services.

        Args:
            audio_data: Audio data to play
            monitor_device: Target monitor device (uses service default if None)
            virtual_device: Target virtual device (uses service default if None)
            volume: Volume level (0.0 to 1.0)

        Returns:
            DualStreamResult: Result of coordinated playback
        """
        if not self._is_initialized:
            return DualStreamResult(
                monitor_task=None,
                virtual_task=None,
                coordination_id="",
                start_time=datetime.now(),
                success=False,
                error_message="AudioCoordinator not initialized"
            )

        # Generate coordination ID
        self._coordination_counter += 1
        coordination_id = f"coord_{self._coordination_counter}_{int(datetime.now().timestamp())}"

        self.logger.info(f"Starting coordinated dual-stream playback {coordination_id}")

        try:
            # Start both services in parallel
            tasks = []

            if self.monitor_service and await self._is_monitor_service_healthy():
                monitor_task = asyncio.create_task(
                    self.monitor_service.play_monitor_audio(audio_data, monitor_device, volume)
                )
                tasks.append(("monitor", monitor_task))

            if self.virtual_service and await self._is_virtual_service_healthy():
                virtual_task = asyncio.create_task(
                    self.virtual_service.play_virtual_microphone(audio_data, virtual_device, volume)
                )
                tasks.append(("virtual", virtual_task))

            if not tasks:
                return DualStreamResult(
                    monitor_task=None,
                    virtual_task=None,
                    coordination_id=coordination_id,
                    start_time=datetime.now(),
                    success=False,
                    error_message="No healthy services available"
                )

            # Wait for all tasks to complete
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

            # Process results
            monitor_task = None
            virtual_task = None

            for i, (service_type, _) in enumerate(tasks):
                result = results[i]

                if isinstance(result, Exception):
                    self.logger.error(f"{service_type} service failed: {result}")
                else:
                    if service_type == "monitor":
                        monitor_task = result
                    else:
                        virtual_task = result

            # Create result
            dual_result = DualStreamResult(
                monitor_task=monitor_task,
                virtual_task=virtual_task,
                coordination_id=coordination_id,
                start_time=datetime.now(),
                success=monitor_task is not None or virtual_task is not None
            )

            # Track active coordination
            self._active_coordinations[coordination_id] = dual_result

            success_msg = []
            if monitor_task:
                success_msg.append("monitor")
            if virtual_task:
                success_msg.append("virtual")

            self.logger.info(f"Coordinated playback {coordination_id} started: {', '.join(success_msg)}")
            return dual_result

        except Exception as e:
            self.logger.error(f"Coordinated dual-stream playback failed: {e}")
            return DualStreamResult(
                monitor_task=None,
                virtual_task=None,
                coordination_id=coordination_id,
                start_time=datetime.now(),
                success=False,
                error_message=str(e)
            )

    async def play_monitor_only(self,
                              audio_data: bytes,
                              device: Optional[AudioDevice] = None,
                              volume: float = 1.0) -> Optional[MonitorPlaybackTask]:
        """
        Play audio through monitor service only.

        Args:
            audio_data: Audio data to play
            device: Target monitor device
            volume: Volume level

        Returns:
            MonitorPlaybackTask: Monitor playback task or None if failed
        """
        if not self.monitor_service or not await self._is_monitor_service_healthy():
            self.logger.warning("Monitor service not available for monitor-only playback")
            return None

        try:
            return await self.monitor_service.play_monitor_audio(audio_data, device, volume)
        except Exception as e:
            self.logger.error(f"Monitor-only playback failed: {e}")
            return None

    async def play_virtual_only(self,
                              audio_data: bytes,
                              device: Optional[AudioDevice] = None,
                              volume: float = 1.0) -> Optional[VirtualPlaybackTask]:
        """
        Play audio through virtual microphone service only.

        Args:
            audio_data: Audio data to play
            device: Target virtual device
            volume: Volume level

        Returns:
            VirtualPlaybackTask: Virtual playback task or None if failed
        """
        if not self.virtual_service or not await self._is_virtual_service_healthy():
            self.logger.warning("Virtual microphone service not available for virtual-only playback")
            return None

        try:
            return await self.virtual_service.play_virtual_microphone(audio_data, device, volume)
        except Exception as e:
            self.logger.error(f"Virtual-only playback failed: {e}")
            return None

    async def stop_coordination(self, coordination_id: str) -> bool:
        """Stop a coordinated playback operation."""
        try:
            if coordination_id not in self._active_coordinations:
                self.logger.warning(f"Coordination {coordination_id} not found")
                return False

            result = self._active_coordinations[coordination_id]

            # Stop both tasks
            stop_tasks = []

            if result.monitor_task and self.monitor_service:
                stop_tasks.append(
                    self.monitor_service.stop_monitor_playback(result.monitor_task.playback_id)
                )

            if result.virtual_task and self.virtual_service:
                stop_tasks.append(
                    self.virtual_service.stop_virtual_playback(result.virtual_task.playback_id)
                )

            if stop_tasks:
                await asyncio.gather(*stop_tasks, return_exceptions=True)

            # Remove from tracking
            del self._active_coordinations[coordination_id]

            self.logger.info(f"Stopped coordination {coordination_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping coordination {coordination_id}: {e}")
            return False

    async def enumerate_all_devices(self) -> Dict[str, List[AudioDevice]]:
        """Enumerate devices from both services."""
        devices = {
            "monitor": [],
            "virtual": []
        }

        try:
            if self.monitor_service:
                devices["monitor"] = await self.monitor_service.enumerate_monitor_devices()

            if self.virtual_service:
                devices["virtual"] = await self.virtual_service.enumerate_virtual_devices()

        except Exception as e:
            self.logger.error(f"Error enumerating devices: {e}")

        return devices

    async def get_coordinator_status(self) -> Dict[str, Any]:
        """Get comprehensive coordinator status."""
        status = {
            "coordinator": {
                "initialized": self._is_initialized,
                "status": self.status.value,
                "active_coordinations": len(self._active_coordinations)
            },
            "monitor_service": None,
            "virtual_service": None
        }

        try:
            if self.monitor_service:
                status["monitor_service"] = await self.monitor_service.get_health()

            if self.virtual_service:
                status["virtual_service"] = await self.virtual_service.get_health()

        except Exception as e:
            self.logger.error(f"Error getting coordinator status: {e}")

        return status

    async def _is_monitor_service_healthy(self) -> bool:
        """Check if monitor service is healthy."""
        if not self.monitor_service:
            self.logger.error("[WIN11-DEBUG] Monitor service health check FAILED: monitor_service is None")
            return False

        try:
            health = await self.monitor_service.health_check()
            is_healthy = health[0]
            self.logger.info(f"[WIN11-DEBUG] Monitor service health check result: {is_healthy}, health={health}")
            return is_healthy
        except Exception as e:
            self.logger.error(f"[WIN11-DEBUG] Monitor service health check EXCEPTION: {e}")
            return False

    async def _is_virtual_service_healthy(self) -> bool:
        """Check if virtual service is healthy."""
        if not self.virtual_service:
            self.logger.error("[WIN11-DEBUG] Virtual service health check FAILED: virtual_service is None")
            return False

        try:
            health = await self.virtual_service.health_check()
            is_healthy = health[0]
            self.logger.info(f"[WIN11-DEBUG] Virtual service health check result: {is_healthy}, health={health}")
            return is_healthy
        except Exception as e:
            self.logger.error(f"[WIN11-DEBUG] Virtual service health check EXCEPTION: {e}")
            return False

    async def update_device_settings(self, new_settings: AppSettings) -> bool:
        """
        Update device settings for both services.

        Args:
            new_settings: New application settings

        Returns:
            bool: True if settings were updated successfully
        """
        try:
            self.logger.info("Updating device settings in AudioCoordinator")

            # Update coordinator settings
            self.app_settings = new_settings

            # Update monitor service settings
            if self.monitor_service:
                self.monitor_service.app_settings = new_settings
                self.logger.debug("Monitor service settings updated")

            # Update virtual service settings
            if self.virtual_service:
                self.virtual_service.app_settings = new_settings
                self.logger.debug("Virtual service settings updated")

            self.logger.info("Device settings updated successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update device settings: {e}")
            return False

    async def start_device_monitoring(self) -> bool:
        """
        Start device change monitoring for both services.

        Returns:
            bool: True if monitoring started successfully
        """
        try:
            self.logger.info("Starting device monitoring")

            # Start monitoring for monitor service
            if self.monitor_service and hasattr(self.monitor_service, 'start_device_monitoring'):
                monitor_result = await self.monitor_service.start_device_monitoring()
                self.logger.debug(f"Monitor service device monitoring: {'started' if monitor_result else 'failed'}")

            # Start monitoring for virtual service
            if self.virtual_service and hasattr(self.virtual_service, 'start_device_monitoring'):
                virtual_result = await self.virtual_service.start_device_monitoring()
                self.logger.debug(f"Virtual service device monitoring: {'started' if virtual_result else 'failed'}")

            self.logger.info("Device monitoring started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start device monitoring: {e}")
            return False

    def add_device_notification_callback(self, callback):
        """
        Add device notification callback to both services.

        Args:
            callback: Callback function for device notifications
        """
        try:
            self.logger.debug("Adding device notification callback")

            # Add callback to monitor service
            if self.monitor_service and hasattr(self.monitor_service, 'add_device_notification_callback'):
                self.monitor_service.add_device_notification_callback(callback)
                self.logger.debug("Callback added to monitor service")

            # Add callback to virtual service
            if self.virtual_service and hasattr(self.virtual_service, 'add_device_notification_callback'):
                self.virtual_service.add_device_notification_callback(callback)
                self.logger.debug("Callback added to virtual service")

        except Exception as e:
            self.logger.error(f"Failed to add device notification callback: {e}")

    # =========================================================================
    # Story 2.1: Playback Complete Callbacks
    # =========================================================================

    def set_playback_complete_callback(self, callback: Callable[[str], None]) -> None:
        """
        Set callback for playback completion notification.

        The callback is propagated to both MonitorAudioService and
        VirtualMicrophoneService so that any playback completion emits the signal.

        Args:
            callback: Function that receives the completed task_id
        """
        self._playback_complete_callback = callback

        # Propagate to services
        if self.monitor_service:
            self.monitor_service.set_playback_complete_callback(callback)
        if self.virtual_service:
            self.virtual_service.set_playback_complete_callback(callback)

    # =========================================================================
    # Story 2.1: Streaming Chunk Playback (FR24 - stream without waiting)
    # =========================================================================

    async def start_streaming_session(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        sample_width: int = 2,
    ) -> Dict[str, Optional[str]]:
        """
        Start streaming sessions on both audio services for immediate chunk playback.

        This enables low-latency dual-stream playback where TTS chunks can be
        played as they are generated without waiting for complete audio (NFR3).

        Args:
            sample_rate: Audio sample rate (default 24000 for Qwen3-TTS)
            channels: Number of audio channels
            sample_width: Bytes per sample (2 for 16-bit)

        Returns:
            Dict with session IDs: {"monitor": id_or_none, "virtual": id_or_none}
        """
        result = {"monitor": None, "virtual": None}

        if not self._is_initialized:
            self.logger.error("AudioCoordinator not initialized")
            return result

        try:
            # Start streaming on monitor service
            if self.monitor_service and await self._is_monitor_service_healthy():
                monitor_session = await self.monitor_service.start_streaming_session(
                    sample_rate=sample_rate,
                    channels=channels,
                    sample_width=sample_width,
                )
                result["monitor"] = monitor_session

            # Start streaming on virtual service (if available)
            if self.virtual_service and await self._is_virtual_service_healthy():
                if hasattr(self.virtual_service, 'start_streaming_session'):
                    virtual_session = await self.virtual_service.start_streaming_session(
                        sample_rate=sample_rate,
                        channels=channels,
                        sample_width=sample_width,
                    )
                    result["virtual"] = virtual_session

            self.logger.info(f"Streaming sessions started: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to start streaming sessions: {e}")
            return result

    async def play_audio_chunk(
        self,
        audio_data: bytes,
        is_final: bool = False,
    ) -> Dict[str, bool]:
        """
        Play an audio chunk immediately to both streaming sessions.

        Chunks are played without waiting for complete audio, enabling
        low-latency streaming playback (NFR3: no stuttering or gaps).

        Args:
            audio_data: Raw audio bytes to play
            is_final: If True, this is the last chunk

        Returns:
            Dict with success status: {"monitor": bool, "virtual": bool}
        """
        result = {"monitor": False, "virtual": False}

        try:
            # Play on monitor service
            if self.monitor_service and self.monitor_service.is_streaming_active():
                result["monitor"] = await self.monitor_service.play_audio_chunk(
                    audio_data, is_final
                )

            # Play on virtual service (if available)
            if self.virtual_service:
                if hasattr(self.virtual_service, 'is_streaming_active') and \
                   self.virtual_service.is_streaming_active():
                    result["virtual"] = await self.virtual_service.play_audio_chunk(
                        audio_data, is_final
                    )

            return result

        except Exception as e:
            self.logger.error(f"Failed to play audio chunk: {e}")
            return result

    async def stop_streaming_session(self) -> Dict[str, bool]:
        """
        Stop streaming sessions on both services.

        Returns:
            Dict with stop status: {"monitor": bool, "virtual": bool}
        """
        result = {"monitor": False, "virtual": False}

        try:
            # Stop monitor streaming
            if self.monitor_service:
                result["monitor"] = await self.monitor_service.stop_streaming_session()

            # Stop virtual streaming (if available)
            if self.virtual_service and hasattr(self.virtual_service, 'stop_streaming_session'):
                result["virtual"] = await self.virtual_service.stop_streaming_session()

            self.logger.info(f"Streaming sessions stopped: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error stopping streaming sessions: {e}")
            return result

    def is_streaming_active(self) -> Dict[str, bool]:
        """
        Check if streaming sessions are active on each service.

        Returns:
            Dict with active status: {"monitor": bool, "virtual": bool}
        """
        result = {"monitor": False, "virtual": False}

        if self.monitor_service:
            result["monitor"] = self.monitor_service.is_streaming_active()

        if self.virtual_service and hasattr(self.virtual_service, 'is_streaming_active'):
            result["virtual"] = self.virtual_service.is_streaming_active()

        return result

    # =========================================================================
    # Story 2.5: Device Resilience - Recovery and Notification Handlers
    # =========================================================================

    def _handle_device_recovery(self, role: DeviceRole, device: AudioDevice) -> None:
        """
        Handle device recovery when a device reconnects or falls back.

        Story 2.5 AC: Device reconnect → auto-recovery, no manual reconfiguration

        Args:
            role: Role of the device (MONITOR or VIRTUAL_MIC)
            device: The recovered or fallback device
        """
        self.logger.info(f"Handling device recovery for {role.value}: {device.name}")

        try:
            if role == DeviceRole.MONITOR and self.monitor_service:
                # Update monitor service's current device
                self.monitor_service.current_monitor_device = device
                self.logger.info(f"Monitor device updated to: {device.name}")

            elif role == DeviceRole.VIRTUAL_MIC and self.virtual_service:
                # Update virtual service's current device
                self.virtual_service.current_virtual_device = device
                self.logger.info(f"Virtual mic device updated to: {device.name}")

        except Exception as e:
            self.logger.error(f"Error handling device recovery for {role.value}: {e}")

    def _handle_device_notification(self, notification: DeviceNotification) -> None:
        """
        Handle device notification and forward to registered callbacks.

        Story 2.5 AC: Warning on disconnect, notification on reconnect

        Args:
            notification: The device notification to handle
        """
        self.logger.info(f"Device notification: {notification.title} - {notification.message}")

        # Forward to all registered callbacks
        for callback in self._device_notification_callbacks:
            try:
                callback(notification)
            except Exception as e:
                self.logger.error(f"Error in device notification callback: {e}")

    def add_device_notification_callback(
        self,
        callback: Callable[[DeviceNotification], None]
    ) -> None:
        """
        Add callback for device notifications.

        Story 2.5: These callbacks receive notifications about device
        disconnect, reconnect, and fallback events for UI display.

        Args:
            callback: Function to call when device notifications occur
        """
        if callback not in self._device_notification_callbacks:
            self._device_notification_callbacks.append(callback)
            self.logger.debug("Added device notification callback")

    def remove_device_notification_callback(
        self,
        callback: Callable[[DeviceNotification], None]
    ) -> None:
        """Remove a device notification callback."""
        if callback in self._device_notification_callbacks:
            self._device_notification_callbacks.remove(callback)
            self.logger.debug("Removed device notification callback")

    def refresh_audio_devices(self) -> bool:
        """
        Refresh the audio device list.

        Story 2.5 AC: New devices detected when settings opened

        Returns:
            bool: True if refresh was successful
        """
        if not self.resilience_manager:
            self.logger.warning("Cannot refresh devices: resilience manager not available")
            return False

        return self.resilience_manager.refresh_devices()

    def register_monitor_device(
        self,
        device: AudioDevice,
        fallback_device: Optional[AudioDevice] = None
    ) -> None:
        """
        Register or update the monitor device for resilience monitoring.

        Args:
            device: Monitor device to track
            fallback_device: Optional fallback device
        """
        if self.resilience_manager:
            self.resilience_manager.register_device(
                DeviceRole.MONITOR,
                device,
                fallback_device
            )
        if self.monitor_service:
            self.monitor_service.current_monitor_device = device

    def register_virtual_mic_device(
        self,
        device: AudioDevice,
        fallback_device: Optional[AudioDevice] = None
    ) -> None:
        """
        Register or update the virtual mic device for resilience monitoring.

        Args:
            device: Virtual mic device to track
            fallback_device: Optional fallback device
        """
        if self.resilience_manager:
            self.resilience_manager.register_device(
                DeviceRole.VIRTUAL_MIC,
                device,
                fallback_device
            )
        if self.virtual_service:
            self.virtual_service.current_virtual_device = device

    def get_device_resilience_status(self) -> Dict[str, Any]:
        """
        Get the current device resilience status.

        Returns:
            Dict with device status information
        """
        if not self.resilience_manager:
            return {"enabled": False, "reason": "Resilience manager not initialized"}

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't await in sync context, return basic status
                return {
                    "enabled": True,
                    "monitoring_active": self.resilience_manager._is_monitoring,
                }
            else:
                return loop.run_until_complete(self.resilience_manager.get_health())
        except Exception as e:
            return {"enabled": True, "error": str(e)}

    # Required BaseService attribute
    _is_initialized = False