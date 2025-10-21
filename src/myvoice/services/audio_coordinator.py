"""
Audio Coordinator Service

This module provides the AudioCoordinator that manages both MonitorAudioService
and VirtualMicrophoneService for coordinated dual-stream audio routing without
resource conflicts.

This replaces the problematic play_synchronized_dual_stream approach with proper
service separation as specified in the dual-service architecture design.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from myvoice.models.audio_device import AudioDevice
from myvoice.models.error import MyVoiceError, ErrorSeverity
from myvoice.models.app_settings import AppSettings
from myvoice.services.monitor_audio_service import MonitorAudioService, MonitorPlaybackTask
from myvoice.services.virtual_microphone_service import VirtualMicrophoneService, VirtualPlaybackTask
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

        # Coordination tracking
        self._active_coordinations: Dict[str, DualStreamResult] = {}
        self._coordination_counter = 0

        # Service health tracking
        self._last_health_check: Optional[datetime] = None
        self._services_healthy = True

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

            # Initialize MonitorAudioService (Audio Service 1)
            self.monitor_service = MonitorAudioService(self.app_settings)
            monitor_success = await self.monitor_service.initialize()

            if monitor_success:
                self.logger.info("MonitorAudioService initialized successfully")
            else:
                self.logger.warning("⚠ MonitorAudioService initialization failed")

            # Initialize VirtualMicrophoneService (Audio Service 2)
            self.virtual_service = VirtualMicrophoneService(self.app_settings)
            virtual_success = await self.virtual_service.initialize()

            if virtual_success:
                self.logger.info("VirtualMicrophoneService initialized successfully")
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
                self.status = ServiceStatus.FAILED
                return False

        except Exception as e:
            self.logger.error(f"Failed to initialize AudioCoordinator: {e}")
            self.status = ServiceStatus.FAILED
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

    # Required BaseService attribute
    _is_initialized = False