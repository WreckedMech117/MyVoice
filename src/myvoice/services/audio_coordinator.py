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

Microphone Mixing Integration:
- MicrophoneCaptureService for real-time mic input capture
- AudioMixerService for mixing mic + TTS audio streams
- Mixed audio routing to virtual microphone for Discord/Zoom/Teams
- Volume controls and mute functionality for mic input
"""

import asyncio
import logging
import threading
import time
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime

if TYPE_CHECKING:
    # Story 12.1: registry import is type-only to avoid forcing PyQt6 onto
    # this module's import chain (the audio coordinator runs in async
    # contexts that don't otherwise require Qt).
    from myvoice.services.sessions.session_registry import SessionRegistry

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
from myvoice.services.microphone_capture_service import MicrophoneCaptureService, MicrophoneCaptureConfig
from myvoice.services.audio_mixer_service import AudioMixerService, MixerConfig
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


class MicrophoneMixingMode(Enum):
    """Modes for audio mixing."""
    TTS_ONLY = "tts_only"          # Only TTS audio (default, backward compatible)
    MIC_ONLY = "mic_only"          # Only microphone audio
    MIXED = "mixed"                 # Mix TTS + microphone together
    MIC_WHEN_SILENT = "mic_when_silent"  # Mic only when TTS is silent


@dataclass
class MicrophoneMixingConfig:
    """Configuration for microphone mixing behavior."""
    # Enable/disable mic mixing
    enabled: bool = False

    # Mixing mode
    mode: MicrophoneMixingMode = field(default=MicrophoneMixingMode.MIXED)

    # Volume settings (0.0 to 1.0)
    mic_volume: float = 1.0
    tts_volume: float = 1.0

    # Ducking: reduce mic volume when TTS plays (0.0 = full ducking, 1.0 = no ducking)
    enable_ducking: bool = False
    duck_amount: float = 0.3  # Reduce mic to 30% during TTS

    # Sample rate for mixed output (should match virtual mic)
    output_sample_rate: int = 48000


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

    # Microphone mixing settings
    mic_mixing: MicrophoneMixingConfig = field(default_factory=MicrophoneMixingConfig)


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

    def __init__(
        self,
        app_settings: Optional[AppSettings] = None,
        *,
        session_registry: Optional["SessionRegistry"] = None,
    ):
        """
        Initialize the Audio Coordinator.

        Args:
            app_settings: Application settings for device preferences
            session_registry: Optional SessionRegistry for Story 12.1 playback-
                side lifecycle wiring. When provided, ``play_dual_stream``
                posts ``mark_playing`` / ``mark_audible`` mutations through
                the registry so the indicator's substate (driven by
                ``MainWindow._redraw_tts_indicator_from_focal``) reflects
                actual audio dispatch. When None, the coordinator runs the
                legacy code path unchanged.
        """
        super().__init__("AudioCoordinator")

        self.app_settings = app_settings
        self.config = AudioCoordinatorConfig()
        self.logger = logging.getLogger(__name__)

        # Story 12.1: SessionRegistry wiring (D-20 Phase 2). Mirrors the
        # injection pattern Story 11.4 used for QwenTTSService.
        self._session_registry = session_registry
        if self._session_registry is not None:
            self.logger.info("AudioCoordinator using SessionRegistry")

        # Service instances
        self.monitor_service: Optional[MonitorAudioService] = None
        self.virtual_service: Optional[VirtualMicrophoneService] = None

        # Microphone mixing services
        self.mic_capture_service: Optional[MicrophoneCaptureService] = None
        self.mixer_service: Optional[AudioMixerService] = None

        # Story 2.5: Device Resilience Manager for graceful disconnect/reconnect
        self.resilience_manager: Optional[DeviceResilienceManager] = None

        # Coordination tracking
        self._active_coordinations: Dict[str, DualStreamResult] = {}
        self._coordination_counter = 0
        # Story 16.5: per-session reverse map populated at play_dual_stream
        # entry and cleaned on completion (success or failure). Lets the
        # registry's cancel-chain hook target a specific session's playback
        # via cancel_playback(session_id) instead of the global
        # stop_all_playback() fan-out.
        self._session_id_to_coordination_id: Dict[str, str] = {}

        # Service health tracking
        self._last_health_check: Optional[datetime] = None
        self._services_healthy = True

        # Callbacks (Story 2.1: playback_complete signal)
        self._playback_complete_callback: Optional[Callable[[str], None]] = None

        # Story 2.5: Device notification callbacks
        self._device_notification_callbacks: List[Callable[[DeviceNotification], None]] = []

        # Microphone mixing state
        self._mic_mixing_enabled = False
        self._tts_sample_rate: Optional[int] = None  # Tracked during streaming

        # Story 17.3 finalization-drain follow-up — track total bytes written
        # and the first-chunk-write timestamp so stop_streaming_session can
        # wait for the PyAudio output buffer to drain before tearing down the
        # underlying stream. Without this, an `is_final` chunk that arrives
        # while the buffer still has un-played audio causes the tail of the
        # last chunk to be cut off (Story 18.3 surfaced this — bf16 + TF32 +
        # cuDNN engagements made the producer fast enough that the consumer
        # was still draining when finalization fired).
        #
        # Story 18.3 M6 follow-up — also track the LAST-chunk write timestamp
        # and bytes. The original math (expected_total - elapsed) goes
        # NEGATIVE on producer-bottleneck workloads (producer 1.62× realtime
        # → elapsed 1.62× expected; remaining always negative; drain skipped).
        # The cut-off-at-end Commander surfaced in the Story 18.3 Task 10
        # bundled smoke is the symptom: PyAudio's device-level buffer (200–
        # 500ms on Windows shared mode) gets truncated by stop_stream() when
        # the math says "no drain needed." The corrected math computes drain
        # from the LAST chunk's playback (last_chunk_duration -
        # time_since_last_write) + safety, so the buffer always drains.
        self._stream_first_write_ts: Optional[float] = None
        self._stream_total_bytes: int = 0
        self._stream_last_write_ts: Optional[float] = None
        self._stream_last_chunk_bytes: int = 0
        self._stream_sample_width: int = 2  # int16 (set authoritatively in start_streaming_session)
        self._stream_channels: int = 1      # mono (set authoritatively in start_streaming_session)

        # Mic monitor state (for "Monitor Mic to Speakers" feature)
        self._mic_monitor_running = False
        self._mic_monitor_task: Optional[asyncio.Task] = None

        # Continuous passthrough state (uses threading to avoid qasync conflicts)
        self._continuous_passthrough_running = False
        self._passthrough_thread = None  # threading.Thread

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
                self.resilience_manager.add_device_change_callback(self._handle_device_change)
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

                # Set current_virtual_device from app_settings if configured
                virtual_device_id = getattr(self.app_settings, 'virtual_microphone_device_id', None) if self.app_settings else None
                if virtual_device_id:
                    try:
                        virtual_devices = await self.virtual_service.enumerate_virtual_devices()
                        for device in virtual_devices:
                            if device.device_id == virtual_device_id:
                                self.virtual_service.current_virtual_device = device
                                self.logger.info(f"Set initial current_virtual_device to: {device.name}")
                                break
                        else:
                            self.logger.warning(f"Configured virtual device {virtual_device_id} not found during initialization")
                    except Exception as e:
                        self.logger.warning(f"Error setting initial virtual device: {e}")

                # Story 2.5: Register virtual mic device with resilience manager
                virtual_device = getattr(self.virtual_service, 'current_virtual_device', None)
                if self.resilience_manager and virtual_device:
                    self.resilience_manager.register_device(
                        DeviceRole.VIRTUAL_MIC,
                        virtual_device
                    )
            else:
                self.logger.warning("⚠ VirtualMicrophoneService initialization failed")

            # Initialize MicrophoneCaptureService for mic input
            mic_config = MicrophoneCaptureConfig(
                sample_rate=self.config.mic_mixing.output_sample_rate,
                channels=1,
                chunk_size=1024,
            )
            self.mic_capture_service = MicrophoneCaptureService(
                app_settings=self.app_settings,
                config=mic_config
            )
            mic_success = await self.mic_capture_service.start()

            if mic_success:
                self.logger.info("MicrophoneCaptureService initialized successfully")
            else:
                self.logger.warning("⚠ MicrophoneCaptureService initialization failed - mic mixing disabled")

            # Initialize AudioMixerService
            mixer_config = MixerConfig(
                default_mic_volume=self.config.mic_mixing.mic_volume,
                default_tts_volume=self.config.mic_mixing.tts_volume,
                enable_limiter=True,
            )
            self.mixer_service = AudioMixerService(config=mixer_config)
            self.logger.info("AudioMixerService initialized")

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

            # Stop mic-related threads FIRST (they read from mic_capture_service)
            # These have their own PyAudio instances that must be terminated
            try:
                if self._continuous_passthrough_running:
                    self.logger.info("Stopping continuous mic passthrough thread...")
                    await self.stop_continuous_mic_passthrough()
            except Exception as e:
                self.logger.warning(f"Error stopping mic passthrough: {e}")
                # Force the flag anyway to signal thread to exit
                self._continuous_passthrough_running = False

            try:
                if self._mic_monitor_running:
                    self.logger.info("Stopping mic monitor to speakers...")
                    await self.stop_mic_monitor_to_speakers()
            except Exception as e:
                self.logger.warning(f"Error stopping mic monitor: {e}")
                self._mic_monitor_running = False

            # Stop all active coordinations
            for coordination_id in list(self._active_coordinations.keys()):
                await self.stop_coordination(coordination_id)

            # Shutdown all services in parallel
            shutdown_tasks = []
            service_names = []

            if self.monitor_service:
                shutdown_tasks.append(self.monitor_service.shutdown())
                service_names.append("Monitor")

            if self.virtual_service:
                shutdown_tasks.append(self.virtual_service.shutdown())
                service_names.append("Virtual")

            if self.mic_capture_service:
                shutdown_tasks.append(self.mic_capture_service.stop())
                service_names.append("MicCapture")

            if shutdown_tasks:
                results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)

                # Log results
                for i, result in enumerate(results):
                    service_name = service_names[i]
                    if isinstance(result, Exception):
                        self.logger.warning(f"{service_name} service shutdown error: {result}")
                    elif result:
                        self.logger.info(f"{service_name} service shutdown successful")
                    else:
                        self.logger.warning(f"⚠ {service_name} service shutdown failed")

            # Clear mixer service
            self.mixer_service = None

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
            # Ensure flags are cleared even on error
            self._continuous_passthrough_running = False
            self._mic_monitor_running = False
            return False

    async def play_dual_stream(self,
                             audio_data: bytes,
                             monitor_device: Optional[AudioDevice] = None,
                             virtual_device: Optional[AudioDevice] = None,
                             volume: float = 1.0,
                             *,
                             session_id: Optional[str] = None) -> DualStreamResult:
        """
        Execute coordinated dual-stream playback to both services.

        Args:
            audio_data: Audio data to play
            monitor_device: Target monitor device (uses service default if None)
            virtual_device: Target virtual device (uses service default if None)
            volume: Volume level (0.0 to 1.0)
            session_id: Story 12.1 — when provided alongside a registry
                injected at construction, the coordinator posts
                ``mark_playing`` before dispatch and ``mark_audible`` once
                the streams are running. When None, no registry mutations
                are posted (legacy callers continue unchanged per D-14).

        Returns:
            DualStreamResult: Result of coordinated playback
        """
        # AI-Review H2 (2026-05-04): mark_playing was previously posted
        # unconditionally at the top of this method, which leaked sessions
        # in PLAYING state on every failure branch (not-initialized,
        # no-healthy-services, exception during dispatch) because no
        # terminal mutation followed. The post is now deferred until we
        # have actually committed to dispatching, and every failure path
        # cleans up via set_error+discard if mark_playing was already
        # queued.
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

        # Story 16.5: register session_id → coordination_id so the registry's
        # cancel-chain hook can target this playback via cancel_playback(sid).
        # The map is cleaned in every completion path below.
        if session_id is not None:
            self._session_id_to_coordination_id[session_id] = coordination_id

        mark_playing_posted = False
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
                # Story 16.5: clean the session→coordination map on the
                # no-healthy-services bail-out so the registry's stale-cancel
                # path returns False instead of fanning out to a dead session.
                if session_id is not None:
                    self._session_id_to_coordination_id.pop(session_id, None)
                return DualStreamResult(
                    monitor_task=None,
                    virtual_task=None,
                    coordination_id=coordination_id,
                    start_time=datetime.now(),
                    success=False,
                    error_message="No healthy services available"
                )

            # Story 12.1: post mark_playing now that we are committed to
            # dispatching. Posted via post_mutation so it crosses the
            # worker→Qt-main boundary safely (P-3). The session is in
            # READY_TO_PLAY when this method is called (qwen_tts_service
            # finalized it), and mark_playing transitions it to PLAYING —
            # required before mark_audible (which checks the state) can be
            # posted below.
            if session_id is not None and self._session_registry is not None:
                self._session_registry.post_mutation('mark_playing', session_id)
                mark_playing_posted = True

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

            # Story 12.1 / AI-Review H2: pair mark_playing with either
            # mark_audible (success) or set_error+discard (failure). Gating
            # mark_audible on `any_successful` rather than `success` is
            # tighter — a task object that came back in 'failed'/'pending'
            # status did not actually start streaming, and posting
            # mark_audible there would leave a session stuck PLAYING+audible
            # with no real playback to ever fire mark_done.
            if mark_playing_posted:
                if dual_result.any_successful:
                    self._session_registry.post_mutation('mark_audible', session_id)
                else:
                    self.logger.warning(
                        f"Coordinated playback {coordination_id} produced "
                        "no audible streams; closing registry session via "
                        "set_error+discard."
                    )
                    self._session_registry.post_mutation('set_error', session_id)
                    self._session_registry.post_mutation('discard', session_id)

            success_msg = []
            if monitor_task:
                success_msg.append("monitor")
            if virtual_task:
                success_msg.append("virtual")

            self.logger.info(f"Coordinated playback {coordination_id} started: {', '.join(success_msg)}")
            # Story 16.5: do NOT clear the session→coordination map on the
            # success path. The MonitorPlaybackTask + VirtualPlaybackTask
            # returned here continue playback in the background after this
            # method returns; the map must remain populated so a mid-
            # playback cancel_playback(session_id) can fan out to the active
            # monitor + virtual services. The map entry is cleared by
            # cancel_playback (the cancel path) per AC #4.
            return dual_result

        except Exception as e:
            self.logger.error(f"Coordinated dual-stream playback failed: {e}")
            # AI-Review H2: clean up the registry session if mark_playing
            # was queued before the exception fired. Without this, the
            # session leaks in PLAYING state forever.
            if mark_playing_posted:
                self._session_registry.post_mutation('set_error', session_id)
                self._session_registry.post_mutation('discard', session_id)
            # Story 16.5: failure-path cleanup — keep the map free of dead
            # entries so cancel_playback returns False cleanly post-failure.
            if session_id is not None:
                self._session_id_to_coordination_id.pop(session_id, None)
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

    async def stop_all_playback(self) -> int:
        """
        Stop every active playback task across both monitor and virtual
        microphone outputs.

        Story 11.4 follow-up: invoked by the main window's Stop button
        (the dual-mode Clear/Stop control) when audio is currently
        playing. Both underlying ``AudioService`` instances expose their
        own ``stop_all_playback`` / ``stop_all_virtual_microphone_playback``
        helpers; this coordinator method just fans out to them and sums
        the count.

        Returns:
            int: Total number of playback tasks stopped (0 if nothing
                 was active — safe to call as a no-op).
        """
        total = 0
        try:
            if self.monitor_service is not None:
                try:
                    total += await self.monitor_service.stop_all_playback()
                except Exception as exc:
                    self.logger.error(
                        f"Error stopping monitor playback: {exc}"
                    )
            if self.virtual_service is not None:
                try:
                    total += await self.virtual_service.stop_all_virtual_microphone_playback()
                except Exception as exc:
                    self.logger.error(
                        f"Error stopping virtual playback: {exc}"
                    )
            if total:
                self.logger.info(
                    f"stop_all_playback: stopped {total} task(s)"
                )
            return total
        except Exception as exc:
            self.logger.error(f"stop_all_playback failed: {exc}")
            return total

    async def cancel_playback(self, session_id: str) -> bool:
        """Story 16.5 — stop monitor + virtual-mic playback for a specific
        session. Per validation gap #3 step (i): the registry calls this
        from request_cancel's hook to stop active playback gracefully when
        a streaming session is cancelled mid-playback.

        For v1 (P-8 session-level serialization — only one session is
        audible at a time), this is approximately stop_all_playback()
        gated on the session_id being in the active map. A stale cancel
        for an already-completed session is a quiet False return.

        Returns:
            bool: True if a stop was actually attempted; False if the
            session was unknown / already finished.
        """
        coordination_id = self._session_id_to_coordination_id.pop(session_id, None)
        if coordination_id is None:
            return False
        attempted = False
        if self.monitor_service is not None:
            try:
                await self.monitor_service.stop_all_playback()
                attempted = True
            except Exception as exc:
                self.logger.error(
                    f"cancel_playback({session_id}): monitor stop failed: {exc}"
                )
        if self.virtual_service is not None:
            try:
                await self.virtual_service.stop_all_virtual_microphone_playback()
                attempted = True
            except Exception as exc:
                self.logger.error(
                    f"cancel_playback({session_id}): virtual stop failed: {exc}"
                )
        return attempted

    async def enumerate_all_devices(self) -> Dict[str, List[AudioDevice]]:
        """Enumerate devices from all services."""
        devices = {
            "monitor": [],
            "virtual": [],
            "mic": []
        }

        try:
            if self.monitor_service:
                devices["monitor"] = await self.monitor_service.enumerate_monitor_devices()

            if self.virtual_service:
                devices["virtual"] = await self.virtual_service.enumerate_virtual_devices()

            if self.mic_capture_service:
                devices["mic"] = await self.mic_capture_service.enumerate_input_devices()

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

            # Update virtual service settings and current_virtual_device
            if self.virtual_service:
                self.virtual_service.app_settings = new_settings

                # Update current_virtual_device if a device is configured
                virtual_device_id = getattr(new_settings, 'virtual_microphone_device_id', None)
                if virtual_device_id:
                    # Find and set the virtual device
                    try:
                        virtual_devices = await self.virtual_service.enumerate_virtual_devices()
                        for device in virtual_devices:
                            if device.device_id == virtual_device_id:
                                self.virtual_service.current_virtual_device = device
                                self.logger.info(f"Set current_virtual_device to: {device.name}")
                                break
                        else:
                            self.logger.warning(f"Virtual device {virtual_device_id} not found in enumeration")
                    except Exception as e:
                        self.logger.warning(f"Error setting virtual device: {e}")
                else:
                    self.virtual_service.current_virtual_device = None
                    self.logger.debug("No virtual device configured")

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
            # Track TTS sample rate for mic resampling
            self._tts_sample_rate = sample_rate

            # Reset drain-tracking counters for the new session.
            self._stream_first_write_ts = None
            self._stream_total_bytes = 0
            self._stream_last_write_ts = None
            self._stream_last_chunk_bytes = 0
            self._stream_sample_width = sample_width
            self._stream_channels = channels

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

        When mic mixing is enabled, microphone audio is mixed with TTS audio
        before being sent to the virtual microphone service.

        Args:
            audio_data: Raw audio bytes to play
            is_final: If True, this is the last chunk

        Returns:
            Dict with success status: {"monitor": bool, "virtual": bool, "mic_mixed": bool}
        """
        result = {"monitor": False, "virtual": False, "mic_mixed": False}

        try:
            # Story 17.3 finalization-drain follow-up — record the first-write
            # timestamp + accumulate total bytes streamed so stop_streaming_session
            # can wait for the PyAudio output buffer to drain on `is_final`.
            # Latency-sensitive: do this BEFORE the dispatch so the timestamp
            # records the moment the producer started filling the output buffer,
            # not the moment after the (blocking) write completed.
            if audio_data:
                now_ts = time.monotonic()
                if self._stream_first_write_ts is None:
                    self._stream_first_write_ts = now_ts
                # Story 18.3 M6 — also stamp the LAST-write timestamp + last
                # chunk bytes so the drain math can compute when THIS chunk
                # finishes playing (the producer-bottleneck case where
                # elapsed >> expected_total leaves the last-chunk audio still
                # in PyAudio's buffer when stop is called).
                self._stream_last_write_ts = now_ts
                self._stream_last_chunk_bytes = len(audio_data)
            self._stream_total_bytes += len(audio_data)

            # Play TTS on monitor service (unaffected by mic mixing)
            if self.monitor_service and self.monitor_service.is_streaming_active():
                result["monitor"] = await self.monitor_service.play_audio_chunk(
                    audio_data, is_final
                )

            # Handle virtual mic output (possibly with mic mixing)
            if self.virtual_service:
                if hasattr(self.virtual_service, 'is_streaming_active') and \
                   self.virtual_service.is_streaming_active():

                    # Determine what audio to send to virtual mic
                    self.logger.debug(f"[MIC_MIX] Virtual mic active, _mic_mixing_enabled={self._mic_mixing_enabled}")
                    virtual_audio = await self._get_virtual_mic_audio(audio_data)
                    if virtual_audio != audio_data:
                        result["mic_mixed"] = True
                        self.logger.debug(f"[MIC_MIX] Audio was mixed! Original: {len(audio_data)}, Mixed: {len(virtual_audio)}")

                    result["virtual"] = await self.virtual_service.play_audio_chunk(
                        virtual_audio, is_final
                    )

            return result

        except Exception as e:
            self.logger.error(f"Failed to play audio chunk: {e}")
            return result

    async def _get_virtual_mic_audio(self, tts_audio: bytes) -> bytes:
        """
        Get audio for virtual microphone, optionally mixing with mic input.

        Args:
            tts_audio: TTS audio bytes

        Returns:
            bytes: Audio to send to virtual mic (TTS-only, mixed, or mic-only)
        """
        # If mic mixing is disabled, just return TTS audio
        if not self._mic_mixing_enabled:
            return tts_audio
        if not self.mic_capture_service:
            self.logger.debug("Mic mixing: capture service unavailable")
            return tts_audio
        if not self.mixer_service:
            self.logger.debug("Mic mixing: mixer service unavailable")
            return tts_audio
        if not self.mic_capture_service.is_capturing:
            self.logger.debug("Mic mixing: not currently capturing")
            return tts_audio

        mode = self.config.mic_mixing.mode
        self.logger.debug(f"[MIC_MIX] Mixing mode: {mode}, getting mic chunk...")

        # TTS-only mode - just return TTS audio
        if mode == MicrophoneMixingMode.TTS_ONLY:
            return tts_audio

        # Get microphone audio chunk
        mic_audio = self.mic_capture_service.get_audio_chunk()
        if mic_audio:
            self.logger.debug(f"[MIC_MIX] Got mic chunk: {len(mic_audio)} bytes")
        else:
            self.logger.debug("[MIC_MIX] No mic audio available")

        # MIC_ONLY mode - return mic audio or silence
        if mode == MicrophoneMixingMode.MIC_ONLY:
            if mic_audio:
                return self._resample_mic_if_needed(mic_audio)
            else:
                # Return silence matching TTS length
                return bytes(len(tts_audio))

        # MIC_WHEN_SILENT mode - use mic only when no TTS
        if mode == MicrophoneMixingMode.MIC_WHEN_SILENT:
            if self._is_silence(tts_audio) and mic_audio:
                return self._resample_mic_if_needed(mic_audio)
            return tts_audio

        # MIXED mode - mix TTS + mic
        if mode == MicrophoneMixingMode.MIXED:
            if not mic_audio:
                return tts_audio

            try:
                # Resample mic if needed
                mic_audio = self._resample_mic_if_needed(mic_audio)

                # Mix using configured volumes
                if self.config.mic_mixing.enable_ducking:
                    mixed = self.mixer_service.mix_with_ducking(
                        mic_audio=mic_audio,
                        tts_audio=tts_audio,
                        duck_amount=self.config.mic_mixing.duck_amount,
                        mic_volume=self.config.mic_mixing.mic_volume,
                        tts_volume=self.config.mic_mixing.tts_volume,
                    )
                else:
                    mixed = self.mixer_service.mix_streams_bytes(
                        mic_audio=mic_audio,
                        tts_audio=tts_audio,
                        mic_volume=self.config.mic_mixing.mic_volume,
                        tts_volume=self.config.mic_mixing.tts_volume,
                    )

                return mixed

            except Exception as e:
                self.logger.warning(f"Mic mixing failed, using TTS only: {e}")
                return tts_audio

        return tts_audio

    def _resample_mic_if_needed(self, mic_audio: bytes) -> bytes:
        """Resample microphone audio if sample rates differ."""
        if not self.mic_capture_service or not self.mixer_service:
            return mic_audio

        mic_rate = self.mic_capture_service.config.sample_rate
        tts_rate = self._tts_sample_rate or 24000  # Default TTS rate

        if mic_rate != tts_rate:
            return self.mixer_service.resample_bytes(mic_audio, mic_rate, tts_rate)

        return mic_audio

    def _is_silence(self, audio_data: bytes, threshold: float = 0.01) -> bool:
        """Check if audio data is essentially silence."""
        import numpy as np
        try:
            samples = np.frombuffer(audio_data, dtype=np.int16)
            rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2)) / 32768.0
            return rms < threshold
        except Exception:
            return False

    # Story 17.3 finalization-drain follow-up — drain-wait constants.
    # Safety buffer covers PyAudio's internal output latency. Windows driver
    # latency varies widely:
    #   * WASAPI exclusive: ~10ms
    #   * WASAPI shared:    ~80-150ms
    #   * WDM-KS:           ~50ms
    #   * DirectSound:      200-500ms
    # 500ms covers all common Windows backends without being absurdly long.
    # Bumped from 150ms (Story 17.3 follow-up landed) to 500ms (Story 18.3
    # M2 follow-up — Commander reported hit-or-miss cut-off-at-end during
    # the NFR1 measurement runs; 150ms was borderline for WASAPI shared and
    # insufficient for DirectSound). Max cap is a hard timeout so a math
    # drift cannot hang the close path indefinitely — 15s is comfortably
    # longer than any single TTS utterance.
    _DRAIN_SAFETY_BUFFER_S = 0.5
    _MAX_DRAIN_WAIT_S = 15.0

    async def stop_streaming_session(
        self,
        *,
        wait_for_drain: bool = False,
    ) -> Dict[str, bool]:
        """
        Stop streaming sessions on both services.

        Args:
            wait_for_drain: When True, wait for the PyAudio output buffer to
                drain before tearing down the underlying stream. Use this on
                the ``is_final`` finalization path so the tail of the last
                chunk plays out cleanly. Cancel / restart paths must keep the
                default False so user-cancel stays prompt — the legacy
                immediate-teardown behavior. Story 18.3 M6 — drain wait is
                computed from the LAST chunk's playback
                (``last_chunk_duration - time_since_last_write``), NOT the
                full-stream-elapsed math the M2 follow-up used. On
                producer-bottleneck workloads (producer ≥ 1.5× realtime),
                the full-stream math goes negative because elapsed
                outpaces expected_total; the corrected math always waits
                for at least the LAST chunk's residual playback plus the
                safety buffer, regardless of producer speed. A safety
                buffer (``_DRAIN_SAFETY_BUFFER_S`` — currently 500ms; bumped
                from 150ms in the Story 18.3 M2 follow-up to cover Windows
                DirectSound's worst-case 200–500ms internal latency) is
                added for PyAudio internal latency, and the wait is capped
                at ``_MAX_DRAIN_WAIT_S`` (15s) so a math drift cannot hang
                the close path.

        Returns:
            Dict with stop status: {"monitor": bool, "virtual": bool}
        """
        result = {"monitor": False, "virtual": False}

        try:
            # Story 17.3 finalization-drain follow-up — wait for the output
            # buffer to drain before tearing down. The producer can outpace
            # PyAudio's playback (notably with bf16 + TF32 + cuDNN engaged
            # post Story 18.2 + 18.3); without this, the last chunk's tail is
            # cut off when the stream is closed mid-playback.
            #
            # Story 18.3 M6 — corrected drain math. The original formula
            # (expected_total - elapsed) goes negative on producer-bottleneck
            # workloads and skipped the drain entirely (Commander's bundled
            # smoke surfaced this — last ~500–800ms of audio truncated). The
            # corrected math computes drain from the LAST chunk's playback
            # state (last_chunk_duration - time_since_last_write), so the
            # PyAudio device-level buffer always gets at least the safety
            # buffer of grace on top of any unplayed audio in the last
            # chunk. The `if remaining > 0` gate is dropped — safety always
            # fires when wait_for_drain is True.
            #
            # Story 18.4 code-review pass follow-up — the last-chunk-only
            # math was correct under producer-SLOWER-than-real-time (Story
            # 18.3's case: the PyAudio buffer is approximately empty when
            # the last chunk arrives, because slow chunks let playback
            # catch up). But under producer-FASTER-than-real-time (Story
            # 18.4's compile-engaged path: torch.compile + CUDA Graph
            # replay produces chunks faster than playback consumes them),
            # multiple prior chunks queue in PyAudio's buffer when the
            # last chunk arrives. The last-chunk-only math underestimates
            # remaining audio by the entire queued depth. Observed in
            # Story 18.4 Task 8 first run (2026-05-11): 18.9 s of audio
            # arrived in 14 s; sessions stopped 566 ms after last chunk
            # while ~4.9 s of audio was still buffered → user heard the
            # audio cut mid-sentence. Fix: compute both estimates and
            # take the max so both producer regimes are covered.
            if (
                wait_for_drain
                and self._stream_last_write_ts is not None
                and self._tts_sample_rate
                and self._stream_last_chunk_bytes > 0
            ):
                bytes_per_second = (
                    self._tts_sample_rate
                    * max(self._stream_channels, 1)
                    * max(self._stream_sample_width, 1)
                )
                if bytes_per_second > 0:
                    last_chunk_duration_s = (
                        self._stream_last_chunk_bytes / bytes_per_second
                    )
                    time_since_last_write = (
                        time.monotonic() - self._stream_last_write_ts
                    )
                    last_chunk_remaining = max(
                        0.0, last_chunk_duration_s - time_since_last_write
                    )
                    # Story 18.4 follow-up: producer-faster regime check.
                    # ``total_audio_duration_s`` is the wall-clock duration
                    # of all audio bytes ever written to the stream;
                    # ``playback_elapsed_s`` is wall-clock since the first
                    # write (PyAudio plays at real-time after a small
                    # device-internal latency we ignore here — the safety
                    # buffer covers it). The difference is how much audio
                    # is still queued in PyAudio's buffer. Under producer-
                    # slower-than-real-time this goes to 0 (playback
                    # caught up); the max() with last_chunk_remaining
                    # preserves Story 18.3's M6 fix for that case.
                    if self._stream_first_write_ts is not None:
                        total_audio_duration_s = (
                            self._stream_total_bytes / bytes_per_second
                        )
                        playback_elapsed_s = (
                            time.monotonic() - self._stream_first_write_ts
                        )
                        total_queued_audio_s = max(
                            0.0, total_audio_duration_s - playback_elapsed_s
                        )
                    else:
                        total_queued_audio_s = 0.0
                    remaining_s = max(last_chunk_remaining, total_queued_audio_s)
                    drain_wait = min(
                        remaining_s + self._DRAIN_SAFETY_BUFFER_S,
                        self._MAX_DRAIN_WAIT_S,
                    )
                    self.logger.debug(
                        "Draining output buffer before close: "
                        "last_chunk_duration=%.3fs time_since_last_write=%.3fs "
                        "last_chunk_remaining=%.3fs total_queued=%.3fs waiting=%.3fs",
                        last_chunk_duration_s, time_since_last_write,
                        last_chunk_remaining, total_queued_audio_s, drain_wait,
                    )
                    await asyncio.sleep(drain_wait)

            # Reset drain-tracking counters and TTS sample rate.
            self._tts_sample_rate = None
            self._stream_first_write_ts = None
            self._stream_total_bytes = 0
            self._stream_last_write_ts = None
            self._stream_last_chunk_bytes = 0

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
            Dict with active status: {"monitor": bool, "virtual": bool, "mic": bool}
        """
        result = {"monitor": False, "virtual": False, "mic": False}

        if self.monitor_service:
            result["monitor"] = self.monitor_service.is_streaming_active()

        if self.virtual_service and hasattr(self.virtual_service, 'is_streaming_active'):
            result["virtual"] = self.virtual_service.is_streaming_active()

        if self.mic_capture_service:
            result["mic"] = self.mic_capture_service.is_capturing

        return result

    # =========================================================================
    # Story 2.5: Device Resilience - Recovery and Notification Handlers
    # =========================================================================

    def _handle_device_recovery(self, role: DeviceRole, device: AudioDevice) -> None:
        """
        Handle device recovery when a device reconnects or falls back.

        Story 2.5 AC: Device reconnect → auto-recovery, no manual reconfiguration

        Args:
            role: Role of the device (MONITOR, VIRTUAL_MIC, or MICROPHONE)
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

            elif role == DeviceRole.MICROPHONE and self.mic_capture_service:
                # Microphone reconnected - attempt to restart capture
                self.logger.info(f"Microphone reconnected: {device.name}")
                # Re-enable mic mixing if it was previously enabled
                if self.config.mic_mixing.enabled:
                    asyncio.create_task(self._restart_mic_capture(device))

        except Exception as e:
            self.logger.error(f"Error handling device recovery for {role.value}: {e}")

    async def _restart_mic_capture(self, device: AudioDevice) -> None:
        """
        Restart microphone capture after device recovery.

        Args:
            device: The recovered microphone device
        """
        try:
            self.logger.info(f"Restarting mic capture on: {device.name}")
            success = await self.mic_capture_service.start_capture(device)
            if success:
                self._mic_mixing_enabled = True
                self.logger.info("Mic capture restarted successfully")
            else:
                self.logger.warning("Failed to restart mic capture")
        except Exception as e:
            self.logger.error(f"Error restarting mic capture: {e}")

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

    def _handle_device_change(self, role: DeviceRole, event) -> None:
        """
        Handle device change events for hot-plug support.

        Args:
            role: Role of the device (MONITOR, VIRTUAL_MIC, or MICROPHONE)
            event: DeviceChangeEvent with event type and device info
        """
        self.logger.debug(f"Device change event: {role.value} - {event.event_type}")

        try:
            # Handle microphone disconnect - disable mixing
            if role == DeviceRole.MICROPHONE and event.event_type == "removed":
                self.logger.warning(f"Microphone disconnected: {event.device_name}")
                self._mic_mixing_enabled = False

                # Stop capture if running
                if self.mic_capture_service and self.mic_capture_service.is_capturing:
                    asyncio.create_task(self._stop_mic_on_disconnect())

        except Exception as e:
            self.logger.error(f"Error handling device change: {e}")

    async def _stop_mic_on_disconnect(self) -> None:
        """Stop mic capture when device is disconnected."""
        try:
            if self.mic_capture_service:
                await self.mic_capture_service.stop_capture()
                self.logger.info("Mic capture stopped due to device disconnect")
        except Exception as e:
            self.logger.error(f"Error stopping mic on disconnect: {e}")

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

    # =========================================================================
    # Microphone Mixing Control Methods
    # =========================================================================

    async def start_mic_capture(self, device: Optional[AudioDevice] = None) -> bool:
        """
        Start microphone capture for mixing with TTS.

        Args:
            device: Input device to capture from (uses default if None)

        Returns:
            bool: True if capture started successfully
        """
        self.logger.info(f"[MIC_DEBUG] start_mic_capture called with device={device}")
        self.logger.info(f"[MIC_DEBUG] config.mic_mixing.enabled={self.config.mic_mixing.enabled}")

        if not self.mic_capture_service:
            self.logger.error("MicrophoneCaptureService not initialized")
            return False

        try:
            success = await self.mic_capture_service.start_capture(device)
            self.logger.info(f"[MIC_DEBUG] mic_capture_service.start_capture returned: {success}")
            if success:
                self._mic_mixing_enabled = self.config.mic_mixing.enabled
                self.logger.info(f"[MIC_DEBUG] _mic_mixing_enabled set to: {self._mic_mixing_enabled}")
                self.logger.info(f"Microphone capture started (mixing: {self._mic_mixing_enabled})")

                # Register mic device with resilience manager for hot-plug handling
                if self.resilience_manager and device:
                    self.resilience_manager.register_device(
                        DeviceRole.MICROPHONE,
                        device
                    )
                    self.logger.debug(f"Registered mic device for hot-plug monitoring: {device.name}")

            return success
        except Exception as e:
            self.logger.error(f"Failed to start mic capture: {e}")
            return False

    async def stop_mic_capture(self) -> bool:
        """
        Stop microphone capture.

        Returns:
            bool: True if capture stopped successfully
        """
        if not self.mic_capture_service:
            return True  # Nothing to stop

        try:
            success = await self.mic_capture_service.stop_capture()
            self._mic_mixing_enabled = False

            # Unregister mic device from resilience manager
            if self.resilience_manager:
                self.resilience_manager.unregister_device(DeviceRole.MICROPHONE)
                self.logger.debug("Unregistered mic device from hot-plug monitoring")

            self.logger.info("Microphone capture stopped")
            return success
        except Exception as e:
            self.logger.error(f"Failed to stop mic capture: {e}")
            return False

    def enable_mic_mixing(self, enabled: bool = True) -> None:
        """
        Enable or disable microphone mixing.

        Args:
            enabled: True to enable mixing, False to disable
        """
        self.config.mic_mixing.enabled = enabled
        if self.mic_capture_service and self.mic_capture_service.is_capturing:
            self._mic_mixing_enabled = enabled
        self.logger.info(f"Mic mixing {'enabled' if enabled else 'disabled'}")

    def set_mixing_mode(self, mode: MicrophoneMixingMode) -> None:
        """
        Set the microphone mixing mode.

        Args:
            mode: MicrophoneMixingMode to use
        """
        self.config.mic_mixing.mode = mode
        self.logger.info(f"Mic mixing mode set to: {mode.value}")

    def set_mic_volume(self, volume: float) -> None:
        """
        Set microphone input volume for mixing.

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.config.mic_mixing.mic_volume = max(0.0, min(1.0, volume))
        if self.mic_capture_service:
            self.mic_capture_service.set_volume(self.config.mic_mixing.mic_volume)
        self.logger.debug(f"Mic volume set to: {self.config.mic_mixing.mic_volume:.2f}")

    def set_tts_volume_for_mixing(self, volume: float) -> None:
        """
        Set TTS volume for the mixed output.

        Note: This affects only the virtual mic output, not the monitor output.

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.config.mic_mixing.tts_volume = max(0.0, min(1.0, volume))
        self.logger.debug(f"TTS mixing volume set to: {self.config.mic_mixing.tts_volume:.2f}")

    def set_mic_muted(self, muted: bool) -> None:
        """
        Mute or unmute microphone input.

        Args:
            muted: True to mute, False to unmute
        """
        if self.mic_capture_service:
            self.mic_capture_service.set_muted(muted)
        self.logger.debug(f"Mic {'muted' if muted else 'unmuted'}")

    def toggle_mic_mute(self) -> bool:
        """
        Toggle microphone mute state.

        Returns:
            bool: New mute state (True = muted)
        """
        if self.mic_capture_service:
            return self.mic_capture_service.toggle_mute()
        return False

    def is_mic_muted(self) -> bool:
        """
        Check if microphone is muted.

        Returns:
            bool: True if muted
        """
        if self.mic_capture_service:
            return self.mic_capture_service.is_muted
        return False

    def is_mic_capturing(self) -> bool:
        """
        Check if microphone capture is active.

        Returns:
            bool: True if capturing
        """
        if self.mic_capture_service:
            return self.mic_capture_service.is_capturing
        return False

    async def enumerate_mic_devices(self) -> List[AudioDevice]:
        """
        Enumerate available microphone input devices.

        Returns:
            List[AudioDevice]: Available input devices
        """
        if not self.mic_capture_service:
            return []

        try:
            return await self.mic_capture_service.enumerate_input_devices()
        except Exception as e:
            self.logger.error(f"Failed to enumerate mic devices: {e}")
            return []

    def get_mic_capture_statistics(self) -> Dict[str, Any]:
        """
        Get microphone capture statistics.

        Returns:
            Dict with capture statistics
        """
        if not self.mic_capture_service:
            return {"enabled": False}

        try:
            stats = self.mic_capture_service.get_statistics()
            return {
                "enabled": True,
                "capturing": self.mic_capture_service.is_capturing,
                "mixing_enabled": self._mic_mixing_enabled,
                "mode": self.config.mic_mixing.mode.value,
                "mic_volume": self.config.mic_mixing.mic_volume,
                "tts_volume": self.config.mic_mixing.tts_volume,
                "muted": self.mic_capture_service.is_muted,
                "chunks_captured": stats.chunks_captured,
                "chunks_dropped": stats.chunks_dropped,
                "buffer_overflows": stats.buffer_overflows,
            }
        except Exception as e:
            return {"enabled": True, "error": str(e)}

    def enable_ducking(self, enabled: bool = True, duck_amount: float = 0.3) -> None:
        """
        Enable or disable ducking (reduce mic volume during TTS).

        Args:
            enabled: True to enable ducking
            duck_amount: How much to reduce mic volume (0.0-1.0)
        """
        self.config.mic_mixing.enable_ducking = enabled
        self.config.mic_mixing.duck_amount = max(0.0, min(1.0, duck_amount))
        self.logger.debug(
            f"Ducking {'enabled' if enabled else 'disabled'} "
            f"(amount: {self.config.mic_mixing.duck_amount:.2f})"
        )

    # =========================================================================
    # Continuous Microphone Passthrough
    # =========================================================================
    #
    # NOTE: Continuous passthrough is disabled for now. The mic mixing during
    # TTS playback works via _get_virtual_mic_audio(). For continuous mic
    # passthrough when TTS is NOT playing, we need a different architecture
    # that doesn't conflict with TTS streaming sessions.
    #
    # Current behavior with mic_mixing_enabled=True:
    # - During TTS playback: mic is mixed with TTS and sent to virtual mic
    # - When TTS is idle: mic audio is NOT sent to virtual mic (limitation)
    #
    # To test mic: Use the "Monitor Mic to Speakers" toggle in settings
    # =========================================================================

    async def start_continuous_mic_passthrough(self) -> bool:
        """
        Start continuous microphone passthrough to virtual mic.

        When TTS is playing, mic is mixed via _get_virtual_mic_audio().
        When TTS is NOT playing, this loop sends mic directly to virtual mic.

        Uses threading to avoid qasync conflicts.

        Returns:
            bool: True if passthrough started successfully
        """
        if self._continuous_passthrough_running:
            self.logger.debug("Continuous mic passthrough already running")
            return True

        if not self.mic_capture_service or not self.mic_capture_service.is_capturing:
            self.logger.warning("Mic capture not active, cannot start passthrough")
            return False

        if not self.virtual_service:
            self.logger.warning("Virtual mic service not available")
            return False

        self._continuous_passthrough_running = True

        # Use a background thread instead of asyncio to avoid qasync conflicts
        self._passthrough_thread = threading.Thread(
            target=self._continuous_passthrough_thread_loop,
            name="MicPassthrough",
            daemon=True
        )
        self._passthrough_thread.start()
        self.logger.info("[MIC_PASSTHROUGH] Continuous mic passthrough thread started")
        return True

    async def stop_continuous_mic_passthrough(self) -> bool:
        """
        Stop continuous microphone passthrough.

        Returns:
            bool: True if stopped successfully
        """
        if not self._continuous_passthrough_running:
            return True

        self._continuous_passthrough_running = False

        # Wait for thread to finish
        if hasattr(self, '_passthrough_thread') and self._passthrough_thread:
            self._passthrough_thread.join(timeout=2.0)
            self._passthrough_thread = None

        self.logger.info("[MIC_PASSTHROUGH] Mic passthrough disabled")
        return True

    def _continuous_passthrough_thread_loop(self):
        """
        Background thread for continuous mic passthrough to virtual mic.

        Uses direct PyAudio calls to avoid asyncio/qasync conflicts.
        Only active when TTS is NOT playing.
        """
        try:
            import pyaudio
        except ImportError:
            self.logger.error("[MIC_PASSTHROUGH] PyAudio not available")
            return

        self.logger.info("[MIC_PASSTHROUGH] Passthrough thread started")
        self.logger.info(f"[MIC_PASSTHROUGH] mic_capture active: {self.mic_capture_service.is_capturing if self.mic_capture_service else False}")
        self.logger.info(f"[MIC_PASSTHROUGH] virtual_service available: {self.virtual_service is not None}")

        passthrough_stream = None
        chunks_sent = 0
        pyaudio_instance = None

        try:
            # Get the virtual device
            virtual_device = getattr(self.virtual_service, 'current_virtual_device', None)
            if not virtual_device:
                self.logger.warning("[MIC_PASSTHROUGH] No virtual device configured, trying to find one")
                # Try to use the first available virtual device
                try:
                    loop = asyncio.new_event_loop()
                    virtual_devices = loop.run_until_complete(self.virtual_service.enumerate_virtual_devices())
                    loop.close()
                    if virtual_devices:
                        virtual_device = virtual_devices[0]
                        self.logger.info(f"[MIC_PASSTHROUGH] Using first available device: {virtual_device.name}")
                except Exception as e:
                    self.logger.error(f"[MIC_PASSTHROUGH] Failed to enumerate virtual devices: {e}")
                    return

            if not virtual_device:
                self.logger.error("[MIC_PASSTHROUGH] No virtual device available")
                return

            # Get device index
            device_id_str = virtual_device.device_id
            if device_id_str.startswith("pyaudio_"):
                device_index = int(device_id_str.split("_")[1])
            else:
                device_index = int(device_id_str)

            sample_rate = self.mic_capture_service.config.sample_rate
            self.logger.info(f"[MIC_PASSTHROUGH] Opening stream to {virtual_device.name} (index {device_index}) at {sample_rate}Hz")

            # Create our own PyAudio instance for the passthrough stream
            pyaudio_instance = pyaudio.PyAudio()

            while self._continuous_passthrough_running:
                # Check if TTS streaming is currently active
                tts_active = (
                    self.virtual_service.is_streaming_active() and
                    self.monitor_service and
                    self.monitor_service.is_streaming_active()
                )

                if tts_active:
                    # TTS is playing - mixing handles mic, we just wait
                    if passthrough_stream:
                        try:
                            passthrough_stream.stop_stream()
                            passthrough_stream.close()
                        except Exception:
                            pass
                        passthrough_stream = None
                        self.logger.debug(f"[MIC_PASSTHROUGH] Paused (TTS active), sent {chunks_sent} chunks")
                        chunks_sent = 0
                    time.sleep(0.05)
                    continue

                # TTS is NOT playing - send mic directly to virtual mic
                if not passthrough_stream:
                    try:
                        passthrough_stream = pyaudio_instance.open(
                            format=pyaudio.paInt16,
                            channels=1,
                            rate=sample_rate,
                            output=True,
                            output_device_index=device_index,
                            frames_per_buffer=1024
                        )
                        self.logger.info(f"[MIC_PASSTHROUGH] Stream opened to device {device_index}")
                    except Exception as e:
                        self.logger.error(f"[MIC_PASSTHROUGH] Failed to open stream: {e}")
                        time.sleep(0.5)
                        continue

                # Get and send mic audio to VIRTUAL MIC (not monitor!)
                if self.mic_capture_service and passthrough_stream:
                    chunk = self.mic_capture_service.get_audio_chunk()
                    if chunk:
                        try:
                            # Write directly to PyAudio stream (synchronous)
                            passthrough_stream.write(chunk)
                            chunks_sent += 1
                            if chunks_sent % 500 == 0:
                                self.logger.debug(f"[MIC_PASSTHROUGH] Sent {chunks_sent} chunks to virtual mic")
                        except Exception as e:
                            self.logger.warning(f"[MIC_PASSTHROUGH] Error sending to virtual: {e}")

                time.sleep(0.005)  # Small sleep to prevent CPU spinning

        except Exception as e:
            self.logger.error(f"[MIC_PASSTHROUGH] Error in thread: {e}")
        finally:
            # Clean up stream
            if passthrough_stream:
                try:
                    passthrough_stream.stop_stream()
                    passthrough_stream.close()
                except Exception:
                    pass
            if pyaudio_instance:
                try:
                    pyaudio_instance.terminate()
                except Exception:
                    pass
            self._continuous_passthrough_running = False
            self.logger.info(f"[MIC_PASSTHROUGH] Thread ended, sent {chunks_sent} total chunks")

    def is_continuous_passthrough_active(self) -> bool:
        """Check if continuous mic passthrough is active."""
        return self._continuous_passthrough_running

    # =========================================================================
    # Mic Monitor to Speakers (for testing/debugging in settings)
    # =========================================================================

    async def start_mic_monitor_to_speakers(
        self,
        mic_device: Optional[AudioDevice] = None
    ) -> bool:
        """
        Start monitoring microphone audio to speakers (for testing).

        This sends mic audio to the monitor speakers so users can verify
        their microphone is working. Separate from virtual mic passthrough.

        Args:
            mic_device: Specific mic device to monitor (uses current if None)

        Returns:
            bool: True if monitoring started successfully
        """
        self.logger.info(f"[MIC_MONITOR] start_mic_monitor_to_speakers called, current running={self._mic_monitor_running}")
        if self._mic_monitor_running:
            self.logger.debug("Mic monitor already running")
            return True

        if not self.monitor_service:
            self.logger.warning("Monitor service not available")
            return False

        try:
            # Start mic capture if not already running
            if not self.mic_capture_service or not self.mic_capture_service.is_capturing:
                success = await self.start_mic_capture(mic_device)
                if not success:
                    self.logger.error("Failed to start mic capture for monitoring")
                    return False

            # Start streaming session on monitor
            sample_rate = self.mic_capture_service.config.sample_rate
            session_id = await self.monitor_service.start_streaming_session(
                sample_rate=sample_rate,
                channels=1,
                sample_width=2
            )

            if not session_id:
                self.logger.error("Failed to start monitor streaming session")
                return False

            self._mic_monitor_running = True
            self._mic_monitor_task = asyncio.create_task(
                self._mic_monitor_loop()
            )
            self.logger.info("Mic monitor to speakers started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start mic monitor: {e}")
            return False

    async def stop_mic_monitor_to_speakers(self) -> bool:
        """
        Stop monitoring microphone audio to speakers.

        Returns:
            bool: True if monitoring stopped successfully
        """
        if not self._mic_monitor_running:
            return True

        try:
            self._mic_monitor_running = False

            # Cancel the monitor task
            if self._mic_monitor_task:
                self._mic_monitor_task.cancel()
                try:
                    await self._mic_monitor_task
                except asyncio.CancelledError:
                    pass
                self._mic_monitor_task = None

            # Stop monitor streaming
            if self.monitor_service:
                await self.monitor_service.stop_streaming_session()

            self.logger.info("Mic monitor to speakers stopped")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping mic monitor: {e}")
            return False

    async def _mic_monitor_loop(self):
        """
        Background loop for mic-to-speaker monitoring.

        Reads mic audio chunks and plays them to the monitor speakers.
        ONLY runs when user explicitly enables "Monitor Mic to Speakers".
        """
        self.logger.warning("[MIC_MONITOR] ====== MONITOR LOOP STARTED ======")
        self.logger.warning("[MIC_MONITOR] This should ONLY happen when checkbox is checked!")

        # Double-check we should actually be running
        if not self._mic_monitor_running:
            self.logger.error("[MIC_MONITOR] Loop started but _mic_monitor_running is False! Exiting.")
            return

        try:
            while self._mic_monitor_running:
                if self.mic_capture_service:
                    chunk = self.mic_capture_service.get_audio_chunk()
                    if chunk:
                        try:
                            await self.monitor_service.play_audio_chunk(chunk, is_final=False)
                        except Exception as e:
                            self.logger.warning(f"Error playing mic chunk to monitor: {e}")

                # Small delay to prevent busy-waiting
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            self.logger.debug("Mic monitor loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in mic monitor loop: {e}")
        finally:
            self._mic_monitor_running = False

    def is_mic_monitor_active(self) -> bool:
        """Check if mic monitor to speakers is active."""
        return self._mic_monitor_running

    # Required BaseService attribute
    _is_initialized = False