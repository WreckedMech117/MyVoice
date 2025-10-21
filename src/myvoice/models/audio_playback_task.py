"""
AudioPlaybackTask Models

This module contains the AudioPlaybackTask data model for managing audio playback
tasks with status tracking and error state management. Supports dual routing to
monitor speakers and virtual microphone devices simultaneously.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime

from myvoice.models.validation import ValidationResult, ValidationIssue, ValidationStatus
from myvoice.models.audio_device import AudioDevice, DeviceType

logger = logging.getLogger(__name__)


class PlaybackStatus(Enum):
    """Audio playback task status tracking."""
    PENDING = "pending"
    PLAYING = "playing"
    COMPLETED = "completed"
    FAILED = "failed"


class DualRoutingMode(Enum):
    """Dual routing configuration modes."""
    MONITOR_ONLY = "monitor_only"
    VIRTUAL_MIC_ONLY = "virtual_mic_only"
    DUAL_ROUTING = "dual_routing"


class SynchronizationStatus(Enum):
    """Synchronization status for dual audio streams."""
    NOT_APPLICABLE = "not_applicable"
    SYNCHRONIZED = "synchronized"
    DRIFT_DETECTED = "drift_detected"
    DESYNCHRONIZED = "desynchronized"


@dataclass
class AudioPlaybackTask:
    """
    Data model for audio playback tasks with dual routing support.

    Represents an audio playback task that can route TTS-generated audio to monitor
    devices (like speakers) and/or virtual microphone devices simultaneously, with
    status tracking, synchronization management, and error handling.

    Attributes:
        audio_data: Binary audio data to be played
        monitor_device: Primary audio device for monitoring (speakers/headphones)
        virtual_mic_device: Optional virtual microphone device for dual routing
        routing_mode: Mode for audio routing (monitor only, virtual mic only, or dual)
        volume_level: Playback volume level (0.0 to 1.0)
        virtual_mic_volume: Volume level for virtual microphone output (0.0 to 1.0)
        playback_id: Unique identifier for this playback task
        status: Current status of the playback task
        sync_status: Synchronization status for dual streams
        error_message: Error message if task failed
        device_errors: Individual device error messages
        created_at: Timestamp when task was created
        started_at: Timestamp when playback started
        completed_at: Timestamp when playback completed
        sync_offset_ms: Detected synchronization offset between streams in milliseconds
        max_allowed_drift_ms: Maximum allowed drift before marking as desynchronized
    """
    audio_data: bytes
    monitor_device: AudioDevice
    virtual_mic_device: Optional[AudioDevice] = None
    routing_mode: DualRoutingMode = DualRoutingMode.MONITOR_ONLY
    volume_level: float = 1.0
    virtual_mic_volume: float = 1.0
    playback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: PlaybackStatus = PlaybackStatus.PENDING
    sync_status: SynchronizationStatus = SynchronizationStatus.NOT_APPLICABLE
    error_message: Optional[str] = None
    device_errors: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    sync_offset_ms: float = 0.0
    max_allowed_drift_ms: float = 50.0

    def __post_init__(self):
        """Initialize and validate the audio playback task."""
        # Auto-configure routing mode based on devices
        self._auto_configure_routing_mode()

        # Set initial synchronization status
        self._update_sync_status()

        # Validate the task
        validation_result = self.validate()
        if not validation_result.is_valid:
            # Set to failed status if validation fails
            self.status = PlaybackStatus.FAILED
            self.error_message = f"Validation failed: {validation_result.summary}"

    def _auto_configure_routing_mode(self) -> None:
        """Auto-configure routing mode based on available devices."""
        # Only auto-configure if routing mode is set to monitor only and we have an available virtual device
        if (self.routing_mode == DualRoutingMode.MONITOR_ONLY and
            self.virtual_mic_device is not None and
            self.virtual_mic_device.is_available):
            self.routing_mode = DualRoutingMode.DUAL_ROUTING

        # Validate that virtual device requirements are met for dual/virtual-only modes
        elif self.routing_mode in [DualRoutingMode.DUAL_ROUTING, DualRoutingMode.VIRTUAL_MIC_ONLY]:
            if (self.virtual_mic_device is None or not self.virtual_mic_device.is_available):
                # Only fallback if auto-configuration is appropriate
                # For explicit dual/virtual-only modes, let validation handle the error
                pass

    def _update_sync_status(self) -> None:
        """Update synchronization status based on current routing mode."""
        if self.routing_mode == DualRoutingMode.DUAL_ROUTING:
            self.sync_status = SynchronizationStatus.SYNCHRONIZED
        else:
            self.sync_status = SynchronizationStatus.NOT_APPLICABLE

    def is_dual_routing_enabled(self) -> bool:
        """Check if dual routing is enabled."""
        return self.routing_mode == DualRoutingMode.DUAL_ROUTING

    def requires_virtual_device(self) -> bool:
        """Check if the task requires a virtual device."""
        return self.routing_mode in [DualRoutingMode.DUAL_ROUTING, DualRoutingMode.VIRTUAL_MIC_ONLY]

    def get_active_devices(self) -> List[AudioDevice]:
        """Get list of active devices based on routing mode."""
        devices = []

        if self.routing_mode in [DualRoutingMode.MONITOR_ONLY, DualRoutingMode.DUAL_ROUTING]:
            devices.append(self.monitor_device)

        if self.routing_mode in [DualRoutingMode.VIRTUAL_MIC_ONLY, DualRoutingMode.DUAL_ROUTING]:
            if self.virtual_mic_device:
                devices.append(self.virtual_mic_device)

        return devices

    def validate_virtual_device_configuration(self) -> ValidationResult:
        """Validate virtual device configuration for dual routing."""
        issues = []
        warnings = []

        # Virtual device requirement validation
        if self.requires_virtual_device():
            if self.virtual_mic_device is None:
                issues.append(ValidationIssue(
                    field="virtual_mic_device",
                    message=f"Virtual microphone device required for {self.routing_mode.value} mode",
                    code="MISSING_VIRTUAL_DEVICE",
                    severity=ValidationStatus.INVALID
                ))
            elif not isinstance(self.virtual_mic_device, AudioDevice):
                issues.append(ValidationIssue(
                    field="virtual_mic_device",
                    message="Virtual microphone device must be an AudioDevice instance",
                    code="INVALID_VIRTUAL_DEVICE",
                    severity=ValidationStatus.INVALID
                ))
            elif not self.virtual_mic_device.is_available:
                issues.append(ValidationIssue(
                    field="virtual_mic_device",
                    message="Virtual microphone device is not available",
                    code="UNAVAILABLE_VIRTUAL_DEVICE",
                    severity=ValidationStatus.INVALID
                ))
            elif self.virtual_mic_device.device_type != DeviceType.VIRTUAL:
                warnings.append(ValidationIssue(
                    field="virtual_mic_device",
                    message="Virtual microphone device is not marked as virtual type",
                    code="NON_VIRTUAL_DEVICE_TYPE",
                    severity=ValidationStatus.WARNING
                ))

        # Virtual microphone volume validation
        if not isinstance(self.virtual_mic_volume, (int, float)):
            issues.append(ValidationIssue(
                field="virtual_mic_volume",
                message="Virtual microphone volume must be a number",
                code="INVALID_VIRTUAL_MIC_VOLUME_TYPE",
                severity=ValidationStatus.INVALID
            ))
        elif self.virtual_mic_volume < 0.0:
            issues.append(ValidationIssue(
                field="virtual_mic_volume",
                message="Virtual microphone volume cannot be negative",
                code="NEGATIVE_VIRTUAL_MIC_VOLUME",
                severity=ValidationStatus.INVALID
            ))
        elif self.virtual_mic_volume > 1.0:
            warnings.append(ValidationIssue(
                field="virtual_mic_volume",
                message="Virtual microphone volume above 1.0 may cause distortion",
                code="HIGH_VIRTUAL_MIC_VOLUME",
                severity=ValidationStatus.WARNING
            ))

        # Synchronization validation for dual routing
        if self.is_dual_routing_enabled():
            if self.max_allowed_drift_ms < 0:
                issues.append(ValidationIssue(
                    field="max_allowed_drift_ms",
                    message="Maximum allowed drift cannot be negative",
                    code="NEGATIVE_MAX_DRIFT",
                    severity=ValidationStatus.INVALID
                ))
            elif self.max_allowed_drift_ms > 500:
                warnings.append(ValidationIssue(
                    field="max_allowed_drift_ms",
                    message="High drift tolerance may affect synchronization quality",
                    code="HIGH_DRIFT_TOLERANCE",
                    severity=ValidationStatus.WARNING
                ))

            if abs(self.sync_offset_ms) > self.max_allowed_drift_ms:
                warnings.append(ValidationIssue(
                    field="sync_offset_ms",
                    message=f"Current sync offset ({self.sync_offset_ms}ms) exceeds tolerance",
                    code="SYNC_OFFSET_EXCEEDS_TOLERANCE",
                    severity=ValidationStatus.WARNING
                ))

        # Determine status
        if issues:
            status = ValidationStatus.INVALID
            is_valid = False
        elif warnings:
            status = ValidationStatus.WARNING
            is_valid = True
        else:
            status = ValidationStatus.VALID
            is_valid = True

        # Get routing mode value safely
        routing_mode_value = self.routing_mode.value if hasattr(self.routing_mode, 'value') else str(self.routing_mode)

        return ValidationResult(
            is_valid=is_valid,
            status=status,
            issues=issues,
            warnings=warnings,
            summary=f"Virtual device validation for {routing_mode_value} mode"
        )

    def validate(self) -> ValidationResult:
        """
        Validate the audio playback task for required fields and constraints.

        Returns:
            ValidationResult: Detailed validation result with issues and warnings
        """
        issues = []
        warnings = []

        try:
            # Audio data validation
            if not self.audio_data:
                issues.append(ValidationIssue(
                    field="audio_data",
                    message="Audio data cannot be empty",
                    code="EMPTY_AUDIO_DATA",
                    severity=ValidationStatus.INVALID
                ))
            elif len(self.audio_data) < 44:  # WAV header is at least 44 bytes
                warnings.append(ValidationIssue(
                    field="audio_data",
                    message="Audio data is very small, may not be valid audio",
                    code="SMALL_AUDIO_DATA",
                    severity=ValidationStatus.WARNING
                ))

            # Monitor device validation
            if not isinstance(self.monitor_device, AudioDevice):
                issues.append(ValidationIssue(
                    field="monitor_device",
                    message="Monitor device must be an AudioDevice instance",
                    code="INVALID_MONITOR_DEVICE",
                    severity=ValidationStatus.INVALID
                ))
            elif not self.monitor_device.is_available:
                issues.append(ValidationIssue(
                    field="monitor_device",
                    message="Monitor device is not available",
                    code="UNAVAILABLE_MONITOR_DEVICE",
                    severity=ValidationStatus.INVALID
                ))

            # Volume level validation
            if not isinstance(self.volume_level, (int, float)):
                issues.append(ValidationIssue(
                    field="volume_level",
                    message="Volume level must be a number",
                    code="INVALID_VOLUME_TYPE",
                    severity=ValidationStatus.INVALID
                ))
            elif self.volume_level < 0.0:
                issues.append(ValidationIssue(
                    field="volume_level",
                    message="Volume level cannot be negative",
                    code="NEGATIVE_VOLUME",
                    severity=ValidationStatus.INVALID
                ))
            elif self.volume_level > 1.0:
                warnings.append(ValidationIssue(
                    field="volume_level",
                    message="Volume level above 1.0 may cause distortion",
                    code="HIGH_VOLUME",
                    severity=ValidationStatus.WARNING
                ))
            elif self.volume_level == 0.0:
                warnings.append(ValidationIssue(
                    field="volume_level",
                    message="Volume level is 0.0, audio will be muted",
                    code="MUTED_VOLUME",
                    severity=ValidationStatus.WARNING
                ))

            # Playback ID validation
            if not self.playback_id or not self.playback_id.strip():
                issues.append(ValidationIssue(
                    field="playback_id",
                    message="Playback ID cannot be empty",
                    code="EMPTY_PLAYBACK_ID",
                    severity=ValidationStatus.INVALID
                ))

            # Status validation
            if not isinstance(self.status, PlaybackStatus):
                issues.append(ValidationIssue(
                    field="status",
                    message="Status must be a valid PlaybackStatus enum",
                    code="INVALID_STATUS",
                    severity=ValidationStatus.INVALID
                ))

            # Timestamp consistency validation
            if self.started_at and self.created_at and self.started_at < self.created_at:
                warnings.append(ValidationIssue(
                    field="started_at",
                    message="Start time is before creation time",
                    code="INVALID_START_TIME",
                    severity=ValidationStatus.WARNING
                ))

            if self.completed_at and self.started_at and self.completed_at < self.started_at:
                warnings.append(ValidationIssue(
                    field="completed_at",
                    message="Completion time is before start time",
                    code="INVALID_COMPLETION_TIME",
                    severity=ValidationStatus.WARNING
                ))

            # Status consistency validation
            if self.status == PlaybackStatus.PLAYING and not self.started_at:
                warnings.append(ValidationIssue(
                    field="status",
                    message="Status is PLAYING but started_at is not set",
                    code="INCONSISTENT_PLAYING_STATUS",
                    severity=ValidationStatus.WARNING
                ))

            if self.status == PlaybackStatus.COMPLETED and not self.completed_at:
                warnings.append(ValidationIssue(
                    field="status",
                    message="Status is COMPLETED but completed_at is not set",
                    code="INCONSISTENT_COMPLETED_STATUS",
                    severity=ValidationStatus.WARNING
                ))

            if self.status == PlaybackStatus.FAILED and not self.error_message:
                warnings.append(ValidationIssue(
                    field="status",
                    message="Status is FAILED but no error message is provided",
                    code="MISSING_ERROR_MESSAGE",
                    severity=ValidationStatus.WARNING
                ))

            # Routing mode validation
            if not isinstance(self.routing_mode, DualRoutingMode):
                issues.append(ValidationIssue(
                    field="routing_mode",
                    message="Routing mode must be a valid DualRoutingMode enum",
                    code="INVALID_ROUTING_MODE",
                    severity=ValidationStatus.INVALID
                ))

            # Virtual device configuration validation
            virtual_validation = self.validate_virtual_device_configuration()
            issues.extend(virtual_validation.issues)
            warnings.extend(virtual_validation.warnings)

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
            logger.exception(f"Error during audio playback task validation: {e}")
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

    def update_sync_offset(self, offset_ms: float) -> None:
        """
        Update the synchronization offset between audio streams.

        Args:
            offset_ms: Detected offset in milliseconds between monitor and virtual mic streams
        """
        self.sync_offset_ms = offset_ms

        # Update synchronization status based on offset
        if not self.is_dual_routing_enabled():
            self.sync_status = SynchronizationStatus.NOT_APPLICABLE
        elif abs(offset_ms) <= self.max_allowed_drift_ms:
            self.sync_status = SynchronizationStatus.SYNCHRONIZED
        elif abs(offset_ms) <= self.max_allowed_drift_ms * 2:
            self.sync_status = SynchronizationStatus.DRIFT_DETECTED
        else:
            self.sync_status = SynchronizationStatus.DESYNCHRONIZED

        logger.debug(f"Updated sync offset for task {self.playback_id}: {offset_ms}ms (status: {self.sync_status.value})")

    def check_device_availability(self) -> Dict[str, bool]:
        """
        Check availability of all required devices.

        Returns:
            Dict[str, bool]: Device availability status keyed by device type
        """
        availability = {}

        # Check monitor device
        availability['monitor'] = self.monitor_device.is_available if self.monitor_device else False

        # Check virtual mic device if required
        if self.requires_virtual_device():
            availability['virtual_mic'] = (
                self.virtual_mic_device.is_available
                if self.virtual_mic_device else False
            )

        return availability

    def add_device_error(self, device_type: str, error_message: str) -> None:
        """
        Add an error message for a specific device.

        Args:
            device_type: Type of device ('monitor' or 'virtual_mic')
            error_message: Error message to record
        """
        self.device_errors[device_type] = error_message
        logger.error(f"Device error for {device_type} in task {self.playback_id}: {error_message}")

    def clear_device_errors(self) -> None:
        """Clear all device-specific error messages."""
        self.device_errors.clear()

    def has_device_errors(self) -> bool:
        """Check if any device has reported errors."""
        return len(self.device_errors) > 0

    def get_device_error_summary(self) -> Optional[str]:
        """
        Get a summary of all device errors.

        Returns:
            Optional[str]: Summary of device errors, None if no errors
        """
        if not self.device_errors:
            return None

        error_parts = []
        for device_type, error_msg in self.device_errors.items():
            error_parts.append(f"{device_type}: {error_msg}")

        return "; ".join(error_parts)

    def start_playback(self) -> bool:
        """
        Mark the task as started and update timestamps.

        Returns:
            bool: True if status transition was successful, False otherwise
        """
        try:
            if self.status != PlaybackStatus.PENDING:
                logger.warning(f"Cannot start playback for task {self.playback_id}: current status is {self.status}")
                return False

            # Check device availability before starting
            device_availability = self.check_device_availability()
            unavailable_devices = [device for device, available in device_availability.items() if not available]

            if unavailable_devices:
                error_msg = f"Cannot start playback: devices unavailable: {', '.join(unavailable_devices)}"
                self.mark_failed(error_msg)
                return False

            # Clear any previous device errors
            self.clear_device_errors()

            self.status = PlaybackStatus.PLAYING
            self.started_at = datetime.now()

            # Initialize synchronization status for dual routing
            if self.is_dual_routing_enabled():
                self.sync_status = SynchronizationStatus.SYNCHRONIZED
                self.sync_offset_ms = 0.0

            logger.info(f"Started playback for task {self.playback_id} (mode: {self.routing_mode.value})")
            return True

        except Exception as e:
            logger.error(f"Failed to start playback for task {self.playback_id}: {e}")
            self.mark_failed(f"Failed to start playback: {str(e)}")
            return False

    def mark_completed(self) -> bool:
        """
        Mark the task as completed and update timestamps.

        Returns:
            bool: True if status transition was successful, False otherwise
        """
        try:
            if self.status not in [PlaybackStatus.PLAYING]:
                logger.warning(f"Cannot mark task {self.playback_id} as completed: current status is {self.status}")
                return False

            self.status = PlaybackStatus.COMPLETED
            self.completed_at = datetime.now()
            self.error_message = None  # Clear any previous error messages
            logger.info(f"Completed playback for task {self.playback_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to mark task {self.playback_id} as completed: {e}")
            return False

    def mark_failed(self, error_message: str, device_type: Optional[str] = None) -> bool:
        """
        Mark the task as failed with an error message.

        Args:
            error_message: Description of the failure
            device_type: Optional device type that caused the failure ('monitor' or 'virtual_mic')

        Returns:
            bool: True if status transition was successful, False otherwise
        """
        try:
            # Can fail from any state except completed
            if self.status == PlaybackStatus.COMPLETED:
                logger.warning(f"Cannot mark completed task {self.playback_id} as failed")
                return False

            self.status = PlaybackStatus.FAILED
            self.error_message = error_message
            self.completed_at = datetime.now()  # Set completion time for failed tasks too

            # Add device-specific error if provided
            if device_type:
                self.add_device_error(device_type, error_message)

            # Include device error summary in main error message if multiple devices
            if self.has_device_errors() and not device_type:
                device_summary = self.get_device_error_summary()
                if device_summary:
                    self.error_message = f"{error_message} | Device errors: {device_summary}"

            logger.error(f"Failed playback for task {self.playback_id}: {error_message}")
            return True

        except Exception as e:
            logger.error(f"Failed to mark task {self.playback_id} as failed: {e}")
            return False

    def get_duration_seconds(self) -> Optional[float]:
        """
        Get the duration of the playback task if completed.

        Returns:
            Optional[float]: Duration in seconds, None if not completed
        """
        if not self.started_at or not self.completed_at:
            return None

        return (self.completed_at - self.started_at).total_seconds()

    def get_task_info(self) -> Dict[str, Any]:
        """
        Get comprehensive task information.

        Returns:
            dict: Task information including all attributes and metadata
        """
        info = {
            'playback_id': self.playback_id,
            'status': self.status.value,
            'routing_mode': self.routing_mode.value,
            'monitor_device': {
                'device_id': self.monitor_device.device_id,
                'name': self.monitor_device.name,
                'device_type': self.monitor_device.device_type.value
            },
            'volume_level': self.volume_level,
            'audio_data_size': len(self.audio_data) if self.audio_data else 0,
            'error_message': self.error_message,
            'device_errors': self.device_errors.copy(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.get_duration_seconds(),
            'is_valid': self.validate().is_valid,
            'active_devices': [device.name for device in self.get_active_devices()]
        }

        # Add virtual mic information if configured
        if self.virtual_mic_device:
            info['virtual_mic_device'] = {
                'device_id': self.virtual_mic_device.device_id,
                'name': self.virtual_mic_device.name,
                'device_type': self.virtual_mic_device.device_type.value,
                'is_available': self.virtual_mic_device.is_available
            }
            info['virtual_mic_volume'] = self.virtual_mic_volume

        # Add synchronization information for dual routing
        if self.is_dual_routing_enabled():
            info['sync_status'] = self.sync_status.value
            info['sync_offset_ms'] = self.sync_offset_ms
            info['max_allowed_drift_ms'] = self.max_allowed_drift_ms
            info['is_synchronized'] = self.sync_status == SynchronizationStatus.SYNCHRONIZED

        return info

    def __str__(self) -> str:
        """String representation of the audio playback task."""
        device_name = self.monitor_device.name if self.monitor_device else "Unknown Device"
        status_display = self.status.value.upper()
        duration = f"({self.get_duration_seconds():.1f}s)" if self.get_duration_seconds() else ""

        # Add routing mode and virtual device info
        routing_info = f"[{self.routing_mode.value}]"
        if self.is_dual_routing_enabled() and self.virtual_mic_device:
            routing_info += f" + {self.virtual_mic_device.name}"
            if self.sync_status != SynchronizationStatus.NOT_APPLICABLE:
                routing_info += f" ({self.sync_status.value})"

        return f"[{status_display}] {routing_info} Playback {self.playback_id[:8]}... -> {device_name} {duration}"

    def __repr__(self) -> str:
        """Developer representation of the audio playback task."""
        virtual_device_name = self.virtual_mic_device.name if self.virtual_mic_device else None
        return (f"AudioPlaybackTask(playback_id='{self.playback_id}', "
                f"status={self.status}, routing_mode={self.routing_mode}, "
                f"monitor_device='{self.monitor_device.name}', "
                f"virtual_mic_device='{virtual_device_name}', "
                f"volume_level={self.volume_level})")

    @classmethod
    def create_from_tts_response(cls, audio_data: bytes, monitor_device: AudioDevice,
                                 volume_level: float = 1.0, playback_id: Optional[str] = None,
                                 virtual_mic_device: Optional[AudioDevice] = None,
                                 virtual_mic_volume: float = 1.0,
                                 routing_mode: Optional[DualRoutingMode] = None) -> 'AudioPlaybackTask':
        """
        Factory method to create an AudioPlaybackTask from TTS response data.

        Args:
            audio_data: Binary audio data from TTS response
            monitor_device: Device to route the audio to for monitoring
            volume_level: Playback volume level (0.0 to 1.0)
            playback_id: Optional custom playback ID
            virtual_mic_device: Optional virtual microphone device for dual routing
            virtual_mic_volume: Volume level for virtual microphone output (0.0 to 1.0)
            routing_mode: Routing mode (auto-detected if None)

        Returns:
            AudioPlaybackTask: Validated audio playback task instance

        Examples:
            # Create monitor-only task
            task = AudioPlaybackTask.create_from_tts_response(
                audio_data=tts_response.audio_data,
                monitor_device=speaker_device,
                volume_level=0.8
            )

            # Create dual routing task
            task = AudioPlaybackTask.create_from_tts_response(
                audio_data=audio_bytes,
                monitor_device=speaker_device,
                virtual_mic_device=vb_cable_device,
                volume_level=0.8,
                virtual_mic_volume=1.0,
                routing_mode=DualRoutingMode.DUAL_ROUTING
            )

            # Create virtual mic only task
            task = AudioPlaybackTask.create_from_tts_response(
                audio_data=audio_bytes,
                monitor_device=speaker_device,
                virtual_mic_device=vb_cable_device,
                routing_mode=DualRoutingMode.VIRTUAL_MIC_ONLY
            )
        """
        # Generate playback ID if not provided
        if playback_id is None:
            playback_id = str(uuid.uuid4())

        # Auto-detect routing mode if not specified
        if routing_mode is None:
            if virtual_mic_device is not None and virtual_mic_device.is_available:
                routing_mode = DualRoutingMode.DUAL_ROUTING
            else:
                routing_mode = DualRoutingMode.MONITOR_ONLY

        logger.debug(f"Creating audio playback task: {playback_id[:8]}... → {monitor_device.name} "
                    f"(mode: {routing_mode.value})")

        # Create and return the task (validation happens in __post_init__)
        task = cls(
            audio_data=audio_data,
            monitor_device=monitor_device,
            virtual_mic_device=virtual_mic_device,
            routing_mode=routing_mode,
            volume_level=volume_level,
            virtual_mic_volume=virtual_mic_volume,
            playback_id=playback_id
        )

        logger.info(f"Created audio playback task: {task}")
        return task

    @classmethod
    def create_dual_routing_task(cls, audio_data: bytes, monitor_device: AudioDevice,
                                virtual_mic_device: AudioDevice, volume_level: float = 1.0,
                                virtual_mic_volume: float = 1.0,
                                max_allowed_drift_ms: float = 50.0,
                                playback_id: Optional[str] = None) -> 'AudioPlaybackTask':
        """
        Factory method to create a dual routing AudioPlaybackTask.

        Args:
            audio_data: Binary audio data from TTS response
            monitor_device: Device for monitoring (speakers/headphones)
            virtual_mic_device: Virtual microphone device for applications
            volume_level: Monitor device volume level (0.0 to 1.0)
            virtual_mic_volume: Virtual microphone volume level (0.0 to 1.0)
            max_allowed_drift_ms: Maximum allowed synchronization drift in milliseconds
            playback_id: Optional custom playback ID

        Returns:
            AudioPlaybackTask: Dual routing task with synchronization support

        Examples:
            # Create dual routing with VB-Cable
            task = AudioPlaybackTask.create_dual_routing_task(
                audio_data=tts_audio,
                monitor_device=speakers,
                virtual_mic_device=vb_cable,
                volume_level=0.8,
                virtual_mic_volume=1.0,
                max_allowed_drift_ms=25.0
            )
        """
        task = cls.create_from_tts_response(
            audio_data=audio_data,
            monitor_device=monitor_device,
            virtual_mic_device=virtual_mic_device,
            routing_mode=DualRoutingMode.DUAL_ROUTING,
            volume_level=volume_level,
            virtual_mic_volume=virtual_mic_volume,
            playback_id=playback_id
        )

        # Set custom drift tolerance
        task.max_allowed_drift_ms = max_allowed_drift_ms
        return task