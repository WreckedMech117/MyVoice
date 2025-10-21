"""
Audio Level Management Service

This service provides consistent audio level normalization and automatic gain control
for virtual microphone output to ensure compatibility with communication applications
like Discord, Zoom, Teams, etc.
"""

import asyncio
import logging
import math
import time
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque
from threading import Lock

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from myvoice.services.core.base_service import BaseService, ServiceStatus
from myvoice.models.error import MyVoiceError, ErrorSeverity


class CommunicationApp(Enum):
    """Common communication applications with specific audio requirements."""
    DISCORD = "discord"
    ZOOM = "zoom"
    TEAMS = "teams"
    SKYPE = "skype"
    WEBEX = "webex"
    GOOGLE_MEET = "google_meet"
    SLACK = "slack"
    GENERIC = "generic"


class NormalizationMode(Enum):
    """Audio normalization modes."""
    PEAK_NORMALIZE = "peak_normalize"      # Normalize to peak level
    RMS_NORMALIZE = "rms_normalize"        # Normalize to RMS level
    LUFS_NORMALIZE = "lufs_normalize"      # Loudness Units relative to Full Scale
    AGC = "agc"                           # Automatic Gain Control
    LIMITER = "limiter"                   # Peak limiting only
    PASSTHROUGH = "passthrough"           # No processing


@dataclass
class AudioLevelMetrics:
    """Audio level measurements and statistics."""
    peak_db: float
    rms_db: float
    lufs: float
    dynamic_range_db: float
    average_level_db: float
    gain_reduction_db: float
    sample_count: int
    timestamp: float


@dataclass
class CommunicationAppProfile:
    """Audio profile for specific communication applications."""
    app_name: CommunicationApp
    target_peak_db: float = -3.0          # Target peak level in dB
    target_rms_db: float = -18.0          # Target RMS level in dB
    target_lufs: float = -16.0            # Target loudness (LUFS)
    max_gain_db: float = 12.0             # Maximum gain boost
    min_gain_db: float = -20.0            # Maximum gain reduction
    attack_time_ms: float = 5.0           # AGC attack time
    release_time_ms: float = 50.0         # AGC release time
    noise_gate_threshold_db: float = -40.0 # Noise gate threshold
    enable_limiter: bool = True           # Enable peak limiter
    limiter_threshold_db: float = -1.0    # Limiter threshold
    normalization_mode: NormalizationMode = NormalizationMode.AGC


@dataclass
class AudioProcessingState:
    """Current state of audio processing."""
    current_gain_db: float = 0.0
    peak_detector_state: float = 0.0
    rms_detector_state: float = 0.0
    gate_envelope: float = 0.0
    limiter_envelope: float = 0.0
    agc_envelope: float = 0.0
    samples_processed: int = 0


class AudioLevelManager(BaseService):
    """
    Audio Level Management Service for virtual microphone consistency.

    This service provides comprehensive audio level management including:
    - Real-time audio level monitoring (peak, RMS, LUFS)
    - Automatic Gain Control (AGC) for consistent levels
    - Communication app-specific audio profiles
    - Peak limiting and noise gating
    - Audio level normalization strategies
    - Performance metrics and monitoring
    """

    def __init__(self, sample_rate: int = 48000, channels: int = 1):
        """
        Initialize the Audio Level Manager.

        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
        """
        super().__init__("AudioLevelManager")
        self.logger = logging.getLogger(self.__class__.__name__)

        # Audio configuration
        self.sample_rate = sample_rate
        self.channels = channels

        # Audio processing state
        self.processing_state = AudioProcessingState()
        self.state_lock = Lock()

        # Communication app profiles
        self.app_profiles = self._initialize_app_profiles()
        self.current_profile = self.app_profiles[CommunicationApp.GENERIC]

        # Audio level monitoring
        self.level_history = deque(maxlen=100)  # Keep last 100 measurements
        self.monitoring_enabled = True

        # Processing parameters
        self.frame_size = 512  # Processing frame size in samples
        self.overlap_factor = 0.5  # Frame overlap factor

        # AGC parameters
        self.agc_attack_coeff = 0.0
        self.agc_release_coeff = 0.0
        self._update_agc_coefficients()

        # Limiter parameters
        self.limiter_attack_coeff = 0.0
        self.limiter_release_coeff = 0.0
        self._update_limiter_coefficients()

        # Noise gate parameters
        self.gate_attack_coeff = 0.0
        self.gate_release_coeff = 0.0
        self._update_gate_coefficients()

        # Level detection filters
        self.peak_detector_tc = 0.001  # Peak detector time constant
        self.rms_detector_tc = 0.050   # RMS detector time constant

        # Callbacks for level changes
        self.level_change_callbacks: List[Callable[[AudioLevelMetrics], None]] = []

        self.logger.info(f"Audio Level Manager initialized (SR: {sample_rate}Hz, Ch: {channels})")

    def _initialize_app_profiles(self) -> Dict[CommunicationApp, CommunicationAppProfile]:
        """
        Initialize communication application profiles with optimized settings.

        Returns:
            Dict[CommunicationApp, CommunicationAppProfile]: App-specific profiles
        """
        profiles = {}

        # Discord - Gaming focused, allows higher levels
        profiles[CommunicationApp.DISCORD] = CommunicationAppProfile(
            app_name=CommunicationApp.DISCORD,
            target_peak_db=-6.0,
            target_rms_db=-16.0,
            target_lufs=-16.0,
            max_gain_db=15.0,
            attack_time_ms=3.0,
            release_time_ms=30.0,
            noise_gate_threshold_db=-45.0,
            limiter_threshold_db=-3.0
        )

        # Zoom - Business focused, conservative levels
        profiles[CommunicationApp.ZOOM] = CommunicationAppProfile(
            app_name=CommunicationApp.ZOOM,
            target_peak_db=-9.0,
            target_rms_db=-20.0,
            target_lufs=-20.0,
            max_gain_db=10.0,
            attack_time_ms=10.0,
            release_time_ms=100.0,
            noise_gate_threshold_db=-40.0,
            limiter_threshold_db=-6.0
        )

        # Microsoft Teams - Corporate focused
        profiles[CommunicationApp.TEAMS] = CommunicationAppProfile(
            app_name=CommunicationApp.TEAMS,
            target_peak_db=-9.0,
            target_rms_db=-20.0,
            target_lufs=-18.0,
            max_gain_db=8.0,
            attack_time_ms=8.0,
            release_time_ms=80.0,
            noise_gate_threshold_db=-42.0,
            limiter_threshold_db=-6.0
        )

        # Skype - General purpose
        profiles[CommunicationApp.SKYPE] = CommunicationAppProfile(
            app_name=CommunicationApp.SKYPE,
            target_peak_db=-8.0,
            target_rms_db=-18.0,
            target_lufs=-18.0,
            max_gain_db=12.0,
            attack_time_ms=5.0,
            release_time_ms=50.0,
            noise_gate_threshold_db=-38.0,
            limiter_threshold_db=-4.0
        )

        # Webex - Corporate focused
        profiles[CommunicationApp.WEBEX] = CommunicationAppProfile(
            app_name=CommunicationApp.WEBEX,
            target_peak_db=-10.0,
            target_rms_db=-22.0,
            target_lufs=-20.0,
            max_gain_db=8.0,
            attack_time_ms=12.0,
            release_time_ms=120.0,
            noise_gate_threshold_db=-40.0,
            limiter_threshold_db=-6.0
        )

        # Google Meet - Balanced approach
        profiles[CommunicationApp.GOOGLE_MEET] = CommunicationAppProfile(
            app_name=CommunicationApp.GOOGLE_MEET,
            target_peak_db=-8.0,
            target_rms_db=-19.0,
            target_lufs=-19.0,
            max_gain_db=10.0,
            attack_time_ms=6.0,
            release_time_ms=60.0,
            noise_gate_threshold_db=-40.0,
            limiter_threshold_db=-5.0
        )

        # Slack - Casual communication
        profiles[CommunicationApp.SLACK] = CommunicationAppProfile(
            app_name=CommunicationApp.SLACK,
            target_peak_db=-7.0,
            target_rms_db=-17.0,
            target_lufs=-17.0,
            max_gain_db=12.0,
            attack_time_ms=4.0,
            release_time_ms=40.0,
            noise_gate_threshold_db=-42.0,
            limiter_threshold_db=-4.0
        )

        # Generic - Safe defaults for unknown apps
        profiles[CommunicationApp.GENERIC] = CommunicationAppProfile(
            app_name=CommunicationApp.GENERIC,
            target_peak_db=-6.0,
            target_rms_db=-18.0,
            target_lufs=-18.0,
            max_gain_db=12.0,
            attack_time_ms=5.0,
            release_time_ms=50.0,
            noise_gate_threshold_db=-40.0,
            limiter_threshold_db=-3.0
        )

        return profiles

    def set_communication_app(self, app: CommunicationApp) -> None:
        """
        Set the target communication application profile.

        Args:
            app: Communication application to optimize for
        """
        try:
            if app in self.app_profiles:
                self.current_profile = self.app_profiles[app]
                self._update_processing_coefficients()
                self.logger.info(f"Set communication app profile: {app.value}")
            else:
                self.logger.warning(f"Unknown communication app: {app.value}, using generic profile")
                self.current_profile = self.app_profiles[CommunicationApp.GENERIC]

        except Exception as e:
            self.logger.error(f"Failed to set communication app profile: {e}")

    def _update_processing_coefficients(self) -> None:
        """Update all processing coefficients based on current profile."""
        self._update_agc_coefficients()
        self._update_limiter_coefficients()
        self._update_gate_coefficients()

    def _update_agc_coefficients(self) -> None:
        """Update AGC attack and release coefficients."""
        profile = self.current_profile

        # Convert time constants to coefficients
        attack_samples = (profile.attack_time_ms / 1000.0) * self.sample_rate
        release_samples = (profile.release_time_ms / 1000.0) * self.sample_rate

        self.agc_attack_coeff = math.exp(-1.0 / max(attack_samples, 1.0))
        self.agc_release_coeff = math.exp(-1.0 / max(release_samples, 1.0))

    def _update_limiter_coefficients(self) -> None:
        """Update limiter attack and release coefficients."""
        # Limiter uses faster time constants
        attack_samples = 0.001 * self.sample_rate  # 1ms attack
        release_samples = 0.010 * self.sample_rate  # 10ms release

        self.limiter_attack_coeff = math.exp(-1.0 / max(attack_samples, 1.0))
        self.limiter_release_coeff = math.exp(-1.0 / max(release_samples, 1.0))

    def _update_gate_coefficients(self) -> None:
        """Update noise gate attack and release coefficients."""
        # Gate uses moderate time constants
        attack_samples = 0.005 * self.sample_rate  # 5ms attack
        release_samples = 0.050 * self.sample_rate  # 50ms release

        self.gate_attack_coeff = math.exp(-1.0 / max(attack_samples, 1.0))
        self.gate_release_coeff = math.exp(-1.0 / max(release_samples, 1.0))

    def process_audio_frame(self, audio_data) -> any:
        """
        Process an audio frame with level management.

        Args:
            audio_data: Input audio data (numpy array if available, list otherwise)

        Returns:
            Processed audio data (same type as input)
        """
        try:
            if not NUMPY_AVAILABLE:
                # Fallback implementation without numpy
                self.logger.warning("NumPy not available, audio processing disabled")
                return audio_data

            if audio_data.size == 0:
                return audio_data

            with self.state_lock:
                # Ensure correct data type and shape
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)

                # Handle different input shapes
                if len(audio_data.shape) == 1:
                    audio_data = audio_data.reshape(-1, 1)
                elif len(audio_data.shape) == 2 and audio_data.shape[1] != self.channels:
                    # Convert to target channel count
                    if self.channels == 1 and audio_data.shape[1] > 1:
                        # Convert to mono
                        audio_data = np.mean(audio_data, axis=1, keepdims=True)
                    elif self.channels == 2 and audio_data.shape[1] == 1:
                        # Convert to stereo
                        audio_data = np.tile(audio_data, (1, 2))

                # Process each channel
                processed_audio = np.zeros_like(audio_data)

                for channel in range(audio_data.shape[1]):
                    channel_data = audio_data[:, channel]
                    processed_channel = self._process_channel(channel_data)
                    processed_audio[:, channel] = processed_channel

                # Update processing statistics
                self.processing_state.samples_processed += audio_data.shape[0]

                # Calculate and store level metrics
                if self.monitoring_enabled:
                    metrics = self._calculate_level_metrics(audio_data, processed_audio)
                    self.level_history.append(metrics)

                    # Notify callbacks
                    for callback in self.level_change_callbacks:
                        try:
                            callback(metrics)
                        except Exception as e:
                            self.logger.warning(f"Level change callback error: {e}")

                return processed_audio

        except Exception as e:
            self.logger.error(f"Error processing audio frame: {e}")
            return audio_data  # Return original data on error

    def _process_channel(self, channel_data) -> any:
        """
        Process a single audio channel.

        Args:
            channel_data: Single channel audio data

        Returns:
            Processed channel data
        """
        if not NUMPY_AVAILABLE:
            return channel_data
        profile = self.current_profile
        state = self.processing_state

        # Apply noise gate
        if profile.noise_gate_threshold_db > -60.0:
            channel_data = self._apply_noise_gate(channel_data)

        # Apply normalization based on mode
        if profile.normalization_mode == NormalizationMode.AGC:
            channel_data = self._apply_agc(channel_data)
        elif profile.normalization_mode == NormalizationMode.PEAK_NORMALIZE:
            channel_data = self._apply_peak_normalization(channel_data)
        elif profile.normalization_mode == NormalizationMode.RMS_NORMALIZE:
            channel_data = self._apply_rms_normalization(channel_data)
        elif profile.normalization_mode == NormalizationMode.LUFS_NORMALIZE:
            channel_data = self._apply_lufs_normalization(channel_data)

        # Apply limiter
        if profile.enable_limiter:
            channel_data = self._apply_limiter(channel_data)

        return channel_data

    def _apply_noise_gate(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply noise gate to reduce background noise."""
        profile = self.current_profile
        state = self.processing_state

        threshold_linear = self._db_to_linear(profile.noise_gate_threshold_db)

        for i, sample in enumerate(audio_data):
            level = abs(sample)

            # Update gate envelope
            if level > threshold_linear:
                state.gate_envelope = level + self.gate_attack_coeff * (state.gate_envelope - level)
            else:
                state.gate_envelope = level + self.gate_release_coeff * (state.gate_envelope - level)

            # Apply gate
            gate_gain = 1.0 if state.gate_envelope > threshold_linear else 0.0
            audio_data[i] = sample * gate_gain

        return audio_data

    def _apply_agc(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply automatic gain control."""
        profile = self.current_profile
        state = self.processing_state

        target_level = self._db_to_linear(profile.target_rms_db)

        # Calculate RMS level over the frame
        rms_level = np.sqrt(np.mean(audio_data ** 2))

        if rms_level > 0:
            # Calculate desired gain
            desired_gain_linear = target_level / rms_level
            desired_gain_db = self._linear_to_db(desired_gain_linear)

            # Limit gain range
            desired_gain_db = np.clip(desired_gain_db, profile.min_gain_db, profile.max_gain_db)
            desired_gain_linear = self._db_to_linear(desired_gain_db)

            # Smooth gain changes
            if desired_gain_linear > state.current_gain_db:
                coeff = self.agc_attack_coeff
            else:
                coeff = self.agc_release_coeff

            state.current_gain_db = desired_gain_linear + coeff * (state.current_gain_db - desired_gain_linear)

            # Apply gain
            audio_data = audio_data * state.current_gain_db

        return audio_data

    def _apply_peak_normalization(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply peak normalization."""
        profile = self.current_profile

        peak_level = np.max(np.abs(audio_data))
        target_peak = self._db_to_linear(profile.target_peak_db)

        if peak_level > 0:
            gain = target_peak / peak_level
            audio_data = audio_data * gain

        return audio_data

    def _apply_rms_normalization(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply RMS normalization."""
        profile = self.current_profile

        rms_level = np.sqrt(np.mean(audio_data ** 2))
        target_rms = self._db_to_linear(profile.target_rms_db)

        if rms_level > 0:
            gain = target_rms / rms_level
            gain_db = self._linear_to_db(gain)
            gain_db = np.clip(gain_db, profile.min_gain_db, profile.max_gain_db)
            gain = self._db_to_linear(gain_db)
            audio_data = audio_data * gain

        return audio_data

    def _apply_lufs_normalization(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply LUFS (loudness) normalization."""
        # Simplified LUFS implementation
        # In a full implementation, this would include K-weighting filter
        profile = self.current_profile

        # Use RMS as approximation for LUFS for simplicity
        rms_level = np.sqrt(np.mean(audio_data ** 2))
        target_lufs = self._db_to_linear(profile.target_lufs)

        if rms_level > 0:
            gain = target_lufs / rms_level
            gain_db = self._linear_to_db(gain)
            gain_db = np.clip(gain_db, profile.min_gain_db, profile.max_gain_db)
            gain = self._db_to_linear(gain_db)
            audio_data = audio_data * gain

        return audio_data

    def _apply_limiter(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply peak limiter."""
        profile = self.current_profile
        state = self.processing_state

        threshold_linear = self._db_to_linear(profile.limiter_threshold_db)

        for i, sample in enumerate(audio_data):
            level = abs(sample)

            # Update limiter envelope
            if level > threshold_linear:
                state.limiter_envelope = level + self.limiter_attack_coeff * (state.limiter_envelope - level)
            else:
                state.limiter_envelope = level + self.limiter_release_coeff * (state.limiter_envelope - level)

            # Calculate gain reduction
            if state.limiter_envelope > threshold_linear:
                gain_reduction = threshold_linear / state.limiter_envelope
            else:
                gain_reduction = 1.0

            # Apply limiting
            audio_data[i] = sample * gain_reduction

        return audio_data

    def _calculate_level_metrics(self, input_audio: np.ndarray, output_audio: np.ndarray) -> AudioLevelMetrics:
        """Calculate comprehensive audio level metrics."""
        try:
            # Calculate peak levels
            input_peak = np.max(np.abs(input_audio))
            output_peak = np.max(np.abs(output_audio))

            # Calculate RMS levels
            input_rms = np.sqrt(np.mean(input_audio ** 2))
            output_rms = np.sqrt(np.mean(output_audio ** 2))

            # Convert to dB (with safety check for zero values)
            peak_db = self._linear_to_db(output_peak) if output_peak > 0 else -100.0
            rms_db = self._linear_to_db(output_rms) if output_rms > 0 else -100.0

            # Calculate dynamic range
            dynamic_range_db = peak_db - rms_db if rms_db > -100.0 else 0.0

            # Calculate gain reduction
            if input_rms > 0 and output_rms > 0:
                gain_reduction_db = self._linear_to_db(output_rms / input_rms)
            else:
                gain_reduction_db = 0.0

            # Simple LUFS approximation (would need proper K-weighting in production)
            lufs = rms_db - 0.691  # Rough approximation

            return AudioLevelMetrics(
                peak_db=peak_db,
                rms_db=rms_db,
                lufs=lufs,
                dynamic_range_db=dynamic_range_db,
                average_level_db=rms_db,
                gain_reduction_db=gain_reduction_db,
                sample_count=input_audio.shape[0],
                timestamp=time.time()
            )

        except Exception as e:
            self.logger.error(f"Error calculating level metrics: {e}")
            return AudioLevelMetrics(
                peak_db=-100.0, rms_db=-100.0, lufs=-100.0,
                dynamic_range_db=0.0, average_level_db=-100.0,
                gain_reduction_db=0.0, sample_count=0, timestamp=time.time()
            )

    def get_current_levels(self) -> Optional[AudioLevelMetrics]:
        """
        Get the most recent audio level metrics.

        Returns:
            Optional[AudioLevelMetrics]: Latest level measurements or None
        """
        if self.level_history:
            return self.level_history[-1]
        return None

    def get_level_statistics(self, window_seconds: float = 10.0) -> Dict[str, float]:
        """
        Get audio level statistics over a time window.

        Args:
            window_seconds: Time window for statistics calculation

        Returns:
            Dict[str, float]: Level statistics
        """
        try:
            current_time = time.time()
            window_start = current_time - window_seconds

            # Filter measurements within time window
            recent_metrics = [
                m for m in self.level_history
                if m.timestamp >= window_start
            ]

            if not recent_metrics:
                return {
                    'avg_peak_db': -100.0,
                    'avg_rms_db': -100.0,
                    'max_peak_db': -100.0,
                    'min_peak_db': -100.0,
                    'avg_gain_reduction_db': 0.0,
                    'sample_count': 0
                }

            peak_levels = [m.peak_db for m in recent_metrics]
            rms_levels = [m.rms_db for m in recent_metrics]
            gain_reductions = [m.gain_reduction_db for m in recent_metrics]

            return {
                'avg_peak_db': np.mean(peak_levels),
                'avg_rms_db': np.mean(rms_levels),
                'max_peak_db': np.max(peak_levels),
                'min_peak_db': np.min(peak_levels),
                'avg_gain_reduction_db': np.mean(gain_reductions),
                'sample_count': len(recent_metrics)
            }

        except Exception as e:
            self.logger.error(f"Error calculating level statistics: {e}")
            return {}

    def add_level_change_callback(self, callback: Callable[[AudioLevelMetrics], None]) -> None:
        """
        Add a callback for level change notifications.

        Args:
            callback: Function to call when levels change
        """
        if callback not in self.level_change_callbacks:
            self.level_change_callbacks.append(callback)

    def remove_level_change_callback(self, callback: Callable[[AudioLevelMetrics], None]) -> None:
        """
        Remove a level change callback.

        Args:
            callback: Callback function to remove
        """
        if callback in self.level_change_callbacks:
            self.level_change_callbacks.remove(callback)

    def reset_processing_state(self) -> None:
        """Reset audio processing state."""
        with self.state_lock:
            self.processing_state = AudioProcessingState()
            self.level_history.clear()
            self.logger.info("Audio processing state reset")

    @staticmethod
    def _db_to_linear(db: float) -> float:
        """Convert dB to linear scale."""
        return 10.0 ** (db / 20.0)

    @staticmethod
    def _linear_to_db(linear: float) -> float:
        """Convert linear to dB scale."""
        if linear <= 0.0:
            return -100.0  # Effectively negative infinity
        return 20.0 * math.log10(linear)

    async def start(self) -> bool:
        """Start the service."""
        try:
            await self._update_status(ServiceStatus.STARTING)

            # Reset processing state
            self.reset_processing_state()

            await self._update_status(ServiceStatus.RUNNING)
            self.logger.info("Audio Level Manager started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start Audio Level Manager: {e}")
            await self._update_status(ServiceStatus.ERROR)
            return False

    async def stop(self) -> bool:
        """Stop the service."""
        try:
            await self._update_status(ServiceStatus.STOPPING)

            # Clear callbacks and history
            self.level_change_callbacks.clear()
            self.level_history.clear()

            await self._update_status(ServiceStatus.STOPPED)
            self.logger.info("Audio Level Manager stopped")
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop Audio Level Manager: {e}")
            await self._update_status(ServiceStatus.ERROR)
            return False

    async def health_check(self) -> Tuple[bool, Optional[MyVoiceError]]:
        """Check service health."""
        try:
            # Check if processing state is valid
            with self.state_lock:
                if self.processing_state is None:
                    raise RuntimeError("Processing state is invalid")

                # Check if coefficients are valid
                if (self.agc_attack_coeff <= 0 or self.agc_release_coeff <= 0 or
                    self.limiter_attack_coeff <= 0 or self.limiter_release_coeff <= 0):
                    raise RuntimeError("Processing coefficients are invalid")

            return True, None

        except Exception as e:
            return False, MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="AUDIO_LEVEL_MANAGER_HEALTH_FAILED",
                user_message="Audio level management service health check failed",
                technical_details=str(e),
                suggested_action="Restart the audio level management service"
            )