"""
Audio Mixer Service Module

This module provides the AudioMixerService for real-time mixing of multiple
audio streams (microphone input + TTS output) with volume control,
clipping prevention, and sample rate conversion.
"""

import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from math import gcd

import numpy as np

try:
    from scipy.signal import resample_poly
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    resample_poly = None


@dataclass
class MixerConfig:
    """Configuration for audio mixing."""
    # Default volume levels
    default_mic_volume: float = 1.0
    default_tts_volume: float = 1.0

    # Limiter settings
    enable_limiter: bool = True
    limiter_threshold: float = 0.95   # Start limiting at 95% of max
    limiter_ratio: float = 4.0        # 4:1 compression above threshold

    # Normalization
    enable_normalization: bool = False
    target_peak: float = 0.9          # Target peak level for normalization


class AudioMixerService:
    """
    Audio Mixer Service

    Mixes multiple audio streams in real-time with volume control,
    clipping prevention, and optional limiting/normalization.

    The mixer operates on int16 audio samples, using float32 internally
    for precision during mixing operations.

    Usage:
        mixer = AudioMixerService()

        # Mix microphone and TTS audio
        mixed = mixer.mix_streams(mic_audio, tts_audio, mic_vol=0.8, tts_vol=1.0)

        # Mix with automatic gain control
        mixed = mixer.mix_with_ducking(mic_audio, tts_audio)
    """

    def __init__(self, config: Optional[MixerConfig] = None):
        """
        Initialize the Audio Mixer Service.

        Args:
            config: Mixer configuration (uses defaults if None)
        """
        self.config = config or MixerConfig()
        self.logger = logging.getLogger(__name__)

        # Volume state
        self._mic_volume = self.config.default_mic_volume
        self._tts_volume = self.config.default_tts_volume

        self.logger.info("AudioMixerService initialized")

    # =========================================================================
    # Core Mixing
    # =========================================================================

    def mix_streams(
        self,
        mic_audio: Optional[np.ndarray],
        tts_audio: Optional[np.ndarray],
        mic_volume: Optional[float] = None,
        tts_volume: Optional[float] = None
    ) -> np.ndarray:
        """
        Mix microphone and TTS audio streams.

        Args:
            mic_audio: Microphone audio (int16 numpy array), or None
            tts_audio: TTS audio (int16 numpy array), or None
            mic_volume: Microphone volume (0.0-1.0), uses instance default if None
            tts_volume: TTS volume (0.0-1.0), uses instance default if None

        Returns:
            np.ndarray: Mixed audio (int16)
        """
        # Use instance volumes if not specified
        mic_vol = mic_volume if mic_volume is not None else self._mic_volume
        tts_vol = tts_volume if tts_volume is not None else self._tts_volume

        # Handle None inputs
        if mic_audio is None and tts_audio is None:
            return np.array([], dtype=np.int16)

        if mic_audio is None:
            return self._apply_volume_int16(tts_audio, tts_vol)

        if tts_audio is None:
            return self._apply_volume_int16(mic_audio, mic_vol)

        # Ensure both are int16
        mic_audio = self._ensure_int16(mic_audio)
        tts_audio = self._ensure_int16(tts_audio)

        # Match lengths (pad shorter with silence)
        mic_audio, tts_audio = self._match_lengths(mic_audio, tts_audio)

        # Convert to float32 for mixing (-1.0 to 1.0 range)
        mic_float = mic_audio.astype(np.float32) / 32768.0
        tts_float = tts_audio.astype(np.float32) / 32768.0

        # Apply volumes
        mic_scaled = mic_float * mic_vol
        tts_scaled = tts_float * tts_vol

        # Mix (simple addition)
        mixed = mic_scaled + tts_scaled

        # Apply limiter if enabled
        if self.config.enable_limiter:
            mixed = self._apply_limiter(mixed)

        # Apply normalization if enabled
        if self.config.enable_normalization:
            mixed = self._apply_normalization(mixed)

        # Clip to valid range
        mixed = np.clip(mixed, -1.0, 1.0)

        # Convert back to int16
        return (mixed * 32767.0).astype(np.int16)

    def mix_streams_bytes(
        self,
        mic_audio: Optional[bytes],
        tts_audio: Optional[bytes],
        mic_volume: Optional[float] = None,
        tts_volume: Optional[float] = None
    ) -> bytes:
        """
        Mix audio streams from bytes input.

        Args:
            mic_audio: Microphone audio bytes (int16 PCM), or None
            tts_audio: TTS audio bytes (int16 PCM), or None
            mic_volume: Microphone volume (0.0-1.0)
            tts_volume: TTS volume (0.0-1.0)

        Returns:
            bytes: Mixed audio (int16 PCM)
        """
        # Convert bytes to numpy arrays
        mic_array = np.frombuffer(mic_audio, dtype=np.int16) if mic_audio else None
        tts_array = np.frombuffer(tts_audio, dtype=np.int16) if tts_audio else None

        # Mix
        mixed = self.mix_streams(mic_array, tts_array, mic_volume, tts_volume)

        # Convert back to bytes
        return mixed.tobytes()

    # =========================================================================
    # Advanced Mixing Modes
    # =========================================================================

    def mix_with_ducking(
        self,
        mic_audio: np.ndarray,
        tts_audio: np.ndarray,
        duck_amount: float = 0.3,
        mic_volume: Optional[float] = None,
        tts_volume: Optional[float] = None
    ) -> np.ndarray:
        """
        Mix with ducking - reduce TTS volume when mic is active.

        This creates a more natural mix where the TTS "steps back"
        when the user is speaking.

        Args:
            mic_audio: Microphone audio (int16)
            tts_audio: TTS audio (int16)
            duck_amount: How much to reduce TTS (0.0-1.0, 0.3 = 30% reduction)
            mic_volume: Microphone volume
            tts_volume: TTS volume

        Returns:
            np.ndarray: Mixed audio with ducking (int16)
        """
        mic_vol = mic_volume if mic_volume is not None else self._mic_volume
        tts_vol = tts_volume if tts_volume is not None else self._tts_volume

        # Detect mic activity (simple RMS threshold)
        mic_rms = self._calculate_rms(mic_audio)
        mic_active = mic_rms > 0.01  # ~-40dB threshold

        # Apply ducking to TTS if mic is active
        if mic_active:
            tts_vol = tts_vol * (1.0 - duck_amount)

        return self.mix_streams(mic_audio, tts_audio, mic_vol, tts_vol)

    def crossfade(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        crossfade_samples: int = 1024
    ) -> np.ndarray:
        """
        Crossfade between two audio segments.

        Args:
            audio_a: First audio segment (int16)
            audio_b: Second audio segment (int16)
            crossfade_samples: Number of samples for crossfade

        Returns:
            np.ndarray: Crossfaded audio (int16)
        """
        audio_a = self._ensure_int16(audio_a)
        audio_b = self._ensure_int16(audio_b)

        # Ensure we have enough samples
        crossfade_samples = min(crossfade_samples, len(audio_a), len(audio_b))

        if crossfade_samples <= 0:
            return np.concatenate([audio_a, audio_b])

        # Convert to float
        a_float = audio_a.astype(np.float32) / 32768.0
        b_float = audio_b.astype(np.float32) / 32768.0

        # Create crossfade region
        fade_out = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)

        # Apply crossfade
        a_end = a_float[-crossfade_samples:] * fade_out
        b_start = b_float[:crossfade_samples] * fade_in
        crossfade_region = a_end + b_start

        # Combine: a (without end) + crossfade + b (without start)
        result = np.concatenate([
            a_float[:-crossfade_samples],
            crossfade_region,
            b_float[crossfade_samples:]
        ])

        # Clip and convert back to int16
        result = np.clip(result, -1.0, 1.0)
        return (result * 32767.0).astype(np.int16)

    # =========================================================================
    # Volume Control
    # =========================================================================

    def set_mic_volume(self, volume: float):
        """Set the default microphone volume (0.0-1.0)."""
        self._mic_volume = max(0.0, min(1.0, volume))

    def get_mic_volume(self) -> float:
        """Get the current microphone volume."""
        return self._mic_volume

    def set_tts_volume(self, volume: float):
        """Set the default TTS volume (0.0-1.0)."""
        self._tts_volume = max(0.0, min(1.0, volume))

    def get_tts_volume(self) -> float:
        """Get the current TTS volume."""
        return self._tts_volume

    # =========================================================================
    # Audio Processing Utilities
    # =========================================================================

    def _ensure_int16(self, audio: np.ndarray) -> np.ndarray:
        """Ensure audio is int16 format."""
        if audio.dtype != np.int16:
            if audio.dtype == np.float32 or audio.dtype == np.float64:
                # Assume normalized float (-1.0 to 1.0)
                return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
            else:
                return audio.astype(np.int16)
        return audio

    def _match_lengths(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match audio array lengths by padding shorter with silence.

        Args:
            audio_a: First audio array
            audio_b: Second audio array

        Returns:
            Tuple of matched-length arrays
        """
        len_a, len_b = len(audio_a), len(audio_b)

        if len_a == len_b:
            return audio_a, audio_b

        if len_a < len_b:
            padding = np.zeros(len_b - len_a, dtype=audio_a.dtype)
            audio_a = np.concatenate([audio_a, padding])
        else:
            padding = np.zeros(len_a - len_b, dtype=audio_b.dtype)
            audio_b = np.concatenate([audio_b, padding])

        return audio_a, audio_b

    def _apply_volume_int16(self, audio: np.ndarray, volume: float) -> np.ndarray:
        """Apply volume to int16 audio."""
        if volume == 1.0:
            return audio

        audio_float = audio.astype(np.float32) / 32768.0
        audio_float = audio_float * volume
        audio_float = np.clip(audio_float, -1.0, 1.0)
        return (audio_float * 32767.0).astype(np.int16)

    def _apply_limiter(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply soft limiter to prevent harsh clipping.

        Uses a simple soft-knee compressor above the threshold.

        Args:
            audio: Audio in float32 format (-1.0 to 1.0)

        Returns:
            Limited audio
        """
        threshold = self.config.limiter_threshold
        ratio = self.config.limiter_ratio

        # Find samples above threshold
        abs_audio = np.abs(audio)
        above_threshold = abs_audio > threshold

        if not np.any(above_threshold):
            return audio

        # Apply compression above threshold
        # Output = threshold + (input - threshold) / ratio
        compressed = np.where(
            above_threshold,
            np.sign(audio) * (threshold + (abs_audio - threshold) / ratio),
            audio
        )

        return compressed

    def _apply_normalization(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to target peak level.

        Args:
            audio: Audio in float32 format

        Returns:
            Normalized audio
        """
        peak = np.max(np.abs(audio))

        if peak < 0.001:  # Essentially silence
            return audio

        target = self.config.target_peak
        gain = target / peak

        # Don't boost, only attenuate if needed
        if gain < 1.0:
            return audio * gain

        return audio

    def _calculate_rms(self, audio: np.ndarray) -> float:
        """
        Calculate RMS (root mean square) level of audio.

        Args:
            audio: Audio samples (int16 or float32)

        Returns:
            RMS level (0.0 to 1.0)
        """
        if len(audio) == 0:
            return 0.0

        # Convert to float if needed
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio

        return float(np.sqrt(np.mean(audio_float ** 2)))

    # =========================================================================
    # Analysis Utilities
    # =========================================================================

    def get_peak_level(self, audio: np.ndarray) -> float:
        """
        Get peak level of audio (0.0 to 1.0).

        Args:
            audio: Audio samples (int16 or float32)

        Returns:
            Peak level
        """
        if len(audio) == 0:
            return 0.0

        if audio.dtype == np.int16:
            return float(np.max(np.abs(audio))) / 32768.0
        else:
            return float(np.max(np.abs(audio)))

    def get_levels(self, audio: np.ndarray) -> dict:
        """
        Get audio level analysis.

        Args:
            audio: Audio samples

        Returns:
            Dict with peak, rms, and estimated dB levels
        """
        peak = self.get_peak_level(audio)
        rms = self._calculate_rms(audio)

        # Convert to dB (with floor at -60dB)
        peak_db = 20 * np.log10(max(peak, 0.001))
        rms_db = 20 * np.log10(max(rms, 0.001))

        return {
            "peak": peak,
            "rms": rms,
            "peak_db": peak_db,
            "rms_db": rms_db,
            "clipping": peak >= 1.0
        }

    # =========================================================================
    # Sample Rate Conversion
    # =========================================================================

    def resample(
        self,
        audio: np.ndarray,
        source_rate: int,
        target_rate: int
    ) -> np.ndarray:
        """
        Resample audio to a different sample rate.

        Uses scipy.signal.resample_poly for high-quality resampling with
        minimal latency. Falls back to linear interpolation if scipy unavailable.

        Args:
            audio: Input audio (int16 or float32)
            source_rate: Source sample rate (e.g., 44100)
            target_rate: Target sample rate (e.g., 48000)

        Returns:
            np.ndarray: Resampled audio (same dtype as input)
        """
        if source_rate == target_rate:
            return audio

        if len(audio) == 0:
            return audio

        original_dtype = audio.dtype
        was_int16 = original_dtype == np.int16

        # Convert to float for processing
        if was_int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)

        # Calculate resampling ratio
        # Find GCD to simplify ratio for resample_poly
        divisor = gcd(source_rate, target_rate)
        up = target_rate // divisor
        down = source_rate // divisor

        if SCIPY_AVAILABLE:
            # High-quality polyphase resampling
            resampled = resample_poly(audio_float, up, down)
        else:
            # Fallback: linear interpolation
            self.logger.warning("scipy not available, using linear interpolation")
            resampled = self._resample_linear(audio_float, source_rate, target_rate)

        # Convert back to original dtype
        if was_int16:
            resampled = np.clip(resampled, -1.0, 1.0)
            return (resampled * 32767.0).astype(np.int16)
        else:
            return resampled.astype(original_dtype)

    def resample_bytes(
        self,
        audio_bytes: bytes,
        source_rate: int,
        target_rate: int
    ) -> bytes:
        """
        Resample audio bytes to a different sample rate.

        Args:
            audio_bytes: Input audio as bytes (int16 PCM)
            source_rate: Source sample rate
            target_rate: Target sample rate

        Returns:
            bytes: Resampled audio (int16 PCM)
        """
        if source_rate == target_rate:
            return audio_bytes

        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        resampled = self.resample(audio, source_rate, target_rate)
        return resampled.tobytes()

    def _resample_linear(
        self,
        audio: np.ndarray,
        source_rate: int,
        target_rate: int
    ) -> np.ndarray:
        """
        Simple linear interpolation resampling (fallback).

        Args:
            audio: Input audio (float32)
            source_rate: Source sample rate
            target_rate: Target sample rate

        Returns:
            np.ndarray: Resampled audio (float32)
        """
        # Calculate output length
        ratio = target_rate / source_rate
        output_length = int(len(audio) * ratio)

        # Create interpolation indices
        indices = np.linspace(0, len(audio) - 1, output_length)

        # Integer and fractional parts
        int_indices = indices.astype(np.int32)
        frac = indices - int_indices

        # Clamp indices
        int_indices = np.clip(int_indices, 0, len(audio) - 2)

        # Linear interpolation
        resampled = audio[int_indices] * (1 - frac) + audio[int_indices + 1] * frac

        return resampled.astype(np.float32)

    def mix_with_resample(
        self,
        mic_audio: Optional[np.ndarray],
        tts_audio: Optional[np.ndarray],
        mic_rate: int = 48000,
        tts_rate: int = 24000,
        output_rate: int = 48000,
        mic_volume: Optional[float] = None,
        tts_volume: Optional[float] = None
    ) -> np.ndarray:
        """
        Mix audio streams with automatic sample rate conversion.

        Resamples both streams to the output rate before mixing.

        Args:
            mic_audio: Microphone audio (int16), or None
            tts_audio: TTS audio (int16), or None
            mic_rate: Microphone sample rate
            tts_rate: TTS sample rate
            output_rate: Target output sample rate
            mic_volume: Microphone volume (0.0-1.0)
            tts_volume: TTS volume (0.0-1.0)

        Returns:
            np.ndarray: Mixed audio at output_rate (int16)
        """
        # Resample mic if needed
        if mic_audio is not None and mic_rate != output_rate:
            mic_audio = self.resample(mic_audio, mic_rate, output_rate)

        # Resample TTS if needed
        if tts_audio is not None and tts_rate != output_rate:
            tts_audio = self.resample(tts_audio, tts_rate, output_rate)

        # Mix at output rate
        return self.mix_streams(mic_audio, tts_audio, mic_volume, tts_volume)

    @staticmethod
    def calculate_resampled_length(source_length: int, source_rate: int, target_rate: int) -> int:
        """
        Calculate the length of resampled audio.

        Args:
            source_length: Number of samples in source
            source_rate: Source sample rate
            target_rate: Target sample rate

        Returns:
            int: Expected number of samples after resampling
        """
        return int(source_length * target_rate / source_rate)
