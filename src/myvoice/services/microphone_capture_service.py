"""
Microphone Capture Service Module

This module provides the MicrophoneCaptureService for real-time microphone input
capture, enabling audio mixing with TTS output for virtual microphone routing.

The service uses a dedicated PyAudio instance with callback-based capture and
thread-safe queue buffering for low-latency audio retrieval.
"""

import asyncio
import logging
import queue
import threading
import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    pyaudio = None

import numpy as np

from myvoice.models.audio_device import AudioDevice, DeviceType
from myvoice.models.error import MyVoiceError, ErrorSeverity
from myvoice.models.app_settings import AppSettings
from myvoice.services.core.base_service import BaseService, ServiceStatus


@dataclass
class MicrophoneCaptureConfig:
    """Configuration for microphone capture."""
    # Audio parameters
    sample_rate: int = 48000      # 48kHz for compatibility with virtual mic
    channels: int = 1             # Mono capture
    sample_width: int = 2         # 16-bit (paInt16)
    chunk_size: int = 1024        # Frames per buffer (~21ms at 48kHz)

    # Buffer configuration
    max_buffer_chunks: int = 10   # Max chunks in queue before dropping old (reduced for low latency)

    # Quality settings
    timeout_seconds: float = 5.0  # Timeout for capture operations


@dataclass
class CaptureStatistics:
    """Statistics for microphone capture monitoring."""
    chunks_captured: int = 0
    chunks_dropped: int = 0
    total_bytes_captured: int = 0
    capture_start_time: Optional[datetime] = None
    last_chunk_time: Optional[datetime] = None
    buffer_overflows: int = 0

    def reset(self):
        """Reset all statistics."""
        self.chunks_captured = 0
        self.chunks_dropped = 0
        self.total_bytes_captured = 0
        self.capture_start_time = None
        self.last_chunk_time = None
        self.buffer_overflows = 0


class MicrophoneCaptureService(BaseService):
    """
    Microphone Capture Service

    Captures real-time audio from a physical microphone device using PyAudio
    with callback-based streaming. Audio chunks are buffered in a thread-safe
    queue for retrieval by the audio mixing service.

    Key Features:
    - Dedicated PyAudio instance (no resource sharing)
    - Callback-based capture for low latency
    - Thread-safe queue buffer
    - Device enumeration and validation
    - Configurable sample rate, channels, buffer size
    - Capture statistics for monitoring

    Usage:
        service = MicrophoneCaptureService()
        await service.start()

        # Start capturing from device
        devices = await service.enumerate_input_devices()
        await service.start_capture(devices[0])

        # Retrieve audio chunks
        while capturing:
            chunk = service.get_audio_chunk()
            if chunk:
                # Process chunk...

        await service.stop_capture()
        await service.stop()
    """

    def __init__(
        self,
        app_settings: Optional[AppSettings] = None,
        config: Optional[MicrophoneCaptureConfig] = None
    ):
        """
        Initialize the Microphone Capture Service.

        Args:
            app_settings: Application settings for device preferences
            config: Optional capture configuration (uses defaults if not provided)
        """
        super().__init__("MicrophoneCaptureService")

        self.app_settings = app_settings
        self.config = config or MicrophoneCaptureConfig()
        self.logger = logging.getLogger(__name__)

        # Dedicated PyAudio instance - no sharing with other services
        self._pyaudio: Optional[pyaudio.PyAudio] = None

        # Capture stream
        self._stream: Optional[Any] = None
        self._stream_lock = threading.Lock()

        # Audio buffer (thread-safe queue)
        self._audio_buffer: queue.Queue = queue.Queue(maxsize=self.config.max_buffer_chunks)

        # Capture state
        self._is_initialized = False
        self._is_capturing = False
        self._current_device: Optional[AudioDevice] = None

        # Statistics
        self._stats = CaptureStatistics()

        # Volume control (0.0 to 1.0)
        self._volume: float = 1.0
        self._muted: bool = False

        # Callbacks
        self._on_capture_error: Optional[Callable[[str], None]] = None
        self._on_device_lost: Optional[Callable[[AudioDevice], None]] = None

        self.logger.info("MicrophoneCaptureService initialized")

    # =========================================================================
    # BaseService Implementation
    # =========================================================================

    async def start(self) -> bool:
        """
        Start the microphone capture service.

        Returns:
            bool: True if service started successfully
        """
        return await self.initialize()

    async def stop(self) -> bool:
        """
        Stop the microphone capture service.

        Returns:
            bool: True if service stopped successfully
        """
        return await self.shutdown()

    async def health_check(self) -> tuple[bool, Optional[MyVoiceError]]:
        """
        Check microphone capture service health.

        Returns:
            tuple[bool, Optional[MyVoiceError]]: (is_healthy, error_if_any)
        """
        try:
            if not self._is_initialized:
                return False, MyVoiceError(
                    severity=ErrorSeverity.WARNING,
                    code="SERVICE_NOT_INITIALIZED",
                    user_message="Microphone capture service is not initialized"
                )

            if not PYAUDIO_AVAILABLE:
                return False, MyVoiceError(
                    severity=ErrorSeverity.CRITICAL,
                    code="PYAUDIO_UNAVAILABLE",
                    user_message="PyAudio library is not available"
                )

            # Check if we have any input devices
            devices = await self.enumerate_input_devices()
            if not devices:
                return False, MyVoiceError(
                    severity=ErrorSeverity.WARNING,
                    code="NO_INPUT_DEVICES",
                    user_message="No microphone devices found"
                )

            return True, None

        except Exception as e:
            return False, MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="HEALTH_CHECK_FAILED",
                user_message="Health check failed",
                technical_details=str(e)
            )

    # =========================================================================
    # Initialization and Shutdown
    # =========================================================================

    async def initialize(self) -> bool:
        """
        Initialize the microphone capture service.

        Returns:
            bool: True if initialization successful
        """
        try:
            if not PYAUDIO_AVAILABLE:
                raise MyVoiceError(
                    severity=ErrorSeverity.CRITICAL,
                    code="PYAUDIO_MISSING",
                    user_message="PyAudio library is not available",
                    technical_details="PyAudio required for microphone capture"
                )

            # Initialize dedicated PyAudio instance
            self._pyaudio = pyaudio.PyAudio()

            self._is_initialized = True
            self.status = ServiceStatus.RUNNING
            self.logger.info("MicrophoneCaptureService initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize MicrophoneCaptureService: {e}")
            self.status = ServiceStatus.ERROR
            return False

    async def shutdown(self) -> bool:
        """
        Shutdown the microphone capture service gracefully.

        Returns:
            bool: True if shutdown successful
        """
        try:
            self.logger.info("Shutting down MicrophoneCaptureService")

            # Stop any active capture
            if self._is_capturing:
                await self.stop_capture()

            # Terminate PyAudio instance
            if self._pyaudio:
                self._pyaudio.terminate()
                self._pyaudio = None

            # Clear buffer
            self._clear_buffer()

            self._is_initialized = False
            self.status = ServiceStatus.STOPPED
            self.logger.info("MicrophoneCaptureService shutdown complete")
            return True

        except Exception as e:
            self.logger.error(f"Error during MicrophoneCaptureService shutdown: {e}")
            return False

    # =========================================================================
    # Device Enumeration
    # =========================================================================

    async def enumerate_input_devices(self) -> List[AudioDevice]:
        """
        Enumerate available microphone input devices.

        Filters for physical input devices, excluding virtual devices
        like VB-Cable Output or Voicemeeter outputs.

        This method works even when the service is not started - it creates
        a temporary PyAudio instance if needed for enumeration.

        Returns:
            List[AudioDevice]: Available microphone devices
        """
        input_devices = []

        # Use existing PyAudio instance or create temporary one for enumeration
        pa = self._pyaudio
        temp_pa = False
        if not pa:
            try:
                pa = pyaudio.PyAudio()
                temp_pa = True
            except Exception as e:
                self.logger.error(f"Failed to create PyAudio for enumeration: {e}")
                return input_devices

        try:
            device_count = pa.get_device_count()
            default_input_index = pa.get_default_input_device_info().get('index', -1)

            for i in range(device_count):
                try:
                    device_info = pa.get_device_info_by_index(i)

                    # Check if device has input channels
                    max_input_channels = device_info.get('maxInputChannels', 0)
                    if max_input_channels <= 0:
                        continue

                    device_name = device_info.get('name', f"Device {i}")

                    # Skip virtual device outputs (we want physical mics)
                    if self._is_virtual_output_device(device_name):
                        continue

                    # Create AudioDevice
                    device = AudioDevice(
                        device_id=f"pyaudio_{i}",
                        name=device_name,
                        device_type=DeviceType.INPUT,
                        is_default=(i == default_input_index),
                        is_available=True
                    )

                    input_devices.append(device)

                except Exception as device_error:
                    self.logger.warning(f"Error checking device {i}: {device_error}")
                    continue

            self.logger.info(f"Found {len(input_devices)} microphone input devices")
            return input_devices

        except Exception as e:
            self.logger.error(f"Error enumerating input devices: {e}")
            return input_devices

        finally:
            # Clean up temporary PyAudio instance
            if temp_pa and pa:
                try:
                    pa.terminate()
                except Exception:
                    pass

    def _is_virtual_output_device(self, device_name: str) -> bool:
        """
        Check if device is a virtual output (should be excluded from mic list).

        We want to exclude virtual device OUTPUTS like "CABLE Output" because
        those are for receiving audio, not capturing from a physical mic.

        Args:
            device_name: Name of the device

        Returns:
            bool: True if device should be excluded
        """
        name_lower = device_name.lower()

        # Exclude virtual device outputs
        exclude_patterns = [
            'cable output',       # VB-Cable output side
            'voicemeeter out',    # Voicemeeter outputs
            'voicemeeter aux',    # Voicemeeter AUX
            'voicemeeter vaio',   # Voicemeeter VAIO
            'virtual audio',      # Generic virtual
            'what u hear',        # Stereo mix / loopback
            'stereo mix',         # Windows stereo mix
            'wave out',           # Wave out mix
        ]

        return any(pattern in name_lower for pattern in exclude_patterns)

    async def get_default_input_device(self) -> Optional[AudioDevice]:
        """
        Get the system default input device.

        Returns:
            Optional[AudioDevice]: Default input device, or None if not found
        """
        devices = await self.enumerate_input_devices()
        for device in devices:
            if device.is_default:
                return device

        # Return first available if no default
        return devices[0] if devices else None

    # =========================================================================
    # Capture Control
    # =========================================================================

    async def start_capture(self, device: Optional[AudioDevice] = None) -> bool:
        """
        Start capturing audio from the specified microphone.

        Args:
            device: Target microphone device (uses default if None)

        Returns:
            bool: True if capture started successfully
        """
        if not self._is_initialized:
            self.logger.error("Service not initialized")
            return False

        if self._is_capturing:
            self.logger.warning("Already capturing, stopping previous capture")
            await self.stop_capture()

        try:
            # Get device to use
            if device is None:
                device = await self.get_default_input_device()
                if device is None:
                    self.logger.error("No microphone device available")
                    return False

            # Extract device index
            device_index = self._get_device_index(device)
            if device_index < 0:
                self.logger.error(f"Invalid device index for {device.name}")
                return False

            # Validate device
            if not self._validate_input_device(device_index):
                self.logger.error(f"Device validation failed for index {device_index}")
                return False

            # Clear buffer before starting
            self._clear_buffer()
            self._stats.reset()

            with self._stream_lock:
                # Open input stream with callback
                self._stream = self._pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=self.config.channels,
                    rate=self.config.sample_rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=self.config.chunk_size,
                    stream_callback=self._audio_callback
                )

                self._stream.start_stream()

            self._is_capturing = True
            self._current_device = device
            self._stats.capture_start_time = datetime.now()

            self.logger.info(f"Started capturing from: {device.name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start capture: {e}")
            self._emit_capture_error(str(e))
            return False

    async def stop_capture(self) -> bool:
        """
        Stop the current capture session.

        Returns:
            bool: True if capture stopped successfully
        """
        if not self._is_capturing:
            return True

        try:
            with self._stream_lock:
                if self._stream:
                    try:
                        self._stream.stop_stream()
                        self._stream.close()
                    except Exception as stream_error:
                        self.logger.warning(f"Error closing stream: {stream_error}")
                    finally:
                        self._stream = None

            self._is_capturing = False

            self.logger.info(f"Stopped capturing. Stats: {self._stats.chunks_captured} chunks, "
                           f"{self._stats.buffer_overflows} overflows")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping capture: {e}")
            return False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback for audio capture.

        This runs in a separate thread - must be fast and thread-safe.

        Args:
            in_data: Captured audio bytes
            frame_count: Number of frames
            time_info: Timing information
            status: Stream status flags

        Returns:
            tuple: (None, pyaudio.paContinue)
        """
        if status:
            self.logger.warning(f"Capture stream status: {status}")

        try:
            # Apply volume and mute
            if self._muted:
                # Return silence
                processed_data = bytes(len(in_data))
            elif self._volume != 1.0:
                # Apply volume
                audio_array = np.frombuffer(in_data, dtype=np.int16)
                audio_array = (audio_array * self._volume).astype(np.int16)
                processed_data = audio_array.tobytes()
            else:
                processed_data = in_data

            # Try to add to buffer (non-blocking)
            try:
                self._audio_buffer.put_nowait(processed_data)
                self._stats.chunks_captured += 1
                self._stats.total_bytes_captured += len(processed_data)
                self._stats.last_chunk_time = datetime.now()
            except queue.Full:
                # Buffer full - drop oldest chunk
                try:
                    self._audio_buffer.get_nowait()
                    self._audio_buffer.put_nowait(processed_data)
                    self._stats.buffer_overflows += 1
                except queue.Empty:
                    pass

        except Exception as e:
            # Don't log in callback - too slow
            self._stats.chunks_dropped += 1

        return (None, pyaudio.paContinue)

    # =========================================================================
    # Audio Retrieval
    # =========================================================================

    def get_audio_chunk(self) -> Optional[bytes]:
        """
        Get the next audio chunk from the buffer (non-blocking).

        Returns:
            Optional[bytes]: Audio chunk, or None if buffer empty
        """
        try:
            return self._audio_buffer.get_nowait()
        except queue.Empty:
            return None

    def get_audio_chunk_blocking(self, timeout: float = 0.1) -> Optional[bytes]:
        """
        Get the next audio chunk from the buffer (blocking with timeout).

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Optional[bytes]: Audio chunk, or None if timeout
        """
        try:
            return self._audio_buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_all_available_chunks(self) -> List[bytes]:
        """
        Get all currently available audio chunks.

        Returns:
            List[bytes]: List of audio chunks
        """
        chunks = []
        while True:
            chunk = self.get_audio_chunk()
            if chunk is None:
                break
            chunks.append(chunk)
        return chunks

    def get_buffer_level(self) -> int:
        """
        Get current buffer fill level.

        Returns:
            int: Number of chunks in buffer
        """
        return self._audio_buffer.qsize()

    def _clear_buffer(self):
        """Clear all chunks from the buffer."""
        while True:
            try:
                self._audio_buffer.get_nowait()
            except queue.Empty:
                break

    # =========================================================================
    # Volume Control
    # =========================================================================

    def set_volume(self, volume: float):
        """
        Set the capture volume.

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self._volume = max(0.0, min(1.0, volume))
        self.logger.debug(f"Microphone volume set to {self._volume:.2f}")

    def get_volume(self) -> float:
        """Get the current capture volume."""
        return self._volume

    def set_muted(self, muted: bool):
        """
        Set the mute state.

        Args:
            muted: True to mute, False to unmute
        """
        self._muted = muted
        self.logger.debug(f"Microphone muted: {muted}")

    def is_muted(self) -> bool:
        """Check if microphone is muted."""
        return self._muted

    def toggle_mute(self) -> bool:
        """
        Toggle mute state.

        Returns:
            bool: New mute state
        """
        self._muted = not self._muted
        self.logger.debug(f"Microphone mute toggled: {self._muted}")
        return self._muted

    # =========================================================================
    # Device Validation
    # =========================================================================

    def _get_device_index(self, device: AudioDevice) -> int:
        """
        Extract PyAudio device index from AudioDevice.

        Args:
            device: AudioDevice instance

        Returns:
            int: Device index, or -1 if invalid
        """
        try:
            device_id = device.device_id
            if device_id.startswith("pyaudio_"):
                return int(device_id.split("_")[1])
            return int(device_id)
        except (ValueError, IndexError):
            return -1

    def _validate_input_device(self, device_index: int) -> bool:
        """
        Validate that an input device is available.

        Args:
            device_index: Device index to validate

        Returns:
            bool: True if device is valid
        """
        if not self._pyaudio:
            return False

        try:
            device_info = self._pyaudio.get_device_info_by_index(device_index)

            # Must have input channels
            if device_info.get('maxInputChannels', 0) <= 0:
                self.logger.warning(f"Device {device_index} has no input channels")
                return False

            # Log device info
            self.logger.info(f"Validated input device {device_index}:")
            self.logger.info(f"  Name: {device_info.get('name')}")
            self.logger.info(f"  Input Channels: {device_info.get('maxInputChannels')}")
            self.logger.info(f"  Default Sample Rate: {device_info.get('defaultSampleRate')}Hz")

            return True

        except Exception as e:
            self.logger.error(f"Device validation failed for index {device_index}: {e}")
            return False

    # =========================================================================
    # Callbacks
    # =========================================================================

    def set_capture_error_callback(self, callback: Callable[[str], None]):
        """Set callback for capture errors."""
        self._on_capture_error = callback

    def set_device_lost_callback(self, callback: Callable[[AudioDevice], None]):
        """Set callback for device disconnection."""
        self._on_device_lost = callback

    def _emit_capture_error(self, error: str):
        """Emit capture error callback."""
        if self._on_capture_error:
            try:
                self._on_capture_error(error)
            except Exception as e:
                self.logger.error(f"Error in capture_error callback: {e}")

    def _emit_device_lost(self, device: AudioDevice):
        """Emit device lost callback."""
        if self._on_device_lost:
            try:
                self._on_device_lost(device)
            except Exception as e:
                self.logger.error(f"Error in device_lost callback: {e}")

    # =========================================================================
    # Status and Statistics
    # =========================================================================

    def is_capturing(self) -> bool:
        """Check if currently capturing."""
        return self._is_capturing

    def get_current_device(self) -> Optional[AudioDevice]:
        """Get the current capture device."""
        return self._current_device

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get capture statistics.

        Returns:
            Dict[str, Any]: Capture statistics
        """
        return {
            "chunks_captured": self._stats.chunks_captured,
            "chunks_dropped": self._stats.chunks_dropped,
            "total_bytes_captured": self._stats.total_bytes_captured,
            "buffer_overflows": self._stats.buffer_overflows,
            "buffer_level": self.get_buffer_level(),
            "capture_start_time": self._stats.capture_start_time.isoformat() if self._stats.capture_start_time else None,
            "last_chunk_time": self._stats.last_chunk_time.isoformat() if self._stats.last_chunk_time else None,
            "is_capturing": self._is_capturing,
            "current_device": self._current_device.name if self._current_device else None,
            "volume": self._volume,
            "muted": self._muted,
        }

    async def get_health(self) -> Dict[str, Any]:
        """
        Get health status of the microphone capture service.

        Returns:
            Dict[str, Any]: Health information
        """
        return {
            "service_name": self.service_name,
            "status": self.status.value,
            "initialized": self._is_initialized,
            "pyaudio_available": PYAUDIO_AVAILABLE,
            "is_capturing": self._is_capturing,
            "current_device": self._current_device.name if self._current_device else None,
            "buffer_level": self.get_buffer_level(),
            "statistics": self.get_statistics(),
        }
