"""
Monitor Audio Service Module

This module provides the MonitorAudioService (Audio Service 1) for handling
monitor speaker audio routing exclusively, eliminating resource competition
with virtual microphone services.

Implements Story 1.4 specifications with dedicated PyAudio instance optimized
for monitor speaker devices.
"""

import asyncio
import logging
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

from myvoice.models.audio_device import AudioDevice, DeviceType
from myvoice.models.audio_playback_task import AudioPlaybackTask, PlaybackStatus
from myvoice.models.error import MyVoiceError, ErrorSeverity
from myvoice.models.app_settings import AppSettings
from myvoice.services.integrations.windows_audio_client import WindowsAudioClient, DeviceChangeEvent
from myvoice.services.core.base_service import BaseService, ServiceStatus


@dataclass
class MonitorAudioConfig:
    """Configuration for monitor speaker audio playback."""
    # Optimized parameters for monitor speakers
    sample_rate: int = 48000  # 48kHz for high quality
    channels: int = 2         # Stereo for monitor speakers
    sample_width: int = 2     # 16-bit
    chunk_size: int = 1024    # Frames per buffer
    buffer_size: int = 4096   # Processing chunk in bytes (1024 samples * 2 channels * 2 bytes)

    # Device configuration
    default_device_index: int = 11  # VoiceMeeter Input from working tests

    # Quality settings
    enable_quality_preservation: bool = True
    timeout_seconds: float = 30.0


@dataclass
class MonitorPlaybackTask:
    """Task for tracking monitor speaker playback operations."""
    playback_id: str
    audio_data: bytes
    device: Optional[AudioDevice]
    status: PlaybackStatus
    volume: float = 1.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None

    def mark_started(self):
        """Mark task as started."""
        self.status = PlaybackStatus.PLAYING
        self.start_time = datetime.now()

    def mark_completed(self):
        """Mark task as completed."""
        self.status = PlaybackStatus.COMPLETED
        self.end_time = datetime.now()

    def mark_failed(self, error: str):
        """Mark task as failed with error message."""
        self.status = PlaybackStatus.FAILED
        self.end_time = datetime.now()
        self.error_message = error


class MonitorAudioService(BaseService):
    """
    Monitor Audio Service (Audio Service 1)

    Dedicated service for routing TTS audio to monitor speakers exclusively,
    completely independent from virtual microphone routing to eliminate resource
    competition and distortion issues.

    Key Features:
    - Dedicated PyAudio instance for monitor speakers only
    - Optimized parameters for monitor audio quality
    - Async interface compatible with TTSService integration
    - No resource sharing with virtual microphone services
    - Monitor device validation and error handling
    """

    def __init__(self, app_settings: Optional[AppSettings] = None):
        """
        Initialize the Monitor Audio Service.

        Args:
            app_settings: Application settings for device preferences
        """
        super().__init__("MonitorAudioService")

        self.app_settings = app_settings
        self.config = MonitorAudioConfig()
        self.logger = logging.getLogger(__name__)

        # Dedicated PyAudio instance - no sharing with other services
        self._pyaudio: Optional[pyaudio.PyAudio] = None

        # Monitor device management
        self.current_monitor_device: Optional[AudioDevice] = None
        self.windows_audio_client: Optional[WindowsAudioClient] = None

        # Task tracking
        self._active_tasks: Dict[str, MonitorPlaybackTask] = {}
        self._playback_threads: Dict[str, threading.Thread] = {}

        # Service state
        self._is_initialized = False
        self._task_counter = 0

        self.logger.info("MonitorAudioService initialized")

    async def start(self) -> bool:
        """Start the monitor audio service."""
        return await self.initialize()

    async def stop(self) -> bool:
        """Stop the monitor audio service."""
        return await self.shutdown()

    async def health_check(self) -> tuple[bool, Optional[MyVoiceError]]:
        """Check monitor audio service health."""
        try:
            if not self._is_initialized:
                return False, MyVoiceError(
                    severity=ErrorSeverity.WARNING,
                    code="SERVICE_NOT_INITIALIZED",
                    user_message="Monitor audio service is not initialized"
                )

            if not PYAUDIO_AVAILABLE:
                return False, MyVoiceError(
                    severity=ErrorSeverity.CRITICAL,
                    code="PYAUDIO_UNAVAILABLE",
                    user_message="PyAudio library is not available"
                )

            # Check if current monitor device is available (if configured)
            # OR if any monitor device is available (for fallback)
            if self.current_monitor_device:
                # Validate configured device
                device_id = self.current_monitor_device.device_id
                if device_id.startswith("pyaudio_"):
                    device_index = int(device_id.split("_")[1])
                else:
                    device_index = int(device_id)

                if not self._validate_monitor_device(device_index):
                    return False, MyVoiceError(
                        severity=ErrorSeverity.WARNING,
                        code="CONFIGURED_DEVICE_UNAVAILABLE",
                        user_message=f"Configured monitor device '{self.current_monitor_device.name}' is not available"
                    )
            else:
                # No configured device - check if any monitor devices are available
                monitor_devices = await self.enumerate_monitor_devices()
                if not monitor_devices:
                    return False, MyVoiceError(
                        severity=ErrorSeverity.WARNING,
                        code="NO_MONITOR_DEVICES",
                        user_message="No monitor audio devices available"
                    )

            return True, None

        except Exception as e:
            return False, MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="HEALTH_CHECK_FAILED",
                user_message="Health check failed",
                technical_details=str(e)
            )

    async def initialize(self) -> bool:
        """Initialize the monitor audio service."""
        try:
            if not PYAUDIO_AVAILABLE:
                raise MyVoiceError(
                    severity=ErrorSeverity.CRITICAL,
                    code="PYAUDIO_MISSING",
                    user_message="PyAudio library is not available",
                    technical_details="PyAudio required for monitor audio service"
                )

            # Initialize dedicated PyAudio instance
            self._pyaudio = pyaudio.PyAudio()

            # Initialize Windows audio client for device management
            self.windows_audio_client = WindowsAudioClient()

            # Log comprehensive audio system info for debugging
            self.windows_audio_client.log_audio_system_info()

            # Validate default monitor device is available
            if not self._validate_monitor_device(self.config.default_device_index):
                self.logger.warning(f"Default monitor device {self.config.default_device_index} not available")

            # Load monitor device from settings
            await self._load_monitor_device_from_settings()

            self._is_initialized = True
            self.status = ServiceStatus.RUNNING
            self.logger.info("MonitorAudioService initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize MonitorAudioService: {e}")
            self.status = ServiceStatus.FAILED
            return False

    async def shutdown(self) -> bool:
        """Shutdown the monitor audio service gracefully."""
        try:
            self.logger.info("Shutting down MonitorAudioService")

            # Stop all active playback tasks
            for task_id in list(self._active_tasks.keys()):
                await self.stop_monitor_playback(task_id)

            # Terminate PyAudio instance
            if self._pyaudio:
                self._pyaudio.terminate()
                self._pyaudio = None

            self._is_initialized = False
            self.status = ServiceStatus.STOPPED
            self.logger.info("MonitorAudioService shutdown complete")
            return True

        except Exception as e:
            self.logger.error(f"Error during MonitorAudioService shutdown: {e}")
            return False

    async def play_monitor_audio(self,
                               audio_data: bytes,
                               device: Optional[AudioDevice] = None,
                               volume: float = 1.0) -> Optional[MonitorPlaybackTask]:
        """
        Route audio to monitor speaker device.

        Args:
            audio_data: WAV audio data to play
            device: Target monitor device (uses current if None)
            volume: Volume level (0.0 to 1.0)

        Returns:
            MonitorPlaybackTask: Task tracking the playback operation
        """
        if not self._is_initialized:
            self.logger.error("MonitorAudioService not initialized")
            return None

        try:
            # Use current device if none specified
            target_device = device or self.current_monitor_device

            if not target_device:
                # Try to get default monitor device
                monitor_devices = await self.enumerate_monitor_devices()
                if monitor_devices:
                    target_device = monitor_devices[0]
                else:
                    self.logger.error("No monitor devices available")
                    return None

            # Update current monitor device if a specific device was provided
            if device:
                self.current_monitor_device = device
                self.logger.debug(f"Updated current monitor device to: {device.name}")

            # Generate unique task ID
            self._task_counter += 1
            task_id = f"monitor_{self._task_counter}_{int(time.time())}"

            # Create playback task
            task = MonitorPlaybackTask(
                playback_id=task_id,
                audio_data=audio_data,
                device=target_device,
                status=PlaybackStatus.PENDING,
                volume=volume
            )

            self._active_tasks[task_id] = task

            # Start playback in separate thread
            self._start_monitor_playback_thread(task)

            self.logger.info(f"Started monitor audio playback task {task_id}")
            return task

        except Exception as e:
            self.logger.error(f"Failed to start monitor audio playback: {e}")
            return None

    async def stop_monitor_playback(self, task_id: str) -> bool:
        """Stop an active monitor audio playback."""
        try:
            if task_id not in self._active_tasks:
                self.logger.warning(f"Task {task_id} not found")
                return False

            task = self._active_tasks[task_id]
            task.status = PlaybackStatus.STOPPED

            # Wait for thread to finish
            if task_id in self._playback_threads:
                thread = self._playback_threads[task_id]
                thread.join(timeout=2.0)

            # Cleanup
            self._active_tasks.pop(task_id, None)
            self._playback_threads.pop(task_id, None)

            self.logger.info(f"Stopped monitor audio playback {task_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping monitor playback {task_id}: {e}")
            return False

    async def enumerate_monitor_devices(self) -> List[AudioDevice]:
        """Enumerate available monitor speaker devices."""
        monitor_devices = []

        if not self._pyaudio:
            return monitor_devices

        try:
            device_count = self._pyaudio.get_device_count()

            for i in range(device_count):
                try:
                    device_info = self._pyaudio.get_device_info_by_index(i)

                    # Check if device supports output
                    if device_info.get('maxOutputChannels', 0) > 0:
                        device_name = device_info.get('name', f"Device {i}")

                        device = AudioDevice(
                            device_id=str(i),
                            name=device_name,
                            device_type=DeviceType.OUTPUT,
                            is_default=False,
                            is_available=True
                        )
                        monitor_devices.append(device)

                except Exception as device_error:
                    self.logger.warning(f"Error checking device {i}: {device_error}")
                    continue

            self.logger.info(f"Found {len(monitor_devices)} monitor devices")
            return monitor_devices

        except Exception as e:
            self.logger.error(f"Error enumerating monitor devices: {e}")
            return monitor_devices

    def _start_monitor_playback_thread(self, task: MonitorPlaybackTask) -> None:
        """Start monitor audio playback in dedicated thread."""
        try:
            playback_thread = threading.Thread(
                target=self._monitor_playback_worker,
                args=(task,),
                daemon=True,
                name=f"Monitor-{task.playback_id[:8]}"
            )

            self._playback_threads[task.playback_id] = playback_thread
            playback_thread.start()

            self.logger.info(f"Started monitor audio thread for {task.playback_id}")

        except Exception as e:
            self.logger.error(f"Failed to start monitor audio thread: {e}")
            task.mark_failed(f"Failed to start thread: {str(e)}")

    def _monitor_playback_worker(self, task: MonitorPlaybackTask) -> None:
        """Worker method for monitor audio playback using optimized parameters."""
        stream = None

        try:
            self.logger.info(f"Monitor audio worker started for {task.playback_id}")
            task.mark_started()

            # Use device from task or default
            # Extract integer index from device_id (format: "pyaudio_12" -> 12)
            if task.device:
                device_id_str = task.device.device_id
                if device_id_str.startswith("pyaudio_"):
                    device_index = int(device_id_str.split("_")[1])
                else:
                    device_index = int(device_id_str)
            else:
                device_index = self.config.default_device_index

            # Parse WAV header to extract audio parameters and data
            try:
                import struct

                # Parse WAV header
                if len(task.audio_data) < 44:
                    raise ValueError("WAV data too short")

                # Find data chunk
                offset = 12  # Skip RIFF header
                data_offset = None
                sample_rate = None
                channels = None
                bits_per_sample = None

                while offset < len(task.audio_data) - 8:
                    chunk_id = task.audio_data[offset:offset + 4]
                    chunk_size = struct.unpack('<I', task.audio_data[offset + 4:offset + 8])[0]

                    if chunk_id == b'fmt ':
                        # Parse format chunk
                        fmt_data = task.audio_data[offset + 8:offset + 8 + chunk_size]
                        if len(fmt_data) >= 16:
                            audio_format, channels, sample_rate, _, _, bits_per_sample = struct.unpack('<HHIIHH', fmt_data[:16])
                            self.logger.info(f"WAV format: {channels}ch, {sample_rate}Hz, {bits_per_sample}bit")

                    elif chunk_id == b'data':
                        data_offset = offset + 8
                        data_size = chunk_size
                        self.logger.info(f"Found data chunk at offset {data_offset}, size {data_size}")
                        break

                    offset += 8 + chunk_size

                if data_offset is None:
                    raise ValueError("No data chunk found in WAV file")

                raw_audio_data = task.audio_data[data_offset:data_offset + data_size]
                self.logger.info(f"Extracted {len(raw_audio_data)} bytes of raw audio data")

            except Exception as extract_error:
                self.logger.error(f"Failed to parse WAV data: {extract_error}")
                task.mark_failed(f"WAV parsing failed: {str(extract_error)}")
                return

            # Validate device exists
            if not self._validate_monitor_device(device_index):
                task.mark_failed(f"Monitor device {device_index} not available")
                return

            # Open stream with audio parameters from WAV file
            try:
                # Use parsed audio parameters or fallback to defaults
                stream_sample_rate = sample_rate or self.config.sample_rate
                stream_channels = channels or self.config.channels

                # Convert bits per sample to PyAudio format
                if bits_per_sample == 16:
                    audio_format = pyaudio.paInt16
                elif bits_per_sample == 24:
                    audio_format = pyaudio.paInt24
                elif bits_per_sample == 32:
                    audio_format = pyaudio.paInt32
                else:
                    audio_format = pyaudio.paInt16  # Default fallback
                    self.logger.warning(f"Unsupported bit depth {bits_per_sample}, using 16-bit")

                # Validate format is supported by device before attempting to open stream
                if self.windows_audio_client:
                    # Log host API being used
                    host_api_name = self.windows_audio_client.get_device_host_api_name(device_index)
                    self.logger.info(f"Device {device_index} uses host API: {host_api_name}")

                    # Check for WASAPI availability
                    wasapi_index = self.windows_audio_client.get_wasapi_host_api_index()
                    if wasapi_index is not None:
                        self.logger.info(f"WASAPI is available at host API index {wasapi_index}")
                    else:
                        self.logger.warning("WASAPI not available - using default host API (may cause silent audio on Windows 11)")

                    # Validate format
                    is_supported, format_error = self.windows_audio_client.validate_stream_format(
                        device_index=device_index,
                        sample_rate=stream_sample_rate,
                        channels=stream_channels,
                        format_type=audio_format,
                        is_input=False
                    )

                    if not is_supported:
                        self.logger.warning(f"Device {device_index} doesn't support requested format: {format_error}")
                        # Format validation failed - this will be handled by fallback logic below

                # Log device capabilities before opening
                if self.windows_audio_client:
                    try:
                        device_info = self._pyaudio.get_device_info_by_index(device_index)
                        self.logger.debug(f"Device capabilities - MaxOutputChannels: {device_info.get('maxOutputChannels')}, "
                                        f"DefaultSampleRate: {device_info.get('defaultSampleRate')}, "
                                        f"DefaultLowOutputLatency: {device_info.get('defaultLowOutputLatency')}")
                    except Exception as e:
                        self.logger.debug(f"Could not get device capabilities: {e}")

                self.logger.info(f"Opening monitor stream: {stream_channels}ch, {stream_sample_rate}Hz, {bits_per_sample}bit on device {device_index}")

                stream = self._pyaudio.open(
                    format=audio_format,
                    channels=stream_channels,
                    rate=stream_sample_rate,
                    output=True,                         # Output stream
                    output_device_index=device_index,
                    frames_per_buffer=self.config.chunk_size  # 1024 frames
                )

                # Log stream state after opening
                try:
                    is_active = stream.is_active()
                    is_stopped = stream.is_stopped()
                    self.logger.info(f"Successfully opened monitor audio stream for device {device_index} - "
                                   f"Active: {is_active}, Stopped: {is_stopped}")
                except Exception as e:
                    self.logger.debug(f"Could not check stream state: {e}")
                    self.logger.info(f"Successfully opened monitor audio stream for device {device_index}")

            except (OSError, IOError) as stream_error:
                # Specific handling for device access errors (error -9996, etc.)
                self.logger.error(f"Failed to open monitor stream on device {device_index}: {stream_error}")

                # Try fallback to default output device
                try:
                    default_device_info = self._pyaudio.get_default_output_device_info()
                    default_device_index = default_device_info.get('index', -1)

                    if default_device_index >= 0 and default_device_index != device_index:
                        self.logger.warning(f"Attempting fallback to default output device {default_device_index}")

                        stream = self._pyaudio.open(
                            format=audio_format,
                            channels=stream_channels,
                            rate=stream_sample_rate,
                            output=True,
                            output_device_index=default_device_index,
                            frames_per_buffer=self.config.chunk_size
                        )

                        self.logger.info(f"Successfully opened stream on default device {default_device_index}")
                        device_index = default_device_index  # Update for error messages
                    else:
                        raise stream_error  # Re-raise if no fallback possible

                except Exception as fallback_error:
                    self.logger.error(f"Fallback to default device also failed: {fallback_error}")
                    task.mark_failed(f"Stream open failed on device {device_index}: {str(stream_error)}. Fallback also failed: {str(fallback_error)}")

                    # Raise MyVoiceError for consistent error handling
                    raise MyVoiceError(
                        severity=ErrorSeverity.ERROR,
                        code="STREAM_OPEN_FAILED",
                        user_message="Could not open audio device for playback",
                        technical_details=f"Primary device {device_index} failed: {str(stream_error)}. Default device fallback failed: {str(fallback_error)}",
                        suggested_action="Try selecting a different audio device in settings or check if the device is being used exclusively by another application"
                    )

            except Exception as stream_error:
                # Catch any other unexpected errors
                self.logger.error(f"Unexpected error opening monitor audio stream: {stream_error}")
                task.mark_failed(f"Stream open failed: {str(stream_error)}")
                return

            # Apply volume adjustment
            volume_adjusted_data = self._apply_volume(raw_audio_data, task.volume)

            # Log audio data characteristics for debugging
            try:
                duration_seconds = len(raw_audio_data) / (stream_sample_rate * stream_channels * (bits_per_sample // 8))

                # Calculate volume peak from raw data
                import numpy as np
                audio_array = np.frombuffer(raw_audio_data[:min(len(raw_audio_data), 4096)], dtype=np.int16)
                volume_peak = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0
                volume_peak_percent = (volume_peak / 32768.0) * 100

                self.logger.info(f"Audio data - Size: {len(raw_audio_data)} bytes, "
                               f"Duration: {duration_seconds:.2f}s, "
                               f"Peak volume: {volume_peak_percent:.1f}%")

                if volume_peak == 0:
                    self.logger.warning("Audio data has ZERO peak volume - audio may be silent!")

            except Exception as e:
                self.logger.debug(f"Could not analyze audio data: {e}")

            # Stream audio data
            chunk_size = self.config.buffer_size  # 4096 bytes
            playback_start_time = time.time()

            offset = 0
            chunks_written = 0
            total_chunks = (len(volume_adjusted_data) + chunk_size - 1) // chunk_size

            while offset < len(volume_adjusted_data):
                # Check if task should be stopped
                if task.status.value in ['failed', 'stopped']:
                    self.logger.debug("Monitor audio playback stopped")
                    break

                # Check for timeout
                elapsed_time = time.time() - playback_start_time
                if elapsed_time > self.config.timeout_seconds:
                    task.mark_failed(f"Playback timed out after {self.config.timeout_seconds} seconds")
                    break

                try:
                    chunk = volume_adjusted_data[offset:offset + chunk_size]
                    if len(chunk) == 0:
                        break

                    # Write chunk to monitor device
                    stream.write(chunk)
                    offset += chunk_size
                    chunks_written += 1

                    # Log progress every 20 chunks for debugging
                    if chunks_written % 20 == 0:
                        progress_percent = (chunks_written / total_chunks) * 100
                        self.logger.debug(f"Playback progress: {chunks_written}/{total_chunks} chunks ({progress_percent:.1f}%)")

                except Exception as chunk_error:
                    self.logger.error(f"Error writing monitor audio chunk {chunks_written}: {chunk_error}")
                    task.mark_failed(f"Chunk write failed: {str(chunk_error)}")
                    break

            # Log final playback stats
            if chunks_written > 0:
                self.logger.debug(f"Playback complete: {chunks_written} chunks written in {elapsed_time:.2f}s")

            # Mark as completed if not already failed
            if task.status.value == 'playing':
                task.mark_completed()
                self.logger.info(f"Monitor audio playback completed for {task.playback_id}")

        except Exception as e:
            self.logger.error(f"Monitor audio worker error for {task.playback_id}: {e}")
            task.mark_failed(f"Playback error: {str(e)}")

        finally:
            # Cleanup stream
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception as cleanup_error:
                    self.logger.warning(f"Error cleaning up stream: {cleanup_error}")

    def _apply_volume(self, audio_data: bytes, volume: float) -> bytes:
        """Apply volume adjustment to audio data."""
        try:
            if volume == 1.0:
                return audio_data

            # Convert to numpy array for volume adjustment
            import numpy as np

            # Convert bytes to int16 array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Apply volume (clamp to prevent overflow)
            volume_adjusted = np.clip(audio_array * volume, -32768, 32767).astype(np.int16)

            return volume_adjusted.tobytes()

        except Exception as e:
            self.logger.warning(f"Volume adjustment failed: {e}, using original audio")
            return audio_data

    def _validate_monitor_device(self, device_index: int) -> bool:
        """Validate that a monitor device is available."""
        if not self._pyaudio:
            return False

        try:
            device_info = self._pyaudio.get_device_info_by_index(device_index)

            # Check if device supports output channels
            max_output_channels = device_info.get('maxOutputChannels', 0)
            if max_output_channels == 0:
                return False

            self.logger.debug(f"Monitor device {device_index} validated: {device_info.get('name')}")
            return True

        except Exception as e:
            self.logger.warning(f"Monitor device {device_index} validation failed: {e}")
            return False

    async def _load_monitor_device_from_settings(self) -> bool:
        """Load monitor device from application settings."""
        try:
            if not self.app_settings:
                return False

            # Try to load from settings
            if hasattr(self.app_settings, 'monitor_device_id'):
                device_id = self.app_settings.monitor_device_id
                if device_id:
                    monitor_devices = await self.enumerate_monitor_devices()
                    for device in monitor_devices:
                        if device.device_id == device_id:
                            self.current_monitor_device = device
                            self.logger.info(f"Loaded monitor device from settings: {device.name}")
                            return True

            return False

        except Exception as e:
            self.logger.error(f"Error loading monitor device from settings: {e}")
            return False

    async def get_health(self) -> Dict[str, Any]:
        """Get health status of the monitor audio service."""
        return {
            "service_name": self.service_name,
            "status": self.status.value,
            "initialized": self._is_initialized,
            "pyaudio_available": PYAUDIO_AVAILABLE,
            "active_tasks": len(self._active_tasks),
            "current_device": self.current_monitor_device.name if self.current_monitor_device else None,
            "default_device_available": self._validate_monitor_device(self.config.default_device_index) if self._pyaudio else False
        }
