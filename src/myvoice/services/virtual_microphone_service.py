"""
Virtual Microphone Service Module

This module provides the VirtualMicrophoneService (Audio Service 2) for handling
virtual microphone audio routing independently from monitor audio, eliminating
resource competition and distortion issues.

Implements Story 1.5 specifications with dedicated PyAudio instance and optimized
virtual microphone parameters.

Story 2.2: Added support for VB-Audio Cable & Voicemeeter detection, playback_complete
callbacks, and real-time streaming for synchronized playback with monitor (NFR16).
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

from myvoice.models.audio_device import AudioDevice, DeviceType, VirtualDeviceDriver
from myvoice.models.audio_playback_task import AudioPlaybackTask, PlaybackStatus
from myvoice.models.error import MyVoiceError, ErrorSeverity
from myvoice.models.app_settings import AppSettings
from myvoice.services.core.base_service import BaseService, ServiceStatus
from myvoice.utils.validation import validate_wav_data


@dataclass
class VirtualMicrophoneConfig:
    """Configuration for virtual microphone audio routing."""
    # Working parameters from successful PyAudio tests
    sample_rate: int = 48000  # 48kHz for communication app compatibility
    channels: int = 1         # Mono for microphone simulation
    sample_width: int = 2     # 16-bit
    chunk_size: int = 1024    # Frames per buffer (matching working test)
    buffer_size: int = 2048   # Processing chunk in bytes (1024 samples * 2 bytes)

    # Device configuration
    default_device_index: int = -1  # No hardcoded default - will auto-detect or use settings

    # Quality settings
    enable_quality_preservation: bool = True
    timeout_seconds: float = 10.0


@dataclass
class VirtualPlaybackTask:
    """Task for tracking virtual microphone playback operations."""
    playback_id: str
    audio_data: bytes
    device: Optional[AudioDevice]
    status: PlaybackStatus
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


class VirtualMicrophoneService(BaseService):
    """
    Virtual Microphone Service (Audio Service 2)

    Dedicated service for routing TTS audio to virtual microphone devices,
    completely independent from monitor audio routing to eliminate resource
    competition and distortion issues.

    Key Features:
    - Dedicated PyAudio instance for virtual microphone only
    - Optimized parameters from successful PyAudio tests
    - Async interface compatible with TTSService integration
    - No resource sharing with monitor audio service
    - Virtual device validation and error handling
    """

    def __init__(self, app_settings: Optional[AppSettings] = None):
        """
        Initialize the Virtual Microphone Service.

        Args:
            app_settings: Application settings for device preferences
        """
        super().__init__("VirtualMicrophoneService")

        self.app_settings = app_settings
        self.config = VirtualMicrophoneConfig()
        self.logger = logging.getLogger(__name__)

        # Dedicated PyAudio instance - no sharing with other services
        self._pyaudio: Optional[pyaudio.PyAudio] = None

        # Task tracking
        self._active_tasks: Dict[str, VirtualPlaybackTask] = {}
        self._playback_threads: Dict[str, threading.Thread] = {}

        # Service state
        self._is_initialized = False
        self._task_counter = 0

        # Callbacks (Story 2.2: playback_complete signal, synchronized with monitor)
        self._playback_complete_callback: Optional[Callable[[str], None]] = None
        self._playback_error_callback: Optional[Callable[[str, str], None]] = None

        # Streaming state (Story 2.2: real-time streaming to virtual mic - NFR16)
        self._streaming_stream: Optional[Any] = None
        self._streaming_lock = threading.Lock()

        # Current device tracking (for AudioCoordinator integration)
        self.current_virtual_device: Optional[AudioDevice] = None

        self.logger.info("VirtualMicrophoneService initialized")

    async def start(self) -> bool:
        """
        Start the virtual microphone service.

        Returns:
            bool: True if service started successfully
        """
        return await self.initialize()

    async def stop(self) -> bool:
        """
        Stop the virtual microphone service.

        Returns:
            bool: True if service stopped successfully
        """
        return await self.shutdown()

    async def health_check(self) -> tuple[bool, Optional[MyVoiceError]]:
        """
        Check virtual microphone service health.

        Returns:
            tuple[bool, Optional[MyVoiceError]]: (is_healthy, error_if_any)
        """
        try:
            if not self._is_initialized:
                return False, MyVoiceError(
                    severity=ErrorSeverity.WARNING,
                    code="SERVICE_NOT_INITIALIZED",
                    user_message="Virtual microphone service is not initialized"
                )

            if not PYAUDIO_AVAILABLE:
                return False, MyVoiceError(
                    severity=ErrorSeverity.CRITICAL,
                    code="PYAUDIO_UNAVAILABLE",
                    user_message="PyAudio library is not available"
                )

            # Don't validate default device if it's -1 (no default set)
            # Device will be provided during playback, so service is healthy if PyAudio is available
            # Only validate if a specific default device is configured
            if self.config.default_device_index != -1:
                if not self._validate_virtual_device(self.config.default_device_index):
                    return False, MyVoiceError(
                        severity=ErrorSeverity.WARNING,
                        code="DEFAULT_DEVICE_UNAVAILABLE",
                        user_message="Default virtual microphone device is not available"
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
        """
        Initialize the virtual microphone service.

        Returns:
            bool: True if initialization successful
        """
        try:
            if not PYAUDIO_AVAILABLE:
                raise MyVoiceError(
                    severity=ErrorSeverity.CRITICAL,
                    code="PYAUDIO_MISSING",
                    user_message="PyAudio library is not available",
                    technical_details="PyAudio required for virtual microphone service"
                )

            # Initialize dedicated PyAudio instance
            self._pyaudio = pyaudio.PyAudio()

            # Validate default virtual device is available
            if not self._validate_virtual_device(self.config.default_device_index):
                self.logger.warning(f"Default virtual device {self.config.default_device_index} not available")

            self._is_initialized = True
            self.status = ServiceStatus.RUNNING
            self.logger.info("VirtualMicrophoneService initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize VirtualMicrophoneService: {e}")
            self.status = ServiceStatus.ERROR
            return False

    async def shutdown(self) -> bool:
        """
        Shutdown the virtual microphone service gracefully.

        Returns:
            bool: True if shutdown successful
        """
        try:
            self.logger.info("Shutting down VirtualMicrophoneService")

            # Stop all active playback tasks
            for task_id in list(self._active_tasks.keys()):
                await self.stop_virtual_playback(task_id)

            # Terminate PyAudio instance
            if self._pyaudio:
                self._pyaudio.terminate()
                self._pyaudio = None

            self._is_initialized = False
            self.status = ServiceStatus.STOPPED
            self.logger.info("VirtualMicrophoneService shutdown complete")
            return True

        except Exception as e:
            self.logger.error(f"Error during VirtualMicrophoneService shutdown: {e}")
            return False

    async def play_virtual_microphone(self,
                                    audio_data: bytes,
                                    device: Optional[AudioDevice] = None,
                                    volume: float = 1.0) -> Optional[VirtualPlaybackTask]:
        """
        Route audio to virtual microphone device.

        Args:
            audio_data: WAV audio data to route
            device: Target virtual device (uses default if None)
            volume: Volume level (0.0 to 1.0)

        Returns:
            VirtualPlaybackTask: Task tracking the playback operation
        """
        if not self._is_initialized:
            self.logger.error("VirtualMicrophoneService not initialized")
            return None

        try:
            # Validate audio data
            if not validate_wav_data(audio_data):
                raise MyVoiceError(
                    severity=ErrorSeverity.ERROR,
                    code="INVALID_AUDIO_DATA",
                    user_message="Audio data is not valid WAV format",
                    technical_details="Virtual microphone requires valid WAV data"
                )

            # Generate unique task ID
            self._task_counter += 1
            task_id = f"virtual_mic_{self._task_counter}_{int(time.time())}"

            # Create playback task
            task = VirtualPlaybackTask(
                playback_id=task_id,
                audio_data=audio_data,
                device=device,
                status=PlaybackStatus.PENDING
            )

            self._active_tasks[task_id] = task

            # Start playback in separate thread
            self._start_virtual_playback_thread(task)

            self.logger.info(f"Started virtual microphone playback task {task_id}")
            return task

        except Exception as e:
            self.logger.error(f"Failed to start virtual microphone playback: {e}")
            return None

    async def stop_virtual_playback(self, task_id: str) -> bool:
        """
        Stop an active virtual microphone playback.

        Args:
            task_id: ID of the task to stop

        Returns:
            bool: True if stopped successfully
        """
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

                if thread.is_alive():
                    self.logger.warning(f"Thread for task {task_id} did not finish gracefully")

            # Cleanup
            self._active_tasks.pop(task_id, None)
            self._playback_threads.pop(task_id, None)

            self.logger.info(f"Stopped virtual microphone playback {task_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping virtual playback {task_id}: {e}")
            return False

    async def enumerate_virtual_devices(self) -> List[AudioDevice]:
        """
        Enumerate available virtual microphone devices.

        Returns:
            List[AudioDevice]: Available virtual devices
        """
        virtual_devices = []

        if not self._pyaudio:
            return virtual_devices

        try:
            device_count = self._pyaudio.get_device_count()

            for i in range(device_count):
                try:
                    device_info = self._pyaudio.get_device_info_by_index(i)

                    # Check if device supports output (we send audio TO virtual devices like VoiceMeeter Output)
                    if device_info.get('maxOutputChannels', 0) > 0:
                        device_name = device_info.get('name', f"Device {i}")

                        # Identify virtual devices by common patterns
                        if self._is_virtual_device(device_name):
                            device = AudioDevice(
                                device_id=str(i),
                                name=device_name,
                                device_type=DeviceType.VIRTUAL,
                                virtual_driver=self._detect_virtual_driver(device_name),
                                is_default=False,
                                is_available=True
                            )
                            virtual_devices.append(device)

                except Exception as device_error:
                    self.logger.warning(f"Error checking device {i}: {device_error}")
                    continue

            self.logger.info(f"Found {len(virtual_devices)} virtual microphone devices")
            return virtual_devices

        except Exception as e:
            self.logger.error(f"Error enumerating virtual devices: {e}")
            return virtual_devices

    def _start_virtual_playback_thread(self, task: VirtualPlaybackTask) -> None:
        """
        Start virtual microphone playback in dedicated thread.

        Args:
            task: Virtual playback task to execute
        """
        try:
            playback_thread = threading.Thread(
                target=self._virtual_playback_worker,
                args=(task,),
                daemon=True,
                name=f"VirtualMic-{task.playback_id[:8]}"
            )

            self._playback_threads[task.playback_id] = playback_thread
            playback_thread.start()

            self.logger.info(f"Started virtual microphone thread for {task.playback_id}")

        except Exception as e:
            self.logger.error(f"Failed to start virtual microphone thread: {e}")
            task.mark_failed(f"Failed to start thread: {str(e)}")

    def _virtual_playback_worker(self, task: VirtualPlaybackTask) -> None:
        """
        Worker method for virtual microphone playback using exact working parameters.

        Args:
            task: VirtualPlaybackTask to execute
        """
        stream = None

        try:
            self.logger.info(f"Virtual microphone worker started for {task.playback_id}")
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
                self.logger.warning("No device specified in task, using default")

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
            if not self._validate_virtual_device(device_index):
                task.mark_failed(f"Virtual device {device_index} not available")
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

                self.logger.info(f"Opening virtual microphone stream: {stream_channels}ch, {stream_sample_rate}Hz, {bits_per_sample}bit on device {device_index}")

                stream = self._pyaudio.open(
                    format=audio_format,
                    channels=stream_channels,
                    rate=stream_sample_rate,
                    output=True,                         # Output stream
                    output_device_index=device_index,
                    frames_per_buffer=self.config.chunk_size  # 1024 frames
                )

                self.logger.info(f"Successfully opened virtual microphone stream for device {device_index}")

            except (OSError, IOError) as stream_error:
                # Specific handling for device access errors (error -9996, etc.)
                error_str = str(stream_error)
                error_code = None

                # Extract PyAudio error code if present
                if '[Errno' in error_str:
                    try:
                        error_code = error_str.split('[Errno')[1].split(']')[0].strip()
                    except:
                        pass

                self.logger.error(f"Failed to open virtual microphone stream on device {device_index}")
                self.logger.error(f"  Error: {stream_error}")
                if error_code:
                    self.logger.error(f"  Error Code: {error_code}")

                # Get device name for better error messages
                device_name = "Unknown"
                try:
                    device_info = self._pyaudio.get_device_info_by_index(device_index)
                    device_name = device_info.get('name', 'Unknown')
                except:
                    pass

                # Determine specific error guidance based on error type
                error_guidance = []
                user_message = "Could not open virtual microphone device for playback"

                # Error -9996: Invalid device or device in use
                if '-9996' in error_str or 'Invalid device' in error_str:
                    self.logger.error("  Diagnosis: Invalid device or device in use by another application")
                    error_guidance.append("The virtual device may be in use by another application")
                    error_guidance.append("Try closing other applications that might be using the virtual audio device")
                    error_guidance.append("Restart your computer if VB-Cable was recently installed")
                    user_message = f"Virtual audio device '{device_name}' is not accessible"

                # Wrong device selected (CABLE Output instead of CABLE Input)
                elif 'cable output' in device_name.lower():
                    self.logger.error("  Diagnosis: Wrong VB-Cable device selected (Output instead of Input)")
                    error_guidance.append("IMPORTANT: You selected 'CABLE Output' but need 'CABLE Input' for virtual microphone")
                    error_guidance.append("In Settings, select 'CABLE Input (VB-Audio Virtual Cable)' instead")
                    user_message = "Wrong VB-Cable device selected"

                # Device not found or driver issue
                elif 'not found' in error_str.lower() or 'does not exist' in error_str.lower():
                    self.logger.error("  Diagnosis: Virtual audio device not found - driver may not be loaded")
                    error_guidance.append("Virtual audio driver may not be properly installed")
                    error_guidance.append("If you just installed VB-Cable, restart your computer")
                    error_guidance.append("Verify VB-Cable appears in Windows Sound settings")
                    user_message = "Virtual audio device not found"

                # Sample rate mismatch or format issue
                elif 'format' in error_str.lower() or 'rate' in error_str.lower():
                    self.logger.error("  Diagnosis: Audio format or sample rate mismatch")
                    error_guidance.append(f"Device may not support {stream_sample_rate}Hz at {bits_per_sample}-bit {stream_channels}ch")
                    error_guidance.append("Try selecting a different virtual device in Settings")
                    user_message = "Audio format not supported by device"

                # Generic error - provide general guidance
                else:
                    error_guidance.append("Check that the virtual audio device is properly installed")
                    error_guidance.append("Restart may be required if you recently installed the virtual audio driver")
                    error_guidance.append("Try selecting a different device in Settings")

                # Log all guidance
                for guidance in error_guidance:
                    self.logger.info(f"  Solution: {guidance}")

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

                        self.logger.info(f"Successfully opened virtual microphone stream on default device {default_device_index}")
                        device_index = default_device_index
                    else:
                        raise stream_error

                except Exception as fallback_error:
                    self.logger.error(f"Fallback to default device also failed: {fallback_error}")

                    # Combine guidance into suggested action
                    suggested_action = " • ".join(error_guidance) if error_guidance else "Try selecting a different virtual microphone device in settings"

                    task.mark_failed(f"{user_message}: {error_str}")

                    # Raise MyVoiceError with specific guidance
                    raise MyVoiceError(
                        severity=ErrorSeverity.ERROR,
                        code="STREAM_OPEN_FAILED",
                        user_message=user_message,
                        technical_details=f"Device {device_index} ({device_name}): {error_str}",
                        suggested_action=suggested_action
                    )

            except Exception as stream_error:
                # Catch any other unexpected errors
                self.logger.error(f"Unexpected error opening virtual microphone stream: {stream_error}")
                task.mark_failed(f"Stream open failed: {str(stream_error)}")
                return

            # Stream audio data
            chunk_size = self.config.buffer_size  # 2048 bytes (1024 samples * 2 bytes)
            playback_start_time = time.time()

            offset = 0
            while offset < len(raw_audio_data):
                # Check if task should be stopped
                if task.status.value in ['failed', 'stopped']:
                    self.logger.debug("Virtual microphone playback stopped")
                    break

                # Check for timeout
                elapsed_time = time.time() - playback_start_time
                if elapsed_time > self.config.timeout_seconds:
                    task.mark_failed(f"Playback timed out after {self.config.timeout_seconds} seconds")
                    break

                try:
                    chunk = raw_audio_data[offset:offset + chunk_size]
                    if len(chunk) == 0:
                        break

                    # Write chunk to virtual microphone device
                    stream.write(chunk)
                    offset += chunk_size

                except Exception as chunk_error:
                    self.logger.error(f"Error writing virtual microphone chunk: {chunk_error}")
                    task.mark_failed(f"Chunk write failed: {str(chunk_error)}")
                    break

            # Mark as completed if not already failed
            if task.status.value == 'playing':
                task.mark_completed()
                self.logger.info(f"Virtual microphone playback completed for {task.playback_id}")
                # Story 2.2: Emit playback_complete signal
                self._emit_playback_complete(task.playback_id)

        except Exception as e:
            self.logger.error(f"Virtual microphone worker error for {task.playback_id}: {e}")
            task.mark_failed(f"Playback error: {str(e)}")
            # Story 2.2: Emit playback_error signal
            self._emit_playback_error(task.playback_id, str(e))

        finally:
            # Cleanup stream
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception as cleanup_error:
                    self.logger.warning(f"Error cleaning up stream: {cleanup_error}")

    def _is_cable_input_device(self, device_name: str) -> bool:
        """
        Check if device name indicates VB-Cable Input (correct device for virtual mic).

        VB-Cable has two devices:
        - CABLE Input: Use this for virtual microphone (we send audio TO it)
        - CABLE Output: Use this for receiving audio (not for virtual mic)

        Args:
            device_name: Name of the device

        Returns:
            bool: True if device is CABLE Input
        """
        name_lower = device_name.lower()
        return 'cable input' in name_lower

    def _validate_virtual_device(self, device_index: int) -> bool:
        """
        Validate that a virtual device is available with detailed logging.

        Args:
            device_index: Device index to validate

        Returns:
            bool: True if device is available
        """
        if not self._pyaudio:
            self.logger.warning("PyAudio not initialized, cannot validate device")
            return False

        try:
            device_info = self._pyaudio.get_device_info_by_index(device_index)
            device_name = device_info.get('name', 'Unknown')
            max_output_channels = device_info.get('maxOutputChannels', 0)
            max_input_channels = device_info.get('maxInputChannels', 0)
            default_sample_rate = device_info.get('defaultSampleRate', 0)

            # Log detailed device information
            self.logger.info(f"Validating virtual device {device_index}:")
            self.logger.info(f"  Name: {device_name}")
            self.logger.info(f"  Max Output Channels: {max_output_channels}")
            self.logger.info(f"  Max Input Channels: {max_input_channels}")
            self.logger.info(f"  Default Sample Rate: {default_sample_rate}Hz")

            # Check if device supports output channels (we send audio TO virtual devices)
            if max_output_channels == 0:
                self.logger.warning(f"Device {device_index} has no output channels - cannot be used as virtual microphone")
                return False

            # Check if this is the correct VB-Cable device (Input, not Output)
            if 'cable' in device_name.lower():
                if not self._is_cable_input_device(device_name):
                    self.logger.warning(f"Device {device_index} appears to be CABLE Output - you need CABLE Input for virtual microphone")

            self.logger.info(f"Virtual device {device_index} validated successfully: {device_name}")
            return True

        except Exception as e:
            self.logger.error(f"Virtual device {device_index} validation failed: {e}")
            return False

    def _is_virtual_device(self, device_name: str) -> bool:
        """
        Check if device name indicates a virtual audio device.

        Args:
            device_name: Name of the device

        Returns:
            bool: True if device appears to be virtual
        """
        virtual_patterns = [
            'voicemeeter',
            'vb-audio',
            'virtual',
            'cable',
            'vac',
            'aux',
            'mic',
            'input'
        ]

        name_lower = device_name.lower()
        return any(pattern in name_lower for pattern in virtual_patterns)

    def _detect_virtual_driver(self, device_name: str) -> VirtualDeviceDriver:
        """
        Detect the virtual device driver type from device name.

        Args:
            device_name: Name of the virtual device

        Returns:
            VirtualDeviceDriver: Detected driver type
        """
        name_lower = device_name.lower()

        if 'voicemeeter' in name_lower:
            return VirtualDeviceDriver.VOICEMEETER
        elif 'vb-audio' in name_lower or 'cable' in name_lower:
            return VirtualDeviceDriver.VB_CABLE
        else:
            return VirtualDeviceDriver.UNKNOWN_VIRTUAL

    async def get_health(self) -> Dict[str, Any]:
        """
        Get health status of the virtual microphone service.

        Returns:
            Dict[str, Any]: Health information
        """
        return {
            "service_name": self.service_name,
            "status": self.status.value,
            "initialized": self._is_initialized,
            "pyaudio_available": PYAUDIO_AVAILABLE,
            "active_tasks": len(self._active_tasks),
            "default_device_available": self._validate_virtual_device(self.config.default_device_index) if self._pyaudio else False
        }

    # =========================================================================
    # Story 2.2: Playback Complete Callbacks (FR25 - synchronized with monitor)
    # =========================================================================

    def set_playback_complete_callback(self, callback: Callable[[str], None]) -> None:
        """
        Set callback for playback completion notification.

        Called when audio playback finishes successfully. Emits the task_id.

        Args:
            callback: Function that receives the completed task_id
        """
        self._playback_complete_callback = callback

    def set_playback_error_callback(self, callback: Callable[[str, str], None]) -> None:
        """
        Set callback for playback error notification.

        Called when audio playback fails. Emits task_id and error message.

        Args:
            callback: Function that receives (task_id, error_message)
        """
        self._playback_error_callback = callback

    def _emit_playback_complete(self, task_id: str) -> None:
        """Emit playback complete callback."""
        if self._playback_complete_callback:
            try:
                self._playback_complete_callback(task_id)
            except Exception as e:
                self.logger.error(f"Error in playback_complete callback: {e}")

    def _emit_playback_error(self, task_id: str, error: str) -> None:
        """Emit playback error callback."""
        if self._playback_error_callback:
            try:
                self._playback_error_callback(task_id, error)
            except Exception as e:
                self.logger.error(f"Error in playback_error callback: {e}")

    # =========================================================================
    # Story 2.2: Real-time Streaming to Virtual Mic (NFR16)
    # =========================================================================

    async def start_streaming_session(
        self,
        device: Optional[AudioDevice] = None,
        sample_rate: int = 24000,
        channels: int = 1,
        sample_width: int = 2,
    ) -> Optional[str]:
        """
        Start a streaming session for real-time audio to virtual microphone.

        This enables low-latency streaming where audio chunks can be sent
        as they are generated, synchronized with monitor playback (FR25, NFR16).

        Args:
            device: Target virtual device (uses first available if None)
            sample_rate: Audio sample rate (default 24000 for Qwen3-TTS)
            channels: Number of audio channels
            sample_width: Bytes per sample (2 for 16-bit)

        Returns:
            Session ID string, or None if failed
        """
        if not self._is_initialized:
            self.logger.error("VirtualMicrophoneService not initialized")
            return None

        with self._streaming_lock:
            # Close existing streaming session if any
            if self._streaming_stream:
                try:
                    self._streaming_stream.stop_stream()
                    self._streaming_stream.close()
                except Exception:
                    pass
                self._streaming_stream = None

            try:
                # Find device index
                if device:
                    device_id_str = device.device_id
                    if device_id_str.startswith("pyaudio_"):
                        device_index = int(device_id_str.split("_")[1])
                    else:
                        device_index = int(device_id_str)
                else:
                    # Try to find first available virtual device
                    virtual_devices = await self.enumerate_virtual_devices()
                    if virtual_devices:
                        device_index = int(virtual_devices[0].device_id)
                    elif self.config.default_device_index >= 0:
                        device_index = self.config.default_device_index
                    else:
                        self.logger.warning("No virtual device available for streaming")
                        return None

                # Convert sample width to PyAudio format
                if sample_width == 2:
                    audio_format = pyaudio.paInt16
                elif sample_width == 4:
                    audio_format = pyaudio.paFloat32
                else:
                    audio_format = pyaudio.paInt16

                # Open streaming output
                self._streaming_stream = self._pyaudio.open(
                    format=audio_format,
                    channels=channels,
                    rate=sample_rate,
                    output=True,
                    output_device_index=device_index,
                    frames_per_buffer=self.config.chunk_size,
                )

                # Generate session ID
                self._task_counter += 1
                session_id = f"vstream_{self._task_counter}_{int(time.time())}"

                self.logger.info(
                    f"Started virtual mic streaming session {session_id}: "
                    f"{channels}ch, {sample_rate}Hz, device={device_index}"
                )
                return session_id

            except Exception as e:
                self.logger.error(f"Failed to start virtual mic streaming session: {e}")
                return None

    async def play_audio_chunk(
        self,
        audio_data: bytes,
        is_final: bool = False,
    ) -> bool:
        """
        Stream an audio chunk immediately to virtual microphone.

        Chunks are played in real-time for synchronized output with
        monitor speakers (NFR16: real-time streaming).

        Args:
            audio_data: Raw audio bytes to stream
            is_final: If True, this is the last chunk

        Returns:
            bool: True if chunk was streamed successfully
        """
        with self._streaming_lock:
            if not self._streaming_stream:
                self.logger.error("No active virtual mic streaming session")
                return False

            try:
                # Write chunk directly to stream
                self._streaming_stream.write(audio_data)

                if is_final:
                    self.logger.debug("Streamed final audio chunk to virtual mic")

                return True

            except Exception as e:
                self.logger.error(f"Failed to stream audio chunk to virtual mic: {e}")
                return False

    async def stop_streaming_session(self) -> bool:
        """
        Stop the current streaming session and close the stream.

        Returns:
            bool: True if session was stopped successfully
        """
        with self._streaming_lock:
            if not self._streaming_stream:
                return True

            try:
                self._streaming_stream.stop_stream()
                self._streaming_stream.close()
                self._streaming_stream = None

                self.logger.info("Virtual mic streaming session stopped")
                return True

            except Exception as e:
                self.logger.error(f"Error stopping virtual mic streaming session: {e}")
                self._streaming_stream = None
                return False

    def is_streaming_active(self) -> bool:
        """Check if a streaming session is currently active."""
        with self._streaming_lock:
            return self._streaming_stream is not None