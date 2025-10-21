"""
Windows Audio Client Integration

This module provides Windows Audio API integration through PyAudio wrapper.
It handles audio device enumeration, device information retrieval, availability detection,
and device change monitoring for Windows systems.
"""

import logging
import platform
import threading
import time
from typing import List, Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    # Use audio backend abstraction (PyAudioWPatch on Windows, PyAudio elsewhere)
    from myvoice.services.integrations.audio_backend import (
        AudioBackend, get_pyaudio_module, IS_WPATCH, BACKEND_NAME
    )
    pyaudio = get_pyaudio_module()  # Get underlying pyaudio module for constants
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    pyaudio = None
    IS_WPATCH = False
    BACKEND_NAME = "None"

# Windows COM support for WASAPI threading
if platform.system() == 'Windows':
    try:
        import pythoncom
        PYTHONCOM_AVAILABLE = True
    except ImportError:
        PYTHONCOM_AVAILABLE = False
        pythoncom = None
else:
    PYTHONCOM_AVAILABLE = False
    pythoncom = None

from myvoice.models.audio_device import AudioDevice, DeviceType, VirtualDeviceDriver
from myvoice.models.error import MyVoiceError, ErrorSeverity
from myvoice.utils.timeout import timeout_decorator, AudioTimeoutConfig, TimeoutError


@dataclass
class DeviceChangeEvent:
    """Represents an audio device change event."""
    event_type: str  # "added", "removed", "changed"
    device_index: Optional[int]
    device_name: str
    timestamp: datetime


class WindowsAudioClient:
    """
    Windows Audio API client with PyAudio wrapper.

    This client provides comprehensive Windows audio device management including
    enumeration, monitoring, and availability detection. It uses PyAudio as the
    underlying audio library for cross-platform compatibility.

    Attributes:
        logger: Logger instance for this client
        pyaudio_instance: PyAudio instance for device operations
        is_monitoring: Whether device change monitoring is active
        device_cache: Cached device information for change detection
        monitor_thread: Background thread for device monitoring
        change_callbacks: List of callbacks for device change events
    """

    def __init__(self, enable_monitoring: bool = False):
        """
        Initialize the Windows Audio Client.

        Args:
            enable_monitoring: Whether to enable automatic device change monitoring

        Raises:
            RuntimeError: If PyAudio is not available
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        if not PYAUDIO_AVAILABLE:
            raise RuntimeError(
                "PyAudio is not available. Please install pyaudio: pip install pyaudio"
            )

        # Log audio backend being used
        self.logger.info(f"Using audio backend: {BACKEND_NAME}" +
                        (" (enhanced WASAPI support)" if IS_WPATCH else ""))

        # Initialize COM for this thread if on Windows
        self._com_initialized = False
        if platform.system() == 'Windows' and PYTHONCOM_AVAILABLE:
            try:
                pythoncom.CoInitialize()
                self._com_initialized = True
                self.logger.debug("COM initialized for Windows Audio Client thread")
            except Exception as e:
                self.logger.warning(f"COM initialization failed: {e}")
                self._com_initialized = False
        elif platform.system() == 'Windows' and not PYTHONCOM_AVAILABLE:
            self.logger.warning("pythoncom not available - WASAPI threading may have issues. Install pywin32: pip install pywin32")

        # Initialize audio backend with retry logic and enhanced error handling
        self.audio_backend: Optional[AudioBackend] = None
        self.pyaudio_instance = None  # Will be set by audio backend
        self._initialize_pyaudio_with_retry()

        # Device monitoring
        self.is_monitoring = False
        self.device_cache: Dict[int, Dict[str, Any]] = {}
        self.monitor_thread: Optional[threading.Thread] = None
        self.change_callbacks: List[Callable[[DeviceChangeEvent], None]] = []
        self._monitor_stop_event = threading.Event()

        # Performance optimization
        self.last_device_scan = None
        self.device_scan_cache_duration = timedelta(seconds=5)  # Cache for 5 seconds

        self.logger.info("Windows Audio Client initialized")

        # Start monitoring if requested
        if enable_monitoring:
            self.start_device_monitoring()

    def refresh_audio_devices(self) -> bool:
        """
        Refresh PyAudio device list by reinitializing PyAudio instance.

        This method addresses the PyAudio limitation where device changes
        are not detected without reinitialization.

        Returns:
            bool: True if refresh was successful
        """
        try:
            self.logger.info("Refreshing audio device list...")

            # Store old instance for cleanup
            old_instance = self.pyaudio_instance

            # Stop monitoring temporarily if running
            was_monitoring = self.is_monitoring
            if was_monitoring:
                self.stop_device_monitoring()

            # Reinitialize PyAudio
            self._initialize_pyaudio_with_retry()

            # Clean up old instance
            if old_instance:
                try:
                    old_instance.terminate()
                except Exception as e:
                    self.logger.warning(f"Error terminating old PyAudio instance: {e}")

            # Restart monitoring if it was running
            if was_monitoring:
                self.start_device_monitoring()

            # Clear device cache to force refresh
            self.device_cache.clear()

            self.logger.info("Audio device list refreshed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to refresh audio devices: {e}")
            return False

    @timeout_decorator(seconds=AudioTimeoutConfig.PYAUDIO_INIT_TIMEOUT)
    def _initialize_pyaudio_with_timeout(self) -> AudioBackend:
        """
        Initialize audio backend with timeout protection.

        Returns:
            AudioBackend: Initialized audio backend instance

        Raises:
            TimeoutError: If initialization exceeds timeout
        """
        return AudioBackend()

    def _initialize_pyaudio_with_retry(self, max_retries: int = 3, retry_delay: float = 1.0) -> None:
        """
        Initialize audio backend with retry logic for better reliability.

        Args:
            max_retries: Maximum number of initialization attempts
            retry_delay: Delay between retries in seconds

        Raises:
            RuntimeError: If audio backend initialization fails after all retries
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Audio backend initialization attempt {attempt + 1}/{max_retries}")

                # Try to initialize audio backend with timeout protection
                self.audio_backend = self._initialize_pyaudio_with_timeout()
                self.pyaudio_instance = self.audio_backend.pa_instance

                # Verify initialization by checking device count
                device_count = self.pyaudio_instance.get_device_count()
                self.logger.debug(f"Audio backend initialized successfully with {device_count} devices")

                # Log backend info
                self.audio_backend.log_backend_info()

                return  # Success!

            except TimeoutError as e:
                last_error = f"Audio backend initialization timed out - Windows Audio Service not responding: {str(e)}"
                self.logger.error(last_error)

            except Exception as e:
                last_error = e
                self.logger.warning(f"Audio backend initialization attempt {attempt + 1} failed: {e}")

                # Clean up failed instance
                if self.audio_backend:
                    try:
                        self.audio_backend.terminate()
                    except Exception:
                        pass
                    self.audio_backend = None
                    self.pyaudio_instance = None

                # Wait before retry (except on last attempt)
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)

        # All retries failed
        error_msg = f"Failed to initialize PyAudio after {max_retries} attempts. Last error: {last_error}"
        self.logger.error(error_msg)

        # Create user-friendly error message based on error type
        if "timed out" in str(last_error).lower() or "timeout" in str(last_error).lower():
            user_msg = "Windows Audio Service is not responding. Try restarting Windows Audio Service or reboot the system."
        elif "ALSA" in str(last_error):
            user_msg = "Audio system initialization failed. Please check that no other applications are using audio devices and try again."
        elif "WASAPI" in str(last_error) or "DirectSound" in str(last_error):
            user_msg = "Windows audio system is not responding. Try restarting the application or check audio driver status."
        elif "Permission" in str(last_error) or "Access" in str(last_error):
            user_msg = "Permission denied accessing audio system. Please run as administrator or check audio device permissions."
        else:
            user_msg = "Audio system initialization failed. Please check audio drivers and restart the application."

        # Create detailed error for technical logging
        raise RuntimeError(f"{user_msg}\n\nTechnical details: {error_msg}")

    def enumerate_audio_devices(self, refresh_cache: bool = False) -> List[AudioDevice]:
        """
        Enumerate all available audio devices.

        Args:
            refresh_cache: Force refresh of device cache

        Returns:
            List[AudioDevice]: List of available audio devices

        Raises:
            MyVoiceError: If device enumeration fails
        """
        try:
            # Check cache first (unless refresh requested)
            if (not refresh_cache and self.last_device_scan and
                datetime.now() - self.last_device_scan < self.device_scan_cache_duration):
                self.logger.debug("Using cached device list")
                return [self._create_audio_device_from_cache(idx, info)
                       for idx, info in self.device_cache.items()]

            self.logger.debug("Enumerating audio devices")
            devices = []
            device_count = self.pyaudio_instance.get_device_count()

            # Clear and rebuild cache
            self.device_cache.clear()

            for device_index in range(device_count):
                try:
                    # Use timeout-protected device info query
                    device_info = self._get_device_info_with_timeout(device_index)

                    # Cache the device info
                    self.device_cache[device_index] = device_info

                    # Create AudioDevice instance
                    audio_device = self._create_audio_device_from_pyaudio(device_index, device_info)
                    if audio_device:
                        devices.append(audio_device)

                except TimeoutError as e:
                    self.logger.error(f"Device {device_index} enumeration timed out - Windows Audio Service may be hung: {e}")
                    continue

                except Exception as e:
                    self.logger.warning(f"Failed to get info for device {device_index}: {e}")
                    continue

            self.last_device_scan = datetime.now()
            self.logger.info(f"Found {len(devices)} audio devices")
            return devices

        except Exception as e:
            self.logger.error(f"Failed to enumerate audio devices: {e}")
            raise MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="DEVICE_ENUMERATION_FAILED",
                user_message="Failed to scan audio devices",
                technical_details=str(e),
                suggested_action="Check Windows audio drivers and restart the application"
            )

    def enumerate_virtual_input_devices(self, refresh_cache: bool = False) -> List[AudioDevice]:
        """
        Enumerate virtual input devices specifically (VB-Audio Cable, Voicemeeter, etc.).

        This method filters the full device list to return only virtual input devices
        that can be used as microphone sources for TTS applications.

        Args:
            refresh_cache: Force refresh of device cache

        Returns:
            List[AudioDevice]: List of available virtual input devices

        Raises:
            MyVoiceError: If virtual device enumeration fails
        """
        try:
            self.logger.debug("Enumerating virtual input devices")

            # Get all devices first
            all_devices = self.enumerate_audio_devices(refresh_cache=refresh_cache)

            # Filter for virtual input devices
            virtual_input_devices = []
            for device in all_devices:
                if self._is_virtual_input_device(device):
                    # Enhance device with virtual-specific information
                    enhanced_device = self._enhance_virtual_device_info(device)
                    virtual_input_devices.append(enhanced_device)

            self.logger.info(f"Found {len(virtual_input_devices)} virtual input devices")
            return virtual_input_devices

        except Exception as e:
            self.logger.error(f"Failed to enumerate virtual input devices: {e}")
            raise MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="VIRTUAL_DEVICE_ENUMERATION_FAILED",
                user_message="Failed to scan virtual audio devices",
                technical_details=str(e),
                suggested_action="Check if VB-Audio Cable or Voicemeeter are installed and running"
            )

    def get_vb_cable_devices(self) -> List[AudioDevice]:
        """
        Get all VB-Audio Cable devices (Input and Output).

        Returns:
            List[AudioDevice]: List of VB-Audio Cable devices
        """
        try:
            virtual_devices = self.enumerate_virtual_input_devices()
            vb_cable_devices = [device for device in virtual_devices if device.is_vb_cable()]

            self.logger.debug(f"Found {len(vb_cable_devices)} VB-Audio Cable devices")
            return vb_cable_devices

        except Exception as e:
            self.logger.error(f"Failed to get VB-Audio Cable devices: {e}")
            return []

    def get_voicemeeter_devices(self) -> List[AudioDevice]:
        """
        Get all Voicemeeter virtual devices (VAIO, AUX, etc.).

        Returns:
            List[AudioDevice]: List of Voicemeeter devices
        """
        try:
            virtual_devices = self.enumerate_virtual_input_devices()
            voicemeeter_devices = [device for device in virtual_devices if device.is_voicemeeter()]

            self.logger.debug(f"Found {len(voicemeeter_devices)} Voicemeeter devices")
            return voicemeeter_devices

        except Exception as e:
            self.logger.error(f"Failed to get Voicemeeter devices: {e}")
            return []

    def validate_virtual_device_availability(self, device: AudioDevice) -> Tuple[bool, Optional[str]]:
        """
        Validate that a virtual audio device is available and functional.

        This extends the base device validation with virtual device specific checks.

        Args:
            device: Virtual AudioDevice to validate

        Returns:
            Tuple[bool, Optional[str]]: (is_available, error_message)
        """
        try:
            self.logger.debug(f"Validating virtual device: {device.name}")

            # First perform standard device validation
            is_available, error_message = self.validate_device_availability(device)
            if not is_available:
                return is_available, error_message

            # Additional virtual device validation
            if not device.is_virtual_device():
                return False, "Device is not a virtual device"

            # Extract device index for PyAudio verification
            device_index = self._extract_device_index(device.device_id)
            if device_index is None:
                return False, "Invalid virtual device ID format"

            # Get current device info
            try:
                current_info = self.pyaudio_instance.get_device_info_by_index(device_index)
            except (OSError, ValueError):
                return False, "Virtual device no longer exists"

            # Check virtual device specific requirements
            max_input_channels = current_info.get('maxInputChannels', 0)
            max_output_channels = current_info.get('maxOutputChannels', 0)

            # For VoiceMeeter "Input" devices (which are actually playback/output devices),
            # we need to check for output channels, not input channels
            device_name = current_info.get('name', '').lower()
            if 'input' in device_name and ('voicemeeter' in device_name or 'vaio' in device_name):
                # This is a VoiceMeeter Input device (actually an output device for playback)
                if max_output_channels <= 0:
                    return False, "Virtual device does not support output channels"
            else:
                # For other virtual devices, check for input channels
                if max_input_channels <= 0:
                    return False, "Virtual device does not support input channels"

            # Validate virtual driver specific requirements
            if device.virtual_driver == VirtualDeviceDriver.VB_CABLE:
                # For virtual microphone devices (recording), we don't need output channels
                # The validation was incorrectly checking for output channels on input devices
                # VB-Cable virtual microphones are input-only devices by design
                if device.device_type == DeviceType.VIRTUAL and max_input_channels <= 0:
                    return False, "VB-Audio Cable virtual microphone missing input channels"

                # Check for typical VB-Cable sample rates
                default_rate = current_info.get('defaultSampleRate', 0)
                if default_rate not in [44100, 48000, 96000]:
                    self.logger.warning(f"VB-Cable device has unusual sample rate: {default_rate}")

            elif device.virtual_driver == VirtualDeviceDriver.VOICEMEETER:
                # Voicemeeter devices should have specific naming patterns
                device_name = current_info.get('name', '').lower()
                if 'voicemeeter' not in device_name and 'vaio' not in device_name:
                    return False, "Device name doesn't match Voicemeeter pattern"

            self.logger.debug(f"Virtual device {device.name} is available and functional")
            return True, None

        except Exception as e:
            self.logger.error(f"Error validating virtual device {device.name}: {e}")
            return False, f"Virtual device validation error: {str(e)}"

    def monitor_virtual_device_availability(self, virtual_devices: List[AudioDevice],
                                          callback: Callable[[AudioDevice, bool], None]) -> None:
        """
        Monitor availability of specific virtual devices.

        Args:
            virtual_devices: List of virtual devices to monitor
            callback: Function called when device availability changes
                     Parameters: (device, is_available)
        """
        def virtual_monitor_callback(event: DeviceChangeEvent) -> None:
            """Handle device change events for virtual device monitoring."""
            try:
                # Check if any of our monitored virtual devices were affected
                for virtual_device in virtual_devices:
                    device_index = self._extract_device_index(virtual_device.device_id)

                    if device_index == event.device_index:
                        # This virtual device was affected
                        if event.event_type == "removed":
                            callback(virtual_device, False)
                        elif event.event_type == "added":
                            # Validate the device is actually available
                            is_available, _ = self.validate_virtual_device_availability(virtual_device)
                            callback(virtual_device, is_available)
                        elif event.event_type == "changed":
                            # Re-validate availability
                            is_available, _ = self.validate_virtual_device_availability(virtual_device)
                            callback(virtual_device, is_available)

            except Exception as e:
                self.logger.error(f"Error in virtual device monitor callback: {e}")

        # Add our virtual device monitor to the general device monitoring
        self.add_device_change_callback(virtual_monitor_callback)

        # Start device monitoring if not already running
        if not self.is_monitoring:
            self.start_device_monitoring()

        self.logger.info(f"Started monitoring {len(virtual_devices)} virtual devices")

    @timeout_decorator(seconds=AudioTimeoutConfig.DEVICE_INFO_TIMEOUT)
    def _get_device_info_with_timeout(self, device_index: int) -> Dict[str, Any]:
        """
        Get device info with timeout protection.

        This method wraps PyAudio's get_device_info_by_index with timeout
        protection to prevent hangs when Windows Audio Service has issues.

        Args:
            device_index: PyAudio device index

        Returns:
            Dict[str, Any]: Device information dictionary

        Raises:
            TimeoutError: If operation exceeds timeout
        """
        return self.pyaudio_instance.get_device_info_by_index(device_index)

    def get_device_info(self, device_index: int) -> Optional[AudioDevice]:
        """
        Get detailed information about a specific audio device.

        Args:
            device_index: Index of the device to query

        Returns:
            Optional[AudioDevice]: Device information or None if not found

        Raises:
            MyVoiceError: If device query fails
        """
        try:
            self.logger.debug(f"Getting device info for index {device_index}")

            # Check cache first
            if device_index in self.device_cache:
                device_info = self.device_cache[device_index]
            else:
                # Query PyAudio directly with timeout protection
                device_info = self._get_device_info_with_timeout(device_index)
                self.device_cache[device_index] = device_info

            return self._create_audio_device_from_pyaudio(device_index, device_info)

        except TimeoutError as e:
            # Device query timed out - Windows Audio Service issue
            self.logger.error(f"Device {device_index} query timed out - audiodg.exe may be hung: {e}")
            raise MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="AUDIO_SERVICE_TIMEOUT",
                user_message="Windows Audio Service is not responding",
                technical_details=f"Device {device_index} query timed out: {str(e)}",
                suggested_action="Try restarting Windows Audio Service or reboot the system"
            )

        except (OSError, ValueError) as e:
            # Device not found or invalid index
            self.logger.warning(f"Device {device_index} not found: {e}")
            return None

        except Exception as e:
            self.logger.error(f"Failed to get device info for {device_index}: {e}")
            raise MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="DEVICE_INFO_FAILED",
                user_message=f"Failed to get information for audio device {device_index}",
                technical_details=str(e),
                suggested_action="Try refreshing the device list or restart the application"
            )

    def validate_device_availability(self, device: AudioDevice) -> Tuple[bool, Optional[str]]:
        """
        Validate that an audio device is available and functional.

        Args:
            device: AudioDevice to validate

        Returns:
            Tuple[bool, Optional[str]]: (is_available, error_message)
        """
        try:
            self.logger.debug(f"Validating device availability: {device.name}")

            # Extract device index from device_id
            device_index = self._extract_device_index(device.device_id)
            if device_index is None:
                return False, "Invalid device ID format"

            # Check if device still exists
            try:
                current_info = self.pyaudio_instance.get_device_info_by_index(device_index)
            except (OSError, ValueError):
                return False, "Device no longer exists in system"

            # Verify device name matches (devices can be reassigned indices)
            if current_info.get('name', '') != device.name:
                return False, "Device name mismatch - device may have been reassigned"

            # Check device capabilities based on type
            if device.device_type == DeviceType.INPUT:
                if current_info.get('maxInputChannels', 0) <= 0:
                    return False, "Device no longer supports input"
            elif device.device_type == DeviceType.OUTPUT:
                if current_info.get('maxOutputChannels', 0) <= 0:
                    return False, "Device no longer supports output"

            # Try to query default sample rate (indicates device is responsive)
            default_sample_rate = current_info.get('defaultSampleRate', 0)
            if default_sample_rate <= 0:
                return False, "Device reports invalid sample rate"

            self.logger.debug(f"Device {device.name} is available")
            return True, None

        except Exception as e:
            self.logger.error(f"Error validating device {device.name}: {e}")
            return False, f"Validation error: {str(e)}"

    def get_default_input_device(self) -> Optional[AudioDevice]:
        """
        Get the default input (microphone) device.

        Returns:
            Optional[AudioDevice]: Default input device or None if not available
        """
        try:
            default_info = self.pyaudio_instance.get_default_input_device_info()
            device_index = default_info.get('index', -1)

            if device_index >= 0:
                return self._create_audio_device_from_pyaudio(device_index, default_info)

        except Exception as e:
            self.logger.warning(f"Failed to get default input device: {e}")

        return None

    def get_default_output_device(self) -> Optional[AudioDevice]:
        """
        Get the default output (speaker) device.

        Returns:
            Optional[AudioDevice]: Default output device or None if not available
        """
        try:
            default_info = self.pyaudio_instance.get_default_output_device_info()
            device_index = default_info.get('index', -1)

            if device_index >= 0:
                return self._create_audio_device_from_pyaudio(device_index, default_info)

        except Exception as e:
            self.logger.warning(f"Failed to get default output device: {e}")

        return None

    def get_wasapi_host_api_index(self) -> Optional[int]:
        """
        Find the WASAPI host API index.

        WASAPI (Windows Audio Session API) provides better Windows 11 compatibility
        than DirectSound. This method locates the WASAPI host API in PyAudio.

        Returns:
            Optional[int]: WASAPI host API index or None if not found

        Example:
            >>> wasapi_index = client.get_wasapi_host_api_index()
            >>> if wasapi_index is not None:
            ...     print(f"WASAPI available at index {wasapi_index}")
        """
        try:
            host_api_count = self.pyaudio_instance.get_host_api_count()

            for i in range(host_api_count):
                try:
                    host_api_info = self.pyaudio_instance.get_host_api_info_by_index(i)
                    api_name = host_api_info.get('name', '')

                    if 'WASAPI' in api_name:
                        self.logger.info(f"Found WASAPI host API at index {i}: {api_name}")
                        return i

                except Exception as e:
                    self.logger.debug(f"Error checking host API {i}: {e}")
                    continue

            self.logger.warning("WASAPI host API not found - will use default host API")
            return None

        except Exception as e:
            self.logger.error(f"Error finding WASAPI host API: {e}")
            return None

    def get_directsound_host_api_index(self) -> Optional[int]:
        """
        Find the DirectSound host API index.

        Returns:
            Optional[int]: DirectSound host API index or None if not found
        """
        try:
            host_api_count = self.pyaudio_instance.get_host_api_count()

            for i in range(host_api_count):
                try:
                    host_api_info = self.pyaudio_instance.get_host_api_info_by_index(i)
                    api_name = host_api_info.get('name', '')

                    if 'DirectSound' in api_name or 'Windows DirectSound' in api_name:
                        self.logger.debug(f"Found DirectSound host API at index {i}: {api_name}")
                        return i

                except Exception as e:
                    self.logger.debug(f"Error checking host API {i}: {e}")
                    continue

            return None

        except Exception as e:
            self.logger.error(f"Error finding DirectSound host API: {e}")
            return None

    def get_device_host_api_name(self, device_index: int) -> str:
        """
        Get the host API name for a specific device.

        Args:
            device_index: PyAudio device index

        Returns:
            str: Host API name (e.g., "Windows WASAPI", "Windows DirectSound")
        """
        try:
            device_info = self.pyaudio_instance.get_device_info_by_index(device_index)
            host_api_index = device_info.get('hostApi', -1)

            if host_api_index >= 0:
                host_api_info = self.pyaudio_instance.get_host_api_info_by_index(host_api_index)
                return host_api_info.get('name', 'Unknown')

            return 'Unknown'

        except Exception as e:
            self.logger.debug(f"Error getting host API name for device {device_index}: {e}")
            return 'Unknown'

    def log_audio_system_info(self) -> None:
        """
        Log detailed audio system information for debugging.

        This method logs comprehensive information about the audio system including:
        - PyAudio/PortAudio version
        - Available host APIs (WASAPI, DirectSound, etc.)
        - Default input/output devices
        - Device counts per host API

        Useful for diagnosing audio issues, especially silent playback problems.
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("Audio System Information")
            self.logger.info("=" * 60)

            # PyAudio/PortAudio version
            try:
                version_text = pyaudio.get_portaudio_version_text()
                version_number = pyaudio.get_portaudio_version()
                self.logger.info(f"PortAudio Version: {version_text} ({version_number})")
            except Exception as e:
                self.logger.warning(f"Could not get PortAudio version: {e}")

            # Host APIs
            try:
                host_count = self.pyaudio_instance.get_host_api_count()
                self.logger.info(f"\nAvailable Host APIs: {host_count}")

                for i in range(host_count):
                    try:
                        host_info = self.pyaudio_instance.get_host_api_info_by_index(i)
                        api_name = host_info.get('name', 'Unknown')
                        device_count = host_info.get('deviceCount', 0)
                        default_input = host_info.get('defaultInputDevice', -1)
                        default_output = host_info.get('defaultOutputDevice', -1)

                        self.logger.info(f"  [{i}] {api_name}")
                        self.logger.info(f"      Devices: {device_count}")
                        if default_input >= 0:
                            self.logger.info(f"      Default Input: {default_input}")
                        if default_output >= 0:
                            self.logger.info(f"      Default Output: {default_output}")

                    except Exception as e:
                        self.logger.debug(f"Error getting info for host API {i}: {e}")

            except Exception as e:
                self.logger.error(f"Error enumerating host APIs: {e}")

            # Default devices
            self.logger.info("\nDefault Devices:")

            try:
                default_input = self.pyaudio_instance.get_default_input_device_info()
                self.logger.info(f"  Input: {default_input.get('name', 'Unknown')} (index {default_input.get('index', -1)})")
            except Exception:
                self.logger.info("  Input: None available")

            try:
                default_output = self.pyaudio_instance.get_default_output_device_info()
                self.logger.info(f"  Output: {default_output.get('name', 'Unknown')} (index {default_output.get('index', -1)})")
            except Exception:
                self.logger.info("  Output: None available")

            # WASAPI availability
            self.logger.info("\nWASAPI Support:")
            wasapi_index = self.get_wasapi_host_api_index()
            if wasapi_index is not None:
                self.logger.info(f"  WASAPI is AVAILABLE at host API index {wasapi_index}")
            else:
                self.logger.warning("  WASAPI is NOT AVAILABLE (may cause silent audio on Windows 11)")

            # DirectSound availability
            directsound_index = self.get_directsound_host_api_index()
            if directsound_index is not None:
                self.logger.info(f"  DirectSound is available at host API index {directsound_index}")

            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"Failed to log audio system info: {e}")

    def enumerate_wasapi_devices(self, device_type: Optional[DeviceType] = None) -> List[AudioDevice]:
        """
        Enumerate devices using WASAPI host API specifically.

        WASAPI devices provide better Windows 11 compatibility than DirectSound devices.
        This method filters devices to only return those using the WASAPI host API.

        Args:
            device_type: Optional filter for INPUT, OUTPUT, or VIRTUAL devices

        Returns:
            List[AudioDevice]: Devices using WASAPI host API

        Example:
            >>> wasapi_outputs = client.enumerate_wasapi_devices(DeviceType.OUTPUT)
            >>> for device in wasapi_outputs:
            ...     print(f"WASAPI device: {device.name}")
        """
        wasapi_devices = []

        try:
            # Find WASAPI host API index
            wasapi_index = self.get_wasapi_host_api_index()
            if wasapi_index is None:
                self.logger.warning("WASAPI host API not available")
                return wasapi_devices

            # Get all devices
            all_devices = self.enumerate_audio_devices(refresh_cache=True)

            # Filter for WASAPI devices
            for device in all_devices:
                device_index = self._extract_device_index(device.device_id)
                if device_index is None:
                    continue

                try:
                    device_info = self.pyaudio_instance.get_device_info_by_index(device_index)
                    device_host_api = device_info.get('hostApi', -1)

                    # Check if device uses WASAPI
                    if device_host_api == wasapi_index:
                        # Apply device type filter if specified
                        if device_type is None or device.device_type == device_type:
                            wasapi_devices.append(device)

                except Exception as e:
                    self.logger.debug(f"Error checking device {device_index} for WASAPI: {e}")
                    continue

            self.logger.info(f"Found {len(wasapi_devices)} WASAPI devices" +
                           (f" of type {device_type.value}" if device_type else ""))
            return wasapi_devices

        except Exception as e:
            self.logger.error(f"Failed to enumerate WASAPI devices: {e}")
            return wasapi_devices

    def validate_stream_format(self, device_index: int, sample_rate: int,
                              channels: int, format_type: int,
                              is_input: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Validate that device supports requested audio format.

        This method checks if a device can handle specific audio parameters
        before attempting to open a stream, preventing format mismatch errors
        that cause silent audio or crashes.

        Args:
            device_index: PyAudio device index
            sample_rate: Desired sample rate (e.g., 48000)
            channels: Number of channels (1=mono, 2=stereo)
            format_type: PyAudio format constant (e.g., pyaudio.paInt16)
            is_input: True for input (recording) stream, False for output (playback)

        Returns:
            Tuple[bool, Optional[str]]: (is_supported, error_message)
                is_supported: True if format is supported
                error_message: Explanation if not supported, None if supported

        Example:
            >>> is_supported, error = client.validate_stream_format(
            ...     device_index=5,
            ...     sample_rate=48000,
            ...     channels=2,
            ...     format_type=pyaudio.paInt16,
            ...     is_input=False
            ... )
            >>> if not is_supported:
            ...     print(f"Format not supported: {error}")
        """
        try:
            # Get device info
            device_info = self.pyaudio_instance.get_device_info_by_index(device_index)
            device_name = device_info.get('name', f'Device {device_index}')

            # Check channel support
            max_channels = device_info.get('maxInputChannels' if is_input else 'maxOutputChannels', 0)
            if channels > max_channels:
                return False, f"Device '{device_name}' supports max {max_channels} channels, requested {channels}"

            # Use PyAudio's is_format_supported for detailed validation
            try:
                if is_input:
                    is_supported = self.pyaudio_instance.is_format_supported(
                        rate=float(sample_rate),
                        input_device=device_index,
                        input_channels=channels,
                        input_format=format_type
                    )
                else:
                    is_supported = self.pyaudio_instance.is_format_supported(
                        rate=float(sample_rate),
                        output_device=device_index,
                        output_channels=channels,
                        output_format=format_type
                    )

                if is_supported:
                    self.logger.debug(f"Format validated for device {device_index}: {sample_rate}Hz, {channels}ch")
                    return True, None
                else:
                    return False, f"Format not supported by device '{device_name}': {sample_rate}Hz, {channels}ch"

            except ValueError as e:
                # PyAudio's is_format_supported raises ValueError if format is invalid
                return False, f"Format validation failed for device '{device_name}': {str(e)}"

        except (OSError, ValueError) as e:
            # Device doesn't exist or invalid device index
            self.logger.error(f"Error accessing device {device_index} for format validation: {e}")
            return False, f"Device access error: {str(e)}"

        except Exception as e:
            self.logger.error(f"Unexpected error validating format for device {device_index}: {e}")
            return False, f"Validation error: {str(e)}"

    def start_device_monitoring(self, poll_interval: float = 2.0) -> None:
        """
        Start monitoring for audio device changes.

        Args:
            poll_interval: How often to check for changes (seconds)
        """
        if self.is_monitoring:
            self.logger.warning("Device monitoring is already active")
            return

        self.logger.info("Starting audio device monitoring")
        self.is_monitoring = True
        self._monitor_stop_event.clear()

        # Initialize baseline device cache
        try:
            self.enumerate_audio_devices(refresh_cache=True)
        except Exception as e:
            self.logger.error(f"Failed to initialize device cache for monitoring: {e}")

        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._device_monitor_loop,
            args=(poll_interval,),
            daemon=True,
            name="AudioDeviceMonitor"
        )
        self.monitor_thread.start()

    def stop_device_monitoring(self) -> None:
        """Stop monitoring for audio device changes."""
        if not self.is_monitoring:
            return

        self.logger.info("Stopping audio device monitoring")
        self.is_monitoring = False
        self._monitor_stop_event.set()

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
            if self.monitor_thread.is_alive():
                self.logger.warning("Monitor thread did not stop gracefully")

    def add_device_change_callback(self, callback: Callable[[DeviceChangeEvent], None]) -> None:
        """
        Add a callback for device change events.

        Args:
            callback: Function to call when devices change
        """
        if callback not in self.change_callbacks:
            self.change_callbacks.append(callback)
            callback_name = getattr(callback, '__name__', str(callback))
        self.logger.debug(f"Added device change callback: {callback_name}")

    def remove_device_change_callback(self, callback: Callable[[DeviceChangeEvent], None]) -> None:
        """
        Remove a device change callback.

        Args:
            callback: Callback function to remove
        """
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
            callback_name = getattr(callback, '__name__', str(callback))
            self.logger.debug(f"Removed device change callback: {callback_name}")

    def _is_virtual_input_device(self, device: AudioDevice) -> bool:
        """
        Check if a device is a virtual input device suitable for microphone routing.

        Args:
            device: AudioDevice to check

        Returns:
            bool: True if device is a virtual input device
        """
        # Must be virtual or have virtual capabilities
        if device.device_type == DeviceType.VIRTUAL:
            return True

        # Check if it's a physical device that supports both input/output (potential virtual)
        if device.device_type == DeviceType.INPUT and device.is_virtual_device():
            return True

        # Check by name patterns for devices that might be misclassified
        if device.name:
            return AudioDevice._is_virtual_device_by_name(device.name, device.driver_name or "")

        return False

    def _enhance_virtual_device_info(self, device: AudioDevice) -> AudioDevice:
        """
        Enhance a virtual device with additional virtual-specific information.

        Args:
            device: AudioDevice to enhance

        Returns:
            AudioDevice: Enhanced device with virtual capabilities
        """
        try:
            # Extract device index to get fresh PyAudio info
            device_index = self._extract_device_index(device.device_id)
            if device_index is None:
                return device

            # Get current device info from PyAudio
            current_info = self.pyaudio_instance.get_device_info_by_index(device_index)

            # Create a new virtual device with enhanced information
            if device.is_vb_cable():
                enhanced_device = AudioDevice.create_vb_cable_device(
                    device_id=device.device_id,
                    name=device.name
                )
            elif device.is_voicemeeter():
                enhanced_device = AudioDevice.create_voicemeeter_device(
                    device_id=device.device_id,
                    name=device.name
                )
            else:
                # Generic virtual device
                enhanced_device = AudioDevice.create_virtual_device(
                    device_id=device.device_id,
                    name=device.name
                )

            # Copy over original properties
            enhanced_device.is_default = device.is_default
            enhanced_device.driver_name = device.driver_name

            # Add additional capabilities based on PyAudio info
            max_input_channels = current_info.get('maxInputChannels', 0)
            max_output_channels = current_info.get('maxOutputChannels', 0)
            default_sample_rate = current_info.get('defaultSampleRate', 0)

            if max_input_channels > 1:
                enhanced_device.capabilities.add("multi_channel_input")
            if max_output_channels > 1:
                enhanced_device.capabilities.add("multi_channel_output")
            if default_sample_rate >= 48000:
                enhanced_device.capabilities.add("high_sample_rate")
            if default_sample_rate >= 96000:
                enhanced_device.capabilities.add("professional_audio")

            return enhanced_device

        except Exception as e:
            self.logger.warning(f"Failed to enhance virtual device {device.name}: {e}")
            return device

    def _create_audio_device_from_pyaudio(self, device_index: int, device_info: Dict[str, Any]) -> Optional[AudioDevice]:
        """
        Create AudioDevice from PyAudio device info.

        Args:
            device_index: PyAudio device index
            device_info: PyAudio device information dict

        Returns:
            Optional[AudioDevice]: Created device or None if invalid
        """
        try:
            # Determine device type
            max_input_channels = device_info.get('maxInputChannels', 0)
            max_output_channels = device_info.get('maxOutputChannels', 0)

            if max_input_channels > 0 and max_output_channels > 0:
                device_type = DeviceType.VIRTUAL  # Can do both
            elif max_input_channels > 0:
                device_type = DeviceType.INPUT
            elif max_output_channels > 0:
                device_type = DeviceType.OUTPUT
            else:
                device_type = DeviceType.UNKNOWN

            # Get device name
            device_name = device_info.get('name', f'Device {device_index}')

            # Check if it's a default device
            is_default = False
            try:
                if device_type in [DeviceType.INPUT, DeviceType.VIRTUAL]:
                    default_input = self.pyaudio_instance.get_default_input_device_info()
                    is_default = device_index == default_input.get('index', -1)
                if not is_default and device_type in [DeviceType.OUTPUT, DeviceType.VIRTUAL]:
                    default_output = self.pyaudio_instance.get_default_output_device_info()
                    is_default = device_index == default_output.get('index', -1)
            except Exception:
                pass  # Ignore errors getting default devices

            # Get host API info for driver name
            driver_name = "Unknown"
            try:
                host_api_index = device_info.get('hostApi', -1)
                if host_api_index >= 0:
                    host_api_info = self.pyaudio_instance.get_host_api_info_by_index(host_api_index)
                    driver_name = host_api_info.get('name', 'Unknown')
            except Exception:
                pass

            # Create AudioDevice using our factory method with host API metadata
            return AudioDevice.create_from_pyaudio_info(device_index, device_info, host_api_name=driver_name)

        except Exception as e:
            self.logger.error(f"Failed to create AudioDevice from PyAudio info: {e}")
            return None

    def _create_audio_device_from_cache(self, device_index: int, device_info: Dict[str, Any]) -> AudioDevice:
        """Create AudioDevice from cached device info."""
        return self._create_audio_device_from_pyaudio(device_index, device_info)

    def _extract_device_index(self, device_id: str) -> Optional[int]:
        """
        Extract PyAudio device index from AudioDevice device_id.

        Args:
            device_id: Device ID string (e.g., "pyaudio_5")

        Returns:
            Optional[int]: Device index or None if invalid format
        """
        if device_id.startswith('pyaudio_'):
            try:
                return int(device_id.split('_')[1])
            except (IndexError, ValueError):
                return None
        return None

    def _device_monitor_loop(self, poll_interval: float) -> None:
        """
        Background loop for monitoring device changes.

        Args:
            poll_interval: How often to check for changes (seconds)
        """
        # Initialize COM for this background thread if on Windows
        com_initialized_in_thread = False
        if platform.system() == 'Windows' and PYTHONCOM_AVAILABLE:
            try:
                pythoncom.CoInitialize()
                com_initialized_in_thread = True
                self.logger.debug("COM initialized for device monitor thread")
            except Exception as e:
                self.logger.warning(f"COM initialization failed in monitor thread: {e}")

        try:
            self.logger.debug(f"Device monitor loop started (interval: {poll_interval}s)")

            while not self._monitor_stop_event.wait(poll_interval):
                try:
                    self._check_for_device_changes()
                except Exception as e:
                    self.logger.error(f"Error in device monitor loop: {e}")

            self.logger.debug("Device monitor loop stopped")

        finally:
            # Clean up COM for this thread
            if platform.system() == 'Windows' and com_initialized_in_thread and PYTHONCOM_AVAILABLE:
                try:
                    pythoncom.CoUninitialize()
                    self.logger.debug("COM uninitialized for device monitor thread")
                except Exception as e:
                    self.logger.warning(f"COM cleanup failed in monitor thread: {e}")

    def _check_for_device_changes(self) -> None:
        """Check for changes in audio devices and notify callbacks."""
        try:
            # Get current device list
            current_device_count = self.pyaudio_instance.get_device_count()
            current_devices = {}

            for device_index in range(current_device_count):
                try:
                    device_info = self.pyaudio_instance.get_device_info_by_index(device_index)
                    current_devices[device_index] = device_info
                except Exception:
                    continue

            # Compare with cached devices
            cached_indices = set(self.device_cache.keys())
            current_indices = set(current_devices.keys())

            # Detect removed devices
            removed_indices = cached_indices - current_indices
            for device_index in removed_indices:
                device_info = self.device_cache[device_index]
                event = DeviceChangeEvent(
                    event_type="removed",
                    device_index=device_index,
                    device_name=device_info.get('name', f'Device {device_index}'),
                    timestamp=datetime.now()
                )
                self._notify_device_change(event)

            # Detect added devices
            added_indices = current_indices - cached_indices
            for device_index in added_indices:
                device_info = current_devices[device_index]
                event = DeviceChangeEvent(
                    event_type="added",
                    device_index=device_index,
                    device_name=device_info.get('name', f'Device {device_index}'),
                    timestamp=datetime.now()
                )
                self._notify_device_change(event)

            # Detect changed devices (name changes, etc.)
            common_indices = cached_indices & current_indices
            for device_index in common_indices:
                cached_info = self.device_cache[device_index]
                current_info = current_devices[device_index]

                # Check if important properties changed
                if (cached_info.get('name') != current_info.get('name') or
                    cached_info.get('maxInputChannels') != current_info.get('maxInputChannels') or
                    cached_info.get('maxOutputChannels') != current_info.get('maxOutputChannels')):

                    event = DeviceChangeEvent(
                        event_type="changed",
                        device_index=device_index,
                        device_name=current_info.get('name', f'Device {device_index}'),
                        timestamp=datetime.now()
                    )
                    self._notify_device_change(event)

            # Update cache
            self.device_cache = current_devices

        except Exception as e:
            self.logger.error(f"Error checking for device changes: {e}")

    def _notify_device_change(self, event: DeviceChangeEvent) -> None:
        """
        Notify all registered callbacks about a device change.

        Args:
            event: Device change event to notify about
        """
        self.logger.info(f"Device {event.event_type}: {event.device_name}")

        for callback in self.change_callbacks:
            try:
                callback(event)
            except Exception as e:
                callback_name = getattr(callback, '__name__', str(callback))
                self.logger.error(f"Error in device change callback {callback_name}: {e}")

    def find_device_by_metadata(self, device_id: Optional[str], device_name: Optional[str],
                                host_api_name: Optional[str] = None) -> Optional[AudioDevice]:
        """
        Find a device using smart matching with fallback strategies.

        This method addresses the issue where PyAudio device indices change between sessions,
        causing saved device selections to fail. It uses multiple matching strategies:
        1. Exact device_id match (if available)
        2. Name + host_api exact match
        3. Name fuzzy match (if device_id and host_api unavailable)

        Args:
            device_id: PyAudio device ID (e.g., "pyaudio_5") - may be stale
            device_name: Device name for fallback matching
            host_api_name: Host API name (e.g., "Windows WASAPI") for precise matching

        Returns:
            Optional[AudioDevice]: Matched device or None if not found

        Example:
            >>> # Try to find previously saved device
            >>> device = client.find_device_by_metadata(
            ...     device_id="pyaudio_5",
            ...     device_name="Speakers (Realtek)",
            ...     host_api_name="Windows WASAPI"
            ... )
        """
        try:
            # Get all current devices
            all_devices = self.enumerate_audio_devices(refresh_cache=True)

            # Strategy 1: Try exact device_id match first
            if device_id:
                for device in all_devices:
                    if device.device_id == device_id:
                        # Verify the name still matches to ensure it's the same device
                        if device_name and device.name == device_name:
                            self.logger.debug(f"Found device by exact ID and name match: {device.name}")
                            return device
                        else:
                            # ID matched but name changed - device may have been reassigned
                            self.logger.warning(f"Device ID {device_id} matched but name changed from '{device_name}' to '{device.name}'")

            # Strategy 2: Match by name + host_api (most reliable cross-session match)
            if device_name and host_api_name:
                for device in all_devices:
                    if device.name == device_name:
                        # Get this device's host API name
                        device_index = self._extract_device_index(device.device_id)
                        if device_index is not None:
                            current_host_api = self.get_device_host_api_name(device_index)
                            if current_host_api == host_api_name:
                                self.logger.info(f"Found device by name + host API match: {device.name} ({host_api_name})")
                                return device

            # Strategy 3: Match by name only (less reliable but better than nothing)
            if device_name:
                for device in all_devices:
                    if device.name == device_name:
                        self.logger.info(f"Found device by name-only match: {device.name}")
                        self.logger.warning(f"Host API may have changed - using name-only match for '{device_name}'")
                        return device

            # Strategy 4: Fuzzy name matching (handle minor name variations)
            if device_name:
                device_name_lower = device_name.lower()
                for device in all_devices:
                    device_current_lower = device.name.lower()
                    # Check if device names are similar (contains or close match)
                    if (device_name_lower in device_current_lower or
                        device_current_lower in device_name_lower):
                        self.logger.info(f"Found device by fuzzy name match: '{device.name}' (looking for '{device_name}')")
                        return device

            # No match found
            self.logger.warning(f"Could not find device: device_id='{device_id}', name='{device_name}', host_api='{host_api_name}'")
            return None

        except Exception as e:
            self.logger.error(f"Error finding device by metadata: {e}")
            return None

    def close(self) -> None:
        """Close the Windows Audio Client and clean up resources."""
        self.logger.info("Closing Windows Audio Client")

        # Stop monitoring
        self.stop_device_monitoring()

        # Clean up audio backend
        if self.audio_backend:
            try:
                self.audio_backend.terminate()
                self.logger.debug("Audio backend terminated")
            except Exception as e:
                self.logger.error(f"Error terminating audio backend: {e}")
            self.audio_backend = None
            self.pyaudio_instance = None

        # Uninitialize COM if we initialized it
        if platform.system() == 'Windows' and self._com_initialized and PYTHONCOM_AVAILABLE:
            try:
                pythoncom.CoUninitialize()
                self._com_initialized = False
                self.logger.debug("COM uninitialized for Windows Audio Client thread")
            except Exception as e:
                self.logger.warning(f"COM cleanup failed: {e}")

        # Clear caches and callbacks
        self.device_cache.clear()
        self.change_callbacks.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()