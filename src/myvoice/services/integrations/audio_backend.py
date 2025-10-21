"""
Audio Backend Abstraction Layer

Provides abstraction between PyAudio and PyAudioWPatch to enable
better Windows WASAPI support while maintaining fallback compatibility.

PyAudioWPatch Features:
- WASAPI loopback device support (record system audio)
- Better Windows 11 WASAPI compatibility
- Context manager support
- Additional device discovery methods
- Fixed device name encoding issues

The abstraction allows seamless switching between backends for testing
and production use.
"""

import logging
from typing import Any, Dict, List, Optional

# Try to import PyAudioWPatch first (better Windows support)
# Fall back to standard PyAudio if not available
try:
    import pyaudiowpatch as pyaudio
    USING_WPATCH = True
    BACKEND_NAME = "PyAudioWPatch"
except ImportError:
    try:
        import pyaudio
        USING_WPATCH = False
        BACKEND_NAME = "PyAudio"
    except ImportError:
        raise ImportError(
            "Neither PyAudioWPatch nor PyAudio could be imported. "
            "Install one with: pip install PyAudioWPatch  (or)  pip install pyaudio"
        )


class AudioBackend:
    """
    Wrapper around PyAudio or PyAudioWPatch.

    Provides unified interface with enhanced features when PyAudioWPatch
    is available (WASAPI loopback, better Windows 11 support).

    Usage:
        backend = AudioBackend()
        print(f"Using: {backend.backend_name}")

        # Standard PyAudio operations work with both backends
        device_count = backend.get_device_count()

        # Enhanced WASAPI features (only if PyAudioWPatch available)
        if backend.supports_wasapi_loopback:
            loopback_device = backend.get_default_wasapi_loopback()
    """

    def __init__(self):
        """Initialize the audio backend."""
        self.logger = logging.getLogger(__name__)
        self._pa_instance: Optional[pyaudio.PyAudio] = None
        self.backend_name = BACKEND_NAME
        self.using_wpatch = USING_WPATCH

        self.logger.info(f"Audio backend initialized: {self.backend_name}")

        if self.using_wpatch:
            self.logger.info("PyAudioWPatch features available: WASAPI loopback, context manager, enhanced device discovery")
        else:
            self.logger.warning("Using standard PyAudio - WASAPI loopback not available. Consider installing PyAudioWPatch for better Windows 11 support")

    @property
    def pa_instance(self) -> pyaudio.PyAudio:
        """
        Get or create PyAudio instance.

        Returns:
            pyaudio.PyAudio: Active PyAudio instance
        """
        if self._pa_instance is None:
            self._pa_instance = pyaudio.PyAudio()
        return self._pa_instance

    @property
    def supports_wasapi_loopback(self) -> bool:
        """
        Check if backend supports WASAPI loopback (system audio recording).

        Returns:
            bool: True if PyAudioWPatch is being used
        """
        return self.using_wpatch

    @property
    def supports_context_manager(self) -> bool:
        """
        Check if backend supports context manager for automatic cleanup.

        Returns:
            bool: True if PyAudioWPatch is being used
        """
        return self.using_wpatch

    # ========================================================================
    # Standard PyAudio Methods (Available in Both Backends)
    # ========================================================================

    def get_device_count(self) -> int:
        """Get number of audio devices."""
        return self.pa_instance.get_device_count()

    def get_device_info_by_index(self, device_index: int) -> Dict[str, Any]:
        """Get device information by index."""
        return self.pa_instance.get_device_info_by_index(device_index)

    def get_default_input_device_info(self) -> Dict[str, Any]:
        """Get default input device information."""
        return self.pa_instance.get_default_input_device_info()

    def get_default_output_device_info(self) -> Dict[str, Any]:
        """Get default output device information."""
        return self.pa_instance.get_default_output_device_info()

    def get_host_api_count(self) -> int:
        """Get number of host APIs."""
        return self.pa_instance.get_host_api_count()

    def get_host_api_info_by_index(self, host_api_index: int) -> Dict[str, Any]:
        """Get host API information by index."""
        return self.pa_instance.get_host_api_info_by_index(host_api_index)

    def is_format_supported(self, *args, **kwargs) -> bool:
        """Check if audio format is supported."""
        return self.pa_instance.is_format_supported(*args, **kwargs)

    def open(self, *args, **kwargs):
        """Open audio stream."""
        return self.pa_instance.open(*args, **kwargs)

    def terminate(self) -> None:
        """Terminate PyAudio instance."""
        if self._pa_instance is not None:
            self._pa_instance.terminate()
            self._pa_instance = None
            self.logger.debug(f"{self.backend_name} instance terminated")

    # ========================================================================
    # PyAudioWPatch-Specific Methods (Only Available with WPatch)
    # ========================================================================

    def get_default_wasapi_loopback(self) -> Optional[Dict[str, Any]]:
        """
        Get default WASAPI loopback device for system audio recording.

        Only available with PyAudioWPatch. Returns None if using standard PyAudio.

        Returns:
            Optional[Dict[str, Any]]: Loopback device info, or None if not available
        """
        if not self.using_wpatch:
            self.logger.debug("get_default_wasapi_loopback not available - using standard PyAudio")
            return None

        try:
            # PyAudioWPatch specific method
            loopback = self.pa_instance.get_default_wasapi_loopback()
            self.logger.debug(f"Found default WASAPI loopback device: {loopback.get('name', 'Unknown')}")
            return loopback
        except Exception as e:
            self.logger.warning(f"Failed to get WASAPI loopback device: {e}")
            return None

    def get_wasapi_loopback_analogue_by_index(self, device_index: int) -> Optional[Dict[str, Any]]:
        """
        Get WASAPI loopback analogue for a given device.

        Only available with PyAudioWPatch. Returns None if using standard PyAudio.

        Args:
            device_index: Index of the output device

        Returns:
            Optional[Dict[str, Any]]: Loopback device info, or None if not available
        """
        if not self.using_wpatch:
            self.logger.debug("get_wasapi_loopback_analogue_by_index not available - using standard PyAudio")
            return None

        try:
            # PyAudioWPatch specific method
            loopback = self.pa_instance.get_wasapi_loopback_analogue_by_index(device_index)
            self.logger.debug(f"Found WASAPI loopback analogue for device {device_index}")
            return loopback
        except Exception as e:
            self.logger.debug(f"No loopback analogue for device {device_index}: {e}")
            return None

    def print_detailed_system_info(self) -> None:
        """
        Print detailed audio system information.

        Only available with PyAudioWPatch. Logs warning if using standard PyAudio.
        """
        if not self.using_wpatch:
            self.logger.warning("print_detailed_system_info not available - using standard PyAudio")
            return

        try:
            # PyAudioWPatch specific method
            self.pa_instance.print_detailed_system_info()
        except Exception as e:
            self.logger.error(f"Failed to print detailed system info: {e}")

    def enumerate_loopback_devices(self) -> List[Dict[str, Any]]:
        """
        Enumerate all WASAPI loopback devices.

        Only available with PyAudioWPatch. Returns empty list if using standard PyAudio.

        Returns:
            List[Dict[str, Any]]: List of loopback device info dictionaries
        """
        if not self.using_wpatch:
            self.logger.debug("Loopback enumeration not available - using standard PyAudio")
            return []

        loopback_devices = []
        try:
            device_count = self.get_device_count()
            for i in range(device_count):
                try:
                    device_info = self.get_device_info_by_index(i)
                    # Check if device is a loopback device
                    if device_info.get('isLoopbackDevice', False):
                        loopback_devices.append(device_info)
                except Exception as e:
                    self.logger.debug(f"Error checking device {i} for loopback: {e}")
                    continue

            self.logger.info(f"Found {len(loopback_devices)} WASAPI loopback devices")
            return loopback_devices

        except Exception as e:
            self.logger.error(f"Failed to enumerate loopback devices: {e}")
            return []

    # ========================================================================
    # Context Manager Support (Automatic Cleanup)
    # ========================================================================

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatic cleanup."""
        self.terminate()
        return False

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get information about the current backend.

        Returns:
            Dict[str, Any]: Backend information
        """
        return {
            'backend_name': self.backend_name,
            'using_wpatch': self.using_wpatch,
            'supports_wasapi_loopback': self.supports_wasapi_loopback,
            'supports_context_manager': self.supports_context_manager,
            'portaudio_version': pyaudio.get_portaudio_version_text(),
            'portaudio_version_number': pyaudio.get_portaudio_version()
        }

    def log_backend_info(self) -> None:
        """Log detailed backend information."""
        info = self.get_backend_info()
        self.logger.info("=" * 60)
        self.logger.info("Audio Backend Information")
        self.logger.info("=" * 60)
        self.logger.info(f"Backend: {info['backend_name']}")
        self.logger.info(f"PortAudio Version: {info['portaudio_version']} ({info['portaudio_version_number']})")
        self.logger.info(f"WASAPI Loopback Support: {info['supports_wasapi_loopback']}")
        self.logger.info(f"Context Manager Support: {info['supports_context_manager']}")
        self.logger.info("=" * 60)


# Convenience function for direct PyAudio module access
def get_pyaudio_module():
    """
    Get the underlying pyaudio module (PyAudio or PyAudioWPatch).

    Useful for accessing module-level constants and functions.

    Returns:
        module: The pyaudio or pyaudiowpatch module
    """
    return pyaudio


# Module-level convenience constants
PYAUDIO_MODULE = pyaudio
IS_WPATCH = USING_WPATCH
