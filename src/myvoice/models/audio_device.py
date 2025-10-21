"""
AudioDevice Models

This module contains the AudioDevice data model for managing Windows audio devices
with validation for device availability and type checking.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Set
from enum import Enum

from myvoice.models.validation import ValidationResult, ValidationIssue, ValidationStatus

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Audio device types."""
    INPUT = "input"
    OUTPUT = "output"
    VIRTUAL = "virtual"
    UNKNOWN = "unknown"


class VirtualDeviceDriver(Enum):
    """Known virtual audio device drivers."""
    VB_CABLE = "vb-cable"
    VOICEMEETER = "voicemeeter"
    OBS_VIRTUAL_CAM = "obs"
    VIRTUAL_AUDIO_CABLE = "virtual-audio-cable"
    UNKNOWN_VIRTUAL = "unknown-virtual"


@dataclass
class AudioDevice:
    """
    Data model for Windows audio devices.

    Represents an audio device (microphone, speaker, or virtual device) available
    in the Windows system. Includes validation for device availability and metadata
    about the device capabilities.

    Attributes:
        device_id: Unique identifier for the audio device
        name: Human-readable name of the device
        device_type: Type of device (input/output/virtual)
        is_default: Whether this is the system default device
        is_available: Whether the device is currently available
        driver_name: Name of the audio driver
        host_api_name: Name of the host API (e.g., "Windows WASAPI", "Windows DirectSound")
        virtual_driver: Type of virtual driver if device is virtual
        capabilities: Set of device capabilities
    """
    device_id: str
    name: str
    device_type: DeviceType
    is_default: bool = False
    is_available: bool = False
    driver_name: Optional[str] = None
    host_api_name: Optional[str] = None
    virtual_driver: Optional[VirtualDeviceDriver] = None
    capabilities: Set[str] = None

    def __post_init__(self):
        """Initialize and validate the audio device."""
        if self.capabilities is None:
            self.capabilities = set()

        # Detect virtual device if not already set
        if self.virtual_driver is None and self.device_type == DeviceType.VIRTUAL:
            self.virtual_driver = self._detect_virtual_driver()

        # Validate the device
        validation_result = self.validate()
        self.is_available = validation_result.is_valid

    def _detect_virtual_driver(self) -> VirtualDeviceDriver:
        """
        Detect the type of virtual audio driver based on device name and driver.

        Returns:
            VirtualDeviceDriver: The detected virtual driver type
        """
        if not self.name:
            return VirtualDeviceDriver.UNKNOWN_VIRTUAL

        name_lower = self.name.lower()
        driver_lower = (self.driver_name or "").lower()

        # Voicemeeter detection (check first - more specific than VB-Audio)
        if any(pattern in name_lower for pattern in ["voicemeeter", "voice meeter", "vaio", "aux"]):
            return VirtualDeviceDriver.VOICEMEETER

        # VB-Audio Cable detection (check after Voicemeeter)
        if any(pattern in name_lower for pattern in ["cable"]) and "vb" in name_lower:
            return VirtualDeviceDriver.VB_CABLE

        # Generic VB-Audio detection (fallback for other VB-Audio products)
        if "vb-audio" in name_lower or "vb audio" in name_lower:
            return VirtualDeviceDriver.VB_CABLE

        # OBS Virtual Camera detection
        if any(pattern in name_lower for pattern in ["obs", "virtual cam"]):
            return VirtualDeviceDriver.OBS_VIRTUAL_CAM

        # Generic virtual audio cable detection
        if any(pattern in name_lower for pattern in ["virtual", "loopback", "line"]):
            return VirtualDeviceDriver.VIRTUAL_AUDIO_CABLE

        return VirtualDeviceDriver.UNKNOWN_VIRTUAL

    def is_virtual_device(self) -> bool:
        """
        Check if this device is a virtual audio device.

        Returns:
            bool: True if this is a virtual device
        """
        return self.device_type == DeviceType.VIRTUAL

    def is_vb_cable(self) -> bool:
        """
        Check if this device is a VB-Audio Cable device.

        Returns:
            bool: True if this is a VB-Audio Cable device
        """
        return self.virtual_driver == VirtualDeviceDriver.VB_CABLE

    def is_voicemeeter(self) -> bool:
        """
        Check if this device is a Voicemeeter device.

        Returns:
            bool: True if this is a Voicemeeter device
        """
        return self.virtual_driver == VirtualDeviceDriver.VOICEMEETER

    def get_virtual_capabilities(self) -> Set[str]:
        """
        Get capabilities specific to virtual audio devices.

        Returns:
            Set[str]: Set of virtual device capabilities
        """
        if not self.is_virtual_device():
            return set()

        base_capabilities = {"loopback", "routing"}

        if self.virtual_driver == VirtualDeviceDriver.VB_CABLE:
            base_capabilities.update({"stereo", "low_latency", "asio_support"})
        elif self.virtual_driver == VirtualDeviceDriver.VOICEMEETER:
            base_capabilities.update({"mixing", "eq", "compressor", "multi_channel"})
        elif self.virtual_driver == VirtualDeviceDriver.OBS_VIRTUAL_CAM:
            base_capabilities.update({"streaming", "recording"})

        return base_capabilities

    def validate_virtual_device_capabilities(self) -> ValidationResult:
        """
        Validate capabilities specific to virtual devices.

        Returns:
            ValidationResult: Validation result for virtual device capabilities
        """
        if not self.is_virtual_device():
            return ValidationResult(
                is_valid=True,
                status=ValidationStatus.VALID,
                issues=[],
                warnings=[],
                summary="Not a virtual device, no virtual validation needed"
            )

        issues = []
        warnings = []

        # Check if virtual driver is detected
        if self.virtual_driver == VirtualDeviceDriver.UNKNOWN_VIRTUAL:
            warnings.append(ValidationIssue(
                field="virtual_driver",
                message="Virtual device driver type could not be determined",
                code="UNKNOWN_VIRTUAL_DRIVER",
                severity=ValidationStatus.WARNING
            ))

        # Check for required capabilities
        virtual_caps = self.get_virtual_capabilities()
        if not virtual_caps:
            warnings.append(ValidationIssue(
                field="capabilities",
                message="No virtual device capabilities detected",
                code="NO_VIRTUAL_CAPABILITIES",
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

        return ValidationResult(
            is_valid=is_valid,
            status=status,
            issues=issues,
            warnings=warnings,
            summary=f"Virtual device validation complete for {self.virtual_driver.value if self.virtual_driver else 'unknown'} driver"
        )

    def validate(self) -> ValidationResult:
        """
        Validate the audio device for required fields and constraints.

        Returns:
            ValidationResult: Detailed validation result with issues and warnings
        """
        issues = []
        warnings = []

        try:
            # Device ID validation
            if not self.device_id or not self.device_id.strip():
                issues.append(ValidationIssue(
                    field="device_id",
                    message="Device ID cannot be empty",
                    code="EMPTY_DEVICE_ID",
                    severity=ValidationStatus.INVALID
                ))

            # Name validation
            if not self.name or not self.name.strip():
                issues.append(ValidationIssue(
                    field="name",
                    message="Device name cannot be empty",
                    code="EMPTY_DEVICE_NAME",
                    severity=ValidationStatus.INVALID
                ))

            # Device type validation
            if not isinstance(self.device_type, DeviceType):
                issues.append(ValidationIssue(
                    field="device_type",
                    message="Device type must be a valid DeviceType enum",
                    code="INVALID_DEVICE_TYPE",
                    severity=ValidationStatus.INVALID
                ))
            elif self.device_type == DeviceType.UNKNOWN:
                warnings.append(ValidationIssue(
                    field="device_type",
                    message="Device type is unknown, functionality may be limited",
                    code="UNKNOWN_DEVICE_TYPE",
                    severity=ValidationStatus.WARNING
                ))

            # Driver name validation
            if self.driver_name is not None and len(self.driver_name.strip()) == 0:
                warnings.append(ValidationIssue(
                    field="driver_name",
                    message="Driver name is empty string, consider setting to None",
                    code="EMPTY_DRIVER_NAME",
                    severity=ValidationStatus.WARNING
                ))

            # Logical validations
            if self.is_default and self.device_type == DeviceType.UNKNOWN:
                warnings.append(ValidationIssue(
                    field="is_default",
                    message="Unknown device type is set as default, this may cause issues",
                    code="DEFAULT_UNKNOWN_TYPE",
                    severity=ValidationStatus.WARNING
                ))

            # Virtual device specific validation
            if self.device_type == DeviceType.VIRTUAL:
                virtual_validation = self.validate_virtual_device_capabilities()
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
            logger.exception(f"Error during audio device validation: {e}")
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

    def check_availability(self) -> bool:
        """
        Check if the audio device is currently available.

        This method can be extended to perform actual Windows API calls
        to verify device availability in real-time.

        Returns:
            bool: True if device is available, False otherwise
        """
        try:
            # For now, return the cached availability status
            # This can be enhanced with actual Windows API calls
            # using sounddevice or PyAudio to check real device status
            return self.is_available
        except Exception as e:
            logger.error(f"Failed to check availability for device {self.device_id}: {e}")
            return False

    def get_device_info(self) -> Dict[str, Any]:
        """
        Get comprehensive device information.

        Returns:
            dict: Device information including all attributes and metadata
        """
        info = {
            'device_id': self.device_id,
            'name': self.name,
            'device_type': self.device_type.value,
            'is_default': self.is_default,
            'is_available': self.is_available,
            'driver_name': self.driver_name,
            'validation_status': 'valid' if self.is_available else 'invalid'
        }

        # Add virtual device specific information
        if self.is_virtual_device():
            info.update({
                'virtual_driver': self.virtual_driver.value if self.virtual_driver else None,
                'virtual_capabilities': list(self.get_virtual_capabilities()),
                'is_vb_cable': self.is_vb_cable(),
                'is_voicemeeter': self.is_voicemeeter()
            })

        if self.capabilities:
            info['capabilities'] = list(self.capabilities)

        return info

    def __str__(self) -> str:
        """String representation of the audio device."""
        status = "AVAILABLE" if self.is_available else "UNAVAILABLE"
        default_marker = " [DEFAULT]" if self.is_default else ""
        return f"[{status}] {self.name} ({self.device_type.value}){default_marker}"

    def __repr__(self) -> str:
        """Developer representation of the audio device."""
        return (f"AudioDevice(device_id='{self.device_id}', name='{self.name}', "
                f"device_type={self.device_type}, is_default={self.is_default}, "
                f"is_available={self.is_available})")

    @classmethod
    def create_from_windows_api(cls, device_data: Dict[str, Any]) -> 'AudioDevice':
        """
        Factory method to create an AudioDevice from Windows API data.

        This method handles the conversion from various Windows audio API formats
        (PyAudio, sounddevice, etc.) to the standardized AudioDevice model.

        Args:
            device_data: Dictionary containing device information from Windows API
                        Expected keys: id, name, type, is_default, driver (optional)

        Returns:
            AudioDevice: Validated audio device instance

        Examples:
            # Create from PyAudio device info
            device = AudioDevice.create_from_windows_api({
                'id': 'dev_1_2',
                'name': 'Microphone (Realtek)',
                'type': 'input',
                'is_default': True,
                'driver': 'MME'
            })

            # Create from sounddevice device info
            device = AudioDevice.create_from_windows_api({
                'id': 'CABLE-A',
                'name': 'CABLE Input (VB-Audio Cable A)',
                'type': 'virtual',
                'is_default': False
            })
        """
        # Extract and normalize device data
        device_id = str(device_data.get('id', ''))
        name = str(device_data.get('name', ''))
        driver_name = device_data.get('driver')

        # Parse device type
        type_str = str(device_data.get('type', 'unknown')).lower()
        try:
            if type_str in ['input', 'microphone', 'mic']:
                device_type = DeviceType.INPUT
            elif type_str in ['output', 'speaker', 'speakers']:
                device_type = DeviceType.OUTPUT
            elif type_str in ['virtual', 'cable', 'loopback']:
                device_type = DeviceType.VIRTUAL
            else:
                device_type = DeviceType.UNKNOWN
        except (AttributeError, ValueError):
            device_type = DeviceType.UNKNOWN

        # Parse default status
        is_default = bool(device_data.get('is_default', False))

        logger.debug(f"Creating audio device from Windows API: {name} ({device_type.value})")

        # Create and return the device (validation happens in __post_init__)
        device = cls(
            device_id=device_id,
            name=name,
            device_type=device_type,
            is_default=is_default,
            is_available=True,  # Assume available if coming from API
            driver_name=driver_name
        )

        logger.debug(f"Created audio device: {device}")
        return device

    @classmethod
    def create_from_pyaudio_info(cls, device_index: int, device_info: Dict[str, Any],
                                 host_api_name: Optional[str] = None) -> 'AudioDevice':
        """
        Factory method to create an AudioDevice from PyAudio device info.

        Args:
            device_index: PyAudio device index
            device_info: PyAudio device info dictionary
            host_api_name: Host API name (e.g., "Windows WASAPI") for metadata matching

        Returns:
            AudioDevice: Validated audio device instance
        """
        # Convert PyAudio info to standardized format
        max_input_channels = device_info.get('maxInputChannels', 0)
        max_output_channels = device_info.get('maxOutputChannels', 0)

        # Determine device type based on channel capabilities
        if max_input_channels > 0 and max_output_channels > 0:
            device_type = DeviceType.VIRTUAL  # Can do both
        elif max_input_channels > 0:
            device_type = DeviceType.INPUT
        elif max_output_channels > 0:
            device_type = DeviceType.OUTPUT
        else:
            device_type = DeviceType.UNKNOWN

        standardized_data = {
            'id': f"pyaudio_{device_index}",
            'name': device_info.get('name', f'Device {device_index}'),
            'type': device_type.value,
            'is_default': device_index == device_info.get('defaultSampleRate', -1),  # PyAudio default logic
            'driver': 'PyAudio',
            'host_api': host_api_name  # Add host API for device matching
        }

        device = cls.create_from_windows_api(standardized_data)
        # Set host_api_name directly since create_from_windows_api doesn't handle it yet
        device.host_api_name = host_api_name
        return device

    @classmethod
    def create_from_sounddevice_info(cls, device_index: int, device_info: Dict[str, Any]) -> 'AudioDevice':
        """
        Factory method to create an AudioDevice from sounddevice device info.

        Args:
            device_index: sounddevice device index
            device_info: sounddevice device info dictionary

        Returns:
            AudioDevice: Validated audio device instance
        """
        # Convert sounddevice info to standardized format
        max_input_channels = device_info.get('max_input_channels', 0)
        max_output_channels = device_info.get('max_output_channels', 0)

        # Determine device type based on channel capabilities
        if max_input_channels > 0 and max_output_channels > 0:
            device_type = DeviceType.VIRTUAL  # Can do both
        elif max_input_channels > 0:
            device_type = DeviceType.INPUT
        elif max_output_channels > 0:
            device_type = DeviceType.OUTPUT
        else:
            device_type = DeviceType.UNKNOWN

        standardized_data = {
            'id': f"sounddevice_{device_index}",
            'name': device_info.get('name', f'Device {device_index}'),
            'type': device_type.value,
            'is_default': False,  # sounddevice handles defaults differently
            'driver': device_info.get('hostapi', 'Unknown')
        }

        return cls.create_from_windows_api(standardized_data)

    @classmethod
    def create_virtual_device(cls, device_id: str, name: str,
                             virtual_driver: Optional[VirtualDeviceDriver] = None,
                             is_default: bool = False) -> 'AudioDevice':
        """
        Factory method to create a virtual audio device.

        Args:
            device_id: Unique identifier for the virtual device
            name: Human-readable name of the virtual device
            virtual_driver: Type of virtual driver (auto-detected if None)
            is_default: Whether this is the default virtual device

        Returns:
            AudioDevice: Validated virtual audio device instance

        Examples:
            # Create VB-Audio Cable device
            vb_cable = AudioDevice.create_virtual_device(
                device_id="vb_cable_1",
                name="CABLE Input (VB-Audio Cable A)",
                virtual_driver=VirtualDeviceDriver.VB_CABLE
            )

            # Create Voicemeeter device with auto-detection
            voicemeeter = AudioDevice.create_virtual_device(
                device_id="vm_input",
                name="Voicemeeter Input (VB-Audio Voicemeeter VAIO)"
            )
        """
        device = cls(
            device_id=device_id,
            name=name,
            device_type=DeviceType.VIRTUAL,
            is_default=is_default,
            is_available=True,
            virtual_driver=virtual_driver
        )

        logger.debug(f"Created virtual audio device: {device}")
        return device

    @classmethod
    def create_vb_cable_device(cls, device_id: str, name: str,
                              cable_letter: str = "A") -> 'AudioDevice':
        """
        Factory method to create a VB-Audio Cable device.

        Args:
            device_id: Unique identifier for the VB-Cable device
            name: Device name (will be normalized if needed)
            cable_letter: Cable identifier letter (A, B, C, etc.)

        Returns:
            AudioDevice: VB-Audio Cable device instance
        """
        # Normalize name to standard VB-Cable format if needed
        if "cable" not in name.lower():
            name = f"CABLE Input (VB-Audio Cable {cable_letter})"

        device = cls.create_virtual_device(
            device_id=device_id,
            name=name,
            virtual_driver=VirtualDeviceDriver.VB_CABLE
        )

        # Add VB-Cable specific capabilities
        device.capabilities.update({"stereo", "low_latency", "asio_support", "loopback"})

        logger.info(f"Created VB-Audio Cable device: {device}")
        return device

    @classmethod
    def create_voicemeeter_device(cls, device_id: str, name: str,
                                 vm_type: str = "VAIO") -> 'AudioDevice':
        """
        Factory method to create a Voicemeeter virtual device.

        Args:
            device_id: Unique identifier for the Voicemeeter device
            name: Device name (will be normalized if needed)
            vm_type: Voicemeeter type (VAIO, AUX, etc.)

        Returns:
            AudioDevice: Voicemeeter device instance
        """
        # Normalize name to standard Voicemeeter format if needed
        if "voicemeeter" not in name.lower():
            name = f"Voicemeeter Input (VB-Audio Voicemeeter {vm_type})"

        device = cls.create_virtual_device(
            device_id=device_id,
            name=name,
            virtual_driver=VirtualDeviceDriver.VOICEMEETER
        )

        # Add Voicemeeter specific capabilities
        device.capabilities.update({"mixing", "eq", "compressor", "multi_channel", "loopback"})

        logger.info(f"Created Voicemeeter device: {device}")
        return device

    @classmethod
    def create_from_system_enumeration(cls, system_devices: List[Dict[str, Any]]) -> List['AudioDevice']:
        """
        Factory method to create AudioDevice instances from system device enumeration.

        This method processes a list of devices from system APIs and creates appropriate
        AudioDevice instances, automatically detecting virtual devices.

        Args:
            system_devices: List of device dictionaries from system enumeration
                          Expected format:
                          {
                              'id': 'device_id',
                              'name': 'Device Name',
                              'type': 'input|output|virtual',
                              'is_default': bool,
                              'driver': 'driver_name' (optional),
                              'max_input_channels': int (optional),
                              'max_output_channels': int (optional)
                          }

        Returns:
            List[AudioDevice]: List of validated AudioDevice instances

        Examples:
            # Process devices from Windows API
            devices = AudioDevice.create_from_system_enumeration([
                {
                    'id': 'mic_1',
                    'name': 'Microphone (Realtek)',
                    'type': 'input',
                    'is_default': True,
                    'driver': 'MME'
                },
                {
                    'id': 'cable_a',
                    'name': 'CABLE Input (VB-Audio Cable A)',
                    'type': 'virtual',
                    'is_default': False,
                    'driver': 'VB-Cable'
                }
            ])
        """
        audio_devices = []

        for device_data in system_devices:
            try:
                # Determine if this is a virtual device by name/driver analysis
                name = device_data.get('name', '')
                driver = device_data.get('driver', '')

                # Auto-detect virtual devices
                is_virtual = cls._is_virtual_device_by_name(name, driver)

                if is_virtual:
                    # Create virtual device with auto-detection
                    device = cls.create_virtual_device(
                        device_id=str(device_data.get('id', '')),
                        name=name,
                        is_default=bool(device_data.get('is_default', False))
                    )
                else:
                    # Create regular device using existing factory method
                    device = cls.create_from_windows_api(device_data)

                audio_devices.append(device)

            except Exception as e:
                logger.error(f"Failed to create device from system data {device_data}: {e}")
                continue

        logger.debug(f"Created {len(audio_devices)} audio devices from system enumeration")
        return audio_devices

    @staticmethod
    def _is_virtual_device_by_name(name: str, driver: str = "") -> bool:
        """
        Determine if a device is virtual based on its name and driver.

        Args:
            name: Device name
            driver: Driver name (optional)

        Returns:
            bool: True if device appears to be virtual
        """
        if not name:
            return False

        name_lower = name.lower()
        driver_lower = driver.lower()

        # Known virtual device patterns (specific to general)
        virtual_patterns = [
            "voicemeeter", "voice meeter", "vaio", "aux",
            "vb-audio", "cable", "obs", "virtual cam",
            "virtual audio", "loopback", "line 1", "line 2"
        ]

        # Check name patterns
        for pattern in virtual_patterns:
            if pattern in name_lower:
                return True

        # Check driver patterns
        virtual_drivers = ["vb-cable", "voicemeeter", "obs"]
        for vdriver in virtual_drivers:
            if vdriver in driver_lower:
                return True

        return False