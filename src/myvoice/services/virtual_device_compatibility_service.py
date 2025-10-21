"""
Virtual Device Compatibility Service

This service provides enhanced compatibility handling for virtual audio devices,
including VB-Audio Cable and Voicemeeter detection, installation guidance, and
graceful degradation when virtual drivers are unavailable.
"""

import logging
import os
import subprocess
import winreg
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from myvoice.models.audio_device import AudioDevice, VirtualDeviceDriver
from myvoice.models.error import MyVoiceError, ErrorSeverity
from myvoice.services.core.base_service import BaseService, ServiceStatus


class VirtualDriverStatus(Enum):
    """Status of virtual audio drivers."""
    INSTALLED = "installed"
    NOT_INSTALLED = "not_installed"
    PARTIALLY_INSTALLED = "partially_installed"
    UNKNOWN = "unknown"
    ERROR = "error"


class FallbackStrategy(Enum):
    """Fallback strategies when virtual drivers are unavailable."""
    MONITOR_ONLY = "monitor_only"
    SYSTEM_DEFAULT = "system_default"
    DISABLE_VIRTUAL = "disable_virtual"
    USER_GUIDANCE = "user_guidance"


@dataclass
class VirtualDriverInfo:
    """Information about a virtual audio driver."""
    driver_type: VirtualDeviceDriver
    status: VirtualDriverStatus
    version: Optional[str] = None
    install_path: Optional[str] = None
    devices_found: List[AudioDevice] = None
    compatibility_issues: List[str] = None
    recommendations: List[str] = None

    def __post_init__(self):
        if self.devices_found is None:
            self.devices_found = []
        if self.compatibility_issues is None:
            self.compatibility_issues = []
        if self.recommendations is None:
            self.recommendations = []


@dataclass
class VirtualDeviceGuidance:
    """User guidance for virtual device installation and configuration."""
    driver_type: VirtualDeviceDriver
    installation_required: bool
    download_url: str
    installation_steps: List[str]
    compatibility_notes: List[str]
    troubleshooting_tips: List[str]


class VirtualDeviceCompatibilityService(BaseService):
    """
    Service for managing virtual audio device compatibility.

    This service provides comprehensive virtual device support including:
    - Detection of VB-Audio Cable and Voicemeeter installations
    - Compatibility checking and validation
    - Fallback mechanisms when drivers are unavailable
    - User guidance for installation and configuration
    - Graceful degradation of functionality
    """

    def __init__(self):
        """Initialize the Virtual Device Compatibility Service."""
        super().__init__("VirtualDeviceCompatibilityService")
        self.logger = logging.getLogger(self.__class__.__name__)

        # Driver detection cache
        self._driver_cache: Dict[VirtualDeviceDriver, VirtualDriverInfo] = {}
        self._cache_valid = False

        # Fallback configuration
        self.fallback_strategy = FallbackStrategy.USER_GUIDANCE
        self.enable_fallback_notifications = True

        # Known registry paths for virtual drivers
        self._registry_paths = {
            VirtualDeviceDriver.VB_CABLE: [
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\VB-Audio Cable",
                r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\VB-Audio Cable",
                r"SOFTWARE\VB-Audio\CABLE"
            ],
            VirtualDeviceDriver.VOICEMEETER: [
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\VB-Audio Voicemeeter",
                r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\VB-Audio Voicemeeter",
                r"SOFTWARE\VB-Audio\Voicemeeter"
            ]
        }

        # Known installation paths
        self._install_paths = {
            VirtualDeviceDriver.VB_CABLE: [
                r"C:\Program Files\VB\CABLE",
                r"C:\Program Files (x86)\VB\CABLE"
            ],
            VirtualDeviceDriver.VOICEMEETER: [
                r"C:\Program Files\VB\Voicemeeter",
                r"C:\Program Files (x86)\VB\Voicemeeter"
            ]
        }

        self.logger.info("Virtual Device Compatibility Service initialized")

    async def detect_virtual_drivers(self, force_refresh: bool = False) -> Dict[VirtualDeviceDriver, VirtualDriverInfo]:
        """
        Detect installed virtual audio drivers.

        Args:
            force_refresh: Force refresh of detection cache

        Returns:
            Dict[VirtualDeviceDriver, VirtualDriverInfo]: Detection results for each driver type
        """
        try:
            if self._cache_valid and not force_refresh:
                self.logger.debug("Using cached virtual driver detection results")
                return self._driver_cache

            self.logger.info("Detecting virtual audio drivers")
            detection_results = {}

            # Detect VB-Audio Cable
            vb_cable_info = await self._detect_vb_cable()
            detection_results[VirtualDeviceDriver.VB_CABLE] = vb_cable_info

            # Detect Voicemeeter
            voicemeeter_info = await self._detect_voicemeeter()
            detection_results[VirtualDeviceDriver.VOICEMEETER] = voicemeeter_info

            # Update cache
            self._driver_cache = detection_results
            self._cache_valid = True

            # Log summary
            installed_count = sum(1 for info in detection_results.values()
                                if info.status == VirtualDriverStatus.INSTALLED)
            self.logger.info(f"Virtual driver detection complete: {installed_count} drivers installed")

            return detection_results

        except Exception as e:
            self.logger.error(f"Failed to detect virtual drivers: {e}")
            raise MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="VIRTUAL_DRIVER_DETECTION_FAILED",
                user_message="Failed to scan for virtual audio drivers",
                technical_details=str(e),
                suggested_action="Try running as administrator or check system permissions"
            )

    async def _detect_vb_cable(self) -> VirtualDriverInfo:
        """
        Detect VB-Audio Cable installation.

        Returns:
            VirtualDriverInfo: VB-Cable detection information
        """
        try:
            self.logger.debug("Detecting VB-Audio Cable")

            # Check registry for installation
            registry_detected, version, install_path = self._check_registry_installation(
                VirtualDeviceDriver.VB_CABLE
            )

            # Check file system for installation
            filesystem_detected, fs_install_path = self._check_filesystem_installation(
                VirtualDeviceDriver.VB_CABLE
            )

            # Use registry path if available, otherwise filesystem path
            detected_path = install_path or fs_install_path

            # Determine status
            if registry_detected and filesystem_detected:
                status = VirtualDriverStatus.INSTALLED
            elif registry_detected or filesystem_detected:
                status = VirtualDriverStatus.PARTIALLY_INSTALLED
            else:
                status = VirtualDriverStatus.NOT_INSTALLED

            # Create driver info
            driver_info = VirtualDriverInfo(
                driver_type=VirtualDeviceDriver.VB_CABLE,
                status=status,
                version=version,
                install_path=detected_path
            )

            # Add compatibility analysis
            self._analyze_vb_cable_compatibility(driver_info)

            self.logger.debug(f"VB-Audio Cable detection: {status.value}")
            return driver_info

        except Exception as e:
            self.logger.error(f"Error detecting VB-Audio Cable: {e}")
            return VirtualDriverInfo(
                driver_type=VirtualDeviceDriver.VB_CABLE,
                status=VirtualDriverStatus.ERROR,
                compatibility_issues=[f"Detection error: {str(e)}"]
            )

    async def _detect_voicemeeter(self) -> VirtualDriverInfo:
        """
        Detect Voicemeeter installation.

        Returns:
            VirtualDriverInfo: Voicemeeter detection information
        """
        try:
            self.logger.debug("Detecting Voicemeeter")

            # Check registry for installation
            registry_detected, version, install_path = self._check_registry_installation(
                VirtualDeviceDriver.VOICEMEETER
            )

            # Check file system for installation
            filesystem_detected, fs_install_path = self._check_filesystem_installation(
                VirtualDeviceDriver.VOICEMEETER
            )

            # Use registry path if available, otherwise filesystem path
            detected_path = install_path or fs_install_path

            # Determine status
            if registry_detected and filesystem_detected:
                status = VirtualDriverStatus.INSTALLED
            elif registry_detected or filesystem_detected:
                status = VirtualDriverStatus.PARTIALLY_INSTALLED
            else:
                status = VirtualDriverStatus.NOT_INSTALLED

            # Create driver info
            driver_info = VirtualDriverInfo(
                driver_type=VirtualDeviceDriver.VOICEMEETER,
                status=status,
                version=version,
                install_path=detected_path
            )

            # Add compatibility analysis
            self._analyze_voicemeeter_compatibility(driver_info)

            self.logger.debug(f"Voicemeeter detection: {status.value}")
            return driver_info

        except Exception as e:
            self.logger.error(f"Error detecting Voicemeeter: {e}")
            return VirtualDriverInfo(
                driver_type=VirtualDeviceDriver.VOICEMEETER,
                status=VirtualDriverStatus.ERROR,
                compatibility_issues=[f"Detection error: {str(e)}"]
            )

    def _check_registry_installation(self, driver_type: VirtualDeviceDriver) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check Windows registry for driver installation.

        Args:
            driver_type: Type of virtual driver to check

        Returns:
            Tuple[bool, Optional[str], Optional[str]]: (detected, version, install_path)
        """
        try:
            registry_paths = self._registry_paths.get(driver_type, [])

            for reg_path in registry_paths:
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as key:
                        # Try to get version
                        version = None
                        try:
                            version, _ = winreg.QueryValueEx(key, "DisplayVersion")
                        except FileNotFoundError:
                            try:
                                version, _ = winreg.QueryValueEx(key, "Version")
                            except FileNotFoundError:
                                pass

                        # Try to get install location
                        install_path = None
                        try:
                            install_path, _ = winreg.QueryValueEx(key, "InstallLocation")
                        except FileNotFoundError:
                            try:
                                install_path, _ = winreg.QueryValueEx(key, "UninstallString")
                                # Extract directory from uninstall string
                                if install_path:
                                    install_path = str(Path(install_path).parent)
                            except FileNotFoundError:
                                pass

                        self.logger.debug(f"Registry detection for {driver_type.value}: version={version}, path={install_path}")
                        return True, version, install_path

                except FileNotFoundError:
                    continue
                except Exception as e:
                    self.logger.warning(f"Error reading registry path {reg_path}: {e}")
                    continue

            return False, None, None

        except Exception as e:
            self.logger.error(f"Error checking registry for {driver_type.value}: {e}")
            return False, None, None

    def _check_filesystem_installation(self, driver_type: VirtualDeviceDriver) -> Tuple[bool, Optional[str]]:
        """
        Check filesystem for driver installation.

        Args:
            driver_type: Type of virtual driver to check

        Returns:
            Tuple[bool, Optional[str]]: (detected, install_path)
        """
        try:
            install_paths = self._install_paths.get(driver_type, [])

            for install_path in install_paths:
                if os.path.exists(install_path):
                    # Check for key files to confirm installation
                    if driver_type == VirtualDeviceDriver.VB_CABLE:
                        key_files = ["vbaudio_cable64.sys", "vbaudio_cable32.sys", "VBCABLE_Setup.exe"]
                    elif driver_type == VirtualDeviceDriver.VOICEMEETER:
                        key_files = ["voicemeeter.exe", "voicemeeter8.exe", "voicemeeter8x64.exe"]
                    else:
                        key_files = []

                    # Check if any key files exist
                    if any(os.path.exists(os.path.join(install_path, file)) for file in key_files):
                        self.logger.debug(f"Filesystem detection for {driver_type.value}: {install_path}")
                        return True, install_path

            return False, None

        except Exception as e:
            self.logger.error(f"Error checking filesystem for {driver_type.value}: {e}")
            return False, None

    def _analyze_vb_cable_compatibility(self, driver_info: VirtualDriverInfo) -> None:
        """
        Analyze VB-Audio Cable compatibility and add recommendations.

        Args:
            driver_info: VB-Cable driver information to analyze
        """
        try:
            if driver_info.status == VirtualDriverStatus.NOT_INSTALLED:
                driver_info.recommendations.extend([
                    "Install VB-Audio Cable for virtual microphone functionality",
                    "Download from: https://vb-audio.com/Cable/",
                    "Restart computer after installation"
                ])
            elif driver_info.status == VirtualDriverStatus.PARTIALLY_INSTALLED:
                driver_info.compatibility_issues.append("VB-Audio Cable installation appears incomplete")
                driver_info.recommendations.extend([
                    "Reinstall VB-Audio Cable",
                    "Run installation as administrator",
                    "Ensure Windows audio service is running"
                ])
            elif driver_info.status == VirtualDriverStatus.INSTALLED:
                driver_info.recommendations.extend([
                    "VB-Audio Cable is properly installed",
                    "Use CABLE Input device for virtual microphone routing"
                ])

                # Check for common issues
                if driver_info.install_path and not os.path.exists(os.path.join(driver_info.install_path, "vbaudio_cable64.sys")):
                    driver_info.compatibility_issues.append("Missing 64-bit driver component")
                    driver_info.recommendations.append("Reinstall with 64-bit support")

        except Exception as e:
            self.logger.warning(f"Error analyzing VB-Cable compatibility: {e}")

    def _analyze_voicemeeter_compatibility(self, driver_info: VirtualDriverInfo) -> None:
        """
        Analyze Voicemeeter compatibility and add recommendations.

        Args:
            driver_info: Voicemeeter driver information to analyze
        """
        try:
            if driver_info.status == VirtualDriverStatus.NOT_INSTALLED:
                driver_info.recommendations.extend([
                    "Install Voicemeeter for advanced virtual audio mixing",
                    "Download from: https://vb-audio.com/Voicemeeter/",
                    "Choose Voicemeeter, Voicemeeter Banana, or Voicemeeter Potato based on needs"
                ])
            elif driver_info.status == VirtualDriverStatus.PARTIALLY_INSTALLED:
                driver_info.compatibility_issues.append("Voicemeeter installation appears incomplete")
                driver_info.recommendations.extend([
                    "Reinstall Voicemeeter",
                    "Run installation as administrator",
                    "Restart Voicemeeter application after installation"
                ])
            elif driver_info.status == VirtualDriverStatus.INSTALLED:
                driver_info.recommendations.extend([
                    "Voicemeeter is properly installed",
                    "Use VAIO device for virtual microphone routing",
                    "Start Voicemeeter application before using virtual devices"
                ])

                # Check if Voicemeeter is currently running
                if not self._is_voicemeeter_running():
                    driver_info.compatibility_issues.append("Voicemeeter application is not running")
                    driver_info.recommendations.append("Start Voicemeeter application for virtual devices to work")

        except Exception as e:
            self.logger.warning(f"Error analyzing Voicemeeter compatibility: {e}")

    def _is_voicemeeter_running(self) -> bool:
        """
        Check if Voicemeeter application is currently running.

        Returns:
            bool: True if Voicemeeter is running
        """
        try:
            # Check for common Voicemeeter process names
            import psutil
            voicemeeter_processes = ["voicemeeter.exe", "voicemeeter8.exe", "voicemeeter8x64.exe"]

            for proc in psutil.process_iter(['name']):
                if proc.info['name'].lower() in [p.lower() for p in voicemeeter_processes]:
                    return True

            return False

        except Exception:
            # If psutil is not available or check fails, assume it might be running
            return True

    async def get_virtual_device_guidance(self, driver_type: VirtualDeviceDriver) -> VirtualDeviceGuidance:
        """
        Get user guidance for virtual device installation and configuration.

        Args:
            driver_type: Type of virtual driver to get guidance for

        Returns:
            VirtualDeviceGuidance: Installation and configuration guidance
        """
        try:
            # Detect current status
            detection_results = await self.detect_virtual_drivers()
            driver_info = detection_results.get(driver_type)

            if driver_type == VirtualDeviceDriver.VB_CABLE:
                return self._create_vb_cable_guidance(driver_info)
            elif driver_type == VirtualDeviceDriver.VOICEMEETER:
                return self._create_voicemeeter_guidance(driver_info)
            else:
                # Generic virtual device guidance
                return VirtualDeviceGuidance(
                    driver_type=driver_type,
                    installation_required=True,
                    download_url="",
                    installation_steps=["Manual installation required"],
                    compatibility_notes=["Unknown virtual device type"],
                    troubleshooting_tips=["Check device manufacturer documentation"]
                )

        except Exception as e:
            self.logger.error(f"Failed to get guidance for {driver_type.value}: {e}")
            raise MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="GUIDANCE_GENERATION_FAILED",
                user_message=f"Failed to generate guidance for {driver_type.value}",
                technical_details=str(e)
            )

    def _create_vb_cable_guidance(self, driver_info: Optional[VirtualDriverInfo]) -> VirtualDeviceGuidance:
        """Create VB-Audio Cable installation guidance."""
        installation_required = not (driver_info and driver_info.status == VirtualDriverStatus.INSTALLED)

        installation_steps = [
            "Download VB-Audio Cable from https://vb-audio.com/Cable/",
            "Run the installer as Administrator",
            "Follow the installation wizard",
            "Restart your computer when prompted",
            "Verify 'CABLE Input' appears in Windows audio devices"
        ]

        compatibility_notes = [
            "Compatible with Windows 7, 8, 10, and 11",
            "Supports both 32-bit and 64-bit systems",
            "Requires administrator privileges for installation",
            "May require Windows audio service restart"
        ]

        troubleshooting_tips = [
            "If devices don't appear, restart Windows Audio service",
            "Disable antivirus temporarily during installation",
            "Run Windows audio troubleshooter if issues persist",
            "Check Windows Event Viewer for driver errors"
        ]

        return VirtualDeviceGuidance(
            driver_type=VirtualDeviceDriver.VB_CABLE,
            installation_required=installation_required,
            download_url="https://vb-audio.com/Cable/",
            installation_steps=installation_steps,
            compatibility_notes=compatibility_notes,
            troubleshooting_tips=troubleshooting_tips
        )

    def _create_voicemeeter_guidance(self, driver_info: Optional[VirtualDriverInfo]) -> VirtualDeviceGuidance:
        """Create Voicemeeter installation guidance."""
        installation_required = not (driver_info and driver_info.status == VirtualDriverStatus.INSTALLED)

        installation_steps = [
            "Download Voicemeeter from https://vb-audio.com/Voicemeeter/",
            "Choose version: Basic, Banana (recommended), or Potato",
            "Run the installer as Administrator",
            "Follow the installation wizard",
            "Restart your computer when prompted",
            "Launch Voicemeeter application",
            "Verify 'VoiceMeeter Input (VAIO)' appears in Windows audio devices"
        ]

        compatibility_notes = [
            "Compatible with Windows 7, 8, 10, and 11",
            "Voicemeeter application must be running for virtual devices to work",
            "Provides advanced mixing and audio processing features",
            "Multiple versions available with different channel counts"
        ]

        troubleshooting_tips = [
            "Start Voicemeeter application before using virtual devices",
            "If devices don't appear, restart Voicemeeter application",
            "Check Voicemeeter System Settings for device configuration",
            "Ensure Windows audio service is running",
            "Use 'Run as Administrator' if permission issues occur"
        ]

        return VirtualDeviceGuidance(
            driver_type=VirtualDeviceDriver.VOICEMEETER,
            installation_required=installation_required,
            download_url="https://vb-audio.com/Voicemeeter/",
            installation_steps=installation_steps,
            compatibility_notes=compatibility_notes,
            troubleshooting_tips=troubleshooting_tips
        )

    async def validate_virtual_device_availability(self, virtual_devices: List[AudioDevice]) -> Dict[str, bool]:
        """
        Validate availability of specific virtual devices.

        Args:
            virtual_devices: List of virtual devices to validate

        Returns:
            Dict[str, bool]: Device ID to availability mapping
        """
        try:
            self.logger.debug(f"Validating {len(virtual_devices)} virtual devices")
            availability_results = {}

            for device in virtual_devices:
                if not device.is_virtual_device():
                    self.logger.warning(f"Device {device.name} is not a virtual device")
                    availability_results[device.device_id] = False
                    continue

                # Check if corresponding driver is installed
                driver_type = device.virtual_driver
                if driver_type in self._driver_cache:
                    driver_info = self._driver_cache[driver_type]
                    is_available = driver_info.status == VirtualDriverStatus.INSTALLED

                    # Additional check for Voicemeeter - must be running
                    if is_available and driver_type == VirtualDeviceDriver.VOICEMEETER:
                        is_available = self._is_voicemeeter_running()

                    availability_results[device.device_id] = is_available
                else:
                    # Driver not in cache, assume unavailable
                    availability_results[device.device_id] = False

            self.logger.debug(f"Virtual device validation complete")
            return availability_results

        except Exception as e:
            self.logger.error(f"Failed to validate virtual device availability: {e}")
            # Return all devices as unavailable on error
            return {device.device_id: False for device in virtual_devices}

    async def apply_fallback_strategy(self, fallback_strategy: FallbackStrategy) -> List[str]:
        """
        Apply fallback strategy when virtual devices are unavailable.

        Args:
            fallback_strategy: Strategy to apply

        Returns:
            List[str]: Actions taken or recommendations
        """
        try:
            self.logger.info(f"Applying fallback strategy: {fallback_strategy.value}")
            actions_taken = []

            if fallback_strategy == FallbackStrategy.MONITOR_ONLY:
                actions_taken.extend([
                    "Disabled virtual microphone functionality",
                    "Audio playback will use monitor speakers only",
                    "Install VB-Audio Cable or Voicemeeter for virtual microphone features"
                ])

            elif fallback_strategy == FallbackStrategy.SYSTEM_DEFAULT:
                actions_taken.extend([
                    "Using system default audio devices",
                    "Virtual microphone features are disabled",
                    "Standard audio playback through default speakers"
                ])

            elif fallback_strategy == FallbackStrategy.DISABLE_VIRTUAL:
                actions_taken.extend([
                    "Virtual audio features disabled",
                    "Application will operate without virtual microphone capability",
                    "Monitor audio playback still available"
                ])

            elif fallback_strategy == FallbackStrategy.USER_GUIDANCE:
                # Get guidance for all virtual driver types
                vb_guidance = await self.get_virtual_device_guidance(VirtualDeviceDriver.VB_CABLE)
                vm_guidance = await self.get_virtual_device_guidance(VirtualDeviceDriver.VOICEMEETER)

                actions_taken.extend([
                    "Virtual audio drivers not detected",
                    "Installation guidance provided for VB-Audio Cable and Voicemeeter",
                    "Application will operate with limited virtual audio functionality"
                ])

                if vb_guidance.installation_required:
                    actions_taken.append(f"VB-Audio Cable: {vb_guidance.download_url}")
                if vm_guidance.installation_required:
                    actions_taken.append(f"Voicemeeter: {vm_guidance.download_url}")

            self.logger.info(f"Fallback strategy applied: {len(actions_taken)} actions")
            return actions_taken

        except Exception as e:
            self.logger.error(f"Failed to apply fallback strategy: {e}")
            return [f"Fallback strategy failed: {str(e)}"]

    async def auto_detect_vb_cable_device(self) -> Optional[AudioDevice]:
        """
        Auto-detect VB-Cable Input device from available audio devices.

        Scans all available audio devices and looks for "CABLE Input" device name
        (case-insensitive) which is the VB-Cable virtual microphone device.

        Returns:
            Optional[AudioDevice]: Detected VB-Cable device or None if not found
        """
        try:
            self.logger.info("Auto-detecting VB-Cable device")

            # First check if VB-Cable driver is installed
            detection_results = await self.detect_virtual_drivers()
            vb_cable_info = detection_results.get(VirtualDeviceDriver.VB_CABLE)

            if not vb_cable_info or vb_cable_info.status != VirtualDriverStatus.INSTALLED:
                self.logger.info("VB-Cable driver not detected as installed")
                return None

            # Import PyAudio for device enumeration
            try:
                import pyaudio
                p = pyaudio.PyAudio()
            except ImportError:
                self.logger.warning("PyAudio not available for device enumeration")
                return None

            try:
                device_count = p.get_device_count()
                self.logger.debug(f"Scanning {device_count} audio devices for VB-Cable")

                for device_index in range(device_count):
                    try:
                        device_info = p.get_device_info_by_index(device_index)
                        device_name = device_info.get('name', '').strip()
                        max_output_channels = device_info.get('maxOutputChannels', 0)

                        # Check if this is CABLE Input (case-insensitive)
                        # CABLE Input has output channels because we send audio TO it
                        if 'cable input' in device_name.lower() and max_output_channels > 0:
                            # Found VB-Cable Input device
                            vb_cable_device = AudioDevice(
                                device_id=str(device_index),
                                name=device_name,
                                device_type=DeviceType.VIRTUAL,
                                virtual_driver=VirtualDeviceDriver.VB_CABLE,
                                is_default=False,
                                is_available=True
                            )

                            self.logger.info(f"Auto-detected VB-Cable device: {device_name} (index={device_index})")
                            return vb_cable_device

                    except Exception as device_error:
                        self.logger.warning(f"Error checking device {device_index}: {device_error}")
                        continue

                self.logger.info("VB-Cable device not found in audio device list")
                return None

            finally:
                p.terminate()

        except Exception as e:
            self.logger.error(f"Error during VB-Cable auto-detection: {e}")
            return None

    async def check_compatibility_issues(self) -> List[Dict[str, Any]]:
        """
        Check for virtual device compatibility issues.

        Returns:
            List[Dict[str, Any]]: List of compatibility issues found
        """
        try:
            self.logger.debug("Checking virtual device compatibility issues")
            issues = []

            # Detect drivers and analyze issues
            detection_results = await self.detect_virtual_drivers()

            for driver_type, driver_info in detection_results.items():
                for issue in driver_info.compatibility_issues:
                    issues.append({
                        'driver_type': driver_type.value,
                        'issue': issue,
                        'severity': 'warning' if driver_info.status == VirtualDriverStatus.PARTIALLY_INSTALLED else 'error',
                        'recommendations': driver_info.recommendations
                    })

            self.logger.info(f"Compatibility check complete: {len(issues)} issues found")
            return issues

        except Exception as e:
            self.logger.error(f"Failed to check compatibility issues: {e}")
            return [{
                'driver_type': 'unknown',
                'issue': f"Compatibility check failed: {str(e)}",
                'severity': 'error',
                'recommendations': ['Restart the application and try again']
            }]

    async def start(self) -> bool:
        """Start the service."""
        try:
            await self._update_status(ServiceStatus.STARTING)

            # Perform initial driver detection
            await self.detect_virtual_drivers(force_refresh=True)

            await self._update_status(ServiceStatus.RUNNING)
            self.logger.info("Virtual Device Compatibility Service started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start service: {e}")
            await self._update_status(ServiceStatus.ERROR)
            return False

    async def stop(self) -> bool:
        """Stop the service."""
        try:
            await self._update_status(ServiceStatus.STOPPING)

            # Clear caches
            self._driver_cache.clear()
            self._cache_valid = False

            await self._update_status(ServiceStatus.STOPPED)
            self.logger.info("Virtual Device Compatibility Service stopped")
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop service: {e}")
            await self._update_status(ServiceStatus.ERROR)
            return False

    async def health_check(self) -> Tuple[bool, Optional[MyVoiceError]]:
        """Check service health."""
        try:
            # Basic health check - try to access Windows registry
            test_key = winreg.HKEY_LOCAL_MACHINE
            winreg.OpenKey(test_key, r"SOFTWARE\Microsoft\Windows\CurrentVersion")

            return True, None

        except Exception as e:
            return False, MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="VIRTUAL_DEVICE_SERVICE_HEALTH_FAILED",
                user_message="Virtual device compatibility service health check failed",
                technical_details=str(e),
                suggested_action="Check Windows registry access permissions"
            )