"""
Application Settings Model

This module contains the AppSettings data model for managing application-wide
configuration including voice profile selection, UI preferences, and system settings.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import json

from myvoice.models.validation import ValidationResult, ValidationIssue, ValidationStatus


logger = logging.getLogger(__name__)


@dataclass
class AppSettings:
    """
    Application settings model for persistent configuration.

    This model manages all application settings including voice profile selection,
    UI preferences, directory paths, system configuration, and training settings.
    Settings are automatically validated and can be serialized to/from JSON.

    Attributes:
        selected_voice_profile: Currently selected voice profile name (optional)
        voice_files_directory: Directory path for voice sample files
        config_directory: Directory path for configuration files
        log_level: Application logging level (DEBUG, INFO, WARNING, ERROR)
        ui_theme: UI theme name (default, dark, light)
        always_on_top: Keep application window always on top (default: True)
        auto_refresh_interval: Auto-refresh interval for voice profiles in seconds
        enable_audio_monitoring: Whether to monitor audio output for debugging
        monitor_device_id: Selected monitor/output device ID for audio playback
        virtual_microphone_device_id: Selected virtual microphone device ID for dual routing
        tts_service_url: TTS service URL (legacy, unused with Qwen3-TTS)
        tts_service_timeout: TTS service request timeout in seconds
        max_voice_duration: Maximum allowed voice file duration in seconds
        recent_voice_profiles: List of recently used voice profile names
        window_geometry: Main window geometry settings
        advanced_settings: Dictionary for advanced/experimental settings
        training_enabled: Whether training features are enabled
        training_workspace_directory: Directory for training data and experiments
    """

    # Voice profile settings
    selected_voice_profile: Optional[str] = "Sarira-F"
    voice_files_directory: str = "voice_files"
    recent_voice_profiles: List[str] = field(default_factory=list)
    max_voice_duration: float = 10.0
    auto_refresh_interval: int = 30

    # Application directories
    config_directory: str = "config"

    # Logging configuration
    log_level: str = "INFO"

    # UI settings
    ui_theme: str = "dark"
    always_on_top: bool = True
    window_geometry: Optional[Dict[str, Any]] = None
    window_transparency: float = 1.0  # Story 6.2: Window opacity (0.0-1.0, FR41)
    # QA3-4: Default to False - X button should close app, not hide to tray
    minimize_to_tray: bool = False  # Story 7.2: Minimize to system tray (FR38)
    tray_notification_shown: bool = False  # Story 7.2: One-time tray notification flag

    # TTS service configuration
    tts_service_url: str = "http://localhost:9880"
    tts_service_timeout: int = 30

    # Audio settings
    enable_audio_monitoring: bool = True
    monitor_device_id: Optional[str] = None
    monitor_device_name: Optional[str] = None
    monitor_device_host_api: Optional[str] = None
    virtual_microphone_device_id: Optional[str] = None
    virtual_microphone_device_name: Optional[str] = None
    virtual_microphone_device_host_api: Optional[str] = None

    # Advanced settings
    advanced_settings: Dict[str, Any] = field(default_factory=dict)

    # Training settings
    training_enabled: bool = True
    training_workspace_directory: str = "training_workspace"

    # Custom emotion settings (Story 3.4: FR7)
    custom_emotion_text: Optional[str] = None
    custom_emotion_presets: List[str] = field(default_factory=lambda: [
        "Rising Frustration",
        "Growing Excitement",
        "Trailing Off Sadly",
        "Building Confidence",
        "Hesitant and Uncertain",
        "Warm and Reassuring",
        "Cold and Distant",
        "Playful Teasing",
        "Sincere Apology",
        "Dramatic Emphasis"
    ])

    # Internal metadata
    _settings_version: str = field(default="1.0", init=False)
    _last_modified: Optional[float] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize and validate settings after creation."""
        # Convert string paths to Path objects for validation
        self.voice_files_directory = str(Path(self.voice_files_directory))
        self.config_directory = str(Path(self.config_directory))
        self.training_workspace_directory = str(Path(self.training_workspace_directory))

        # Validate settings
        validation_result = self.validate()
        if not validation_result.is_valid:
            logger.warning(f"Settings validation issues found: {len(validation_result.issues)} issues")
            for issue in validation_result.issues:
                logger.warning(f"  - {issue.field}: {issue.message}")

        # Set modification time
        import time
        self._last_modified = time.time()

    def validate(self) -> ValidationResult:
        """
        Validate the application settings for consistency and correctness.

        Returns:
            ValidationResult: Detailed validation result with issues and warnings
        """
        issues = []
        warnings = []

        try:
            # Validate voice profile selection
            if self.selected_voice_profile is not None:
                if not isinstance(self.selected_voice_profile, str):
                    issues.append(ValidationIssue(
                        field="selected_voice_profile",
                        message="Selected voice profile must be a string",
                        code="INVALID_TYPE",
                        severity=ValidationStatus.INVALID
                    ))
                elif not self.selected_voice_profile.strip():
                    warnings.append(ValidationIssue(
                        field="selected_voice_profile",
                        message="Selected voice profile is empty",
                        code="EMPTY_VALUE",
                        severity=ValidationStatus.WARNING
                    ))

            # Validate directory paths
            for field_name, directory_path in [
                ("voice_files_directory", self.voice_files_directory),
                ("config_directory", self.config_directory)
            ]:
                if not directory_path or not directory_path.strip():
                    issues.append(ValidationIssue(
                        field=field_name,
                        message=f"{field_name} cannot be empty",
                        code="EMPTY_PATH",
                        severity=ValidationStatus.INVALID
                    ))
                else:
                    # Check for invalid path characters (basic validation)
                    try:
                        Path(directory_path)
                    except Exception:
                        issues.append(ValidationIssue(
                            field=field_name,
                            message=f"Invalid path format for {field_name}",
                            code="INVALID_PATH",
                            severity=ValidationStatus.INVALID
                        ))

            # Validate log level
            valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if self.log_level not in valid_log_levels:
                issues.append(ValidationIssue(
                    field="log_level",
                    message=f"Log level must be one of: {', '.join(valid_log_levels)}",
                    code="INVALID_LOG_LEVEL",
                    severity=ValidationStatus.INVALID
                ))

            # Validate UI theme and auto-migrate from old "default" theme
            valid_themes = ["dark", "light"]
            if self.ui_theme == "default":
                # Auto-migrate from old default theme to dark
                self.ui_theme = "dark"
                warnings.append(ValidationIssue(
                    field="ui_theme",
                    message="Migrated theme from 'default' to 'dark'",
                    code="THEME_MIGRATION",
                    severity=ValidationStatus.WARNING
                ))
            elif self.ui_theme not in valid_themes:
                # Set to dark for any other invalid theme
                original_theme = self.ui_theme
                self.ui_theme = "dark"
                warnings.append(ValidationIssue(
                    field="ui_theme",
                    message=f"Unknown UI theme '{original_theme}', switched to 'dark'",
                    code="UNKNOWN_THEME",
                    severity=ValidationStatus.WARNING
                ))

            # Validate always_on_top setting
            if not isinstance(self.always_on_top, bool):
                issues.append(ValidationIssue(
                    field="always_on_top",
                    message="Always on top setting must be a boolean value",
                    code="INVALID_TYPE",
                    severity=ValidationStatus.INVALID
                ))

            # Validate window_transparency (Story 7.3: FR41)
            # Valid range: 0.0-1.0, but UI enforces minimum 0.2 (20%)
            if not isinstance(self.window_transparency, (int, float)):
                issues.append(ValidationIssue(
                    field="window_transparency",
                    message="Window transparency must be a number",
                    code="INVALID_TYPE",
                    severity=ValidationStatus.INVALID
                ))
            elif self.window_transparency < 0.0 or self.window_transparency > 1.0:
                issues.append(ValidationIssue(
                    field="window_transparency",
                    message="Window transparency must be between 0.0 and 1.0",
                    code="INVALID_TRANSPARENCY_RANGE",
                    severity=ValidationStatus.INVALID
                ))
            elif self.window_transparency < 0.2:
                # Story 7.3: Warn about low transparency (will be clamped to 20% at runtime)
                warnings.append(ValidationIssue(
                    field="window_transparency",
                    message="Window transparency below 20% will be clamped to 20% (minimum)",
                    code="LOW_TRANSPARENCY",
                    severity=ValidationStatus.WARNING
                ))

            # Validate numeric settings
            if self.max_voice_duration <= 0:
                issues.append(ValidationIssue(
                    field="max_voice_duration",
                    message="Maximum voice duration must be positive",
                    code="INVALID_DURATION",
                    severity=ValidationStatus.INVALID
                ))
            elif self.max_voice_duration > 60.0:
                warnings.append(ValidationIssue(
                    field="max_voice_duration",
                    message="Maximum voice duration is very high (>60s)",
                    code="HIGH_DURATION",
                    severity=ValidationStatus.WARNING
                ))

            if self.auto_refresh_interval <= 0:
                issues.append(ValidationIssue(
                    field="auto_refresh_interval",
                    message="Auto-refresh interval must be positive",
                    code="INVALID_INTERVAL",
                    severity=ValidationStatus.INVALID
                ))

            if self.tts_service_timeout <= 0:
                issues.append(ValidationIssue(
                    field="tts_service_timeout",
                    message="TTS service timeout must be positive",
                    code="INVALID_TIMEOUT",
                    severity=ValidationStatus.INVALID
                ))

            # Validate monitor device ID
            if self.monitor_device_id is not None:
                if not isinstance(self.monitor_device_id, str):
                    issues.append(ValidationIssue(
                        field="monitor_device_id",
                        message="Monitor device ID must be a string",
                        code="INVALID_TYPE",
                        severity=ValidationStatus.INVALID
                    ))
                elif not self.monitor_device_id.strip():
                    warnings.append(ValidationIssue(
                        field="monitor_device_id",
                        message="Monitor device ID is empty",
                        code="EMPTY_VALUE",
                        severity=ValidationStatus.WARNING
                    ))

            # Validate virtual microphone device ID
            if self.virtual_microphone_device_id is not None:
                if not isinstance(self.virtual_microphone_device_id, str):
                    issues.append(ValidationIssue(
                        field="virtual_microphone_device_id",
                        message="Virtual microphone device ID must be a string",
                        code="INVALID_TYPE",
                        severity=ValidationStatus.INVALID
                    ))
                elif not self.virtual_microphone_device_id.strip():
                    warnings.append(ValidationIssue(
                        field="virtual_microphone_device_id",
                        message="Virtual microphone device ID is empty",
                        code="EMPTY_VALUE",
                        severity=ValidationStatus.WARNING
                    ))

            # Validate TTS service URL
            if not self.tts_service_url or not self.tts_service_url.strip():
                issues.append(ValidationIssue(
                    field="tts_service_url",
                    message="TTS service URL cannot be empty",
                    code="EMPTY_URL",
                    severity=ValidationStatus.INVALID
                ))
            elif not (self.tts_service_url.startswith("http://") or self.tts_service_url.startswith("https://")):
                warnings.append(ValidationIssue(
                    field="tts_service_url",
                    message="TTS service URL should start with http:// or https://",
                    code="INVALID_URL_SCHEME",
                    severity=ValidationStatus.WARNING
                ))

            # Validate recent voice profiles list
            if len(self.recent_voice_profiles) > 20:
                warnings.append(ValidationIssue(
                    field="recent_voice_profiles",
                    message="Recent voice profiles list is very long (>20 items)",
                    code="LONG_RECENT_LIST",
                    severity=ValidationStatus.WARNING
                ))

            # Check for duplicate recent profiles
            if len(self.recent_voice_profiles) != len(set(self.recent_voice_profiles)):
                warnings.append(ValidationIssue(
                    field="recent_voice_profiles",
                    message="Recent voice profiles list contains duplicates",
                    code="DUPLICATE_RECENT_PROFILES",
                    severity=ValidationStatus.WARNING
                ))

            # Validate training settings
            if not isinstance(self.training_enabled, bool):
                issues.append(ValidationIssue(
                    field="training_enabled",
                    message="Training enabled must be a boolean value",
                    code="INVALID_TYPE",
                    severity=ValidationStatus.INVALID
                ))

            # Validate training workspace path
            if not self.training_workspace_directory or not self.training_workspace_directory.strip():
                issues.append(ValidationIssue(
                    field="training_workspace_directory",
                    message="Training workspace directory cannot be empty",
                    code="EMPTY_PATH",
                    severity=ValidationStatus.INVALID
                ))
            else:
                try:
                    Path(self.training_workspace_directory)
                except Exception:
                    issues.append(ValidationIssue(
                        field="training_workspace_directory",
                        message="Invalid path format for training_workspace_directory",
                        code="INVALID_PATH",
                        severity=ValidationStatus.INVALID
                    ))

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
                warnings=warnings,
                summary=f"Settings validation: {len(issues)} issues, {len(warnings)} warnings"
            )

        except Exception as e:
            logger.exception(f"Error during settings validation: {e}")
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
                summary="Settings validation failed due to internal error"
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to dictionary for serialization.

        Returns:
            Dict[str, Any]: Settings as dictionary
        """
        try:
            return {
                "selected_voice_profile": self.selected_voice_profile,
                "voice_files_directory": self.voice_files_directory,
                "recent_voice_profiles": self.recent_voice_profiles.copy(),
                "max_voice_duration": self.max_voice_duration,
                "auto_refresh_interval": self.auto_refresh_interval,
                "config_directory": self.config_directory,
                "log_level": self.log_level,
                "ui_theme": self.ui_theme,
                "always_on_top": self.always_on_top,
                "window_geometry": self.window_geometry.copy() if self.window_geometry else None,
                "window_transparency": self.window_transparency,
                "minimize_to_tray": self.minimize_to_tray,
                "tray_notification_shown": self.tray_notification_shown,
                "tts_service_url": self.tts_service_url,
                "tts_service_timeout": self.tts_service_timeout,
                "enable_audio_monitoring": self.enable_audio_monitoring,
                "monitor_device_id": self.monitor_device_id,
                "monitor_device_name": self.monitor_device_name,
                "monitor_device_host_api": self.monitor_device_host_api,
                "virtual_microphone_device_id": self.virtual_microphone_device_id,
                "virtual_microphone_device_name": self.virtual_microphone_device_name,
                "virtual_microphone_device_host_api": self.virtual_microphone_device_host_api,
                "advanced_settings": self.advanced_settings.copy(),
                "training_enabled": self.training_enabled,
                "training_workspace_directory": self.training_workspace_directory,
                "custom_emotion_text": self.custom_emotion_text,
                "custom_emotion_presets": self.custom_emotion_presets.copy(),
                "_settings_version": self._settings_version,
                "_last_modified": self._last_modified
            }
        except Exception as e:
            logger.error(f"Error converting settings to dict: {e}")
            raise

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppSettings':
        """
        Create settings from dictionary (deserialization).

        Args:
            data: Dictionary containing settings data

        Returns:
            AppSettings: Settings instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        try:
            # Check settings version compatibility
            settings_version = data.get("_settings_version", "0.0")
            if settings_version != "1.0":
                logger.warning(f"Settings version {settings_version} may be incompatible with current version 1.0")

            # Extract fields with defaults
            settings = cls(
                selected_voice_profile=data.get("selected_voice_profile"),
                voice_files_directory=data.get("voice_files_directory", "voice_files"),
                recent_voice_profiles=data.get("recent_voice_profiles", []),
                max_voice_duration=data.get("max_voice_duration", 10.0),
                auto_refresh_interval=data.get("auto_refresh_interval", 30),
                config_directory=data.get("config_directory", "config"),
                log_level=data.get("log_level", "INFO"),
                ui_theme=data.get("ui_theme", "dark"),
                always_on_top=data.get("always_on_top", True),
                window_geometry=data.get("window_geometry"),
                window_transparency=data.get("window_transparency", 1.0),
                minimize_to_tray=data.get("minimize_to_tray", True),
                tray_notification_shown=data.get("tray_notification_shown", False),
                tts_service_url=data.get("tts_service_url", "http://localhost:9880"),
                tts_service_timeout=data.get("tts_service_timeout", 30),
                enable_audio_monitoring=data.get("enable_audio_monitoring", True),
                monitor_device_id=data.get("monitor_device_id"),
                monitor_device_name=data.get("monitor_device_name"),
                monitor_device_host_api=data.get("monitor_device_host_api"),
                virtual_microphone_device_id=data.get("virtual_microphone_device_id"),
                virtual_microphone_device_name=data.get("virtual_microphone_device_name"),
                virtual_microphone_device_host_api=data.get("virtual_microphone_device_host_api"),
                advanced_settings=data.get("advanced_settings", {}),
                training_enabled=data.get("training_enabled", True),
                training_workspace_directory=data.get("training_workspace_directory", "training_workspace"),
                custom_emotion_text=data.get("custom_emotion_text"),
                custom_emotion_presets=data.get("custom_emotion_presets", [
                    "Rising Frustration",
                    "Growing Excitement",
                    "Trailing Off Sadly",
                    "Building Confidence",
                    "Hesitant and Uncertain",
                    "Warm and Reassuring",
                    "Cold and Distant",
                    "Playful Teasing",
                    "Sincere Apology",
                    "Dramatic Emphasis"
                ])
            )

            # Restore internal metadata
            settings._last_modified = data.get("_last_modified")

            logger.debug("Successfully created settings from dictionary")
            return settings

        except Exception as e:
            logger.exception(f"Error creating settings from dict: {e}")
            raise ValueError(f"Failed to create settings from data: {str(e)}")

    def update_recent_voice_profile(self, profile_name: str, max_recent: int = 10):
        """
        Update the recent voice profiles list with a new selection.

        Args:
            profile_name: Name of the voice profile to add to recent list
            max_recent: Maximum number of recent profiles to maintain (default: 10)
        """
        try:
            if not profile_name or not profile_name.strip():
                return

            # Remove if already in list
            if profile_name in self.recent_voice_profiles:
                self.recent_voice_profiles.remove(profile_name)

            # Add to beginning of list
            self.recent_voice_profiles.insert(0, profile_name)

            # Trim to max length
            if len(self.recent_voice_profiles) > max_recent:
                self.recent_voice_profiles = self.recent_voice_profiles[:max_recent]

            # Update modification time
            import time
            self._last_modified = time.time()

            logger.debug(f"Updated recent voice profiles, now has {len(self.recent_voice_profiles)} items")

        except Exception as e:
            logger.error(f"Error updating recent voice profile: {e}")

    def get_voice_files_path(self) -> Path:
        """
        Get the voice files directory as a Path object.

        Returns:
            Path: Voice files directory path
        """
        return Path(self.voice_files_directory)

    def get_config_path(self) -> Path:
        """
        Get the config directory as a Path object.

        Returns:
            Path: Config directory path
        """
        return Path(self.config_directory)

    def clear_recent_voice_profiles(self):
        """Clear the recent voice profiles list."""
        self.recent_voice_profiles.clear()
        import time
        self._last_modified = time.time()
        logger.debug("Cleared recent voice profiles")

    def get_training_workspace_path(self) -> Path:
        """
        Get the training workspace directory as a Path object.

        Returns:
            Path: Training workspace directory path
        """
        return Path(self.training_workspace_directory)

    def reset_to_defaults(self):
        """Reset all settings to their default values."""
        logger.info("Resetting settings to defaults")

        # Create new default instance
        defaults = AppSettings()

        # Copy all default values
        for field_name in [
            "selected_voice_profile", "voice_files_directory", "recent_voice_profiles",
            "max_voice_duration", "auto_refresh_interval", "config_directory",
            "log_level", "ui_theme", "always_on_top", "window_geometry",
            "window_transparency", "tts_service_url",
            "tts_service_timeout", "enable_audio_monitoring", "monitor_device_id",
            "monitor_device_name", "monitor_device_host_api",
            "virtual_microphone_device_id", "virtual_microphone_device_name",
            "virtual_microphone_device_host_api", "advanced_settings",
            "training_enabled", "training_workspace_directory",
            "custom_emotion_text", "custom_emotion_presets"
        ]:
            setattr(self, field_name, getattr(defaults, field_name))

        # Update metadata
        import time
        self._last_modified = time.time()

    def __str__(self) -> str:
        """String representation of settings."""
        return (f"AppSettings(voice_profile='{self.selected_voice_profile}', "
                f"theme='{self.ui_theme}', log_level='{self.log_level}')")

    def __repr__(self) -> str:
        """Developer representation of settings."""
        return (f"AppSettings(selected_voice_profile='{self.selected_voice_profile}', "
                f"voice_files_directory='{self.voice_files_directory}', "
                f"ui_theme='{self.ui_theme}', log_level='{self.log_level}')")