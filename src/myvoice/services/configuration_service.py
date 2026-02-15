"""
Configuration Service

This module implements the ConfigurationManager service for loading, saving,
and managing application settings with automatic persistence and graceful error handling.

Story 2.3: Added audio device selection methods with persistence and guidance
for missing virtual microphone detection (FR26, FR27, FR44).
"""

import asyncio
import logging
import json
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from concurrent.futures import ThreadPoolExecutor

from myvoice.models.app_settings import AppSettings
from myvoice.models.error import MyVoiceError, ErrorSeverity
from myvoice.models.validation import ValidationResult

# Import base service after other models to avoid circular imports
from myvoice.services.core.base_service import BaseService, ServiceStatus


logger = logging.getLogger(__name__)


class ConfigurationManager(BaseService):
    """
    Service for managing application configuration with persistent storage.

    This service provides centralized management of application settings including:
    - Loading and saving settings to JSON files
    - Settings validation and error recovery
    - Voice profile selection persistence
    - Graceful handling of missing or corrupted configuration files
    - Atomic file operations to prevent corruption
    - Automatic backup and recovery mechanisms

    Features:
    - Thread-safe operations for settings access
    - Atomic file writes with temporary files
    - Configuration versioning and migration support
    - Settings validation with detailed error reporting
    - Integration with existing error handling patterns
    """

    def __init__(
        self,
        config_file: Optional[Path] = None,
        backup_count: int = 3,
        auto_save: bool = True,
        max_workers: int = 2
    ):
        """
        Initialize the ConfigurationManager.

        Args:
            config_file: Path to configuration file (default: config/settings.json)
            backup_count: Number of backup files to maintain (default: 3)
            auto_save: Whether to automatically save settings on changes (default: True)
            max_workers: Maximum thread pool workers for file operations
        """
        super().__init__("ConfigurationManager")

        # Configuration
        self.config_file = config_file or Path("config/settings.json")
        self.backup_count = backup_count
        self.auto_save = auto_save
        self.max_workers = max_workers

        # Service state
        self._settings: Optional[AppSettings] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._save_lock = threading.Lock()
        self._load_lock = threading.Lock()

        # Metrics
        self._load_count = 0
        self._save_count = 0
        self._last_save_time: Optional[float] = None
        self._last_load_time: Optional[float] = None

        self.logger.debug(f"ConfigurationManager initialized with file: {self.config_file}")
        self.logger.debug(f"Auto-save: {self.auto_save}, Backup count: {self.backup_count}")

    async def start(self) -> bool:
        """
        Start the ConfigurationManager service.

        Returns:
            bool: True if service started successfully, False otherwise
        """
        try:
            await self._update_status(ServiceStatus.STARTING)
            self.logger.info("Starting ConfigurationManager service")

            # Initialize thread executor
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="ConfigManager"
            )

            # Create config directory if it doesn't exist
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            # Load existing settings or create defaults
            await self.load_settings()

            await self._update_status(ServiceStatus.RUNNING)
            self.logger.info("ConfigurationManager service started successfully")
            return True

        except Exception as e:
            self.logger.exception(f"Failed to start ConfigurationManager service: {e}")
            await self._update_status(ServiceStatus.ERROR)
            return False

    async def stop(self) -> bool:
        """
        Stop the ConfigurationManager service.

        Returns:
            bool: True if service stopped successfully, False otherwise
        """
        try:
            await self._update_status(ServiceStatus.STOPPING)
            self.logger.info("Stopping ConfigurationManager service")

            # Save current settings before stopping
            if self._settings and self.auto_save:
                await self.save_settings()

            # Cleanup resources - QA Round 2 Item #8: Non-blocking shutdown
            if self._executor:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None

            await self._update_status(ServiceStatus.STOPPED)
            self.logger.info("ConfigurationManager service stopped successfully")
            return True

        except Exception as e:
            self.logger.exception(f"Error stopping ConfigurationManager service: {e}")
            await self._update_status(ServiceStatus.ERROR)
            return False

    async def health_check(self) -> tuple[bool, Optional[MyVoiceError]]:
        """
        Check ConfigurationManager service health.

        Returns:
            tuple[bool, Optional[MyVoiceError]]: (is_healthy, error_if_any)
        """
        try:
            # Check service status
            if not self.is_running():
                return False, MyVoiceError(
                    severity=ErrorSeverity.ERROR,
                    code="SERVICE_NOT_RUNNING",
                    user_message="ConfigurationManager service is not running",
                    suggested_action="Start the ConfigurationManager service"
                )

            # Check config file accessibility
            if self.config_file.exists():
                if not self.config_file.is_file():
                    return False, MyVoiceError(
                        severity=ErrorSeverity.ERROR,
                        code="CONFIG_FILE_INVALID",
                        user_message=f"Config file path is not a file: {self.config_file}",
                        suggested_action="Remove the invalid config path"
                    )

                # Try to read the config file
                try:
                    self.config_file.read_text(encoding='utf-8')
                except PermissionError:
                    return False, MyVoiceError(
                        severity=ErrorSeverity.ERROR,
                        code="CONFIG_FILE_NO_ACCESS",
                        user_message=f"Cannot read config file: {self.config_file}",
                        suggested_action="Check file permissions"
                    )

            # Check if settings are loaded
            if self._settings is None:
                return False, MyVoiceError(
                    severity=ErrorSeverity.WARNING,
                    code="NO_SETTINGS_LOADED",
                    user_message="No settings are currently loaded",
                    suggested_action="Load settings from file or create defaults"
                )

            return True, None

        except Exception as e:
            self.logger.exception(f"Health check failed: {e}")
            return False, MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="HEALTH_CHECK_FAILED",
                user_message="Failed to check ConfigurationManager service health",
                technical_details=str(e)
            )

    async def load_settings(self, config_file: Optional[Path] = None) -> AppSettings:
        """
        Load application settings from file or create defaults.

        Args:
            config_file: Optional path to config file (defaults to configured file)

        Returns:
            AppSettings: Loaded or default settings

        Raises:
            Exception: If loading fails and cannot create defaults
        """
        load_file = config_file or self.config_file

        try:
            self.logger.info(f"Loading settings from: {load_file}")
            import time
            start_time = time.time()

            if load_file.exists():
                # Load from file
                loop = asyncio.get_running_loop()
                settings_data = await loop.run_in_executor(
                    self._executor,
                    self._load_settings_sync,
                    load_file
                )

                if settings_data:
                    self._settings = AppSettings.from_dict(settings_data)
                    self.logger.info(f"Settings loaded successfully from {load_file}")
                else:
                    # File exists but failed to load, create defaults
                    self.logger.warning(f"Failed to load settings from {load_file}, using defaults")
                    self._settings = AppSettings()
                    # Save defaults to file
                    if self.auto_save:
                        await self.save_settings()
            else:
                # No file exists, create defaults
                self.logger.info(f"No config file found at {load_file}, creating defaults")
                self._settings = AppSettings()
                # Save defaults to file
                if self.auto_save:
                    await self.save_settings(load_file)

            # Validate loaded settings
            validation_result = self._settings.validate()
            if not validation_result.is_valid:
                self.logger.warning(f"Loaded settings have validation issues: {validation_result.summary}")
                # Continue with invalid settings but log warnings
                for issue in validation_result.issues:
                    self.logger.warning(f"  Settings issue: {issue.field} - {issue.message}")

            # Update metrics
            self._load_count += 1
            self._last_load_time = time.time()
            load_duration = self._last_load_time - start_time

            self.logger.info(f"Settings loading completed in {load_duration:.3f}s")
            return self._settings

        except Exception as e:
            self.logger.exception(f"Error loading settings from {load_file}: {e}")

            # Try to create default settings as fallback
            try:
                self.logger.info("Creating default settings as fallback")
                self._settings = AppSettings()
                return self._settings
            except Exception as fallback_error:
                self.logger.exception(f"Failed to create default settings: {fallback_error}")
                raise Exception(f"Failed to load settings and cannot create defaults: {str(e)}")

    def _load_settings_sync(self, config_file: Path) -> Optional[Dict[str, Any]]:
        """
        Synchronous settings loading for thread pool execution with enhanced error recovery.

        Args:
            config_file: Path to the config file

        Returns:
            Optional[Dict[str, Any]]: Loaded settings data or None if failed
        """
        try:
            # Try main config file first
            if config_file.exists():
                content = config_file.read_text(encoding='utf-8')
                data = json.loads(content)
                self.logger.debug(f"Successfully loaded settings from {config_file}")
                return data

            # Try backup files if main file fails
            for i in range(1, self.backup_count + 1):
                backup_file = config_file.with_suffix(f'.bak{i}')
                if backup_file.exists():
                    try:
                        content = backup_file.read_text(encoding='utf-8')
                        data = json.loads(content)
                        self.logger.warning(f"Loaded settings from backup file: {backup_file}")
                        return data
                    except Exception as backup_error:
                        self.logger.warning(f"Failed to load backup {backup_file}: {backup_error}")
                        continue

            return None

        except json.JSONDecodeError as e:
            self.logger.error(f"Settings file corruption detected in {config_file}: {e}")

            # Try to repair the corrupted file
            try:
                content = config_file.read_text(encoding='utf-8')
                repaired_content = self._attempt_json_repair(content)
                if repaired_content:
                    data = json.loads(repaired_content)
                    # Create backup of corrupted file
                    corrupted_backup = config_file.with_suffix(f'.corrupted_{int(time.time())}')
                    config_file.rename(corrupted_backup)
                    # Write repaired content
                    config_file.write_text(repaired_content, encoding='utf-8')
                    self.logger.info(f"Successfully repaired corrupted settings file. Backup saved as: {corrupted_backup}")
                    return data
            except Exception as repair_error:
                self.logger.warning(f"Failed to repair corrupted file: {repair_error}")

            return None
        except Exception as e:
            self.logger.error(f"Error reading config file {config_file}: {e}")
            return None

    def _attempt_json_repair(self, content: str) -> Optional[str]:
        """
        Attempt to repair corrupted JSON content.

        Args:
            content: Original JSON content

        Returns:
            Optional[str]: Repaired JSON content or None if repair failed
        """
        import re

        # Strategy 1: Remove trailing commas
        repaired = re.sub(r',\s*}', '}', content)
        repaired = re.sub(r',\s*]', ']', repaired)

        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            pass

        # Strategy 2: Fix truncated JSON by adding missing closing braces
        open_braces = content.count('{') - content.count('}')
        if open_braces > 0:
            repaired = content + '}' * open_braces
            try:
                json.loads(repaired)
                return repaired
            except json.JSONDecodeError:
                pass

        # Strategy 3: Extract valid JSON from partial content
        try:
            # Find the last complete key-value pair
            lines = content.split('\n')
            for i in range(len(lines) - 1, -1, -1):
                partial_content = '\n'.join(lines[:i]) + '\n}'
                try:
                    json.loads(partial_content)
                    return partial_content
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass

        return None

    async def save_settings(self, config_file: Optional[Path] = None) -> bool:
        """
        Save current settings to file with atomic operation.

        Args:
            config_file: Optional path to config file (defaults to configured file)

        Returns:
            bool: True if save successful, False otherwise
        """
        if not self._settings:
            self.logger.warning("No settings to save")
            return False

        save_file = config_file or self.config_file

        try:
            self.logger.debug(f"Saving settings to: {save_file}")
            import time
            start_time = time.time()

            # Validate settings before saving
            validation_result = self._settings.validate()
            if not validation_result.is_valid:
                self.logger.warning(f"Saving settings with validation issues: {validation_result.summary}")

            # Save using thread pool
            loop = asyncio.get_running_loop()
            success = await loop.run_in_executor(
                self._executor,
                self._save_settings_sync,
                save_file,
                self._settings.to_dict()
            )

            if success:
                # Update metrics
                self._save_count += 1
                self._last_save_time = time.time()
                save_duration = self._last_save_time - start_time

                self.logger.info(f"Settings saved successfully to {save_file} in {save_duration:.3f}s")
                return True
            else:
                self.logger.error(f"Failed to save settings to {save_file}")
                return False

        except Exception as e:
            self.logger.exception(f"Error saving settings to {save_file}: {e}")
            return False

    def _save_settings_sync(self, config_file: Path, settings_data: Dict[str, Any]) -> bool:
        """
        Synchronous settings saving for thread pool execution.

        Args:
            config_file: Path to the config file
            settings_data: Settings data to save

        Returns:
            bool: True if save successful, False otherwise
        """
        # Acquire lock for thread-safe file operations
        with self._save_lock:
            try:
                # Create directory if it doesn't exist
                config_file.parent.mkdir(parents=True, exist_ok=True)

                # Create backup of existing file
                if config_file.exists():
                    self._create_backup(config_file)

                # Write to temporary file first (atomic operation)
                temp_file = config_file.with_suffix('.tmp')
                json_content = json.dumps(settings_data, indent=2, ensure_ascii=False)
                temp_file.write_text(json_content, encoding='utf-8')

                # Replace original file with temporary file
                temp_file.replace(config_file)

                self.logger.debug(f"Settings written successfully to {config_file}")
                return True

            except Exception as e:
                self.logger.error(f"Error writing settings file {config_file}: {e}")
                return False

    def _create_backup(self, config_file: Path):
        """
        Create backup copies of config file.

        Args:
            config_file: Path to the config file to backup
        """
        try:
            # Rotate existing backups
            for i in range(self.backup_count, 1, -1):
                old_backup = config_file.with_suffix(f'.bak{i-1}')
                new_backup = config_file.with_suffix(f'.bak{i}')
                if old_backup.exists():
                    if new_backup.exists():
                        new_backup.unlink()
                    old_backup.rename(new_backup)

            # Create new backup
            backup_file = config_file.with_suffix('.bak1')
            if backup_file.exists():
                backup_file.unlink()
            config_file.replace(backup_file)

            self.logger.debug(f"Created backup: {backup_file}")

        except Exception as e:
            self.logger.warning(f"Failed to create backup of {config_file}: {e}")

    def get_settings(self) -> Optional[AppSettings]:
        """
        Get current application settings.

        Returns:
            Optional[AppSettings]: Current settings or None if not loaded
        """
        return self._settings

    async def update_voice_selection(self, profile_name: Optional[str]) -> bool:
        """
        Update the selected voice profile in settings.

        Args:
            profile_name: Name of the voice profile to select (None to clear selection)

        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            if not self._settings:
                self.logger.error("No settings loaded, cannot update voice selection")
                return False

            old_selection = self._settings.selected_voice_profile
            self._settings.selected_voice_profile = profile_name

            # Update recent profiles list if selecting a profile
            if profile_name:
                self._settings.update_recent_voice_profile(profile_name)

            self.logger.info(f"Voice selection updated: '{old_selection}' -> '{profile_name}'")

            # Auto-save if enabled
            if self.auto_save:
                return await self.save_settings()

            return True

        except Exception as e:
            self.logger.exception(f"Error updating voice selection: {e}")
            return False

    async def get_voice_selection(self) -> Optional[str]:
        """
        Get the currently selected voice profile.

        Returns:
            Optional[str]: Selected voice profile name or None
        """
        if self._settings:
            return self._settings.selected_voice_profile
        return None

    async def restore_voice_selection(self) -> Optional[str]:
        """
        Restore voice selection from saved settings.

        This method checks if the saved voice profile still exists and is valid.
        If the saved profile is missing or invalid, it falls back to bundled voice.

        Story 6.3: FR47 - If selected voice deleted externally, switch to bundled.

        Returns:
            Optional[str]: Restored voice profile name or bundled fallback
        """
        try:
            if not self._settings or not self._settings.selected_voice_profile:
                self.logger.debug("No voice profile selected in settings")
                return await self._select_default_bundled_voice()

            selected_profile = self._settings.selected_voice_profile
            self.logger.info(f"Attempting to restore voice selection: {selected_profile}")

            # Check if the voice file still exists
            voice_files_path = self._settings.get_voice_files_path()
            if not voice_files_path.exists():
                self.logger.warning(f"Voice files directory not found: {voice_files_path}")
                return await self._select_default_bundled_voice()

            # Check for bundled voices first (they don't have .wav files)
            from myvoice.models.voice_profile import BUNDLED_SPEAKERS
            if selected_profile in BUNDLED_SPEAKERS:
                self.logger.info(f"Successfully restored bundled voice selection: {selected_profile}")
                return selected_profile

            # Check for embedding voices (they have embedding directories, not .wav files)
            embeddings_path = voice_files_path / "embeddings" / selected_profile
            if embeddings_path.exists() and embeddings_path.is_dir():
                # Verify embedding has at least a neutral embedding or root embedding
                has_embedding = (
                    (embeddings_path / "neutral" / "embedding.pt").exists() or
                    (embeddings_path / "embedding.pt").exists()
                )
                if has_embedding:
                    self.logger.info(f"Successfully restored embedding voice selection: {selected_profile}")
                    return selected_profile
                else:
                    self.logger.warning(f"Embedding voice directory exists but no embedding.pt found: {selected_profile}")

            # Look for the voice file (cloned voices)
            voice_file_pattern = f"{selected_profile}.wav"
            voice_files = list(voice_files_path.rglob(voice_file_pattern))

            if not voice_files:
                self.logger.warning(f"Voice file not found for profile: {selected_profile}")
                # Story 6.3: Fallback to bundled voice
                return await self._select_default_bundled_voice()

            # Verify the voice file is still valid
            voice_file = voice_files[0]
            if not voice_file.exists() or voice_file.stat().st_size == 0:
                self.logger.warning(f"Voice file is missing or empty: {voice_file}")
                # Story 6.3: Fallback to bundled voice
                return await self._select_default_bundled_voice()

            self.logger.info(f"Successfully restored voice selection: {selected_profile}")
            return selected_profile

        except Exception as e:
            self.logger.exception(f"Error restoring voice selection: {e}")
            # Story 6.3: Fallback to bundled voice on error
            return await self._select_default_bundled_voice()

    async def _select_default_bundled_voice(self) -> Optional[str]:
        """
        Select default bundled voice as fallback (Story 6.3: FR47).

        This method selects the first available bundled voice when the
        previously selected voice is no longer available.

        Returns:
            Optional[str]: Name of bundled voice selected, or None if none available
        """
        try:
            # Default bundled voice name
            default_bundled = "Sarira-F"

            voice_files_path = self._settings.get_voice_files_path() if self._settings else Path("voice_files")

            # Check if default bundled voice exists
            default_voice_file = voice_files_path / f"{default_bundled}.wav"
            if default_voice_file.exists():
                self.logger.info(f"Falling back to default bundled voice: {default_bundled}")
                await self.update_voice_selection(default_bundled)
                return default_bundled

            # Try to find any bundled voice file
            if voice_files_path.exists():
                wav_files = list(voice_files_path.glob("*.wav"))
                if wav_files:
                    first_voice = wav_files[0].stem
                    self.logger.info(f"Falling back to first available voice: {first_voice}")
                    await self.update_voice_selection(first_voice)
                    return first_voice

            # No voices available
            self.logger.warning("No bundled voices available for fallback")
            await self.update_voice_selection(None)
            return None

        except Exception as e:
            self.logger.error(f"Error selecting default bundled voice: {e}")
            if self._settings:
                await self.update_voice_selection(None)
            return None

    def get_service_metrics(self) -> Dict[str, Any]:
        """
        Get service performance metrics.

        Returns:
            Dict[str, Any]: Service metrics and statistics
        """
        metrics = self.get_status_info()
        metrics.update({
            "config_file": str(self.config_file),
            "auto_save_enabled": self.auto_save,
            "backup_count": self.backup_count,
            "load_count": self._load_count,
            "save_count": self._save_count,
            "last_load_time": self._last_load_time,
            "last_save_time": self._last_save_time,
            "settings_loaded": self._settings is not None,
            "selected_voice": self._settings.selected_voice_profile if self._settings else None
        })
        return metrics

    async def reload_settings(self) -> bool:
        """
        Reload settings from file, discarding current changes.

        Returns:
            bool: True if reload successful, False otherwise
        """
        try:
            self.logger.info("Reloading settings from file")
            await self.load_settings()
            return True
        except Exception as e:
            self.logger.exception(f"Error reloading settings: {e}")
            return False

    async def export_settings(self, export_file: Path, include_metadata: bool = True) -> tuple[bool, Optional[MyVoiceError]]:
        """
        Export current settings to a JSON backup file with metadata and validation.

        Args:
            export_file: Path to export the settings to
            include_metadata: Whether to include export metadata (default: True)

        Returns:
            tuple[bool, Optional[MyVoiceError]]: (success, error_if_any)
        """
        try:
            if not self._settings:
                error = MyVoiceError(
                    severity=ErrorSeverity.ERROR,
                    code="NO_SETTINGS_TO_EXPORT",
                    user_message="No settings are currently loaded to export",
                    suggested_action="Load settings first before exporting"
                )
                return False, error

            self.logger.info(f"Exporting settings to: {export_file}")
            import time
            start_time = time.time()

            # Validate settings before export
            validation_result = self._settings.validate()
            if not validation_result.is_valid:
                self.logger.warning(f"Exporting settings with validation issues: {validation_result.summary}")

            # Prepare export data
            export_data = {
                "settings": self._settings.to_dict()
            }

            # Add metadata if requested
            if include_metadata:
                export_data["export_metadata"] = {
                    "export_timestamp": time.time(),
                    "export_date_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "application_version": "1.0.0",  # TODO: Get from app version
                    "settings_version": self._settings._settings_version,
                    "exported_by": "ConfigurationManager",
                    "validation_status": validation_result.status.value,
                    "validation_summary": validation_result.summary,
                    "validation_issues": len(validation_result.issues),
                    "validation_warnings": len(validation_result.warnings),
                    "total_settings_count": len([k for k in export_data["settings"].keys() if not k.startswith("_")])
                }

            # Save using thread pool with enhanced error handling
            loop = asyncio.get_running_loop()
            success = await loop.run_in_executor(
                self._executor,
                self._export_settings_sync,
                export_file,
                export_data
            )

            if success:
                export_duration = time.time() - start_time
                self.logger.info(f"Settings exported successfully to {export_file} in {export_duration:.3f}s")
                return True, None
            else:
                error = MyVoiceError(
                    severity=ErrorSeverity.ERROR,
                    code="EXPORT_FAILED",
                    user_message=f"Failed to export settings to {export_file}",
                    suggested_action="Check file permissions and disk space"
                )
                return False, error

        except Exception as e:
            self.logger.exception(f"Error exporting settings: {e}")
            error = MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="EXPORT_ERROR",
                user_message="An error occurred while exporting settings",
                technical_details=str(e),
                suggested_action="Check the application logs for details"
            )
            return False, error

    def _export_settings_sync(self, export_file: Path, export_data: Dict[str, Any]) -> bool:
        """
        Synchronous settings export for thread pool execution.

        Args:
            export_file: Path to the export file
            export_data: Complete export data including settings and metadata

        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            export_file.parent.mkdir(parents=True, exist_ok=True)

            # Write to temporary file first (atomic operation)
            temp_file = export_file.with_suffix('.tmp')
            json_content = json.dumps(export_data, indent=2, ensure_ascii=False)
            temp_file.write_text(json_content, encoding='utf-8')

            # Replace original file with temporary file
            temp_file.replace(export_file)

            self.logger.debug(f"Settings export written successfully to {export_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error writing export file {export_file}: {e}")
            return False

    async def import_settings(self, import_file: Path, validate_before_apply: bool = True,
                            create_backup: bool = True) -> tuple[bool, Optional[MyVoiceError], Optional[Dict[str, Any]]]:
        """
        Import settings from a JSON backup file with validation and migration support.

        Args:
            import_file: Path to import the settings from
            validate_before_apply: Whether to validate settings before applying (default: True)
            create_backup: Whether to backup current settings before import (default: True)

        Returns:
            tuple[bool, Optional[MyVoiceError], Optional[Dict]]: (success, error_if_any, import_summary)
        """
        try:
            if not import_file.exists():
                error = MyVoiceError(
                    severity=ErrorSeverity.ERROR,
                    code="IMPORT_FILE_NOT_FOUND",
                    user_message=f"Import file not found: {import_file}",
                    suggested_action="Check the file path and ensure the file exists"
                )
                return False, error, None

            self.logger.info(f"Importing settings from: {import_file}")
            import time
            start_time = time.time()

            # Create backup of current settings if requested
            if create_backup and self._settings:
                backup_file = self.config_file.with_suffix(f'.backup_{int(time.time())}')
                backup_success, backup_error = await self.export_settings(backup_file, include_metadata=True)
                if backup_success:
                    self.logger.info(f"Created backup of current settings: {backup_file}")
                else:
                    self.logger.warning(f"Failed to create backup: {backup_error.user_message if backup_error else 'Unknown error'}")

            # Load import data
            loop = asyncio.get_running_loop()
            import_data = await loop.run_in_executor(
                self._executor,
                self._load_import_sync,
                import_file
            )

            if not import_data:
                error = MyVoiceError(
                    severity=ErrorSeverity.ERROR,
                    code="IMPORT_DATA_INVALID",
                    user_message=f"Could not read valid data from import file: {import_file}",
                    suggested_action="Check that the file is a valid MyVoice settings backup"
                )
                return False, error, None

            # Extract settings data and metadata
            settings_data = import_data.get("settings")
            import_metadata = import_data.get("export_metadata", {})

            if not settings_data:
                error = MyVoiceError(
                    severity=ErrorSeverity.ERROR,
                    code="NO_SETTINGS_IN_IMPORT",
                    user_message="Import file does not contain settings data",
                    suggested_action="Ensure the file is a valid MyVoice settings backup"
                )
                return False, error, None

            # Check version compatibility and migrate if needed
            imported_version = settings_data.get("_settings_version", "0.0")
            current_version = "1.0"  # Current version

            if imported_version != current_version:
                self.logger.info(f"Migrating settings from version {imported_version} to {current_version}")
                settings_data = await self._migrate_settings(settings_data, imported_version, current_version)

            # Create AppSettings instance from imported data
            try:
                imported_settings = AppSettings.from_dict(settings_data)
            except ValueError as e:
                error = MyVoiceError(
                    severity=ErrorSeverity.ERROR,
                    code="IMPORT_SETTINGS_INVALID",
                    user_message="Imported settings data is invalid",
                    technical_details=str(e),
                    suggested_action="Check that the import file is not corrupted"
                )
                return False, error, None

            # Validate imported settings if requested
            validation_issues = []
            validation_warnings = []
            if validate_before_apply:
                validation_result = imported_settings.validate()
                validation_issues = validation_result.issues
                validation_warnings = validation_result.warnings

                if not validation_result.is_valid:
                    self.logger.warning(f"Imported settings have validation issues: {validation_result.summary}")
                    # Continue with import but log the issues

            # Apply the imported settings
            self._settings = imported_settings

            # Save to main config file if auto-save is enabled
            if self.auto_save:
                save_success = await self.save_settings()
                if not save_success:
                    self.logger.warning("Failed to save imported settings to main config file")

            # Prepare import summary
            import_duration = time.time() - start_time
            import_summary = {
                "import_successful": True,
                "import_duration_seconds": round(import_duration, 3),
                "imported_from": str(import_file),
                "import_timestamp": time.time(),
                "original_export_date": import_metadata.get("export_date_iso", "Unknown"),
                "settings_migrated": imported_version != current_version,
                "original_version": imported_version,
                "current_version": current_version,
                "validation_issues": len(validation_issues),
                "validation_warnings": len(validation_warnings),
                "total_settings_imported": len([k for k in settings_data.keys() if not k.startswith("_")]),
                "backup_created": create_backup
            }

            self.logger.info(f"Settings imported successfully from {import_file} in {import_duration:.3f}s")
            return True, None, import_summary

        except Exception as e:
            self.logger.exception(f"Error importing settings: {e}")
            error = MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="IMPORT_ERROR",
                user_message="An error occurred while importing settings",
                technical_details=str(e),
                suggested_action="Check the application logs for details"
            )
            return False, error, None

    def _load_import_sync(self, import_file: Path) -> Optional[Dict[str, Any]]:
        """
        Synchronous import data loading for thread pool execution.

        Args:
            import_file: Path to the import file

        Returns:
            Optional[Dict[str, Any]]: Loaded import data or None if failed
        """
        try:
            content = import_file.read_text(encoding='utf-8')
            data = json.loads(content)

            # Handle both new format (with metadata) and legacy format (just settings)
            if "settings" in data:
                # New format with metadata
                return data
            elif "export_metadata" in data and "settings" not in data:
                # File has metadata but no settings - invalid
                return None
            else:
                # Legacy format - assume the entire content is settings
                return {"settings": data}

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in import file {import_file}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading import file {import_file}: {e}")
            return None

    async def _migrate_settings(self, settings_data: Dict[str, Any], from_version: str, to_version: str) -> Dict[str, Any]:
        """
        Migrate settings between versions.

        Args:
            settings_data: Original settings data
            from_version: Source version
            to_version: Target version

        Returns:
            Dict[str, Any]: Migrated settings data
        """
        try:
            self.logger.info(f"Migrating settings from version {from_version} to {to_version}")
            migrated_data = settings_data.copy()

            # Version-specific migrations
            if from_version == "0.0" and to_version == "1.0":
                # Migration from pre-versioned to version 1.0
                # Add any missing fields with defaults
                defaults = AppSettings()
                default_dict = defaults.to_dict()

                for key, default_value in default_dict.items():
                    if key not in migrated_data:
                        migrated_data[key] = default_value
                        self.logger.debug(f"Added missing field '{key}' with default value")

                # Update version
                migrated_data["_settings_version"] = "1.0"

            # Add more migration paths as needed
            # elif from_version == "1.0" and to_version == "1.1":
            #     # Future migration example
            #     pass

            self.logger.info(f"Settings migration completed: {from_version} -> {to_version}")
            return migrated_data

        except Exception as e:
            self.logger.exception(f"Error during settings migration: {e}")
            # Return original data if migration fails
            return settings_data

    async def create_settings_backup(self, backup_name: Optional[str] = None) -> tuple[bool, Optional[MyVoiceError], Optional[Path]]:
        """
        Create a timestamped backup of current settings.

        Args:
            backup_name: Optional custom name for the backup (default: auto-generated)

        Returns:
            tuple[bool, Optional[MyVoiceError], Optional[Path]]: (success, error_if_any, backup_file_path)
        """
        try:
            import time
            if backup_name:
                # Use custom name but ensure it has timestamp
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                backup_filename = f"{backup_name}_{timestamp}.json"
            else:
                # Auto-generated name
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                backup_filename = f"myvoice_settings_backup_{timestamp}.json"

            backup_file = self.config_file.parent / "backups" / backup_filename
            success, error = await self.export_settings(backup_file, include_metadata=True)

            if success:
                self.logger.info(f"Settings backup created: {backup_file}")
                return True, None, backup_file
            else:
                return False, error, None

        except Exception as e:
            self.logger.exception(f"Error creating settings backup: {e}")
            error = MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="BACKUP_ERROR",
                user_message="Failed to create settings backup",
                technical_details=str(e)
            )
            return False, error, None

    async def restore_settings_backup(self, backup_file: Path) -> tuple[bool, Optional[MyVoiceError], Optional[Dict[str, Any]]]:
        """
        Restore settings from a backup file.

        Args:
            backup_file: Path to the backup file to restore

        Returns:
            tuple[bool, Optional[MyVoiceError], Optional[Dict]]: (success, error_if_any, restore_summary)
        """
        try:
            self.logger.info(f"Restoring settings from backup: {backup_file}")
            return await self.import_settings(backup_file, validate_before_apply=True, create_backup=True)

        except Exception as e:
            self.logger.exception(f"Error restoring settings backup: {e}")
            error = MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="RESTORE_ERROR",
                user_message="Failed to restore settings from backup",
                technical_details=str(e)
            )
            return False, error, None

    def list_available_backups(self) -> List[Dict[str, Any]]:
        """
        List all available settings backup files.

        Returns:
            List[Dict[str, Any]]: List of backup file information
        """
        try:
            backup_dir = self.config_file.parent / "backups"
            if not backup_dir.exists():
                return []

            backups = []
            for backup_file in backup_dir.glob("*.json"):
                try:
                    stat = backup_file.stat()
                    import time
                    backup_info = {
                        "filename": backup_file.name,
                        "full_path": str(backup_file),
                        "size_bytes": stat.st_size,
                        "created_timestamp": stat.st_ctime,
                        "created_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_ctime)),
                        "modified_timestamp": stat.st_mtime,
                        "modified_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
                    }

                    # Try to read metadata if available
                    try:
                        content = backup_file.read_text(encoding='utf-8')
                        data = json.loads(content)
                        metadata = data.get("export_metadata", {})
                        if metadata:
                            backup_info.update({
                                "export_date": metadata.get("export_date_iso", "Unknown"),
                                "settings_version": metadata.get("settings_version", "Unknown"),
                                "application_version": metadata.get("application_version", "Unknown"),
                                "validation_status": metadata.get("validation_status", "Unknown"),
                                "total_settings": metadata.get("total_settings_count", "Unknown")
                            })
                    except Exception:
                        # If we can't read metadata, just use file info
                        pass

                    backups.append(backup_info)

                except Exception as e:
                    self.logger.warning(f"Could not read backup file info for {backup_file}: {e}")
                    continue

            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x["created_timestamp"], reverse=True)
            return backups

        except Exception as e:
            self.logger.exception(f"Error listing backups: {e}")
            return []

    async def validate_backup_file(self, backup_file: Path) -> tuple[bool, Optional[MyVoiceError], Optional[Dict[str, Any]]]:
        """
        Validate a backup file without importing it.

        Args:
            backup_file: Path to the backup file to validate

        Returns:
            tuple[bool, Optional[MyVoiceError], Optional[Dict]]: (is_valid, error_if_any, validation_summary)
        """
        try:
            if not backup_file.exists():
                error = MyVoiceError(
                    severity=ErrorSeverity.ERROR,
                    code="BACKUP_FILE_NOT_FOUND",
                    user_message=f"Backup file not found: {backup_file}",
                    suggested_action="Check the file path"
                )
                return False, error, None

            self.logger.debug(f"Validating backup file: {backup_file}")

            # Load and validate the backup file structure
            loop = asyncio.get_running_loop()
            import_data = await loop.run_in_executor(
                self._executor,
                self._load_import_sync,
                backup_file
            )

            if not import_data:
                error = MyVoiceError(
                    severity=ErrorSeverity.ERROR,
                    code="INVALID_BACKUP_FORMAT",
                    user_message="Backup file format is invalid or corrupted",
                    suggested_action="Ensure the file is a valid MyVoice backup"
                )
                return False, error, None

            settings_data = import_data.get("settings")
            if not settings_data:
                error = MyVoiceError(
                    severity=ErrorSeverity.ERROR,
                    code="NO_SETTINGS_IN_BACKUP",
                    user_message="Backup file does not contain settings data",
                    suggested_action="Ensure the file is a valid MyVoice backup"
                )
                return False, error, None

            # Try to create AppSettings instance to validate structure
            try:
                test_settings = AppSettings.from_dict(settings_data)
                validation_result = test_settings.validate()
            except ValueError as e:
                error = MyVoiceError(
                    severity=ErrorSeverity.ERROR,
                    code="INVALID_SETTINGS_DATA",
                    user_message="Settings data in backup is invalid",
                    technical_details=str(e)
                )
                return False, error, None

            # Prepare validation summary
            import_metadata = import_data.get("export_metadata", {})
            validation_summary = {
                "backup_valid": True,
                "settings_version": settings_data.get("_settings_version", "Unknown"),
                "export_date": import_metadata.get("export_date_iso", "Unknown"),
                "total_settings": len([k for k in settings_data.keys() if not k.startswith("_")]),
                "validation_issues": len(validation_result.issues),
                "validation_warnings": len(validation_result.warnings),
                "validation_status": validation_result.status.value,
                "requires_migration": settings_data.get("_settings_version", "0.0") != "1.0"
            }

            self.logger.debug(f"Backup file validation successful: {backup_file}")
            return True, None, validation_summary

        except Exception as e:
            self.logger.exception(f"Error validating backup file: {e}")
            error = MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="VALIDATION_ERROR",
                user_message="Error occurred while validating backup file",
                technical_details=str(e)
            )
            return False, error, None

    async def validate_and_report_settings(self, settings: Optional[AppSettings] = None) -> tuple[bool, Dict[str, Any]]:
        """
        Validate settings and generate comprehensive error report with fix suggestions.

        Args:
            settings: Settings to validate (defaults to current settings)

        Returns:
            tuple[bool, Dict[str, Any]]: (is_valid, validation_report)
        """
        try:
            target_settings = settings or self._settings
            if not target_settings:
                return False, {
                    "status": "error",
                    "message": "No settings available for validation",
                    "issues": [],
                    "suggestions": ["Load settings first before validation"]
                }

            # Perform validation
            validation_result = target_settings.validate()

            # Build comprehensive report
            report = {
                "status": "valid" if validation_result.is_valid else "invalid",
                "summary": validation_result.summary,
                "total_issues": len(validation_result.issues),
                "total_warnings": len(validation_result.warnings),
                "issues": [],
                "warnings": [],
                "auto_fixable": [],
                "manual_fixes": [],
                "suggestions": []
            }

            # Process validation issues with fix suggestions
            for issue in validation_result.issues:
                issue_info = {
                    "field": issue.field,
                    "message": issue.message,
                    "code": issue.code,
                    "severity": issue.severity.value,
                    "current_value": getattr(target_settings, issue.field, "N/A"),
                    "auto_fixable": self._is_auto_fixable(issue),
                    "fix_suggestion": self._get_fix_suggestion(issue),
                    "user_guidance": self._get_user_guidance_for_issue(issue)
                }

                report["issues"].append(issue_info)

                if issue_info["auto_fixable"]:
                    report["auto_fixable"].append(issue_info)
                else:
                    report["manual_fixes"].append(issue_info)

            # Process warnings
            for warning in validation_result.warnings:
                warning_info = {
                    "field": warning.field,
                    "message": warning.message,
                    "code": warning.code,
                    "current_value": getattr(target_settings, warning.field, "N/A"),
                    "recommendation": self._get_warning_recommendation(warning)
                }
                report["warnings"].append(warning_info)

            # Add general suggestions
            if report["total_issues"] > 0:
                report["suggestions"].extend([
                    "Consider creating a backup before applying fixes",
                    "Review settings after automatic fixes are applied",
                    "Contact support if issues persist after fixes"
                ])

            if report["auto_fixable"]:
                report["suggestions"].append(f"{len(report['auto_fixable'])} issues can be automatically fixed")

            self.logger.info(f"Settings validation completed: {report['status']} ({report['total_issues']} issues, {report['total_warnings']} warnings)")
            return validation_result.is_valid, report

        except Exception as e:
            self.logger.exception(f"Error during settings validation: {e}")
            return False, {
                "status": "error",
                "message": f"Validation failed: {str(e)}",
                "issues": [],
                "suggestions": ["Check application logs for details", "Try reloading settings"]
            }

    async def apply_automatic_fixes(self, settings: Optional[AppSettings] = None) -> tuple[bool, Dict[str, Any]]:
        """
        Apply automatic fixes to validation issues where possible.

        Args:
            settings: Settings to fix (defaults to current settings)

        Returns:
            tuple[bool, Dict[str, Any]]: (success, fix_report)
        """
        try:
            target_settings = settings or self._settings
            if not target_settings:
                return False, {
                    "status": "error",
                    "message": "No settings available for fixing",
                    "fixes_applied": [],
                    "fixes_failed": []
                }

            # Get validation results
            is_valid, validation_report = await self.validate_and_report_settings(target_settings)

            fix_report = {
                "status": "completed",
                "fixes_applied": [],
                "fixes_failed": [],
                "settings_updated": False
            }

            # Apply automatic fixes
            for issue_info in validation_report.get("auto_fixable", []):
                try:
                    fix_result = await self._apply_specific_fix(issue_info, target_settings)
                    if fix_result[0]:
                        fix_report["fixes_applied"].append({
                            "field": issue_info["field"],
                            "issue": issue_info["message"],
                            "fix_applied": fix_result[1],
                            "new_value": getattr(target_settings, issue_info["field"], "N/A")
                        })
                        fix_report["settings_updated"] = True
                    else:
                        fix_report["fixes_failed"].append({
                            "field": issue_info["field"],
                            "issue": issue_info["message"],
                            "error": fix_result[1]
                        })

                except Exception as fix_error:
                    fix_report["fixes_failed"].append({
                        "field": issue_info["field"],
                        "issue": issue_info["message"],
                        "error": str(fix_error)
                    })

            # Save settings if any fixes were applied
            if fix_report["settings_updated"] and self.auto_save:
                save_success = await self.save_settings()
                if not save_success:
                    fix_report["status"] = "warning"
                    fix_report["message"] = "Fixes applied but failed to save settings"

            success = len(fix_report["fixes_applied"]) > 0 and len(fix_report["fixes_failed"]) == 0
            self.logger.info(f"Automatic fixes completed: {len(fix_report['fixes_applied'])} applied, {len(fix_report['fixes_failed'])} failed")

            return success, fix_report

        except Exception as e:
            self.logger.exception(f"Error applying automatic fixes: {e}")
            return False, {
                "status": "error",
                "message": f"Fix application failed: {str(e)}",
                "fixes_applied": [],
                "fixes_failed": []
            }

    def _is_auto_fixable(self, issue: 'ValidationIssue') -> bool:
        """Check if a validation issue can be automatically fixed."""
        auto_fixable_codes = {
            "INVALID_LOG_LEVEL",
            "INVALID_VOLUME_RANGE",
            "INVALID_DURATION",
            "INVALID_INTERVAL",
            "INVALID_TIMEOUT",
            "EMPTY_PATH",
            "UNKNOWN_THEME",
            "HIGH_DURATION",
            "LONG_RECENT_LIST",
            "DUPLICATE_RECENT_PROFILES",
            "INVALID_TRANSPARENCY_RANGE"
        }
        return issue.code in auto_fixable_codes

    def _get_fix_suggestion(self, issue: 'ValidationIssue') -> str:
        """Get fix suggestion for a validation issue."""
        suggestions = {
            "INVALID_LOG_LEVEL": "Will reset to 'INFO'",
            "INVALID_VOLUME_RANGE": "Will clamp to range 0.0-1.0",
            "INVALID_DURATION": "Will reset to default duration",
            "INVALID_INTERVAL": "Will reset to default interval",
            "INVALID_TIMEOUT": "Will reset to default timeout",
            "EMPTY_PATH": "Will reset to default path",
            "UNKNOWN_THEME": "Will reset to 'dark' theme",
            "HIGH_DURATION": "Will reduce to reasonable value",
            "LONG_RECENT_LIST": "Will trim to 10 most recent items",
            "DUPLICATE_RECENT_PROFILES": "Will remove duplicates",
            "INVALID_TRANSPARENCY_RANGE": "Will reset to 1.0 (fully opaque)"
        }
        return suggestions.get(issue.code, "Manual fix required")

    def _get_user_guidance_for_issue(self, issue: 'ValidationIssue') -> str:
        """Get user guidance for a validation issue."""
        guidance = {
            "INVALID_LOG_LEVEL": "Choose from: DEBUG, INFO, WARNING, ERROR, CRITICAL",
            "INVALID_VOLUME_RANGE": "Volume must be between 0.0 (mute) and 1.0 (full)",
            "INVALID_DURATION": "Duration must be positive and reasonable (<60 seconds)",
            "INVALID_INTERVAL": "Interval must be positive (suggested: 10-60 seconds)",
            "EMPTY_PATH": "Path cannot be empty, provide a valid directory path",
            "UNKNOWN_THEME": "Choose from: default, dark, light",
            "HIGH_DURATION": "Very high durations may cause memory issues"
        }
        return guidance.get(issue.code, "Please review and correct this setting manually")

    def _get_warning_recommendation(self, warning: 'ValidationIssue') -> str:
        """Get recommendation for a validation warning."""
        recommendations = {
            "UNKNOWN_THEME": "Consider using a supported theme for better experience",
            "HIGH_DURATION": "Consider reducing for better performance",
            "LONG_RECENT_LIST": "Consider clearing old entries for better performance",
            "DUPLICATE_RECENT_PROFILES": "Remove duplicates to clean up the list"
        }
        return recommendations.get(warning.code, "Review this setting when convenient")

    async def _apply_specific_fix(self, issue_info: Dict[str, Any], settings: AppSettings) -> tuple[bool, str]:
        """Apply a specific fix to a validation issue."""
        try:
            field = issue_info["field"]
            code = issue_info["code"]

            if code == "INVALID_LOG_LEVEL":
                settings.log_level = "INFO"
                return True, "Reset to INFO"

            elif code == "INVALID_DURATION" and field == "max_voice_duration":
                settings.max_voice_duration = 10.0
                return True, "Reset to 10 seconds"

            elif code == "INVALID_INTERVAL" and field == "auto_refresh_interval":
                settings.auto_refresh_interval = 30
                return True, "Reset to 30 seconds"

            elif code == "INVALID_TIMEOUT" and field == "tts_service_timeout":
                settings.tts_service_timeout = 30
                return True, "Reset to 30 seconds"

            elif code == "EMPTY_PATH":
                if field == "voice_files_directory":
                    settings.voice_files_directory = "voice_files"
                    return True, "Reset to 'voice_files'"
                elif field == "config_directory":
                    settings.config_directory = "config"
                    return True, "Reset to 'config'"

            elif code == "UNKNOWN_THEME":
                settings.ui_theme = "dark"
                return True, "Reset to 'dark'"

            elif code == "HIGH_DURATION" and field == "max_voice_duration":
                if settings.max_voice_duration > 60.0:
                    settings.max_voice_duration = 30.0
                    return True, "Reduced to 30 seconds"

            elif code == "LONG_RECENT_LIST":
                settings.recent_voice_profiles = settings.recent_voice_profiles[:10]
                return True, "Trimmed to 10 items"

            elif code == "DUPLICATE_RECENT_PROFILES":
                settings.recent_voice_profiles = list(dict.fromkeys(settings.recent_voice_profiles))
                return True, "Removed duplicates"

            elif code == "INVALID_TRANSPARENCY_RANGE":
                settings.window_transparency = 1.0
                return True, "Reset to 1.0 (fully opaque)"

            return False, f"No fix available for {code}"

        except Exception as e:
            return False, f"Fix failed: {str(e)}"

    # =========================================================================
    # Story 2.3: Audio Device Selection (FR26, FR27, FR44)
    # =========================================================================

    async def update_monitor_device(
        self,
        device_id: str,
        device_name: str,
        host_api: Optional[str] = None
    ) -> bool:
        """
        Update the selected monitor audio device in settings.

        Args:
            device_id: Device ID (PyAudio index)
            device_name: Human-readable device name
            host_api: Host API name (e.g., "Windows WASAPI")

        Returns:
            bool: True if update successful
        """
        try:
            if not self._settings:
                self.logger.error("No settings loaded, cannot update monitor device")
                return False

            old_device = self._settings.monitor_device_name
            self._settings.monitor_device_id = device_id
            self._settings.monitor_device_name = device_name
            self._settings.monitor_device_host_api = host_api

            self.logger.info(f"Monitor device updated: '{old_device}' -> '{device_name}' (id={device_id})")

            # Auto-save if enabled
            if self.auto_save:
                return await self.save_settings()

            return True

        except Exception as e:
            self.logger.exception(f"Error updating monitor device: {e}")
            return False

    async def update_virtual_microphone_device(
        self,
        device_id: str,
        device_name: str,
        host_api: Optional[str] = None
    ) -> bool:
        """
        Update the selected virtual microphone device in settings.

        Args:
            device_id: Device ID (PyAudio index)
            device_name: Human-readable device name
            host_api: Host API name (e.g., "Windows WASAPI")

        Returns:
            bool: True if update successful
        """
        try:
            if not self._settings:
                self.logger.error("No settings loaded, cannot update virtual microphone device")
                return False

            old_device = self._settings.virtual_microphone_device_name
            self._settings.virtual_microphone_device_id = device_id
            self._settings.virtual_microphone_device_name = device_name
            self._settings.virtual_microphone_device_host_api = host_api

            self.logger.info(f"Virtual mic device updated: '{old_device}' -> '{device_name}' (id={device_id})")

            # Auto-save if enabled
            if self.auto_save:
                return await self.save_settings()

            return True

        except Exception as e:
            self.logger.exception(f"Error updating virtual microphone device: {e}")
            return False

    def get_monitor_device(self) -> Optional[Dict[str, Any]]:
        """
        Get the currently selected monitor device.

        Returns:
            Dict with device_id, device_name, host_api, or None if not set
        """
        if not self._settings or not self._settings.monitor_device_id:
            return None

        return {
            "device_id": self._settings.monitor_device_id,
            "device_name": self._settings.monitor_device_name,
            "host_api": self._settings.monitor_device_host_api,
        }

    def get_virtual_microphone_device(self) -> Optional[Dict[str, Any]]:
        """
        Get the currently selected virtual microphone device.

        Returns:
            Dict with device_id, device_name, host_api, or None if not set
        """
        if not self._settings or not self._settings.virtual_microphone_device_id:
            return None

        return {
            "device_id": self._settings.virtual_microphone_device_id,
            "device_name": self._settings.virtual_microphone_device_name,
            "host_api": self._settings.virtual_microphone_device_host_api,
        }

    async def clear_monitor_device(self) -> bool:
        """
        Clear the selected monitor device (use system default).

        Returns:
            bool: True if successful
        """
        try:
            if not self._settings:
                return False

            self._settings.monitor_device_id = None
            self._settings.monitor_device_name = None
            self._settings.monitor_device_host_api = None

            self.logger.info("Monitor device selection cleared")

            if self.auto_save:
                return await self.save_settings()

            return True

        except Exception as e:
            self.logger.exception(f"Error clearing monitor device: {e}")
            return False

    async def clear_virtual_microphone_device(self) -> bool:
        """
        Clear the selected virtual microphone device.

        Returns:
            bool: True if successful
        """
        try:
            if not self._settings:
                return False

            self._settings.virtual_microphone_device_id = None
            self._settings.virtual_microphone_device_name = None
            self._settings.virtual_microphone_device_host_api = None

            self.logger.info("Virtual microphone device selection cleared")

            if self.auto_save:
                return await self.save_settings()

            return True

        except Exception as e:
            self.logger.exception(f"Error clearing virtual microphone device: {e}")
            return False

    # =========================================================================
    # Story 6.2: Window Settings Persistence (FR41)
    # =========================================================================

    async def update_window_geometry(
        self,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> bool:
        """
        Update window geometry (position and size) in settings.

        Args:
            x: Window X position
            y: Window Y position
            width: Window width
            height: Window height

        Returns:
            bool: True if update successful
        """
        try:
            if not self._settings:
                self.logger.error("No settings loaded, cannot update window geometry")
                return False

            self._settings.window_geometry = {
                "x": x,
                "y": y,
                "width": width,
                "height": height
            }

            self.logger.debug(f"Window geometry updated: x={x}, y={y}, width={width}, height={height}")

            # Auto-save if enabled
            if self.auto_save:
                return await self.save_settings()

            return True

        except Exception as e:
            self.logger.exception(f"Error updating window geometry: {e}")
            return False

    async def update_window_transparency(self, transparency: float) -> bool:
        """
        Update window transparency in settings.

        Args:
            transparency: Window opacity (0.0 = fully transparent, 1.0 = fully opaque)

        Returns:
            bool: True if update successful
        """
        try:
            if not self._settings:
                self.logger.error("No settings loaded, cannot update window transparency")
                return False

            # Validate range
            if transparency < 0.0 or transparency > 1.0:
                self.logger.error(f"Invalid transparency value: {transparency}. Must be 0.0-1.0")
                return False

            old_transparency = self._settings.window_transparency
            self._settings.window_transparency = transparency

            self.logger.info(f"Window transparency updated: {old_transparency} -> {transparency}")

            # Auto-save if enabled
            if self.auto_save:
                return await self.save_settings()

            return True

        except Exception as e:
            self.logger.exception(f"Error updating window transparency: {e}")
            return False

    async def update_always_on_top(self, always_on_top: bool) -> bool:
        """
        Update always-on-top setting.

        Args:
            always_on_top: Whether window should stay on top

        Returns:
            bool: True if update successful
        """
        try:
            if not self._settings:
                self.logger.error("No settings loaded, cannot update always_on_top")
                return False

            old_value = self._settings.always_on_top
            self._settings.always_on_top = always_on_top

            self.logger.info(f"Always on top updated: {old_value} -> {always_on_top}")

            # Auto-save if enabled
            if self.auto_save:
                return await self.save_settings()

            return True

        except Exception as e:
            self.logger.exception(f"Error updating always_on_top: {e}")
            return False

    def get_window_geometry(self) -> Optional[Dict[str, int]]:
        """
        Get the saved window geometry.

        Returns:
            Dict with x, y, width, height or None if not set
        """
        if not self._settings or not self._settings.window_geometry:
            return None

        return self._settings.window_geometry.copy()

    def get_window_transparency(self) -> float:
        """
        Get the saved window transparency.

        Returns:
            Transparency value (0.0-1.0), defaults to 1.0
        """
        if not self._settings:
            return 1.0

        return self._settings.window_transparency

    def get_always_on_top(self) -> bool:
        """
        Get the always-on-top setting.

        Returns:
            Always on top setting, defaults to True
        """
        if not self._settings:
            return True

        return self._settings.always_on_top

    def get_no_virtual_mic_guidance(self) -> Dict[str, Any]:
        """
        Get guidance for when no virtual microphone is detected (FR44).

        Returns:
            Dict with guidance information and links
        """
        return {
            "title": "No Virtual Microphone Detected",
            "message": (
                "To use MyVoice in voice chat applications, you need a virtual "
                "audio cable installed. This creates a virtual microphone that "
                "other apps can use as their audio input."
            ),
            "recommended_software": [
                {
                    "name": "VB-Audio Virtual Cable",
                    "url": "https://vb-audio.com/Cable/",
                    "description": "Free, lightweight virtual audio cable",
                    "free": True,
                },
                {
                    "name": "VoiceMeeter",
                    "url": "https://vb-audio.com/Voicemeeter/",
                    "description": "Free audio mixer with virtual cables",
                    "free": True,
                },
            ],
            "setup_steps": [
                "1. Download and install VB-Audio Virtual Cable",
                "2. Restart your computer after installation",
                "3. In MyVoice Settings, select 'CABLE Input' as virtual microphone",
                "4. In your voice chat app, select 'CABLE Output' as microphone",
            ],
            "troubleshooting": [
                "If device doesn't appear, restart your computer",
                "Make sure to select 'CABLE Input' (not 'CABLE Output') in MyVoice",
                "Check Windows Sound settings to verify device is enabled",
            ],
        }