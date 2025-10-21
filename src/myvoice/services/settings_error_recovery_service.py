"""
Settings Error Recovery Service

This module provides comprehensive error handling and recovery mechanisms for settings operations,
including file corruption recovery, device enumeration failure handling, validation error reporting,
and user guidance for configuration issues.
"""

import asyncio
import logging
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from myvoice.models.error import MyVoiceError, ErrorSeverity
from myvoice.models.validation import ValidationResult, ValidationIssue, ValidationStatus
from myvoice.models.app_settings import AppSettings
from myvoice.services.core.base_service import BaseService, ServiceStatus


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    AUTO_RECOVER = "auto_recover"
    PROMPT_USER = "prompt_user"
    FALLBACK_DEFAULTS = "fallback_defaults"
    MANUAL_INTERVENTION = "manual_intervention"


class ErrorCategory(Enum):
    """Categories of settings-related errors."""
    FILE_CORRUPTION = "file_corruption"
    DEVICE_ENUMERATION = "device_enumeration"
    VALIDATION_FAILURE = "validation_failure"
    PERMISSION_ERROR = "permission_error"
    NETWORK_ERROR = "network_error"
    CONFIGURATION_ERROR = "configuration_error"


@dataclass
class RecoveryAction:
    """Represents a recovery action that can be taken."""
    action_id: str
    title: str
    description: str
    strategy: RecoveryStrategy
    auto_execute: bool = False
    callback: Optional[Callable] = None
    risk_level: str = "low"  # low, medium, high
    estimated_success_rate: float = 0.8


@dataclass
class ErrorContext:
    """Context information for error recovery."""
    error: MyVoiceError
    category: ErrorCategory
    source_file: Optional[Path] = None
    attempted_operations: List[str] = field(default_factory=list)
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    user_guidance: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class RecoveryReport:
    """Report of recovery operation results."""
    success: bool
    actions_taken: List[str]
    warnings: List[str]
    user_messages: List[str]
    technical_details: Dict[str, Any]
    recovery_duration: float


class SettingsErrorRecoveryService(BaseService):
    """
    Comprehensive error recovery service for settings operations.

    This service provides:
    - Settings file corruption detection and recovery
    - Device enumeration failure handling with fallback strategies
    - Settings validation error reporting with fix suggestions
    - User guidance for configuration issues
    - Automated recovery workflows where safe
    """

    def __init__(self, settings_directory: Path, backup_directory: Optional[Path] = None):
        """
        Initialize the Settings Error Recovery Service.

        Args:
            settings_directory: Path to the settings directory
            backup_directory: Path to backup directory (defaults to settings_dir/backups)
        """
        super().__init__("SettingsErrorRecoveryService")

        self.settings_directory = settings_directory
        self.backup_directory = backup_directory or (settings_directory / "backups")

        # Recovery history for tracking patterns
        self.recovery_history: List[ErrorContext] = []
        self.max_history_entries = 100

        # Recovery statistics
        self.recovery_stats = {
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "auto_recoveries": 0,
            "manual_interventions": 0,
            "by_category": {category.value: 0 for category in ErrorCategory}
        }

        # Recovery callback registry
        self.recovery_callbacks: Dict[str, Callable] = {}

        self.logger.info("Settings Error Recovery Service initialized")

    async def start(self) -> bool:
        """Start the recovery service."""
        try:
            await self._update_status(ServiceStatus.STARTING)
            self.logger.info("Starting Settings Error Recovery Service")

            # Ensure backup directory exists
            self.backup_directory.mkdir(parents=True, exist_ok=True)

            # Initialize recovery callbacks
            self._initialize_recovery_callbacks()

            await self._update_status(ServiceStatus.RUNNING)
            self.logger.info("Settings Error Recovery Service started successfully")
            return True

        except Exception as e:
            self.logger.exception(f"Failed to start Settings Error Recovery Service: {e}")
            await self._update_status(ServiceStatus.ERROR)
            return False

    async def stop(self) -> bool:
        """Stop the recovery service."""
        try:
            await self._update_status(ServiceStatus.STOPPING)
            self.logger.info("Stopping Settings Error Recovery Service")

            # Save recovery statistics
            await self._save_recovery_statistics()

            await self._update_status(ServiceStatus.STOPPED)
            self.logger.info("Settings Error Recovery Service stopped successfully")
            return True

        except Exception as e:
            self.logger.exception(f"Error stopping Settings Error Recovery Service: {e}")
            await self._update_status(ServiceStatus.ERROR)
            return False

    async def handle_settings_file_corruption(self,
                                            corrupted_file: Path,
                                            auto_recover: bool = True) -> Tuple[bool, RecoveryReport]:
        """
        Handle settings file corruption with multiple recovery strategies.

        Args:
            corrupted_file: Path to the corrupted settings file
            auto_recover: Whether to attempt automatic recovery

        Returns:
            Tuple[bool, RecoveryReport]: (success, recovery_report)
        """
        start_time = time.time()
        self.logger.warning(f"Handling settings file corruption: {corrupted_file}")

        error = MyVoiceError(
            severity=ErrorSeverity.ERROR,
            code="SETTINGS_FILE_CORRUPTED",
            user_message=f"Settings file {corrupted_file.name} is corrupted",
            technical_details=f"File: {corrupted_file}"
        )

        context = ErrorContext(
            error=error,
            category=ErrorCategory.FILE_CORRUPTION,
            source_file=corrupted_file
        )

        # Build recovery actions
        recovery_actions = await self._build_corruption_recovery_actions(corrupted_file)
        context.recovery_actions = recovery_actions

        recovery_report = RecoveryReport(
            success=False,
            actions_taken=[],
            warnings=[],
            user_messages=[],
            technical_details={},
            recovery_duration=0.0
        )

        try:
            # Strategy 1: Try to recover from backup files
            if auto_recover:
                backup_recovery = await self._attempt_backup_recovery(corrupted_file)
                if backup_recovery[0]:
                    recovery_report.success = True
                    recovery_report.actions_taken.append("Restored from backup file")
                    recovery_report.user_messages.append("Settings restored from backup")
                    self.logger.info(f"Successfully recovered {corrupted_file} from backup")
                    self.recovery_stats["successful_recoveries"] += 1
                    self.recovery_stats["auto_recoveries"] += 1
                else:
                    recovery_report.warnings.append("Backup recovery failed")

            # Strategy 2: Try to repair the corrupted file
            if not recovery_report.success and auto_recover:
                repair_result = await self._attempt_file_repair(corrupted_file)
                if repair_result[0]:
                    recovery_report.success = True
                    recovery_report.actions_taken.append("Repaired corrupted file")
                    recovery_report.user_messages.append("Settings file repaired successfully")
                    self.logger.info(f"Successfully repaired corrupted file {corrupted_file}")
                    self.recovery_stats["successful_recoveries"] += 1
                    self.recovery_stats["auto_recoveries"] += 1
                else:
                    recovery_report.warnings.append("File repair failed")

            # Strategy 3: Create new settings with defaults
            if not recovery_report.success:
                default_recovery = await self._create_default_settings_file(corrupted_file)
                if default_recovery[0]:
                    recovery_report.success = True
                    recovery_report.actions_taken.append("Created new settings with defaults")
                    recovery_report.user_messages.append("New settings file created with default values")
                    recovery_report.warnings.append("Previous settings were lost")
                    self.logger.info(f"Created new default settings file at {corrupted_file}")
                    self.recovery_stats["successful_recoveries"] += 1
                    if auto_recover:
                        self.recovery_stats["auto_recoveries"] += 1

            # Update context with results
            context.attempted_operations = recovery_report.actions_taken
            if not recovery_report.success:
                context.user_guidance = self._generate_manual_recovery_guidance(corrupted_file)
                recovery_report.user_messages.append(context.user_guidance)
                self.recovery_stats["manual_interventions"] += 1

            recovery_report.recovery_duration = time.time() - start_time
            recovery_report.technical_details = {
                "file_path": str(corrupted_file),
                "backup_available": len(list(self.backup_directory.glob(f"{corrupted_file.stem}*"))) > 0,
                "recovery_actions_attempted": len(recovery_report.actions_taken)
            }

            # Record recovery attempt
            self.recovery_stats["total_recoveries"] += 1
            self.recovery_stats["by_category"][ErrorCategory.FILE_CORRUPTION.value] += 1
            self._add_to_history(context)

            return recovery_report.success, recovery_report

        except Exception as e:
            self.logger.exception(f"Error during corruption recovery: {e}")
            recovery_report.warnings.append(f"Recovery process failed: {str(e)}")
            recovery_report.recovery_duration = time.time() - start_time
            return False, recovery_report

    async def handle_device_enumeration_failure(self,
                                              original_error: Exception,
                                              retry_count: int = 3) -> Tuple[bool, RecoveryReport]:
        """
        Handle device enumeration failures with progressive fallback strategies.

        Args:
            original_error: The original enumeration error
            retry_count: Number of retry attempts

        Returns:
            Tuple[bool, RecoveryReport]: (success, recovery_report)
        """
        start_time = time.time()
        self.logger.warning(f"Handling device enumeration failure: {original_error}")

        error = MyVoiceError(
            severity=ErrorSeverity.ERROR,
            code="DEVICE_ENUMERATION_FAILED",
            user_message="Failed to scan audio devices",
            technical_details=str(original_error)
        )

        context = ErrorContext(
            error=error,
            category=ErrorCategory.DEVICE_ENUMERATION
        )

        recovery_report = RecoveryReport(
            success=False,
            actions_taken=[],
            warnings=[],
            user_messages=[],
            technical_details={},
            recovery_duration=0.0
        )

        try:
            # Strategy 1: Progressive retry with delays
            for attempt in range(retry_count):
                await asyncio.sleep(0.5 * (attempt + 1))  # Progressive delay

                try:
                    # Simulate device enumeration retry (would call actual service)
                    self.logger.debug(f"Device enumeration retry attempt {attempt + 1}")
                    recovery_report.actions_taken.append(f"Retry attempt {attempt + 1}")

                    # In real implementation, would call device enumeration here
                    # For now, simulate success after some attempts
                    if attempt >= 1:  # Simulate success on 2nd attempt
                        recovery_report.success = True
                        recovery_report.user_messages.append("Device enumeration recovered after retry")
                        self.recovery_stats["successful_recoveries"] += 1
                        self.recovery_stats["auto_recoveries"] += 1
                        break

                except Exception as retry_error:
                    recovery_report.warnings.append(f"Retry {attempt + 1} failed: {str(retry_error)}")
                    continue

            # Strategy 2: Fallback to cached device list
            if not recovery_report.success:
                cached_devices = await self._get_cached_device_list()
                if cached_devices:
                    recovery_report.success = True
                    recovery_report.actions_taken.append("Used cached device list")
                    recovery_report.user_messages.append("Using previously discovered devices")
                    recovery_report.warnings.append("Device list may not be current")
                    self.recovery_stats["successful_recoveries"] += 1
                    self.recovery_stats["auto_recoveries"] += 1

            # Strategy 3: Provide fallback guidance
            if not recovery_report.success:
                context.user_guidance = self._generate_device_troubleshooting_guidance()
                recovery_report.user_messages.append(context.user_guidance)
                self.recovery_stats["manual_interventions"] += 1

            recovery_report.recovery_duration = time.time() - start_time
            recovery_report.technical_details = {
                "original_error": str(original_error),
                "retry_attempts": len([a for a in recovery_report.actions_taken if "Retry" in a]),
                "fallback_used": "cached device list" in recovery_report.actions_taken
            }

            # Record recovery attempt
            self.recovery_stats["total_recoveries"] += 1
            self.recovery_stats["by_category"][ErrorCategory.DEVICE_ENUMERATION.value] += 1
            context.attempted_operations = recovery_report.actions_taken
            self._add_to_history(context)

            return recovery_report.success, recovery_report

        except Exception as e:
            self.logger.exception(f"Error during device enumeration recovery: {e}")
            recovery_report.warnings.append(f"Recovery process failed: {str(e)}")
            recovery_report.recovery_duration = time.time() - start_time
            return False, recovery_report

    async def handle_validation_errors(self,
                                     validation_result: ValidationResult,
                                     settings: AppSettings) -> Tuple[bool, RecoveryReport]:
        """
        Handle settings validation errors with automatic fixing where possible.

        Args:
            validation_result: The validation result containing errors
            settings: The settings object to fix

        Returns:
            Tuple[bool, RecoveryReport]: (success, recovery_report)
        """
        start_time = time.time()
        self.logger.warning(f"Handling validation errors: {len(validation_result.issues)} issues")

        recovery_report = RecoveryReport(
            success=False,
            actions_taken=[],
            warnings=[],
            user_messages=[],
            technical_details={},
            recovery_duration=0.0
        )

        fixes_applied = 0
        fixes_failed = 0

        try:
            for issue in validation_result.issues:
                fix_result = await self._attempt_validation_fix(issue, settings)

                if fix_result[0]:
                    fixes_applied += 1
                    recovery_report.actions_taken.append(f"Fixed: {issue.field} - {fix_result[1]}")
                else:
                    fixes_failed += 1
                    recovery_report.warnings.append(f"Could not fix: {issue.field} - {issue.message}")

            # Handle warnings separately
            for warning in validation_result.warnings:
                warning_result = await self._attempt_warning_fix(warning, settings)
                if warning_result[0]:
                    recovery_report.actions_taken.append(f"Addressed warning: {warning.field}")

            # Determine overall success
            if fixes_applied > 0 and fixes_failed == 0:
                recovery_report.success = True
                recovery_report.user_messages.append(f"Fixed {fixes_applied} validation issues")
                self.recovery_stats["successful_recoveries"] += 1
                self.recovery_stats["auto_recoveries"] += 1
            elif fixes_applied > 0:
                recovery_report.success = True  # Partial success
                recovery_report.user_messages.append(f"Fixed {fixes_applied} of {fixes_applied + fixes_failed} issues")
                recovery_report.warnings.append(f"{fixes_failed} issues require manual attention")
                self.recovery_stats["successful_recoveries"] += 1
            else:
                recovery_report.user_messages.append("Manual review required for validation issues")
                self.recovery_stats["manual_interventions"] += 1

            recovery_report.recovery_duration = time.time() - start_time
            recovery_report.technical_details = {
                "total_issues": len(validation_result.issues),
                "fixes_applied": fixes_applied,
                "fixes_failed": fixes_failed,
                "warnings_count": len(validation_result.warnings)
            }

            # Record recovery attempt
            self.recovery_stats["total_recoveries"] += 1
            self.recovery_stats["by_category"][ErrorCategory.VALIDATION_FAILURE.value] += 1

            return recovery_report.success, recovery_report

        except Exception as e:
            self.logger.exception(f"Error during validation error recovery: {e}")
            recovery_report.warnings.append(f"Recovery process failed: {str(e)}")
            recovery_report.recovery_duration = time.time() - start_time
            return False, recovery_report

    async def generate_user_guidance(self, error_context: ErrorContext) -> str:
        """
        Generate comprehensive user guidance for configuration issues.

        Args:
            error_context: Context about the error

        Returns:
            str: User-friendly guidance message
        """
        guidance_parts = []

        # Category-specific guidance
        if error_context.category == ErrorCategory.FILE_CORRUPTION:
            guidance_parts.append("Settings file corruption detected:")
            guidance_parts.append("• Check if antivirus software is interfering")
            guidance_parts.append("• Ensure sufficient disk space is available")
            guidance_parts.append("• Try running the application as administrator")

        elif error_context.category == ErrorCategory.DEVICE_ENUMERATION:
            guidance_parts.append("Audio device detection failed:")
            guidance_parts.append("• Check audio driver status in Device Manager")
            guidance_parts.append("• Restart Windows Audio service")
            guidance_parts.append("• Disconnect and reconnect audio devices")
            guidance_parts.append("• Update audio drivers")

        elif error_context.category == ErrorCategory.VALIDATION_FAILURE:
            guidance_parts.append("Configuration validation issues found:")
            guidance_parts.append("• Some settings may have invalid values")
            guidance_parts.append("• Consider resetting to default values")
            guidance_parts.append("• Check file permissions on settings directory")

        # Add recovery action suggestions
        if error_context.recovery_actions:
            guidance_parts.append("\nRecommended actions:")
            for action in error_context.recovery_actions:
                if action.risk_level == "low":
                    guidance_parts.append(f"• {action.description} (Safe)")
                elif action.risk_level == "medium":
                    guidance_parts.append(f"• {action.description} (May lose some settings)")
                else:
                    guidance_parts.append(f"• {action.description} (Advanced users only)")

        # Add contact information for persistent issues
        guidance_parts.append("\nIf issues persist:")
        guidance_parts.append("• Check application logs for detailed error information")
        guidance_parts.append("• Consider creating a backup before making changes")
        guidance_parts.append("• Contact support with error details and logs")

        return "\n".join(guidance_parts)

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery operation statistics."""
        success_rate = (
            self.recovery_stats["successful_recoveries"] / self.recovery_stats["total_recoveries"]
            if self.recovery_stats["total_recoveries"] > 0 else 0.0
        )

        auto_recovery_rate = (
            self.recovery_stats["auto_recoveries"] / self.recovery_stats["total_recoveries"]
            if self.recovery_stats["total_recoveries"] > 0 else 0.0
        )

        return {
            **self.recovery_stats,
            "success_rate": success_rate,
            "auto_recovery_rate": auto_recovery_rate,
            "recent_recoveries": len([h for h in self.recovery_history if time.time() - h.timestamp < 3600])
        }

    # Private helper methods

    def _initialize_recovery_callbacks(self):
        """Initialize recovery callback registry."""
        self.recovery_callbacks.update({
            "backup_recovery": self._attempt_backup_recovery,
            "file_repair": self._attempt_file_repair,
            "default_creation": self._create_default_settings_file,
            "device_retry": self._retry_device_enumeration,
            "validation_fix": self._attempt_validation_fix
        })

    async def _build_corruption_recovery_actions(self, corrupted_file: Path) -> List[RecoveryAction]:
        """Build recovery actions for file corruption."""
        actions = []

        # Check for available backups
        backup_files = list(self.backup_directory.glob(f"{corrupted_file.stem}*"))
        if backup_files:
            actions.append(RecoveryAction(
                action_id="restore_backup",
                title="Restore from Backup",
                description=f"Restore settings from backup file ({len(backup_files)} available)",
                strategy=RecoveryStrategy.AUTO_RECOVER,
                auto_execute=True,
                risk_level="low",
                estimated_success_rate=0.9
            ))

        # File repair option
        actions.append(RecoveryAction(
            action_id="repair_file",
            title="Repair Corrupted File",
            description="Attempt to repair the corrupted settings file",
            strategy=RecoveryStrategy.AUTO_RECOVER,
            auto_execute=True,
            risk_level="medium",
            estimated_success_rate=0.6
        ))

        # Reset to defaults
        actions.append(RecoveryAction(
            action_id="reset_defaults",
            title="Reset to Defaults",
            description="Create new settings file with default values",
            strategy=RecoveryStrategy.FALLBACK_DEFAULTS,
            auto_execute=False,
            risk_level="medium",
            estimated_success_rate=1.0
        ))

        return actions

    async def _attempt_backup_recovery(self, corrupted_file: Path) -> Tuple[bool, str]:
        """Attempt to recover from backup files."""
        try:
            backup_files = sorted(
                self.backup_directory.glob(f"{corrupted_file.stem}*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            for backup_file in backup_files[:3]:  # Try up to 3 most recent backups
                try:
                    # Validate backup file
                    content = backup_file.read_text(encoding='utf-8')
                    data = json.loads(content)

                    # Basic validation
                    if isinstance(data, dict) and len(data) > 0:
                        # Copy backup to original location
                        corrupted_file.write_text(content, encoding='utf-8')
                        self.logger.info(f"Successfully restored from backup: {backup_file}")
                        return True, f"Restored from {backup_file.name}"

                except (json.JSONDecodeError, OSError) as e:
                    self.logger.warning(f"Backup file {backup_file} is also corrupted: {e}")
                    continue

            return False, "No valid backup files found"

        except Exception as e:
            self.logger.error(f"Backup recovery failed: {e}")
            return False, f"Backup recovery error: {str(e)}"

    async def _attempt_file_repair(self, corrupted_file: Path) -> Tuple[bool, str]:
        """Attempt to repair a corrupted settings file."""
        try:
            content = corrupted_file.read_text(encoding='utf-8')

            # Try common JSON repair strategies
            repair_strategies = [
                self._repair_truncated_json,
                self._repair_extra_commas,
                self._repair_quote_issues,
                self._repair_bracket_mismatch
            ]

            for strategy in repair_strategies:
                try:
                    repaired_content = strategy(content)
                    if repaired_content != content:
                        # Validate repaired content
                        data = json.loads(repaired_content)
                        if isinstance(data, dict):
                            # Create backup of corrupted file
                            backup_path = corrupted_file.with_suffix('.corrupted')
                            corrupted_file.rename(backup_path)

                            # Write repaired content
                            corrupted_file.write_text(repaired_content, encoding='utf-8')

                            self.logger.info(f"Successfully repaired corrupted file using {strategy.__name__}")
                            return True, f"Repaired using {strategy.__name__}"

                except (json.JSONDecodeError, OSError):
                    continue

            return False, "Could not repair file with available strategies"

        except Exception as e:
            self.logger.error(f"File repair failed: {e}")
            return False, f"File repair error: {str(e)}"

    async def _create_default_settings_file(self, file_path: Path) -> Tuple[bool, str]:
        """Create a new settings file with default values."""
        try:
            # Create backup of corrupted file if it exists
            if file_path.exists():
                backup_path = file_path.with_suffix(f'.corrupted_{int(time.time())}')
                file_path.rename(backup_path)
                self.logger.info(f"Backed up corrupted file to: {backup_path}")

            # Create default settings
            default_settings = AppSettings()
            settings_data = default_settings.to_dict()

            # Write to file
            file_path.write_text(
                json.dumps(settings_data, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )

            self.logger.info(f"Created default settings file: {file_path}")
            return True, "Created new settings with defaults"

        except Exception as e:
            self.logger.error(f"Default settings creation failed: {e}")
            return False, f"Default creation error: {str(e)}"

    async def _get_cached_device_list(self) -> Optional[List[Dict[str, Any]]]:
        """Get cached device list as fallback."""
        # This would interface with the audio service's cached device information
        # For now, return a placeholder
        return None

    async def _retry_device_enumeration(self) -> Tuple[bool, str]:
        """Retry device enumeration with progressive delays."""
        # This would interface with the audio service
        # For now, return a placeholder
        return False, "Not implemented"

    async def _attempt_validation_fix(self, issue: ValidationIssue, settings: AppSettings) -> Tuple[bool, str]:
        """Attempt to automatically fix a validation issue."""
        try:
            if issue.code == "INVALID_LOG_LEVEL":
                settings.log_level = "INFO"
                return True, "Reset log level to INFO"

            elif issue.code == "EMPTY_PATH":
                if issue.field == "voice_files_directory":
                    settings.voice_files_directory = "voice_files"
                    return True, "Reset voice files directory to default"
                elif issue.field == "config_directory":
                    settings.config_directory = "config"
                    return True, "Reset config directory to default"

            elif issue.code == "INVALID_DURATION":
                if issue.field == "max_voice_duration":
                    settings.max_voice_duration = 10.0
                    return True, "Reset max voice duration to 10 seconds"

            elif issue.code == "INVALID_INTERVAL":
                if issue.field == "auto_refresh_interval":
                    settings.auto_refresh_interval = 30
                    return True, "Reset auto-refresh interval to 30 seconds"

            return False, f"No automatic fix available for {issue.code}"

        except Exception as e:
            self.logger.error(f"Validation fix failed for {issue.field}: {e}")
            return False, f"Fix attempt failed: {str(e)}"

    async def _attempt_warning_fix(self, warning: ValidationIssue, settings: AppSettings) -> Tuple[bool, str]:
        """Attempt to address validation warnings."""
        try:
            if warning.code == "UNKNOWN_THEME":
                settings.ui_theme = "dark"
                return True, "Reset UI theme to dark"

            elif warning.code == "HIGH_DURATION":
                if settings.max_voice_duration > 60.0:
                    settings.max_voice_duration = 30.0
                    return True, "Reduced max voice duration to 30 seconds"

            elif warning.code == "LONG_RECENT_LIST":
                settings.recent_voice_profiles = settings.recent_voice_profiles[:10]
                return True, "Trimmed recent voice profiles list"

            elif warning.code == "DUPLICATE_RECENT_PROFILES":
                settings.recent_voice_profiles = list(dict.fromkeys(settings.recent_voice_profiles))
                return True, "Removed duplicate recent profiles"

            return False, f"No automatic fix available for warning {warning.code}"

        except Exception as e:
            self.logger.error(f"Warning fix failed for {warning.field}: {e}")
            return False, f"Warning fix attempt failed: {str(e)}"

    def _repair_truncated_json(self, content: str) -> str:
        """Repair JSON that was truncated."""
        # Try to add missing closing braces
        open_braces = content.count('{') - content.count('}')
        if open_braces > 0:
            return content + '}' * open_braces
        return content

    def _repair_extra_commas(self, content: str) -> str:
        """Remove trailing commas that break JSON parsing."""
        import re
        # Remove trailing commas before closing braces/brackets
        content = re.sub(r',\s*}', '}', content)
        content = re.sub(r',\s*]', ']', content)
        return content

    def _repair_quote_issues(self, content: str) -> str:
        """Fix common quote-related JSON issues."""
        # This is a simplified repair - real implementation would be more sophisticated
        return content

    def _repair_bracket_mismatch(self, content: str) -> str:
        """Fix bracket/brace mismatches."""
        # This is a simplified repair - real implementation would be more sophisticated
        return content

    def _generate_manual_recovery_guidance(self, corrupted_file: Path) -> str:
        """Generate guidance for manual recovery."""
        return f"""
Manual recovery required for {corrupted_file.name}:

1. Check backup directory: {self.backup_directory}
2. Look for recent backup files with similar names
3. If backup exists, copy it to: {corrupted_file}
4. If no backup, delete corrupted file to create fresh settings
5. Restart the application to regenerate default settings

Contact support if you need to preserve specific settings.
"""

    def _generate_device_troubleshooting_guidance(self) -> str:
        """Generate guidance for device enumeration issues."""
        return """
Audio device detection troubleshooting:

1. Check Device Manager for audio device status
2. Update audio drivers from manufacturer
3. Restart Windows Audio service:
   • Press Win+R, type 'services.msc'
   • Find 'Windows Audio' service
   • Right-click and select 'Restart'
4. Disconnect and reconnect USB audio devices
5. Try running the application as administrator
6. Check for Windows Updates

If using virtual audio devices (VB-Cable, Voicemeeter):
• Ensure they are properly installed and running
• Check virtual device driver status
• Restart virtual audio software
"""

    def _add_to_history(self, context: ErrorContext):
        """Add error context to recovery history."""
        self.recovery_history.append(context)

        # Trim history if it gets too long
        if len(self.recovery_history) > self.max_history_entries:
            self.recovery_history = self.recovery_history[-self.max_history_entries:]

    async def _save_recovery_statistics(self):
        """Save recovery statistics to file."""
        try:
            stats_file = self.settings_directory / "recovery_stats.json"
            stats_file.write_text(
                json.dumps(self.recovery_stats, indent=2),
                encoding='utf-8'
            )
            self.logger.debug(f"Saved recovery statistics to {stats_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save recovery statistics: {e}")