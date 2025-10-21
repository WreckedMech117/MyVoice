"""
Validation Models

This module contains validation result models and related data structures
for the MyVoice application services.
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

from myvoice.models.error import MyVoiceError


class ValidationStatus(Enum):
    """Validation result status."""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"


@dataclass
class ValidationIssue:
    """
    Represents a single validation issue.

    Attributes:
        field: Field name that has the issue
        message: Human-readable description of the issue
        code: Error code for programmatic handling
        severity: Severity level of the issue
    """
    field: str
    message: str
    code: str
    severity: ValidationStatus


@dataclass
class ValidationResult:
    """
    Result of validation operation.

    Attributes:
        is_valid: Whether validation passed
        status: Overall validation status
        issues: List of validation issues found
        warnings: List of non-blocking warnings
        summary: Summary message for the validation result
    """
    is_valid: bool
    status: ValidationStatus
    issues: List[ValidationIssue]
    warnings: List[ValidationIssue]
    summary: Optional[str] = None

    def __post_init__(self):
        """Set summary if not provided."""
        if self.summary is None:
            if self.is_valid and not self.warnings:
                self.summary = "Validation passed successfully"
            elif self.is_valid and self.warnings:
                self.summary = f"Validation passed with {len(self.warnings)} warning(s)"
            else:
                self.summary = f"Validation failed with {len(self.issues)} error(s)"

    def has_errors(self) -> bool:
        """Check if there are any validation errors."""
        return len(self.issues) > 0

    def has_warnings(self) -> bool:
        """Check if there are any validation warnings."""
        return len(self.warnings) > 0

    def get_error_messages(self) -> List[str]:
        """Get list of error messages."""
        return [issue.message for issue in self.issues]

    def get_warning_messages(self) -> List[str]:
        """Get list of warning messages."""
        return [issue.message for issue in self.warnings]


@dataclass
class UserFriendlyMessage:
    """
    User-friendly message for displaying to end users.

    This provides a clean interface for showing validation results,
    errors, and other messages in the UI.

    Attributes:
        title: Message title/header
        message: Main message content
        details: Optional detailed information
        action_suggestions: List of suggested actions user can take
        severity: Message severity for UI styling
        is_recoverable: Whether the user can retry/fix the issue
    """
    title: str
    message: str
    details: Optional[str] = None
    action_suggestions: Optional[List[str]] = None
    severity: str = "info"  # info, warning, error, success
    is_recoverable: bool = True

    def __post_init__(self):
        """Initialize default action suggestions if none provided."""
        if self.action_suggestions is None:
            self.action_suggestions = []