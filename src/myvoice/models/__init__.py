"""
MyVoice Data Models Package

This package contains all data models and business entities for the MyVoice application.
"""

from .ui_state import (
    UIState, ServiceStatusInfo, ServiceHealthStatus, NotificationMessage
)
from .error import MyVoiceError, ErrorSeverity
from .validation import ValidationResult, ValidationIssue, ValidationStatus, UserFriendlyMessage
from .retry_config import RetryConfig, RetryAttempt, RetryableErrorType, RetryConfigs
from .app_settings import AppSettings
from .voice_profile import VoiceProfile, TranscriptionStatus
from .audio_device import AudioDevice, DeviceType
from .audio_playback_task import AudioPlaybackTask, PlaybackStatus
from .transcription_result import (
    TranscriptionResult, TranscriptionSegment, WordTimestamp,
    LanguageConfidence
)
from .transcription_result import TranscriptionStatus as TranscriptionResultStatus
from .service_enums import ServiceStatus, ModelState, QwenModelType

__all__ = [
    'UIState', 'ServiceStatusInfo', 'ServiceHealthStatus', 'NotificationMessage',
    'MyVoiceError', 'ErrorSeverity',
    'ValidationResult', 'ValidationIssue', 'ValidationStatus', 'UserFriendlyMessage',
    'RetryConfig', 'RetryAttempt', 'RetryableErrorType', 'RetryConfigs',
    'AppSettings', 'VoiceProfile', 'TranscriptionStatus', 'AudioDevice', 'DeviceType',
    'AudioPlaybackTask', 'PlaybackStatus',
    'TranscriptionResult', 'TranscriptionSegment', 'WordTimestamp',
    'TranscriptionResultStatus', 'LanguageConfidence',
    # V2 Qwen3-TTS enums
    'ServiceStatus', 'ModelState', 'QwenModelType',
]