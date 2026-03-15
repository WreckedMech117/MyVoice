"""
MyVoice Services Package

This package contains all business logic services for the MyVoice application.
"""

# Import services without causing circular imports
__all__ = [
    'ConfigurationManager',
    'VoiceProfileManager',
    'WhisperService',
    'QwenTTSService',
    'ModelRegistry',
    # Audio services
    'AudioCoordinator',
    'MonitorAudioService',
    'VirtualMicrophoneService',
    'MicrophoneCaptureService',
    'AudioMixerService',
    # Microphone mixing types
    'MicrophoneMixingMode',
    'MicrophoneMixingConfig',
    # Device resilience (Story 2.5)
    'DeviceResilienceManager',
    'DeviceResilienceConfig',
    'DeviceRole',
    # Streaming types
    'AudioChunk',
    'QwenTTSRequest',
    'QwenTTSResponse',
    'GenerationMode',
    'GenerationState',
    # Error types
    'TTSError',
    'TTSErrorCode',
    # Validation types
    'TextValidationResult',
    'TextValidationStatus',
    # Startup types (Story 1.5)
    'StartupState',
    'StartupProgress',
    'BundledVoiceConfig',
]

def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == 'ConfigurationManager':
        from .configuration_service import ConfigurationManager
        return ConfigurationManager
    elif name == 'VoiceProfileManager':
        from .voice_profile_service import VoiceProfileManager
        return VoiceProfileManager
    elif name == 'WhisperService':
        from .whisper_service import WhisperService
        return WhisperService
    elif name == 'QwenTTSService':
        from .qwen_tts_service import QwenTTSService
        return QwenTTSService
    elif name == 'ModelRegistry':
        from .model_registry import ModelRegistry
        return ModelRegistry
    elif name == 'AudioCoordinator':
        from .audio_coordinator import AudioCoordinator
        return AudioCoordinator
    elif name == 'MonitorAudioService':
        from .monitor_audio_service import MonitorAudioService
        return MonitorAudioService
    elif name == 'VirtualMicrophoneService':
        from .virtual_microphone_service import VirtualMicrophoneService
        return VirtualMicrophoneService
    elif name == 'MicrophoneCaptureService':
        from .microphone_capture_service import MicrophoneCaptureService
        return MicrophoneCaptureService
    elif name == 'AudioMixerService':
        from .audio_mixer_service import AudioMixerService
        return AudioMixerService
    elif name in ('MicrophoneMixingMode', 'MicrophoneMixingConfig'):
        from .audio_coordinator import MicrophoneMixingMode, MicrophoneMixingConfig
        return locals()[name]
    elif name in ('DeviceResilienceManager', 'DeviceResilienceConfig', 'DeviceRole'):
        from .device_resilience_manager import (
            DeviceResilienceManager, DeviceResilienceConfig, DeviceRole
        )
        return locals()[name]
    elif name in ('AudioChunk', 'QwenTTSRequest', 'QwenTTSResponse',
                  'GenerationMode', 'GenerationState', 'TTSError', 'TTSErrorCode',
                  'TextValidationResult', 'TextValidationStatus',
                  'StartupState', 'StartupProgress', 'BundledVoiceConfig'):
        from .qwen_tts_service import (
            AudioChunk, QwenTTSRequest, QwenTTSResponse,
            GenerationMode, GenerationState, TTSError, TTSErrorCode,
            TextValidationResult, TextValidationStatus,
            StartupState, StartupProgress, BundledVoiceConfig
        )
        return locals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")