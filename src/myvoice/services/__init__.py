"""
MyVoice Services Package

This package contains all business logic services for the MyVoice application.
"""

# Import services without causing circular imports
__all__ = [
    'ConfigurationManager', 'VoiceProfileManager', 'TTSService', 'WhisperService'
]

def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == 'ConfigurationManager':
        from .configuration_service import ConfigurationManager
        return ConfigurationManager
    elif name == 'VoiceProfileManager':
        from .voice_profile_service import VoiceProfileManager
        return VoiceProfileManager
    elif name == 'TTSService':
        from .tts_service import TTSService
        return TTSService
    elif name == 'WhisperService':
        from .whisper_service import WhisperService
        return WhisperService
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")