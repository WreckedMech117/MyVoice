"""
MyVoice Service Integrations Package

This package contains external service integration clients.
"""

from .windows_audio_client import WindowsAudioClient, DeviceChangeEvent

__all__ = [
    'WindowsAudioClient',
    'DeviceChangeEvent'
]