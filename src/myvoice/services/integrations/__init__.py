"""
MyVoice Service Integrations Package

This package contains external service integration clients.
"""

from .gpt_sovits_client import GPTSoVITSClient
from .windows_audio_client import WindowsAudioClient, DeviceChangeEvent

__all__ = [
    'GPTSoVITSClient',
    'WindowsAudioClient',
    'DeviceChangeEvent'
]