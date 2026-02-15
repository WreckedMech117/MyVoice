"""
MyVoice V2 - Expressive Voice Communication for Everyone

A PyQt6-based desktop application for generating emotionally expressive text-to-speech
using embedded Qwen3-TTS with voice cloning, Voice Design, and dual audio routing.

Version: 2.0.0
Author: MyVoice Development Team
License: MIT
"""

__version__ = "2.0.0"
__author__ = "MyVoice Development Team"
__email__ = "support@myvoice.local"
__license__ = "MIT"

# Package metadata
__title__ = "MyVoice"
__description__ = "Expressive Voice Communication with Qwen3-TTS, Emotion Control, and Voice Design"
__url__ = "https://github.com/myvoice/myvoice"

# Import main application components
from myvoice.app import MyVoiceApp

__all__ = [
    "MyVoiceApp",
    "__version__",
    "__author__",
    "__title__",
    "__description__"
]