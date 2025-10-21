"""
MyVoice - Desktop Text-to-Speech Application

A PyQt6-based desktop application for generating high-quality text-to-speech
using GPT-SoVITS integration with dual audio routing capabilities.

Version: 1.0.0
Author: MyVoice Development Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "MyVoice Development Team"
__email__ = "support@myvoice.local"
__license__ = "MIT"

# Package metadata
__title__ = "MyVoice"
__description__ = "Desktop Text-to-Speech Application with GPT-SoVITS Integration"
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