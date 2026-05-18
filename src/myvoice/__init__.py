"""
MyVoice V2 - Expressive Voice Communication for Everyone

A PyQt6-based desktop application for generating emotionally expressive text-to-speech
using embedded Qwen3-TTS with voice cloning, Voice Design, and dual audio routing.

Version: 2.2.0
Author: MyVoice Development Team
License: MIT
"""

__version__ = "2.2.0"
__author__ = "MyVoice Development Team"
__email__ = "support@myvoice.local"
__license__ = "MIT"

# Package metadata
__title__ = "MyVoice"
__description__ = "Expressive Voice Communication with Qwen3-TTS, Emotion Control, and Voice Design"
__url__ = "https://github.com/myvoice/myvoice"

# Lazy re-export of MyVoiceApp via PEP 562 `__getattr__`. Eager import
# would trigger `myvoice.app` → PyQt6 + torch on every `import myvoice.*`,
# including lightweight entrypoints like the install-time model pre-
# download (see `myvoice.utils.predownload_models`) which must NOT
# depend on torch. `from myvoice import MyVoiceApp` still works — the
# import path is materialized on first attribute access instead of at
# package import time.
def __getattr__(name):
    if name == "MyVoiceApp":
        from myvoice.app import MyVoiceApp as _MyVoiceApp
        return _MyVoiceApp
    raise AttributeError(f"module 'myvoice' has no attribute {name!r}")


__all__ = [
    "MyVoiceApp",
    "__version__",
    "__author__",
    "__title__",
    "__description__"
]