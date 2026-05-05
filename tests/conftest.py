"""
Pytest configuration for MyVoice tests.

Sets up the Python path to include the src directory and mirrors the
torch-before-PyQt6 DLL-ordering preamble from ``src/myvoice/main.py`` so
the test suite can import ``myvoice.services.qwen_tts_service`` (which
transitively imports torch) on any Windows machine that has CUDA installed
— regardless of GPU architecture (RTX 30xx Ampere, 40xx Ada Lovelace,
or 50xx Blackwell). The ordering is identical to the production launcher.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# CRITICAL: Register DLL directories BEFORE importing torch (Python 3.8+
# Windows requirement). Mirrors src/myvoice/main.py:25-40. Required for the
# bundled portable Python environment where DLL paths aren't in system PATH.
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _repo_root = Path(__file__).parent.parent
    _torch_lib = _repo_root / "python310" / "Lib" / "site-packages" / "torch" / "lib"
    if _torch_lib.exists():
        os.add_dll_directory(str(_torch_lib))

    # CUDA toolkit bin directories — pinned to v12.8 to match the bundled
    # torch wheel (cu128). Newer CUDA toolkits coexist; we only need the
    # one torch was built against to satisfy DLL dependencies.
    for _cuda_path in (
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp"),
    ):
        if _cuda_path.exists():
            os.add_dll_directory(str(_cuda_path))

# CRITICAL: Import torch BEFORE PyQt6 to avoid the DLL loading conflict that
# breaks ``c10.dll`` initialization on Windows (PyQt6 pre-loads CRT DLLs
# whose initialization order conflicts with torch's). Same workaround as
# the production launcher. Failures are swallowed — TTS-dependent tests
# already guard their own imports via ``pytest.importorskip("torch")`` or
# ``try/except``, so a torch-less environment still runs the rest of the
# suite.
try:
    import torch  # noqa: F401
except (ImportError, OSError):
    pass
