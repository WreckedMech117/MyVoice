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


# MainWindow's closeEvent (src/myvoice/ui/main_window.py:2301-2340) shows a
# confirm-close QMessageBox.question gated by self._force_quit. Without
# bypass, every test that lets a MainWindow tear down blocks pytest waiting
# for a manual click. The production flag (_force_quit=True) is the intended
# skip mechanism; this autouse fixture wraps closeEvent so every test
# instance flips it before the original runs. Tests that already set
# _force_quit=True explicitly remain correct (the flag is idempotent).
import pytest


@pytest.fixture(autouse=True)
def _suppress_main_window_close_confirm(monkeypatch):
    try:
        from myvoice.ui.main_window import MainWindow
    except (ImportError, OSError):
        return

    original_close_event = MainWindow.closeEvent

    def _patched_close_event(self, event):
        # Only auto-flip _force_quit when the QMessageBox would actually
        # fire. Tests that exercise the minimize-to-tray branch require
        # _force_quit=False AND tray_icon AND app_settings.minimize_to_tray
        # — under those exact conditions, closeEvent skips the dialog
        # before reaching it, so the fixture must not touch the flag or
        # the minimize-to-tray test's assertions break.
        would_minimize_to_tray = (
            self.tray_icon is not None
            and self.app_settings is not None
            and getattr(self.app_settings, "minimize_to_tray", False)
        )
        if not would_minimize_to_tray:
            self._force_quit = True
        return original_close_event(self, event)

    monkeypatch.setattr(MainWindow, "closeEvent", _patched_close_event)
