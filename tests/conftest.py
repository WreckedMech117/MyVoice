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


# ---------------------------------------------------------------------------
# Story tooling-4 (Site A) -- a quick-speak service stub that is safe to hand
# to SettingsDialog.
#
# A bare MagicMock() returns another MagicMock for every attribute.
# QuickSpeakSettingsWidget._load_entries (run from SettingsDialog.__init__ via
# _create_quick_speak_tab) calls get_profiles() / get_current_profile() /
# get_entries() and feeds the results straight into Qt setters;
# QComboBox.setCurrentText(MagicMock) raises TypeError, _load_entries catches
# it and pops a *modal* QMessageBox.warning from inside dialog construction,
# and the test hangs waiting for a click (24 of tooling-3's 27 timeouts).
# This fixture returns values of the right type for every service call the
# widget makes during construction, and for the reset path
# (SettingsDialog._reset_quick_speak_entries -> _create_default_profile() /
# load_entries() -> widget.refresh_entries() -> _load_entries()). spec= means
# a mistyped attribute raises AttributeError instead of silently returning a
# MagicMock. Every test that constructs SettingsDialog with a stub service
# must use this fixture rather than hand-rolling a MagicMock.
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock


@pytest.fixture
def quick_speak_service_stub():
    from myvoice.services.quick_speak_service import QuickSpeakService

    stub = MagicMock(spec=QuickSpeakService)
    # Construction-time reads (QuickSpeakSettingsWidget._load_entries).
    stub.get_profiles.return_value = ["default"]      # List[str] -> addItems
    stub.get_current_profile.return_value = "default"  # str      -> setCurrentText
    stub.get_entries.return_value = []                 # List[QuickSpeakEntry] -> _populate_table
    # Reset path (SettingsDialog._reset_quick_speak_entries).
    stub.load_entries.return_value = []
    stub._create_default_profile.return_value = None
    # Profile mutators the widget's slots read as bool.
    stub.switch_profile.return_value = True
    stub.create_profile.return_value = True
    stub.delete_profile.return_value = True
    return stub


# ---------------------------------------------------------------------------
# Story tooling-4 (AC #3) -- an unexpected modal under test must FAIL, never
# hang.
#
# Both tooling-3 stall sites were the same shape: production code reached a
# blocking QMessageBox on an error path that a test stub had put it on, and
# pytest sat waiting for a click until pytest-timeout fired os._exit(1) 60 s
# later. Nothing stops the next test file from hand-rolling a bad stub, so
# this net turns every *unpatched* blocking QMessageBox entry point into an
# immediate, named failure:
#
#   * the four static modals -- warning / critical / information / question
#   * QMessageBox.exec -- the instance route (settings_dialog.py:1561 and
#     app.py:794 build a QMessageBox and call .exec() on it)
#
# The replacements are installed ONCE per session, on the class, so they are
# active during module/class-scoped fixture setup as well as inside tests. A
# test that legitimately exercises a modal keeps doing what it does today --
# monkeypatch.setattr(QMessageBox, "warning", ...) or
# patch.object(QMessageBox, "question", ...) -- and its patch wins for the
# duration of the test (both undo back to the raiser, not to the original).
# A test that must reach the real Qt implementation (e.g. it drives the box
# with a QTimer) opts out with @pytest.mark.qmessagebox_passthrough.
#
# The raiser derives from BaseException, not Exception, on purpose: most of
# these modals sit inside `try: ... except Exception:` blocks (app.py:793 wraps
# the .exec() itself in one), and an `except Exception` would swallow the
# failure and let the test pass silently. Only a bare `except:` can hide it.
# pytest reports a BaseException subclass as an ordinary failure, and PyQt6
# routes an exception raised inside a C++-invoked slot to sys.excepthook,
# which pytest-qt captures and reports as a failure for every test (except
# the two marked qt_no_exception_capture in test_streaming_tts_smoke.py,
# where it would abort the process -- still not a hang).
#
# This is a net, not a replacement for the MainWindow close-confirm bypass
# above: that one flips the production _force_quit flag so closeEvent never
# reaches its QMessageBox.question; this one only catches what does reach a
# QMessageBox.
# ---------------------------------------------------------------------------
class UnexpectedModalDialogError(BaseException):
    """A blocking QMessageBox was reached under test without being patched."""


_QMESSAGEBOX_STATIC_MODALS = ("warning", "critical", "information", "question")
_QMESSAGEBOX_PASSTHROUGH_MARK = "qmessagebox_passthrough"


def _describe_unexpected_modal(entry_point, title, text):
    return (
        f"QMessageBox.{entry_point} reached under test: title={title!r}, "
        f"text={text!r}. A blocking modal would hang pytest. If the test "
        f"means to exercise this dialog, patch QMessageBox.{entry_point} in "
        f"the test; otherwise fix the stub that put the code under test on "
        f"this error path (see tests/conftest.py, Story tooling-4)."
    )


def _make_static_modal_raiser(name):
    def _raise(parent=None, title="", text="", *args, **kwargs):
        raise UnexpectedModalDialogError(
            _describe_unexpected_modal(name, title, text)
        )

    _raise.__name__ = f"unexpected_modal_{name}"
    return staticmethod(_raise)


def _unexpected_modal_exec(self, *args, **kwargs):
    raise UnexpectedModalDialogError(
        _describe_unexpected_modal("exec", self.windowTitle(), self.text())
    )


@pytest.fixture(scope="session", autouse=True)
def _fail_on_unexpected_qmessagebox():
    """Install the raisers on QMessageBox for the whole session."""
    try:
        from PyQt6.QtWidgets import QMessageBox
    except (ImportError, OSError):
        yield None
        return

    originals = {
        name: getattr(QMessageBox, name) for name in _QMESSAGEBOX_STATIC_MODALS
    }
    mp = pytest.MonkeyPatch()
    for name in _QMESSAGEBOX_STATIC_MODALS:
        mp.setattr(QMessageBox, name, _make_static_modal_raiser(name))
    # exec is inherited from QDialog; setting it on QMessageBox shadows it
    # for message boxes only. MonkeyPatch.undo() delattr()s it again.
    mp.setattr(QMessageBox, "exec", _unexpected_modal_exec, raising=False)
    try:
        yield originals
    finally:
        mp.undo()


@pytest.fixture(autouse=True)
def _qmessagebox_passthrough(request, monkeypatch, _fail_on_unexpected_qmessagebox):
    """Per-test opt-out: @pytest.mark.qmessagebox_passthrough restores the
    real Qt implementations for that one test."""
    originals = _fail_on_unexpected_qmessagebox
    if originals is None or request.node.get_closest_marker(
        _QMESSAGEBOX_PASSTHROUGH_MARK
    ) is None:
        return
    from PyQt6.QtWidgets import QMessageBox

    for name, original in originals.items():
        monkeypatch.setattr(QMessageBox, name, original)
    monkeypatch.delattr(QMessageBox, "exec")  # back to the inherited QDialog.exec
