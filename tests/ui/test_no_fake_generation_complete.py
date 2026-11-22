"""Generate must not fake its own completion.

A pre-integration demo stub in ``MainWindow._start_tts_generation`` fired
2 s after every Generate and called ``set_generation_status("Speech
generated successfully!", is_generating=False)`` -- in production, while a
longer generation was still running. With no audio playing yet, the Clear
button then fell back from Stop (cancel) to clear-text, and the status bar
claimed success early.

Only the app controller may settle a generation.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qtbot):
    from myvoice.ui.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    return win


def test_generation_still_in_flight_after_two_seconds(qtbot, window):
    cancels = []
    window.cancel_generation_requested.connect(lambda: cancels.append(1))

    window._start_tts_generation("a long utterance that takes a while")
    qtbot.wait(2600)  # past the old stub's 2 s

    assert window._is_generating is True
    assert "successfully" not in window.status_bar.currentMessage()
    # Clear is still Stop: it cancels instead of clearing text
    window._on_clear_clicked()
    assert cancels == [1]


def test_settled_status_reverts_to_ready_message(qtbot, window, monkeypatch):
    monkeypatch.setattr(type(window), "SETTLED_STATUS_REVERT_MS", 50)
    window.set_generation_status("Speech generated successfully", False)
    qtbot.waitUntil(
        lambda: window.status_bar.currentMessage() == window._get_ready_message(),
        timeout=2000,
    )


def test_revert_does_not_clobber_a_newer_message(qtbot, window, monkeypatch):
    monkeypatch.setattr(type(window), "SETTLED_STATUS_REVERT_MS", 50)
    window.set_generation_status("Speech generated successfully", False)
    window.status_bar.showMessage("Replaying last audio...")
    qtbot.wait(200)
    assert window.status_bar.currentMessage() == "Replaying last audio..."


def test_revert_skipped_when_a_new_generation_started(qtbot, window, monkeypatch):
    monkeypatch.setattr(type(window), "SETTLED_STATUS_REVERT_MS", 50)
    window.set_generation_status("Generation failed: x", False)
    window.set_generation_status("Generating speech...", True)
    qtbot.wait(200)
    assert window.status_bar.currentMessage() == "Generating speech..."
