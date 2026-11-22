"""ui-5: after Generate the text stays in the box greyed out ("ghost text").

Requested behaviour (2026-11-22):
  * Enter / Generate -> the text greys out instead of disappearing and the
    cursor goes to the beginning;
  * Tab -> the text becomes live ("hot") again;
  * typing something else -> the grey text disappears and the new input is
    taken.

Real key presses go through MainWindow's event filter (pytest-qt).
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtGui import QPalette, QTextCursor  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qtbot):
    from myvoice.ui.main_window import MainWindow

    win = MainWindow()
    # _start_tts_generation schedules a 2 s demo-stub timer that then
    # schedules a 3 s status-bar update; both would fire after this window
    # is deleted and fail whichever test runs next. No-op it (resolved at
    # fire time via the instance attribute).
    win._complete_tts_generation_stub = lambda text: None
    qtbot.addWidget(win)
    win.show()
    win.text_input.setFocus()
    generated = []
    win.text_generate_requested.connect(generated.append)
    win._generated = generated
    return win


def _char_colors(win):
    """Foreground colour of every character in the box."""
    doc = win.text_input.document()
    cur = QTextCursor(doc)
    colors = []
    for i in range(len(win.text_input.toPlainText())):
        cur.setPosition(i)
        cur.setPosition(i + 1, QTextCursor.MoveMode.KeepAnchor)
        colors.append(cur.charFormat().foreground().color().name())
    return colors


def _grey(win):
    return win._ghost_text_color().name()


def test_ghost_colour_differs_from_live_text(window):
    live = window.text_input.palette().color(QPalette.ColorRole.Text).name()
    assert _grey(window) != live


def _type_and_enter(qtbot, win, text):
    qtbot.keyClicks(win.text_input, text)
    qtbot.keyClick(win.text_input, Qt.Key.Key_Return)


def test_enter_greys_text_and_puts_cursor_at_start(qtbot, window):
    _type_and_enter(qtbot, window, "hello there")

    assert window._generated == ["hello there"]
    assert window.text_input.toPlainText() == "hello there"
    assert window._ghost_text_active
    assert set(_char_colors(window)) == {_grey(window)}
    assert window.text_input.textCursor().position() == 0


def test_generate_button_also_greys(qtbot, window):
    qtbot.keyClicks(window.text_input, "button path")
    window.generate_button.setEnabled(True)
    qtbot.mouseClick(window.generate_button, Qt.MouseButton.LeftButton)

    assert window._generated == ["button path"]
    assert window._ghost_text_active
    assert window.text_input.toPlainText() == "button path"


def test_tab_makes_text_hot_again(qtbot, window):
    _type_and_enter(qtbot, window, "say it again")
    qtbot.keyClick(window.text_input, Qt.Key.Key_Tab)

    assert not window._ghost_text_active
    assert window.text_input.toPlainText() == "say it again"  # no tab char
    assert _grey(window) not in set(_char_colors(window))
    # and it is live input: Enter sends it again
    qtbot.keyClick(window.text_input, Qt.Key.Key_Return)
    assert window._generated == ["say it again", "say it again"]


def test_typing_replaces_ghost_text(qtbot, window):
    _type_and_enter(qtbot, window, "old line")
    qtbot.keyClicks(window.text_input, "new")

    assert not window._ghost_text_active
    assert window.text_input.toPlainText() == "new"
    assert _grey(window) not in set(_char_colors(window))


def test_enter_on_ghost_text_does_not_resend(qtbot, window):
    _type_and_enter(qtbot, window, "once only")
    qtbot.keyClick(window.text_input, Qt.Key.Key_Return)

    assert window._generated == ["once only"]
    assert window._ghost_text_active


def test_backspace_clears_ghost_text(qtbot, window):
    _type_and_enter(qtbot, window, "gone")
    qtbot.keyClick(window.text_input, Qt.Key.Key_Backspace)

    assert window.text_input.toPlainText() == ""
    assert not window._ghost_text_active


def test_navigation_keys_keep_ghost(qtbot, window):
    _type_and_enter(qtbot, window, "stay grey")
    qtbot.keyClick(window.text_input, Qt.Key.Key_Right)
    qtbot.keyClick(window.text_input, Qt.Key.Key_Shift)

    assert window._ghost_text_active
    assert window.text_input.toPlainText() == "stay grey"


def test_external_set_text_ends_ghost_live(qtbot, window):
    """Stop's restore / Quick Speak / dictation write the box directly."""
    _type_and_enter(qtbot, window, "first")
    window.text_input.setPlainText("restored")

    assert not window._ghost_text_active
    assert _grey(window) not in set(_char_colors(window))


def test_escape_clears_ghost(qtbot, window):
    _type_and_enter(qtbot, window, "esc me")
    qtbot.keyClick(window, Qt.Key.Key_Escape)

    assert window.text_input.toPlainText() == ""
    assert not window._ghost_text_active


def test_ghost_text_not_persisted_as_draft(qtbot, window):
    _type_and_enter(qtbot, window, "said already")
    assert window.get_ui_state().last_text_input == ""


def test_quick_speak_ignores_ghost_for_append_mode(qtbot, window):
    """A ghost ending in an apostrophe is the previous utterance, not a
    prefix the user is building: Quick Speak replaces, never appends."""
    _type_and_enter(qtbot, window, "they'")
    assert window._ghost_text_active

    window._on_quick_speak_entry_selected("hello")

    assert window.text_input.toPlainText() == "hello"
    assert window._generated[-1] == "hello"


def test_clear_button_removes_ghost(qtbot, window):
    _type_and_enter(qtbot, window, "don't keep")
    window._is_generating = False
    window._is_playing = False
    window._on_clear_clicked()

    assert window.text_input.toPlainText() == ""
    assert not window._ghost_text_active
