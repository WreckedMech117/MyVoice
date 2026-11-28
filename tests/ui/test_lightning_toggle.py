"""Epic 19 — the Lightning mode toggle beside the text entry."""
import pytest

from myvoice.ui.main_window import MainWindow


@pytest.fixture
def main_window(qapp):
    window = MainWindow()
    yield window
    window._force_quit = True
    window.deleteLater()


@pytest.fixture
def toggles(main_window):
    seen = []
    main_window.lightning_mode_toggled.connect(seen.append)
    return seen


def test_toggle_shows_a_lightning_bolt_and_starts_disabled(main_window):
    box = main_window.lightning_checkbox
    assert box.text() == "⚡"
    assert not box.isEnabled() and not box.isChecked()


def test_available_engine_enables_the_toggle(main_window):
    main_window.set_lightning_available(True)
    assert main_window.lightning_checkbox.isEnabled()
    assert "Lightning mode" in main_window.lightning_checkbox.toolTip()


def test_unavailable_engine_explains_why_in_the_tooltip(main_window):
    main_window.set_lightning_available(True)
    main_window.set_lightning_checked(True)
    main_window.set_lightning_available(False, "Lightning mode needs an NVIDIA GPU.")
    box = main_window.lightning_checkbox
    assert not box.isEnabled() and not box.isChecked()
    assert box.toolTip() == "Lightning mode needs an NVIDIA GPU."


def test_user_click_emits_the_new_state(main_window, toggles):
    main_window.set_lightning_available(True)
    main_window.lightning_checkbox.click()
    main_window.lightning_checkbox.click()
    assert toggles == [True, False]


def test_reflecting_engine_state_does_not_emit(main_window, toggles):
    """The app sets the box to match the engine (startup, a failed switch);
    that must not come back as a second switch request."""
    main_window.set_lightning_available(True)
    main_window.set_lightning_checked(True)
    main_window.set_lightning_checked(False)
    assert toggles == []


def test_busy_locks_the_toggle_until_the_swap_finishes(main_window):
    main_window.set_lightning_available(True)
    main_window.set_lightning_busy(True)
    assert not main_window.lightning_checkbox.isEnabled()
    main_window.set_lightning_busy(False)
    assert main_window.lightning_checkbox.isEnabled()


def test_busy_does_not_enable_an_unavailable_toggle(main_window):
    main_window.set_lightning_available(False, "not installed")
    main_window.set_lightning_busy(True)
    main_window.set_lightning_busy(False)
    assert not main_window.lightning_checkbox.isEnabled()
