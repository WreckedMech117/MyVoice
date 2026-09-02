"""Tests for Story ui-1: Close-to-Tray Toggle (and the minimize button).

Covers:

* AC #1 — ``minimize_to_tray`` is surfaced on the Settings dialog's
  **Interface** tab, hydrates from the persisted value, and saves through the
  dialog's existing path.
* AC #2 — ``MainWindow.closeEvent`` honours the toggle in both directions.
* AC #3 — the title-bar minimize button honours the toggle too. The original
  guard was ``hasattr(parent, '_minimize_to_tray')``, which is *always* true on
  ``MainWindow``, so the ``showMinimized()`` branch was unreachable and the
  minimize button always hid to the tray. The taskbar tests below fail against
  that old guard — that is the exact bug class they exist to pin.
* AC #4 — ``MYVOICE_AUTO_QUIT_ON_CLOSE=1`` still forces a real quit (no
  tray-minimize, no confirm dialog) regardless of the toggle.

Per ``memory/main_window_close_confirm_dialog_in_tests.md`` the confirm dialog
itself is never weakened; tests that must get past it patch
``QMessageBox.question`` or use the production ``_force_quit`` path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QMessageBox,
    QSystemTrayIcon,
    QTabWidget,
)

from PyQt6.QtGui import QCloseEvent  # noqa: E402

from myvoice.models.app_settings import AppSettings  # noqa: E402
from myvoice.ui.components.custom_title_bar import CustomTitleBar  # noqa: E402
from myvoice.ui.components.settings_dialog import SettingsDialog  # noqa: E402
from myvoice.ui.main_window import MainWindow  # noqa: E402

# Captured at import time, before ``tests/conftest.py``'s autouse fixture wraps
# ``MainWindow.closeEvent`` for the duration of each test. See
# ``_invoke_close_event`` below.
_ORIGINAL_CLOSE_EVENT = MainWindow.closeEvent


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def skip_if_no_tray():
    """Skip when the host has no usable system tray."""
    try:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            pytest.skip("System tray not available on this platform")
    except Exception:  # pragma: no cover - platform dependent
        pytest.skip("System tray check failed")


# ---------------------------------------------------------------------------
# AC #3 — title-bar minimize button
# ---------------------------------------------------------------------------


class _FakeSettings:
    """Minimal stand-in for AppSettings with just the field under test."""

    def __init__(self, minimize_to_tray: bool):
        self.minimize_to_tray = minimize_to_tray


class _FakeWindowWithTray:
    """Parent window that supports tray-minimize (like ``MainWindow``)."""

    def __init__(self, app_settings):
        self.app_settings = app_settings
        self.tray_calls = 0
        self.taskbar_calls = 0

    def _minimize_to_tray(self):
        self.tray_calls += 1

    def showMinimized(self):
        self.taskbar_calls += 1


class _FakeWindowNoTraySupport:
    """Parent window without the tray helper at all."""

    def __init__(self):
        self.app_settings = _FakeSettings(minimize_to_tray=True)
        self.taskbar_calls = 0

    def showMinimized(self):
        self.taskbar_calls += 1


@pytest.fixture
def title_bar(qapp, qtbot):
    bar = CustomTitleBar(parent=None)
    qtbot.addWidget(bar)
    return bar


class TestTitleBarMinimizeHonoursSetting:
    """The minimize button reads the setting, not ``hasattr``."""

    def test_minimizes_to_tray_when_setting_enabled(self, title_bar):
        parent = _FakeWindowWithTray(_FakeSettings(minimize_to_tray=True))
        title_bar._parent_window = parent

        title_bar._on_minimize_clicked()

        assert parent.tray_calls == 1
        assert parent.taskbar_calls == 0

    def test_minimizes_to_taskbar_when_setting_disabled(self, title_bar):
        """Regression for the always-true ``hasattr`` guard (AC #3).

        Under the old guard this branch never executed, so ``tray_calls``
        would be 1 and ``taskbar_calls`` 0.
        """
        parent = _FakeWindowWithTray(_FakeSettings(minimize_to_tray=False))
        title_bar._parent_window = parent

        title_bar._on_minimize_clicked()

        assert parent.taskbar_calls == 1
        assert parent.tray_calls == 0

    def test_falls_back_to_taskbar_when_settings_missing(self, title_bar):
        """No settings wired up -> the safer taskbar branch (Task 2.2)."""
        parent = _FakeWindowWithTray(app_settings=None)
        title_bar._parent_window = parent

        title_bar._on_minimize_clicked()

        assert parent.taskbar_calls == 1
        assert parent.tray_calls == 0

    def test_falls_back_to_taskbar_when_parent_has_no_tray_helper(self, title_bar):
        parent = _FakeWindowNoTraySupport()
        title_bar._parent_window = parent

        title_bar._on_minimize_clicked()

        assert parent.taskbar_calls == 1

    def test_no_parent_window_is_a_no_op(self, title_bar):
        title_bar._parent_window = None

        title_bar._on_minimize_clicked()  # must not raise

    def test_helper_reports_both_branches(self, title_bar):
        """``_should_minimize_to_tray`` is a real read of the setting."""
        title_bar._parent_window = _FakeWindowWithTray(
            _FakeSettings(minimize_to_tray=True)
        )
        assert title_bar._should_minimize_to_tray() is True

        title_bar._parent_window = _FakeWindowWithTray(
            _FakeSettings(minimize_to_tray=False)
        )
        assert title_bar._should_minimize_to_tray() is False


# ---------------------------------------------------------------------------
# AC #1 — the Interface tab control
# ---------------------------------------------------------------------------


@pytest.fixture
def settings_factory(tmp_path):
    def _make(minimize_to_tray: bool = False) -> AppSettings:
        config_dir = tmp_path / "config"
        config_dir.mkdir(exist_ok=True)
        voice_dir = tmp_path / "voices"
        voice_dir.mkdir(exist_ok=True)
        return AppSettings(
            config_directory=str(config_dir),
            voice_files_directory=str(voice_dir),
            minimize_to_tray=minimize_to_tray,
        )

    return _make


def _make_dialog(settings, qtbot) -> SettingsDialog:
    quick_speak_stub = MagicMock()
    quick_speak_stub.load_entries = MagicMock()
    dlg = SettingsDialog(settings, parent=None, quick_speak_service=quick_speak_stub)
    qtbot.addWidget(dlg)
    return dlg


class TestInterfaceTabCloseBehaviorControl:
    """AC #1 — visible, labelled, hydrated, and saved."""

    def test_control_exists_and_is_a_two_option_combo(
        self, qapp, qtbot, settings_factory
    ):
        dialog = _make_dialog(settings_factory(), qtbot)

        assert hasattr(dialog, "close_behavior_combo"), (
            "No close-behavior control on the Settings dialog — "
            "minimize_to_tray is unreachable from the UI (Story ui-1 AC #1)"
        )
        combo: QComboBox = dialog.close_behavior_combo
        assert combo.count() == 2
        assert {combo.itemData(0), combo.itemData(1)} == {True, False}

    def test_control_lives_on_the_interface_tab(self, qapp, qtbot, settings_factory):
        dialog = _make_dialog(settings_factory(), qtbot)

        tab_widget: QTabWidget = dialog.tab_widget
        interface_index = -1
        for i in range(tab_widget.count()):
            if tab_widget.tabText(i) == "Interface":
                interface_index = i
                break
        assert interface_index >= 0, "Interface tab missing"

        interface_tab = tab_widget.widget(interface_index)
        combos = interface_tab.findChildren(QComboBox)
        assert dialog.close_behavior_combo in combos

    def test_option_labels_are_user_facing(self, qapp, qtbot, settings_factory):
        """Worded from the user's point of view, not after the field name."""
        dialog = _make_dialog(settings_factory(), qtbot)
        combo = dialog.close_behavior_combo
        labels = [combo.itemText(i).lower() for i in range(combo.count())]

        assert any("tray" in label for label in labels)
        assert any("quit" in label for label in labels)
        assert not any("minimize_to_tray" in label for label in labels)

    @pytest.mark.parametrize("persisted", [True, False])
    def test_reflects_persisted_value_on_open(
        self, qapp, qtbot, settings_factory, persisted
    ):
        dialog = _make_dialog(settings_factory(minimize_to_tray=persisted), qtbot)

        assert dialog.close_behavior_combo.currentData() is persisted

    @pytest.mark.parametrize("chosen", [True, False])
    def test_selection_saves_through_existing_path(
        self, qapp, qtbot, settings_factory, chosen
    ):
        dialog = _make_dialog(settings_factory(minimize_to_tray=not chosen), qtbot)

        index = dialog.close_behavior_combo.findData(chosen)
        assert index >= 0
        dialog.close_behavior_combo.setCurrentIndex(index)

        dialog._save_current_settings()

        assert dialog.current_settings.minimize_to_tray is chosen

    def test_saved_value_round_trips_through_persistence(
        self, qapp, qtbot, settings_factory
    ):
        """No new persistence mechanism — the existing dict round-trip carries it."""
        dialog = _make_dialog(settings_factory(minimize_to_tray=True), qtbot)

        dialog.close_behavior_combo.setCurrentIndex(
            dialog.close_behavior_combo.findData(False)
        )
        dialog._save_current_settings()

        restored = AppSettings.from_dict(dialog.current_settings.to_dict())
        assert restored.minimize_to_tray is False

    def test_default_setting_value_is_unchanged(self):
        """Dev Notes: do not change the default in this story."""
        assert AppSettings().minimize_to_tray is False


# ---------------------------------------------------------------------------
# AC #2 / AC #4 — closeEvent
# ---------------------------------------------------------------------------


@pytest.fixture
def main_window(qapp):
    window = MainWindow()
    yield window
    window._force_quit = True
    window.deleteLater()


def _invoke_close_event(window):
    """Run the *real* ``closeEvent`` and hand back the event.

    ``tests/conftest.py`` installs an autouse fixture that wraps
    ``MainWindow.closeEvent`` and flips ``_force_quit`` whenever the confirm
    dialog would otherwise fire — that is the sanctioned pytest bypass
    (``memory/main_window_close_confirm_dialog_in_tests.md``). These tests need
    the unwrapped implementation so they can observe the minimize-vs-confirm
    decision itself, so they call the reference captured at import time,
    before any fixture patches the class.
    """
    event = QCloseEvent()
    _ORIGINAL_CLOSE_EVENT(window, event)
    return event


class TestCloseEventHonoursToggle:
    """AC #2 — X quits or hides depending on the toggle."""

    def test_close_hides_to_tray_when_toggle_enabled(self, main_window):
        settings = AppSettings()
        settings.minimize_to_tray = True
        settings.tray_notification_shown = True  # skip the first-use notification
        main_window.app_settings = settings
        main_window.tray_icon = main_window.tray_icon or MagicMock()
        main_window._minimize_to_tray = MagicMock()

        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Yes,
        ) as confirm:
            event = _invoke_close_event(main_window)

        main_window._minimize_to_tray.assert_called_once()
        assert not event.isAccepted()
        # The tray branch returns before the confirm dialog is ever reached.
        confirm.assert_not_called()

    def test_close_hides_to_tray_with_a_real_tray_icon(self, main_window):
        """End-to-end variant against the window's actual tray icon."""
        skip_if_no_tray()
        if main_window.tray_icon is None:
            pytest.skip("No tray icon on this window")

        settings = AppSettings()
        settings.minimize_to_tray = True
        settings.tray_notification_shown = True
        main_window.app_settings = settings
        main_window.show()

        main_window.close()

        assert not main_window.isVisible()
        assert main_window.tray_icon.isVisible()

    def test_close_quits_when_toggle_disabled(self, main_window):
        settings = AppSettings()
        settings.minimize_to_tray = False
        main_window.app_settings = settings
        main_window.tray_icon = main_window.tray_icon or MagicMock()
        main_window._minimize_to_tray = MagicMock()

        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Yes,
        ) as confirm:
            event = _invoke_close_event(main_window)

        main_window._minimize_to_tray.assert_not_called()
        assert event.isAccepted()
        # AC #2: this story changes *whether we minimize*, never *whether we
        # confirm* — the confirm dialog still runs on the quit path.
        confirm.assert_called_once()

    def test_confirm_dialog_can_still_cancel_the_close(self, main_window):
        """The confirm dialog is untouched: No still aborts the close."""
        settings = AppSettings()
        settings.minimize_to_tray = False
        main_window.app_settings = settings
        main_window._minimize_to_tray = MagicMock()

        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.No,
        ):
            event = _invoke_close_event(main_window)

        assert not event.isAccepted()
        main_window._minimize_to_tray.assert_not_called()


class TestMeasurementModeUnaffected:
    """AC #4 — MYVOICE_AUTO_QUIT_ON_CLOSE=1 still wins over the toggle."""

    @pytest.mark.parametrize("toggle", [True, False])
    def test_env_var_forces_quit_regardless_of_toggle(
        self, main_window, monkeypatch, toggle
    ):
        monkeypatch.setenv("MYVOICE_AUTO_QUIT_ON_CLOSE", "1")

        settings = AppSettings()
        settings.minimize_to_tray = toggle
        settings.tray_notification_shown = True
        main_window.app_settings = settings
        main_window.tray_icon = main_window.tray_icon or MagicMock()
        main_window._minimize_to_tray = MagicMock()

        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Yes,
        ) as confirm:
            event = _invoke_close_event(main_window)

        assert main_window._force_quit is True
        main_window._minimize_to_tray.assert_not_called()
        confirm.assert_not_called()
        assert event.isAccepted()

    def test_without_env_var_the_toggle_still_governs(self, main_window, monkeypatch):
        """Guard against the bypass leaking into normal operation."""
        monkeypatch.delenv("MYVOICE_AUTO_QUIT_ON_CLOSE", raising=False)

        settings = AppSettings()
        settings.minimize_to_tray = True
        settings.tray_notification_shown = True
        main_window.app_settings = settings
        main_window.tray_icon = main_window.tray_icon or MagicMock()
        main_window._minimize_to_tray = MagicMock()

        event = _invoke_close_event(main_window)

        main_window._minimize_to_tray.assert_called_once()
        assert main_window._force_quit is False
        assert not event.isAccepted()


class TestLiveSettingsUpdate:
    """AC #1 — the change takes effect without an app restart (Task 1.3)."""

    def test_window_and_title_bar_follow_a_live_settings_update(self, main_window):
        main_window.app_settings = AppSettings(minimize_to_tray=True)
        assert main_window.title_bar._should_minimize_to_tray() is True

        # This is the same call app.py makes when the dialog emits
        # settings_changed — no restart, no new persistence mechanism.
        main_window.update_settings(AppSettings(minimize_to_tray=False))

        assert main_window.app_settings.minimize_to_tray is False
        assert main_window.title_bar._should_minimize_to_tray() is False
