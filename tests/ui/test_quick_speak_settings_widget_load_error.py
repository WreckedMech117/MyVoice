"""Story ui-2 -- a Quick Speak load failure must not modal from inside
``SettingsDialog.__init__``.

``QuickSpeakSettingsWidget._load_entries`` runs during dialog construction.
Before ui-2 its ``except`` branch called ``QMessageBox.warning``, so a bad
``quickspeak.csv`` (or any load error) blocked the *entire* Settings dialog
behind a modal that appeared before the dialog itself did (tooling-4 AC #4
finding). These tests mirror that exact shape: the loader raises, the dialog
is constructed, and the tooling-4 raise-on-modal net (tests/conftest.py)
fails the test outright if any blocking ``QMessageBox`` is reached.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QInputDialog, QMessageBox, QTabWidget  # noqa: E402

from myvoice.models.app_settings import AppSettings  # noqa: E402
from myvoice.ui.components.quick_speak_settings_widget import (  # noqa: E402
    QuickSpeakSettingsWidget,
)
from myvoice.ui.components.settings_dialog import SettingsDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def default_settings(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    voice_dir = tmp_path / "voices"
    voice_dir.mkdir()
    return AppSettings(
        config_directory=str(config_dir),
        voice_files_directory=str(voice_dir),
    )


@pytest.fixture
def raising_service(quick_speak_service_stub):
    """The tooling-4 typed stub, with the entries read failing.

    ``get_entries`` is the last construction-time read in ``_load_entries``
    -- profiles and the current profile have already been applied to the
    combo -- so this is the shape a corrupted ``quickspeak.csv`` produces.
    """
    quick_speak_service_stub.get_entries.side_effect = RuntimeError(
        "quickspeak.csv is corrupted"
    )
    return quick_speak_service_stub


@pytest.fixture
def modal_spies(monkeypatch):
    """Explicit call-count spies on every static modal, WITHOUT disarming the
    tooling-4 net: each spy's side_effect is the net's raiser, so a reached
    modal still fails the test immediately *and* is recorded here."""
    spies = {}
    for name in ("warning", "critical", "information", "question"):
        net_raiser = getattr(QMessageBox, name)
        spy = MagicMock(name=f"QMessageBox.{name}", side_effect=net_raiser)
        monkeypatch.setattr(QMessageBox, name, spy)
        spies[name] = spy
    return spies


def _quick_speak_tab(dialog: SettingsDialog):
    tab_widget: QTabWidget = dialog.tab_widget
    for i in range(tab_widget.count()):
        if tab_widget.tabText(i) == "Quick Speak":
            return tab_widget.widget(i)
    raise AssertionError("No Quick Speak tab in SettingsDialog")


class TestDialogConstructsWhenQuickSpeakLoadFails:
    """AC #1 + AC #3 -- the former hang shape, at the dialog layer."""

    def test_dialog_constructs_and_no_modal_is_reached(
        self, qapp, qtbot, default_settings, raising_service, modal_spies
    ):
        dlg = SettingsDialog(
            default_settings,
            parent=None,
            quick_speak_service=raising_service,
        )
        qtbot.addWidget(dlg)

        # The dialog exists and every tab is present -- Settings stayed
        # reachable despite the Quick Speak data problem.
        labels = [dlg.tab_widget.tabText(i) for i in range(dlg.tab_widget.count())]
        assert "Quick Speak" in labels
        assert len(labels) > 1, labels
        for name, spy in modal_spies.items():
            assert not spy.called, f"QMessageBox.{name} was reached: {spy.call_args}"

    def test_quick_speak_tab_shows_error_in_place(
        self, qapp, qtbot, default_settings, raising_service
    ):
        dlg = SettingsDialog(
            default_settings,
            parent=None,
            quick_speak_service=raising_service,
        )
        qtbot.addWidget(dlg)

        widget = _quick_speak_tab(dlg)
        assert isinstance(widget, QuickSpeakSettingsWidget)
        assert widget is dlg.quick_speak_widget

        label = widget.load_error_label
        assert not label.isHidden()
        assert "Failed to load Quick Speak entries" in label.text()
        assert "quickspeak.csv is corrupted" in label.text()
        assert label.property("status") == "error"

        # The table is neutralised, not left half-populated behind a modal.
        assert widget.entries_table.rowCount() == 0
        assert not widget.entries_table.isEnabled()
        assert not widget.add_button.isEnabled()

    def test_rest_of_dialog_is_usable(
        self, qapp, qtbot, default_settings, raising_service
    ):
        dlg = SettingsDialog(
            default_settings,
            parent=None,
            quick_speak_service=raising_service,
        )
        qtbot.addWidget(dlg)

        # Switching to another tab and back must work: nothing is blocking.
        tab_widget = dlg.tab_widget
        quick_speak_index = next(
            i for i in range(tab_widget.count())
            if tab_widget.tabText(i) == "Quick Speak"
        )
        other_index = 0 if quick_speak_index != 0 else 1
        tab_widget.setCurrentIndex(other_index)
        assert tab_widget.currentIndex() == other_index
        assert tab_widget.widget(other_index).isEnabled()
        tab_widget.setCurrentIndex(quick_speak_index)
        assert tab_widget.currentIndex() == quick_speak_index


class TestWidgetLoadErrorPath:
    """AC #1 detail, at the widget layer."""

    def test_error_is_logged_at_error_level(
        self, qapp, qtbot, raising_service, caplog
    ):
        caplog.set_level(logging.ERROR, logger="QuickSpeakSettingsWidget")
        widget = QuickSpeakSettingsWidget(raising_service)
        qtbot.addWidget(widget)

        errors = [
            r for r in caplog.records
            if r.name == "QuickSpeakSettingsWidget" and r.levelno == logging.ERROR
        ]
        assert len(errors) == 1, [r.getMessage() for r in caplog.records]
        assert "Error loading Quick Speak entries" in errors[0].getMessage()
        assert "quickspeak.csv is corrupted" in errors[0].getMessage()

    def test_error_label_hidden_when_load_succeeds(
        self, qapp, qtbot, quick_speak_service_stub
    ):
        widget = QuickSpeakSettingsWidget(quick_speak_service_stub)
        qtbot.addWidget(widget)

        assert widget.load_error_label.isHidden()
        assert widget.load_error_label.text() == ""
        assert widget.entries_table.isEnabled()
        assert widget.add_button.isEnabled()

    def test_successful_reload_clears_the_error(
        self, qapp, qtbot, raising_service
    ):
        widget = QuickSpeakSettingsWidget(raising_service)
        qtbot.addWidget(widget)
        assert not widget.load_error_label.isHidden()

        # Recovery: the service works again (e.g. after a profile switch or
        # the dialog's reset path), and refresh_entries() is the public route.
        raising_service.get_entries.side_effect = None
        raising_service.get_entries.return_value = []
        widget.refresh_entries()

        assert widget.load_error_label.isHidden()
        assert widget.load_error_label.text() == ""
        assert widget.entries_table.isEnabled()
        assert widget.add_button.isEnabled()

    def test_failure_at_first_read_also_stays_in_place(
        self, qapp, qtbot, quick_speak_service_stub, modal_spies
    ):
        # The earliest construction-time read failing (get_profiles) is the
        # other end of the try block; it must take the same in-place path.
        quick_speak_service_stub.get_profiles.side_effect = OSError("no such file")
        widget = QuickSpeakSettingsWidget(quick_speak_service_stub)
        qtbot.addWidget(widget)

        assert not widget.load_error_label.isHidden()
        assert "no such file" in widget.load_error_label.text()
        for name, spy in modal_spies.items():
            assert not spy.called, f"QMessageBox.{name} was reached"


class TestUserInitiatedModalsUnchanged:
    """AC #2 -- a modal in response to a click the user just made is fine."""

    def test_delete_default_profile_still_warns_modally(
        self, qapp, qtbot, quick_speak_service_stub, monkeypatch
    ):
        widget = QuickSpeakSettingsWidget(quick_speak_service_stub)
        qtbot.addWidget(widget)

        # Deliberately exercised: patch the modal (the tooling-4 net's own
        # documented opt-in) and assert the user-initiated path still reaches it.
        warning = MagicMock(return_value=QMessageBox.StandardButton.Ok)
        monkeypatch.setattr(QMessageBox, "warning", warning)

        quick_speak_service_stub.get_current_profile.return_value = "default"
        widget._on_delete_profile()

        warning.assert_called_once()
        args = warning.call_args.args
        assert args[0] is widget
        assert args[1] == "Cannot Delete"
        assert "default profile" in args[2]

    def test_add_entry_failure_still_reports_modally(
        self, qapp, qtbot, quick_speak_service_stub, monkeypatch
    ):
        widget = QuickSpeakSettingsWidget(quick_speak_service_stub)
        qtbot.addWidget(widget)

        monkeypatch.setattr(
            QInputDialog, "getText", MagicMock(return_value=("hello", True))
        )
        critical = MagicMock(return_value=QMessageBox.StandardButton.Ok)
        monkeypatch.setattr(QMessageBox, "critical", critical)
        quick_speak_service_stub.add_entry.side_effect = RuntimeError("disk full")

        widget._on_add_entry()

        critical.assert_called_once()
        assert critical.call_args.args[1] == "Add Error"
        assert "disk full" in critical.call_args.args[2]
        # And the construction-time path was never involved: no inline error.
        assert widget.load_error_label.isHidden()
