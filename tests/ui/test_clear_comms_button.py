"""Tests for Story 15.2: Clear Comms Button with Interrupt-by-Default.

Covers AC #1 through #6 (widget surface, set_state matrix, module-
boundary discipline, components re-export, MainWindow integration).
The dispatch chain (AC #7-#11) is tested in
``tests/integration/test_clear_comms_dispatch.py``; the AppSettings
schema (AC #12) is tested in
``tests/unit/models/test_app_settings_clear_comms.py``.

Test class layout (mirrors ``tests/ui/test_save_button.py``):

  - ``TestClearCommsButtonConstruction``       — AC #1, #2 widget surface.
  - ``TestClearCommsButtonSetState``           — AC #3 enablement matrix.
  - ``TestClearCommsButtonModuleBoundary``     — AC #1 import discipline.
  - ``TestClearCommsButtonInComponentsExports``— AC #1 components re-export.
  - ``TestClearCommsButtonMainWindowIntegration`` — AC #4, #5, #6.
  - ``TestClearCommsButtonAccessibility``      — AC #5 accessibility.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# PyQt6 imports — skip the whole file if PyQt6 is not available.
pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSize, Qt, pyqtBoundSignal
from PyQt6.QtWidgets import QApplication, QPushButton

from myvoice.ui.components import ClearCommsButton as ClearCommsButtonFromComponents
from myvoice.ui.components.clear_comms_button import (
    ClearCommsButton,
    _TOOLTIP_DISABLED_FILE_MISSING,
    _TOOLTIP_DISABLED_NO_SAVEABLE,
    _TOOLTIP_INTERRUPT_FILE,
    _TOOLTIP_INTERRUPT_LAST_GEN,
    _TOOLTIP_QUEUE_FILE,
    _TOOLTIP_QUEUE_LAST_GEN,
    _compute_enabled_and_tooltip,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def app(qtbot):
    """Ensure QApplication exists (mirrors test_save_button.py)."""
    return QApplication.instance() or QApplication([])


@pytest.fixture
def clear_comms_button(qtbot):
    """Fresh ClearCommsButton with qtbot cleanup."""
    widget = ClearCommsButton()
    qtbot.addWidget(widget)
    return widget


# --------------------------------------------------------------------------- #
# AC #1, #2 — TestClearCommsButtonConstruction
# --------------------------------------------------------------------------- #


class TestClearCommsButtonConstruction:
    """Widget-construction contract: subclass, default state, icon, size."""

    def test_clear_comms_button_is_qpushbutton_subclass(self):
        # AC #1 — QPushButton subclass for the same reason SaveButton is
        # (action-toolbar visual consistency, button-box frame + hover).
        assert issubclass(ClearCommsButton, QPushButton)

    def test_clear_comms_button_default_disabled(self, clear_comms_button):
        assert clear_comms_button.isEnabled() is False

    def test_clear_comms_button_default_tooltip(self, clear_comms_button):
        assert clear_comms_button.toolTip() == _TOOLTIP_DISABLED_NO_SAVEABLE
        assert (
            clear_comms_button.toolTip()
            == "Generate audio first to use Clear Comms"
        )

    def test_clear_comms_button_default_object_name(self, clear_comms_button):
        assert clear_comms_button.objectName() == "clear_comms_button"

    def test_clear_comms_button_fixed_size_24x24(self, clear_comms_button):
        assert clear_comms_button.size() == QSize(24, 24)
        assert clear_comms_button.width() == 24
        assert clear_comms_button.height() == 24

    def test_clear_comms_button_has_icon(self, clear_comms_button):
        assert clear_comms_button.icon().isNull() is False

    def test_clear_comms_button_logger_attached(self, clear_comms_button):
        assert clear_comms_button._logger is not None
        assert clear_comms_button._logger.name == "ClearCommsButton"


# --------------------------------------------------------------------------- #
# AC #3 — TestClearCommsButtonSetState
# --------------------------------------------------------------------------- #


_MATRIX_CASES = [
    # (source_kind, has_saveable, file_path_valid, queue_mode,
    #  expected_enabled, expected_tooltip)
    ("last_generation", False, False, False, False, _TOOLTIP_DISABLED_NO_SAVEABLE),
    ("last_generation", False, True, False, False, _TOOLTIP_DISABLED_NO_SAVEABLE),
    ("last_generation", True, False, False, True, _TOOLTIP_INTERRUPT_LAST_GEN),
    ("last_generation", True, False, True, True, _TOOLTIP_QUEUE_LAST_GEN),
    ("last_generation", True, True, False, True, _TOOLTIP_INTERRUPT_LAST_GEN),
    ("last_generation", True, True, True, True, _TOOLTIP_QUEUE_LAST_GEN),
    ("file", False, False, False, False, _TOOLTIP_DISABLED_FILE_MISSING),
    ("file", True, False, False, False, _TOOLTIP_DISABLED_FILE_MISSING),
    ("file", False, True, False, True, _TOOLTIP_INTERRUPT_FILE),
    ("file", False, True, True, True, _TOOLTIP_QUEUE_FILE),
    ("file", True, True, False, True, _TOOLTIP_INTERRUPT_FILE),
    ("file", True, True, True, True, _TOOLTIP_QUEUE_FILE),
]


class TestClearCommsButtonSetState:
    """``set_state`` enforces the AC #3 enablement matrix exactly."""

    @pytest.mark.parametrize(
        "source_kind,has_saveable,file_path_valid,queue_mode,expected_enabled,expected_tooltip",
        _MATRIX_CASES,
    )
    def test_set_state_matrix(
        self,
        clear_comms_button,
        source_kind,
        has_saveable,
        file_path_valid,
        queue_mode,
        expected_enabled,
        expected_tooltip,
    ):
        clear_comms_button.set_state(
            source_kind=source_kind,
            has_saveable=has_saveable,
            file_path_valid=file_path_valid,
            queue_mode=queue_mode,
        )
        assert clear_comms_button.isEnabled() is expected_enabled
        assert clear_comms_button.toolTip() == expected_tooltip

    def test_unrecognized_source_kind_falls_back_to_disabled(self, clear_comms_button):
        clear_comms_button.set_state(
            source_kind="bogus",
            has_saveable=True,
            file_path_valid=True,
            queue_mode=False,
        )
        assert clear_comms_button.isEnabled() is False
        assert clear_comms_button.toolTip() == _TOOLTIP_DISABLED_NO_SAVEABLE

    def test_set_state_idempotent_repeated_call(self, clear_comms_button):
        kwargs = dict(
            source_kind="last_generation",
            has_saveable=True,
            file_path_valid=False,
            queue_mode=False,
        )
        clear_comms_button.set_state(**kwargs)
        first_enabled = clear_comms_button.isEnabled()
        first_tooltip = clear_comms_button.toolTip()
        clear_comms_button.set_state(**kwargs)
        assert clear_comms_button.isEnabled() == first_enabled
        assert clear_comms_button.toolTip() == first_tooltip

    def test_set_state_pure_helper_matches_matrix(self):
        """``_compute_enabled_and_tooltip`` is the testable pure helper."""
        for case in _MATRIX_CASES:
            (
                source_kind,
                has_saveable,
                file_path_valid,
                queue_mode,
                expected_enabled,
                expected_tooltip,
            ) = case
            enabled, tooltip = _compute_enabled_and_tooltip(
                source_kind=source_kind,
                has_saveable=has_saveable,
                file_path_valid=file_path_valid,
                queue_mode=queue_mode,
            )
            assert enabled is expected_enabled, (
                f"Matrix mismatch for {source_kind!r}, "
                f"has_saveable={has_saveable}, file_path_valid={file_path_valid}, "
                f"queue_mode={queue_mode}: expected enabled={expected_enabled}, got {enabled}"
            )
            assert tooltip == expected_tooltip


# --------------------------------------------------------------------------- #
# AC #1 — TestClearCommsButtonModuleBoundary
# --------------------------------------------------------------------------- #


_FORBIDDEN_IMPORTS = [
    "myvoice.services",
    "myvoice.models",
    "clear_comms_settings_panel",
    "qwen_tts_service",
    "audio_coordinator",
    "tts_streaming",
]


class TestClearCommsButtonModuleBoundary:
    """Static-scan: ``clear_comms_button.py`` is a pure presentation widget.

    Per AC #1 it must not import from ``myvoice.services.*``,
    ``myvoice.models.*``, or 15.1's ``clear_comms_settings_panel`` —
    state arrives via ``set_state``.
    """

    @pytest.fixture
    def source_text(self) -> str:
        from myvoice.ui.components import clear_comms_button as button_module
        path = Path(button_module.__file__)
        return path.read_text(encoding="utf-8")

    @pytest.mark.parametrize("forbidden", _FORBIDDEN_IMPORTS)
    def test_clear_comms_button_does_not_import_forbidden(self, source_text, forbidden):
        # Allow forbidden tokens to appear inside the docstring/comments
        # — only the real import statements would matter — but a string-
        # match scan is sufficient for the v1 boundary check (mirrors the
        # save_button equivalent test). If a future contributor inlines
        # one of the forbidden tokens in a comment for legitimate reasons,
        # this test will need a more discerning AST scan.
        for line in source_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                assert forbidden not in stripped, (
                    f"clear_comms_button.py must not import {forbidden!r}; "
                    "the widget is a pure presentation component per AC #1"
                )


# --------------------------------------------------------------------------- #
# AC #1 — TestClearCommsButtonInComponentsExports
# --------------------------------------------------------------------------- #


class TestClearCommsButtonInComponentsExports:
    """``ClearCommsButton`` is re-exported from ``myvoice.ui.components``."""

    def test_clear_comms_button_exported_from_components_package(self):
        assert ClearCommsButtonFromComponents is ClearCommsButton

    def test_clear_comms_button_in_components_all(self):
        import myvoice.ui.components as components_pkg
        assert "ClearCommsButton" in components_pkg.__all__


# --------------------------------------------------------------------------- #
# AC #4, #5, #6 — TestClearCommsButtonMainWindowIntegration
# --------------------------------------------------------------------------- #


def _process_events(app: QApplication, iterations: int = 3) -> None:
    for _ in range(iterations):
        app.processEvents()


def _find_layout_containing(root_layout, widget):
    """Walk a layout tree depth-first and return the innermost layout
    that directly contains ``widget``. Mirrors the helper in
    ``test_save_button.py``."""
    for i in range(root_layout.count()):
        item = root_layout.itemAt(i)
        if item.widget() is widget:
            return root_layout
        sub_layout = item.layout()
        if sub_layout is not None:
            found = _find_layout_containing(sub_layout, widget)
            if found is not None:
                return found
    return None


class TestClearCommsButtonMainWindowIntegration:
    """End-to-end wiring: signal flow + layout placement + state seed."""

    @pytest.fixture
    def main_window_no_registry(self, qtbot):
        from myvoice.ui.main_window import MainWindow
        window = MainWindow()
        qtbot.addWidget(window)
        return window

    @pytest.fixture
    def main_window_with_registry(self, qtbot):
        from myvoice.services.sessions import SessionRegistry
        from myvoice.ui.main_window import MainWindow
        registry = SessionRegistry()
        window = MainWindow(session_registry=registry)
        qtbot.addWidget(window)
        return window, registry

    def test_main_window_has_clear_comms_button_attribute(self, main_window_with_registry):
        window, _ = main_window_with_registry
        assert hasattr(window, "clear_comms_button")
        assert isinstance(window.clear_comms_button, ClearCommsButton)

    def test_clear_comms_button_in_action_buttons_layout(self, main_window_with_registry):
        window, _ = main_window_with_registry
        outer = window.replay_button.parentWidget().layout()
        action_layout = _find_layout_containing(outer, window.replay_button)
        assert action_layout is not None
        assert action_layout.indexOf(window.clear_comms_button) >= 0

    def test_clear_comms_button_layout_order(self, main_window_with_registry):
        """Per AC #5: order is quick_speak, generate, replay, save,
        clear_comms, clear (Clear Comms inserts between Save and Clear)."""
        window, _ = main_window_with_registry
        outer = window.replay_button.parentWidget().layout()
        action_layout = _find_layout_containing(outer, window.replay_button)
        assert action_layout is not None
        expected_order = [
            window.quick_speak_button,
            window.generate_button,
            window.replay_button,
            window.save_button,
            window.clear_comms_button,
            window.clear_button,
        ]
        actual_widgets = [
            action_layout.itemAt(i).widget()
            for i in range(action_layout.count())
            if action_layout.itemAt(i).widget() is not None
        ]
        for expected in expected_order:
            assert expected in actual_widgets
        ordered_indices = [actual_widgets.index(w) for w in expected_order]
        assert ordered_indices == sorted(ordered_indices)

    def test_main_window_declares_clear_comms_requested_signal(self):
        from myvoice.ui.main_window import MainWindow
        assert hasattr(MainWindow, "clear_comms_requested")

    def test_clear_comms_requested_is_bound_signal_on_instance(
        self, main_window_with_registry
    ):
        window, _ = main_window_with_registry
        assert isinstance(window.clear_comms_requested, pyqtBoundSignal)

    def test_click_emits_clear_comms_requested(self, main_window_with_registry, app):
        """Force the button enabled; click; verify the signal fires."""
        window, _ = main_window_with_registry
        captured: list[None] = []
        window.clear_comms_requested.connect(lambda: captured.append(None))
        # Force the button enabled by calling set_state directly.
        window.clear_comms_button.set_state(
            source_kind="last_generation",
            has_saveable=True,
            file_path_valid=False,
            queue_mode=False,
        )
        assert window.clear_comms_button.isEnabled() is True
        window.clear_comms_button.click()
        _process_events(app)
        assert len(captured) == 1

    def test_click_when_disabled_does_not_emit(self, main_window_with_registry, app):
        window, _ = main_window_with_registry
        captured: list[None] = []
        window.clear_comms_requested.connect(lambda: captured.append(None))
        # Default state — no saveable, no file — disabled.
        assert window.clear_comms_button.isEnabled() is False
        window.clear_comms_button.click()
        _process_events(app)
        assert captured == []

    def test_set_clear_comms_config_snapshot_drives_button_state(
        self, main_window_with_registry, app
    ):
        """The public snapshot setter (AC #6) updates button state."""
        window, _ = main_window_with_registry
        # File-source with a valid file path → button should enable
        # (interrupt mode by default).
        window.set_clear_comms_config_snapshot(
            source_kind="file",
            file_path="/tmp/some.wav",  # path content irrelevant; valid flag drives state
            file_path_valid=True,
            queue_mode=False,
        )
        assert window.clear_comms_button.isEnabled() is True
        assert window.clear_comms_button.toolTip() == _TOOLTIP_INTERRUPT_FILE

    def test_saveable_session_changed_propagates_to_clear_comms_button(
        self, main_window_with_registry, app
    ):
        """When a saveable session lands, the Clear Comms button enables for
        the default ``"last_generation"`` source kind. Verifies AC #6's
        extension of ``_on_saveable_session_changed``."""
        from myvoice.services.sessions import SessionSource
        import numpy as np
        window, registry = main_window_with_registry
        # Default source_kind is "last_generation"; button should be disabled
        # initially (no saveable yet).
        assert window.clear_comms_button.isEnabled() is False

        # Drive a session through to READY_TO_PLAY so the saveable slot
        # is promoted.
        sid = registry.create_session(
            text="hi", voice="v", model_type="m", source=SessionSource.GENERATED,
        )
        registry.start_generation(sid)
        registry.append_chunk(sid, np.array([1.0, 2.0, 3.0], dtype=np.float32))
        registry.finalize(sid)
        _process_events(app)
        assert window.clear_comms_button.isEnabled() is True

    def test_legacy_no_registry_button_disabled(self, main_window_no_registry):
        """Without a registry the button still exists, default disabled."""
        window = main_window_no_registry
        assert hasattr(window, "clear_comms_button")
        assert isinstance(window.clear_comms_button, ClearCommsButton)
        assert window.clear_comms_button.isEnabled() is False


# --------------------------------------------------------------------------- #
# AC #5 — TestClearCommsButtonAccessibility
# --------------------------------------------------------------------------- #


class TestClearCommsButtonAccessibility:
    """Accessibility metadata: name, description, focus policy, tab order."""

    @pytest.fixture
    def window(self, qtbot):
        from myvoice.services.sessions import SessionRegistry
        from myvoice.ui.main_window import MainWindow
        registry = SessionRegistry()
        w = MainWindow(session_registry=registry)
        qtbot.addWidget(w)
        return w

    def test_clear_comms_accessible_name_set(self, window):
        assert window.clear_comms_button.accessibleName() == "Clear Comms"

    def test_clear_comms_accessible_description_set(self, window):
        assert (
            window.clear_comms_button.accessibleDescription()
            == "Interrupt current playback and replay your configured Clear Comms audio source"
        )

    def test_clear_comms_focus_policy_strong(self, window):
        assert window.clear_comms_button.focusPolicy() == Qt.FocusPolicy.StrongFocus

    def test_clear_comms_in_tab_order_save_to_clear_comms_to_clear(self):
        """Tab chain: save → clear_comms → clear, replacing the old direct
        save → clear edge from Story 14.2."""
        from myvoice.ui import main_window as main_window_module
        source = Path(main_window_module.__file__).read_text(encoding="utf-8")
        assert "setTabOrder(self.save_button, self.clear_comms_button)" in source
        assert "setTabOrder(self.clear_comms_button, self.clear_button)" in source
        # The pre-15.2 single-step setTabOrder(save, clear) must not
        # remain — it would short-circuit the new chain.
        assert "setTabOrder(self.save_button, self.clear_button)" not in source
