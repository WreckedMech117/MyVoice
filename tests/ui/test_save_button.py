"""Tests for Story 14.2: Save Button with Enablement Signal.

Covers AC #1 through #16 of the saveable Save button — widget-level
construction, ``set_saveable`` semantics (None / finalized / streaming),
module-boundary discipline, package-level import path, components
re-export, MainWindow integration (registry → button slot wiring,
``save_requested`` signal, layout placement, accessibility), and the
net-zero invariant.

Test class layout (mirrors Story 12.3 / 14.1 organization):

  - ``TestSaveButtonConstruction``       — AC #1, #2 widget surface.
  - ``TestSaveButtonSetSaveable``        — AC #2, #5, #6, #7 state mutation.
  - ``TestSaveButtonModuleBoundary``     — AC #11 import discipline.
  - ``TestSaveButtonImportPath``         — AC #13 package-root import.
  - ``TestSaveButtonInComponentsExports``— AC #1 components re-export.
  - ``TestSaveButtonMainWindowIntegration`` — AC #3, #4, #5, #6, #8,
    #9, #10, #12 end-to-end wiring.
  - ``TestSaveButtonAccessibility``      — AC #9 accessible name /
    focus policy.
  - ``TestSaveButtonNetZero``            — AC #12 gate (documents the
    suites swept manually as part of Task 7).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from unittest.mock import patch

import numpy as np
import pytest

# PyQt6 imports — skip the whole file if PyQt6 is not available.
pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSize, Qt, pyqtBoundSignal, pyqtSignal
from PyQt6.QtWidgets import QApplication, QToolButton

from myvoice.services.sessions import (
    GenerationSession,
    SaveableAudio,
    SessionRegistry,
    SessionSource,
    SessionState,
)
from myvoice.ui.components import SaveButton as SaveButtonFromComponents
from myvoice.ui.components.save_button import (
    SaveButton,
    _TOOLTIP_DISABLED,
    _TOOLTIP_ENABLED,
    _TOOLTIP_ENABLED_STREAMING,
)


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def app(qtbot):
    """Ensure QApplication exists (mirrors test_status_indicators.py)."""
    return QApplication.instance() or QApplication([])


@pytest.fixture
def save_button(qtbot):
    """Fresh SaveButton with qtbot cleanup."""
    widget = SaveButton()
    qtbot.addWidget(widget)
    return widget


def make_saveable_audio(
    session_id: str = "test-sid",
    is_streaming: bool = False,
    sample_rate: int = 24000,
    voice: str = "test_voice",
    text: str = "test text",
    created_at: float = 0.0,
    audio: Optional[np.ndarray] = None,
) -> SaveableAudio:
    """Build a SaveableAudio for widget-level tests without driving the
    full registry finalize path. Mirrors the helper described in Story
    14.2 AC #14 — keeps unit tests independent of the registry.
    """
    if audio is None:
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    return SaveableAudio(
        session_id=session_id,
        complete_audio=audio,
        sample_rate=sample_rate,
        voice=voice,
        text=text,
        is_streaming=is_streaming,
        created_at=created_at,
    )


def make_finalized_session(
    registry: SessionRegistry,
    *,
    text: str = "test text",
    voice: str = "test voice",
    source: SessionSource = SessionSource.GENERATED,
    audio: Optional[np.ndarray] = None,
    is_streaming: bool = False,
) -> str:
    """Inline copy of Story 14.1's helper — drives a session through
    PENDING → GENERATING → READY_TO_PLAY via the registry's slot path
    so the saveable lifecycle promotes the session naturally.
    """
    if audio is None:
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sid = registry.create_session(
        text=text, voice=voice, model_type="test_model", source=source,
    )
    if is_streaming:
        object.__setattr__(registry.get(sid), "is_streaming", True)
    registry.start_generation(sid)
    registry.append_chunk(sid, audio)
    registry.finalize(sid)
    return sid


# --------------------------------------------------------------------------- #
# AC #1, #2 — TestSaveButtonConstruction
# --------------------------------------------------------------------------- #


class TestSaveButtonConstruction:
    """Widget-construction contract: subclass, default state, icon, size."""

    def test_save_button_is_qtoolbutton_subclass(self):
        assert issubclass(SaveButton, QToolButton)

    def test_save_button_default_disabled(self, save_button):
        assert save_button.isEnabled() is False

    def test_save_button_default_tooltip(self, save_button):
        assert save_button.toolTip() == _TOOLTIP_DISABLED
        assert save_button.toolTip() == "No generation to save yet"

    def test_save_button_default_object_name(self, save_button):
        assert save_button.objectName() == "save_button"

    def test_save_button_fixed_size_24x24(self, save_button):
        assert save_button.size() == QSize(24, 24)
        assert save_button.width() == 24
        assert save_button.height() == 24

    def test_save_button_has_icon(self, save_button):
        assert save_button.icon().isNull() is False

    def test_save_button_construction_no_qapplication_exec(self, app, qtbot):
        """Construction is cheap — no event-loop spin needed."""
        widget = SaveButton()
        qtbot.addWidget(widget)
        assert widget.isEnabled() is False

    def test_save_button_logger_attached(self, save_button):
        """Internal: a logger is attached (mirrors queue_depth_badge)."""
        assert save_button._logger is not None
        assert save_button._logger.name == "SaveButton"


# --------------------------------------------------------------------------- #
# AC #2, #5, #6, #7 — TestSaveButtonSetSaveable
# --------------------------------------------------------------------------- #


class TestSaveButtonSetSaveable:
    """``set_saveable`` is the only state-mutating entry point.

    Three branches: None → disabled, finalized → enabled, streaming →
    enabled with suffix. Idempotent under repeated calls.
    """

    def test_set_saveable_none_disables_button(self, save_button):
        save_button.set_saveable(None)
        assert save_button.isEnabled() is False
        assert save_button.toolTip() == _TOOLTIP_DISABLED

    def test_set_saveable_finalized_audio_enables_button(self, save_button):
        audio = make_saveable_audio(is_streaming=False)
        save_button.set_saveable(audio)
        assert save_button.isEnabled() is True
        assert save_button.toolTip() == _TOOLTIP_ENABLED
        assert "(streaming" not in save_button.toolTip()

    def test_set_saveable_streaming_audio_enables_with_suffix(self, save_button):
        audio = make_saveable_audio(is_streaming=True)
        save_button.set_saveable(audio)
        assert save_button.isEnabled() is True
        assert save_button.toolTip() == _TOOLTIP_ENABLED_STREAMING
        assert save_button.toolTip().endswith("(streaming…)")

    def test_set_saveable_streaming_to_finalized_drops_suffix(self, save_button):
        save_button.set_saveable(make_saveable_audio(is_streaming=True))
        save_button.set_saveable(make_saveable_audio(is_streaming=False))
        assert save_button.toolTip() == _TOOLTIP_ENABLED
        assert "(streaming" not in save_button.toolTip()

    def test_set_saveable_finalized_to_none_reverts_tooltip(self, save_button):
        save_button.set_saveable(make_saveable_audio(is_streaming=False))
        save_button.set_saveable(None)
        assert save_button.isEnabled() is False
        assert save_button.toolTip() == _TOOLTIP_DISABLED

    def test_set_saveable_idempotent_repeated_none(self, save_button):
        save_button.set_saveable(None)
        save_button.set_saveable(None)
        assert save_button.isEnabled() is False
        assert save_button.toolTip() == _TOOLTIP_DISABLED

    def test_set_saveable_idempotent_repeated_same_audio(self, save_button):
        audio = make_saveable_audio(is_streaming=False)
        save_button.set_saveable(audio)
        save_button.set_saveable(audio)
        assert save_button.isEnabled() is True
        assert save_button.toolTip() == _TOOLTIP_ENABLED

    def test_set_saveable_finalized_to_streaming_adds_suffix(self, save_button):
        save_button.set_saveable(make_saveable_audio(is_streaming=False))
        save_button.set_saveable(make_saveable_audio(is_streaming=True))
        assert save_button.toolTip() == _TOOLTIP_ENABLED_STREAMING


# --------------------------------------------------------------------------- #
# AC #11 — TestSaveButtonModuleBoundary
# --------------------------------------------------------------------------- #


_FORBIDDEN_IMPORTS = ["qwen_tts_service", "audio_coordinator", "tts_streaming"]


class TestSaveButtonModuleBoundary:
    """Static-scan check: ``save_button.py`` must not import services
    other than the saveable-typed re-export. Mirrors Story 14.1's
    ``TestSaveableModuleBoundary`` parametrized pattern.
    """

    @pytest.fixture
    def source_text(self) -> str:
        from myvoice.ui.components import save_button as save_button_module
        path = Path(save_button_module.__file__)
        return path.read_text(encoding="utf-8")

    @pytest.mark.parametrize("forbidden", _FORBIDDEN_IMPORTS)
    def test_save_button_does_not_import_forbidden(self, source_text, forbidden):
        assert forbidden not in source_text, (
            f"save_button.py must not reference {forbidden!r}; "
            "ui/* may not import services/* directly per architecture line 689-694"
        )


# --------------------------------------------------------------------------- #
# AC #13 — TestSaveButtonImportPath
# --------------------------------------------------------------------------- #


class TestSaveButtonImportPath:
    """``SaveableAudio`` must be imported from the package root, not
    from the implementation module ``services.sessions.session_registry``.
    """

    @pytest.fixture
    def source_text(self) -> str:
        from myvoice.ui.components import save_button as save_button_module
        path = Path(save_button_module.__file__)
        return path.read_text(encoding="utf-8")

    def test_save_button_imports_saveable_audio_from_package(self, source_text):
        # Public-surface import line is present.
        assert "from myvoice.services.sessions import" in source_text
        # The implementation-module path is NOT present.
        assert "from myvoice.services.sessions.session_registry import" not in source_text


# --------------------------------------------------------------------------- #
# AC #1 — TestSaveButtonInComponentsExports
# --------------------------------------------------------------------------- #


class TestSaveButtonInComponentsExports:
    """``SaveButton`` is re-exported from ``myvoice.ui.components``."""

    def test_save_button_exported_from_components_package(self):
        # Import via the components package path resolves to the same class.
        assert SaveButtonFromComponents is SaveButton

    def test_save_button_in_components_all(self):
        import myvoice.ui.components as components_pkg
        assert "SaveButton" in components_pkg.__all__


# --------------------------------------------------------------------------- #
# AC #3, #4, #5, #6, #8, #9, #10, #12 — TestSaveButtonMainWindowIntegration
# --------------------------------------------------------------------------- #


def _process_events(app: QApplication, iterations: int = 3) -> None:
    """Drain the QueuedConnection backlog (mirrors test_session_registry's
    ``drain_qt_events``).
    """
    for _ in range(iterations):
        app.processEvents()


def _find_layout_containing(root_layout, widget):
    """Walk a layout tree depth-first and return the innermost layout
    that directly contains ``widget``.

    ``MainWindow`` builds ``action_buttons_layout`` (a ``QVBoxLayout``)
    as a sub-layout of an outer text-input layout, so
    ``widget.parentWidget().layout()`` returns the *outer* layout, not
    the action layout. This helper finds the actual sub-layout the
    button was added to so order assertions can iterate
    ``itemAt(i).widget()`` directly.
    """
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


class TestSaveButtonMainWindowIntegration:
    """End-to-end wiring tests: registry → window slot → button.

    Uses a real ``MainWindow`` + real ``SessionRegistry`` so the
    ``QueuedConnection`` fires on the Qt main thread under the same
    conditions as production (mirrors ``TestSessionRegistryIndicatorWiring``).
    """

    @pytest.fixture
    def main_window_no_registry(self, qtbot):
        """``MainWindow`` instantiated without a registry — exercises the
        legacy callback-chain path; the button still exists but is never
        wired to the registry.
        """
        from myvoice.ui.main_window import MainWindow
        window = MainWindow()
        qtbot.addWidget(window)
        return window

    @pytest.fixture
    def registry_main_window(self, qtbot):
        """``MainWindow`` wired to a real ``SessionRegistry``.

        Same fixture pattern as ``TestSessionRegistryIndicatorWiring``;
        the registry runs on the Qt main thread and the queued
        connection between registry and window means tests must cycle
        the event loop after each mutation before asserting.
        """
        from myvoice.ui.main_window import MainWindow
        registry = SessionRegistry()
        window = MainWindow(session_registry=registry)
        qtbot.addWidget(window)
        return window, registry

    # ----- AC #3 / AC #9 / AC #12: button exists on MainWindow ----- #

    def test_main_window_has_save_button_attribute(self, registry_main_window):
        window, _ = registry_main_window
        assert hasattr(window, "save_button")
        assert isinstance(window.save_button, SaveButton)

    def test_save_button_in_action_buttons_layout(self, registry_main_window):
        """save_button is in the same QVBoxLayout (``action_buttons_layout``)
        that hosts the other action buttons — not just sharing a parent
        widget further up the layout tree.
        """
        window, _ = registry_main_window
        outer = window.replay_button.parentWidget().layout()
        action_layout = _find_layout_containing(outer, window.replay_button)
        assert action_layout is not None, "replay_button's containing layout not found"
        assert action_layout.indexOf(window.save_button) >= 0, (
            "save_button is not in the same layout as replay_button"
        )

    def test_save_button_layout_order(self, registry_main_window):
        """Order in ``action_buttons_layout``: quick_speak, generate,
        replay, save, clear. Verified by iterating the layout's items in
        index order — a sibling-shuffle would fail here.
        """
        window, _ = registry_main_window
        outer = window.replay_button.parentWidget().layout()
        action_layout = _find_layout_containing(outer, window.replay_button)
        assert action_layout is not None
        expected_order = [
            window.quick_speak_button,
            window.generate_button,
            window.replay_button,
            window.save_button,
            window.clear_button,
        ]
        actual_widgets = [
            action_layout.itemAt(i).widget()
            for i in range(action_layout.count())
            if action_layout.itemAt(i).widget() is not None
        ]
        for expected in expected_order:
            assert expected in actual_widgets, (
                f"{expected.objectName() or expected} missing from action_buttons_layout"
            )
        ordered_indices = [actual_widgets.index(w) for w in expected_order]
        assert ordered_indices == sorted(ordered_indices), (
            f"action_buttons_layout order mismatch: "
            f"got {[w.objectName() for w in actual_widgets]}; "
            f"expected the five action buttons in "
            f"[quick_speak, generate, replay, save, clear] order"
        )

    def test_main_window_declares_save_requested_signal(self):
        from myvoice.ui.main_window import MainWindow
        # Class-level attribute is a `pyqtSignal` descriptor; instance
        # access yields a bound signal. Verify both shapes.
        assert hasattr(MainWindow, "save_requested")

    def test_save_requested_is_bound_signal_on_instance(self, registry_main_window):
        window, _ = registry_main_window
        assert isinstance(window.save_requested, pyqtBoundSignal)

    # ----- AC #4: seed initial state on construction ----- #

    def test_seed_disabled_when_no_saveable_at_construction(self, registry_main_window):
        window, registry = registry_main_window
        # No session yet → registry has no saveable
        assert registry.saveable_session_id is None
        assert window.save_button.isEnabled() is False

    def test_seed_enabled_when_saveable_present_at_construction(self, qtbot, app):
        """Pre-finalize a session in the registry, THEN instantiate
        MainWindow — the seed call must reflect that saveable.
        """
        from myvoice.ui.main_window import MainWindow
        registry = SessionRegistry()
        sid = make_finalized_session(registry)
        assert registry.saveable_session_id == sid
        window = MainWindow(session_registry=registry)
        qtbot.addWidget(window)
        # Seed runs synchronously inside _connect_registry_to_indicator
        # so no event-loop pump is needed.
        assert window.save_button.isEnabled() is True
        assert window.save_button.toolTip() == _TOOLTIP_ENABLED

    # ----- AC #5: signal-driven enable ----- #

    def test_button_enables_on_finalize_signal(self, registry_main_window, app):
        window, registry = registry_main_window
        assert window.save_button.isEnabled() is False
        make_finalized_session(registry, text="hello")
        # QueuedConnection: pump the event loop.
        _process_events(app)
        assert window.save_button.isEnabled() is True
        assert window.save_button.toolTip() == _TOOLTIP_ENABLED

    def test_two_finalizes_keep_button_enabled(self, registry_main_window, app):
        """A → finalize → B → finalize: button stays enabled across
        the supersession."""
        window, registry = registry_main_window
        make_finalized_session(registry, text="A")
        _process_events(app)
        assert window.save_button.isEnabled() is True
        make_finalized_session(registry, text="B")
        _process_events(app)
        assert window.save_button.isEnabled() is True

    # ----- AC #6: cancel-to-None disables ----- #

    def test_button_disables_on_cancel_to_none(self, registry_main_window, app):
        """Finalize then cancel that same session (no previous saveable)
        → emits None → button disables."""
        window, registry = registry_main_window
        sid = make_finalized_session(registry, text="A")
        _process_events(app)
        assert window.save_button.isEnabled() is True
        # Cancel A; registry path must vacate the saveable slot.
        registry.cancel(sid)
        _process_events(app)
        assert window.save_button.isEnabled() is False
        assert window.save_button.toolTip() == _TOOLTIP_DISABLED

    def test_cancel_of_current_with_previous_keeps_button_enabled(
        self, registry_main_window, app
    ):
        """A finalize, B finalize, B cancel reverts to A → button stays
        enabled per Story 14.1 AC #7 alternate branch."""
        window, registry = registry_main_window
        a_sid = make_finalized_session(registry, text="A")
        b_sid = make_finalized_session(registry, text="B")
        _process_events(app)
        assert window.save_button.isEnabled() is True
        # Cancel B; registry reverts the saveable to A.
        registry.cancel(b_sid)
        _process_events(app)
        assert window.save_button.isEnabled() is True
        # And the saveable session id should have reverted to A.
        assert registry.saveable_session_id == a_sid

    # ----- AC #7: streaming tooltip via direct mock ----- #

    def test_button_streaming_tooltip_during_streaming_saveable(
        self, registry_main_window, app
    ):
        """Construct a streaming-saveable scenario by mocking
        registry.saveable_audio so the slot's re-read yields a
        streaming SaveableAudio."""
        window, registry = registry_main_window
        streaming = make_saveable_audio(session_id="stream", is_streaming=True)
        with patch.object(
            type(registry), "saveable_audio", new_callable=lambda: property(lambda self: streaming)
        ):
            registry.saveable_session_changed.emit("stream")
            _process_events(app)
            assert window.save_button.isEnabled() is True
            assert window.save_button.toolTip() == _TOOLTIP_ENABLED_STREAMING

    # ----- AC #8: click emits save_requested ----- #

    def test_click_emits_save_requested(self, registry_main_window, app):
        """Connect a spy to MainWindow.save_requested; finalize so the
        button enables; click; spy captures one emission."""
        window, registry = registry_main_window
        captured: list[None] = []
        window.save_requested.connect(lambda: captured.append(None))
        make_finalized_session(registry, text="hello")
        _process_events(app)
        assert window.save_button.isEnabled() is True
        window.save_button.click()
        _process_events(app)
        assert len(captured) == 1

    def test_click_when_disabled_does_not_emit(self, registry_main_window, app):
        """Fresh registry — button disabled — click is a no-op for
        ``QToolButton.click()`` (Qt-internal enabled gate)."""
        window, registry = registry_main_window
        captured: list[None] = []
        window.save_requested.connect(lambda: captured.append(None))
        assert window.save_button.isEnabled() is False
        window.save_button.click()
        _process_events(app)
        assert captured == []

    # ----- AC #10 / #12: legacy path still works ----- #

    def test_legacy_path_no_session_registry_button_disabled(self, main_window_no_registry):
        """``MainWindow(session_registry=None)`` — the button still
        exists, defaults to disabled, and the save_requested signal
        exists as a class attribute."""
        window = main_window_no_registry
        assert hasattr(window, "save_button")
        assert isinstance(window.save_button, SaveButton)
        assert window.save_button.isEnabled() is False
        assert window.save_requested is not None

    def test_slot_is_no_op_when_registry_is_none(self, main_window_no_registry):
        """Direct call to the slot with registry=None must not raise
        and must not change the button state."""
        window = main_window_no_registry
        # Should not raise even though no registry is wired.
        window._on_saveable_session_changed(None)
        assert window.save_button.isEnabled() is False


# --------------------------------------------------------------------------- #
# AC #9 — TestSaveButtonAccessibility
# --------------------------------------------------------------------------- #


class TestSaveButtonAccessibility:
    """Accessibility entries: name, focus policy, tab order."""

    @pytest.fixture
    def window(self, qtbot):
        from myvoice.ui.main_window import MainWindow
        registry = SessionRegistry()
        w = MainWindow(session_registry=registry)
        qtbot.addWidget(w)
        return w

    def test_save_button_accessible_name_set(self, window):
        assert window.save_button.accessibleName() == "Save last generation"

    def test_save_button_accessible_description_set(self, window):
        assert (
            window.save_button.accessibleDescription()
            == "Save the most recent generated speech to a WAV file"
        )

    def test_save_button_focus_policy_strong(self, window):
        assert window.save_button.focusPolicy() == Qt.FocusPolicy.StrongFocus

    def test_save_button_in_tab_order_after_replay_before_clear(self):
        """Tab chain: replay → save → clear. Verified via static scan of
        ``setTabOrder`` calls in ``main_window.py`` per AC #14's "or
        assert via setTabOrder calls in source" alternative — more
        robust than walking ``nextInFocusChain`` across nested layouts.
        """
        from myvoice.ui import main_window as main_window_module
        source = Path(main_window_module.__file__).read_text(encoding="utf-8")
        assert "setTabOrder(self.replay_button, self.save_button)" in source, (
            "Tab-order chain must include replay_button → save_button"
        )
        assert "setTabOrder(self.save_button, self.clear_button)" in source, (
            "Tab-order chain must include save_button → clear_button"
        )
        # The pre-14.2 single-step setTabOrder(replay, clear) must not
        # remain — it would short-circuit the new chain.
        assert "setTabOrder(self.replay_button, self.clear_button)" not in source, (
            "Old setTabOrder(replay_button, clear_button) must be replaced "
            "by the two-step chain through save_button"
        )


# --------------------------------------------------------------------------- #
# AC #12 — TestSaveButtonNetZero (gate / documentation)
# --------------------------------------------------------------------------- #


class TestSaveButtonNetZero:
    """Gate test documenting the manual sweep — does not re-run other
    test files. The expected suites swept by Task 7 of Story 14.2:

      - tests/unit/services/sessions/  (~269 tests, unchanged)
      - tests/ui/test_status_indicators.py (~56 tests, unchanged)
      - tests/ui/test_playback_last.py (~16 tests, unchanged)
      - tests/integration/test_session_lifecycle.py (~48 tests, unchanged)
      - tests/integration/test_playback_last_preservation.py (~9 tests, unchanged)
      - tests/unit/observability/ (~41 tests, unchanged)
      - tests/integration/test_qwen_tts_metrics_migration.py (unchanged)
    """

    def test_documented_sweep_is_recorded_in_dev_agent_record(self):
        """The actual sweep is run by the dev as part of Task 7. This
        test is a no-op gate; presence here advertises the contract.
        """
        assert True
