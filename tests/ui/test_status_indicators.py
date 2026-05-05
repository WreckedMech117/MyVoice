"""
Tests for Story 7.4: Status Indicators

Tests the TTS and virtual microphone status indicators including:
- Emoji-based indicators (FR42, FR43)
- Loading state for TTS model loading
- Pulsing animation during generation
- Virtual mic setup dialog (FR44)
- Tooltips and click handlers

Story 12.1 — `TestSessionRegistryIndicatorWiring` and the migrated
``TestMainWindowStatusIndicators`` tests verify the rewire from the
legacy callback chain (``set_generation_*`` / ``set_playback_active``)
to ``SessionRegistry``-driven substate. The other test groups continue
to exercise the widget itself, which Story 12.1 does not touch.

Test Suite Layout (Story 12.3)
------------------------------

After Story 12.3 this file is the single, authoritative regression net
for OFR-D (the registry-driven indicator contract). The class layout is:

* Widget-internal — exercise the widget's own emoji/dot/health rendering.
  Story 12.x does not touch these classes.

  - ``TestServiceStatusIndicatorEmoji``
  - ``TestServiceStatusIndicatorDotMode``
  - ``TestServiceStatusIndicatorPulsing``
  - ``TestServiceStatusIndicatorLoading``
  - ``TestServiceStatusIndicatorClick``
  - ``TestServiceStatusIndicatorTooltip``
  - ``TestServiceStatusIndicatorCleanup``
  - ``TestServiceStatusBar``
  - ``TestVirtualMicSetupDialog``

* ``TestMainWindowStatusIndicators`` — main-window-level tests. Story 12.1
  migrated three pulsing tests onto the registry path; the
  health/loading/mic tests stay on the legacy path (carved out by AC #1
  of Story 12.1 and unchanged in 12.3).

* ``TestSessionRegistryIndicatorWiring`` — the per-OFR-D-bug regression
  suite. Covers all four AC #6 lifecycle scenarios (idle → done,
  cancel-mid-gen, error-mid-gen, rapid-fire focal handoff), the
  AI-Review H3 race regression, and — newly added in Story 12.3 — the
  5-second focal-decay path and the strict-equality focal-handoff
  contract enabled by Story 12.2's ``time.perf_counter()`` migration.
"""

import time

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt

from myvoice.ui.components.queue_depth_badge import QueueDepthBadge
from myvoice.ui.components.service_status_indicator import (
    ServiceStatusIndicator,
    ServiceStatusBar,
    STATUS_EMOJI,
    LOADING_EMOJI,
)
from myvoice.ui.components.virtual_mic_setup_dialog import VirtualMicSetupDialog
from myvoice.models.ui_state import ServiceStatusInfo, ServiceHealthStatus
from myvoice.models.service_enums import ServiceStatus
# Story 12.3 Task 6: monkeypatch target for the registry's perf_counter clock
# (the focal-decay timer test fast-forwards the registry's wall-clock without
# ``time.sleep(5)``).
from myvoice.services.sessions import session_registry as registry_module
# AI-Review M3 (Story 12.3): assert the timer interval against the same
# derived constant the production code uses, so a drift in
# ``_FOCAL_DECAY_SECONDS`` propagates symmetrically.
from myvoice.ui.main_window import _FOCAL_DECAY_TIMER_INTERVAL_MS


@pytest.fixture
def app(qtbot):
    """Ensure QApplication exists."""
    return QApplication.instance() or QApplication([])


@pytest.fixture
def indicator(qtbot):
    """Create a ServiceStatusIndicator widget."""
    widget = ServiceStatusIndicator("TTS", use_emoji=True)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def indicator_dot_mode(qtbot):
    """Create a ServiceStatusIndicator widget in dot mode."""
    widget = ServiceStatusIndicator("TTS", use_emoji=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def status_bar(qtbot):
    """Create a ServiceStatusBar widget."""
    widget = ServiceStatusBar(use_emoji=True)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def badge(qtbot):
    """Create a QueueDepthBadge widget (Story 13.4)."""
    widget = QueueDepthBadge()
    qtbot.addWidget(widget)
    return widget


class TestServiceStatusIndicatorEmoji:
    """Tests for emoji-based status indicators."""

    def test_indicator_created_with_emoji_mode(self, indicator):
        """Indicator should be created in emoji mode by default."""
        assert indicator._use_emoji is True

    def test_indicator_shows_unknown_emoji_initially(self, indicator):
        """Indicator should show unknown emoji (⚪) initially."""
        assert indicator._status_dot.text() == STATUS_EMOJI[ServiceHealthStatus.UNKNOWN]

    def test_indicator_shows_green_emoji_when_healthy(self, indicator):
        """Indicator should show green emoji (🟢) when healthy."""
        status_info = ServiceStatusInfo(
            service_name="TTS",
            status=ServiceStatus.RUNNING,
            health_status=ServiceHealthStatus.HEALTHY,
        )
        indicator.update_status(status_info)
        assert indicator._status_dot.text() == STATUS_EMOJI[ServiceHealthStatus.HEALTHY]

    def test_indicator_shows_yellow_emoji_when_loading(self, indicator):
        """Indicator should show yellow emoji (🟡) when loading."""
        indicator.set_loading(True)
        assert indicator._status_dot.text() == LOADING_EMOJI

    def test_indicator_shows_red_emoji_when_error(self, indicator):
        """Indicator should show red emoji (🔴) when error."""
        status_info = ServiceStatusInfo(
            service_name="TTS",
            status=ServiceStatus.ERROR,
            health_status=ServiceHealthStatus.ERROR,
        )
        indicator.update_status(status_info)
        assert indicator._status_dot.text() == STATUS_EMOJI[ServiceHealthStatus.ERROR]

    def test_indicator_shows_warning_emoji_when_warning(self, indicator):
        """Indicator should show warning emoji (⚠️) when warning."""
        status_info = ServiceStatusInfo(
            service_name="TTS",
            status=ServiceStatus.RUNNING,
            health_status=ServiceHealthStatus.WARNING,
        )
        indicator.update_status(status_info)
        assert indicator._status_dot.text() == STATUS_EMOJI[ServiceHealthStatus.WARNING]

    def test_loading_state_overrides_health_status(self, indicator):
        """Loading state should override health status display."""
        # Set healthy status
        status_info = ServiceStatusInfo(
            service_name="TTS",
            status=ServiceStatus.RUNNING,
            health_status=ServiceHealthStatus.HEALTHY,
        )
        indicator.update_status(status_info)
        assert indicator._status_dot.text() == STATUS_EMOJI[ServiceHealthStatus.HEALTHY]

        # Enable loading - should show yellow
        indicator.set_loading(True)
        assert indicator._status_dot.text() == LOADING_EMOJI

        # Disable loading - should return to green
        indicator.set_loading(False)
        assert indicator._status_dot.text() == STATUS_EMOJI[ServiceHealthStatus.HEALTHY]


class TestServiceStatusIndicatorDotMode:
    """Tests for dot-mode (non-emoji) indicators."""

    def test_indicator_created_in_dot_mode(self, indicator_dot_mode):
        """Indicator should be created in dot mode."""
        assert indicator_dot_mode._use_emoji is False

    def test_indicator_uses_pixmap_in_dot_mode(self, indicator_dot_mode):
        """Indicator should use pixmap in dot mode."""
        # In dot mode, the status dot should have a pixmap, not text
        assert indicator_dot_mode._status_dot.pixmap() is not None


class TestServiceStatusIndicatorPulsing:
    """Tests for pulsing animation during generation."""

    def test_pulsing_disabled_by_default(self, indicator):
        """Pulsing should be disabled by default."""
        assert indicator._is_pulsing is False
        assert indicator.is_pulsing() is False

    def test_pulsing_can_be_enabled(self, indicator):
        """Pulsing should be enabled when set_pulsing(True) is called."""
        indicator.set_pulsing(True)
        assert indicator._is_pulsing is True
        assert indicator.is_pulsing() is True

    def test_pulsing_can_be_disabled(self, indicator):
        """Pulsing should be disabled when set_pulsing(False) is called."""
        indicator.set_pulsing(True)
        indicator.set_pulsing(False)
        assert indicator._is_pulsing is False
        assert indicator.is_pulsing() is False

    def test_pulsing_timer_starts_when_enabled(self, indicator):
        """Pulse timer should start when pulsing is enabled."""
        indicator.set_pulsing(True)
        assert indicator._pulse_timer is not None
        assert indicator._pulse_timer.isActive() is True

    def test_pulsing_timer_stops_when_disabled(self, indicator):
        """Pulse timer should stop when pulsing is disabled."""
        indicator.set_pulsing(True)
        indicator.set_pulsing(False)
        assert indicator._pulse_timer.isActive() is False


class TestServiceStatusIndicatorLoading:
    """Tests for loading state indicator."""

    def test_loading_disabled_by_default(self, indicator):
        """Loading should be disabled by default."""
        assert indicator._is_loading is False
        assert indicator.is_loading() is False

    def test_loading_can_be_enabled(self, indicator):
        """Loading should be enabled when set_loading(True) is called."""
        indicator.set_loading(True)
        assert indicator._is_loading is True
        assert indicator.is_loading() is True

    def test_loading_can_be_disabled(self, indicator):
        """Loading should be disabled when set_loading(False) is called."""
        indicator.set_loading(True)
        indicator.set_loading(False)
        assert indicator._is_loading is False
        assert indicator.is_loading() is False


class TestServiceStatusIndicatorClick:
    """Tests for indicator click handling."""

    def test_click_emits_signal(self, indicator, qtbot):
        """Clicking indicator should emit status_clicked signal."""
        with qtbot.waitSignal(indicator.status_clicked, timeout=1000) as blocker:
            qtbot.mouseClick(indicator, Qt.MouseButton.LeftButton)
        assert blocker.args == ["TTS"]


class TestServiceStatusIndicatorTooltip:
    """Tests for indicator tooltips."""

    def test_initial_tooltip_shows_unknown(self, indicator):
        """Initial tooltip should indicate unknown status."""
        assert "unknown" in indicator.toolTip().lower()

    def test_tooltip_updates_on_status_change(self, indicator):
        """Tooltip should update when status changes."""
        status_info = ServiceStatusInfo(
            service_name="TTS",
            status=ServiceStatus.RUNNING,
            health_status=ServiceHealthStatus.HEALTHY,
        )
        indicator.update_status(status_info)
        assert "TTS" in indicator.toolTip()


class TestServiceStatusIndicatorCleanup:
    """Tests for indicator cleanup."""

    def test_cleanup_stops_timers(self, indicator):
        """Cleanup should stop all timers."""
        indicator.set_pulsing(True)
        indicator.cleanup()
        assert indicator._tooltip_timer.isActive() is False
        assert indicator._pulse_timer.isActive() is False


class TestServiceStatusBar:
    """Tests for ServiceStatusBar widget."""

    def test_status_bar_created_with_emoji_mode(self, status_bar):
        """Status bar should be created with emoji mode."""
        assert status_bar._use_emoji is True

    def test_add_service_creates_indicator(self, status_bar):
        """Adding a service should create an indicator."""
        indicator = status_bar.add_service("TTS")
        assert indicator is not None
        assert "TTS" in status_bar.get_service_names()

    def test_add_service_uses_emoji_mode(self, status_bar):
        """Added service should use emoji mode from status bar."""
        indicator = status_bar.add_service("TTS")
        assert indicator._use_emoji is True

    def test_get_indicator_returns_correct_indicator(self, status_bar):
        """get_indicator should return the correct indicator."""
        status_bar.add_service("TTS")
        indicator = status_bar.get_indicator("TTS")
        assert indicator is not None
        assert indicator.service_name == "TTS"

    def test_get_indicator_returns_none_for_unknown(self, status_bar):
        """get_indicator should return None for unknown service."""
        indicator = status_bar.get_indicator("Unknown")
        assert indicator is None

    def test_set_service_loading(self, status_bar):
        """set_service_loading should update indicator loading state."""
        status_bar.add_service("TTS")
        status_bar.set_service_loading("TTS", True)
        indicator = status_bar.get_indicator("TTS")
        assert indicator.is_loading() is True

    def test_set_service_pulsing(self, status_bar):
        """set_service_pulsing should update indicator pulsing state."""
        status_bar.add_service("TTS")
        status_bar.set_service_pulsing("TTS", True)
        indicator = status_bar.get_indicator("TTS")
        assert indicator.is_pulsing() is True


class TestQueueDepthBadge:
    """Story 13.4: Tests for the QueueDepthBadge widget.

    The badge is a sink for ``SessionRegistry.playback_queue_depth_changed``
    (the queue→registry forwarding lives in ``app.py:338-340``). Driven by
    ``MainWindow._on_playback_queue_depth_changed``; the badge itself emits
    nothing.

    Visibility note: ``qtbot.addWidget(badge)`` re-parents the badge to a
    hidden test widget; ``isVisible()`` returns False unless an ancestor is
    shown. The widget tests therefore assert against ``badge._depth`` and
    ``badge.text()`` (which ``set_depth`` calls regardless of parent
    visibility) and use ``isVisible()`` only where the visibility change is
    a synchronous side-effect of a ``hide()`` call (depth==0).
    """

    def test_badge_initial_state_is_hidden_with_depth_zero(self, badge):
        """Fresh badge: depth defaults to 0 and the widget is hidden."""
        assert badge.depth == 0
        assert badge.isVisible() is False

    def test_set_depth_zero_hides_badge(self, badge, qtbot):
        """``set_depth(0)`` from any prior depth hides the badge."""
        badge.set_depth(2)
        assert badge.depth == 2
        assert badge.text().endswith("2")
        badge.set_depth(0)
        assert badge.depth == 0
        assert badge.isVisible() is False

    def test_set_depth_one_shows_badge_with_digit(self, badge):
        """``set_depth(1)`` puts the digit "1" into the badge text."""
        badge.set_depth(1)
        assert badge.depth == 1
        assert badge.text().endswith("1")

    def test_set_depth_increment_updates_text_no_flicker(self, badge, monkeypatch):
        """Depth N → N+1: text updates; the badge does not hide-then-show.

        ``set_depth`` must call ``setText`` and ``show()`` (idempotent) on
        every non-zero depth — never ``hide()`` between the two calls.

        Spies on ``badge.hide`` and ``badge.show`` for the second
        ``set_depth`` call. ``isHidden()`` after-the-fact would miss a
        ``hide()`` → ``show()`` flicker (the trailing state would still be
        not-hidden); the spy catches the call directly.
        """
        badge.set_depth(2)
        assert badge.text().endswith("2")
        assert badge.isHidden() is False

        hide_calls: list[None] = []
        show_calls: list[None] = []
        original_hide = badge.hide
        original_show = badge.show

        def spy_hide() -> None:
            hide_calls.append(None)
            original_hide()

        def spy_show() -> None:
            show_calls.append(None)
            original_show()

        monkeypatch.setattr(badge, "hide", spy_hide)
        monkeypatch.setattr(badge, "show", spy_show)

        badge.set_depth(3)

        assert badge.text().endswith("3")
        assert badge.depth == 3
        assert badge.isHidden() is False
        # No hide() between the two depths >= 1 — that's the flicker contract.
        assert hide_calls == [], (
            f"set_depth(N→N+1) regressed: hide() was called {len(hide_calls)} time(s)"
        )
        # show() is called once per set_depth(>=1) (idempotent on already-visible).
        assert len(show_calls) == 1, (
            f"set_depth(N→N+1) called show() {len(show_calls)} times; expected 1"
        )

    def test_set_depth_negative_clamps_to_zero(self, badge):
        """``set_depth(-1)`` clamps to 0, hides the badge, raises nothing."""
        badge.set_depth(-1)
        assert badge.depth == 0
        assert badge.isVisible() is False

    def test_set_depth_negative_logs_warning(self, badge, caplog):
        """``set_depth(-1)`` logs a ``WARNING`` so a regression is visible."""
        import logging as _logging
        caplog.set_level(_logging.WARNING, logger="QueueDepthBadge")
        badge.set_depth(-3)
        warnings = [r for r in caplog.records if r.levelno == _logging.WARNING]
        assert any("clamping" in r.getMessage().lower() for r in warnings)

    def test_set_depth_large_value_displays_verbatim(self, badge):
        """Large depth (e.g. 999) is rendered verbatim — no "99+" capping."""
        badge.set_depth(999)
        assert badge.text().endswith("999")
        assert badge.depth == 999

    def test_set_depth_zero_from_zero_is_idempotent(self, badge, monkeypatch):
        """Two consecutive ``set_depth(0)`` calls leave state unchanged.

        Spy on ``setText`` to verify the second call does not push redundant
        text. The contract: depth==0 ⇒ hidden + no text update; this test
        pins the "no work" behavior so a future regression that, e.g.,
        toggles visibility or pushes redundant text on every signal would
        surface.
        """
        text_calls: list[str] = []
        original_set_text = badge.setText

        def spy_set_text(t: str) -> None:
            text_calls.append(t)
            original_set_text(t)

        monkeypatch.setattr(badge, "setText", spy_set_text)

        badge.set_depth(0)
        first_text_count = len(text_calls)
        badge.set_depth(0)
        # Second call must not push more text than the first (contract:
        # no text update on depth==0). Hidden state and depth are
        # synchronously stable, so no ``hide`` spy is needed here.
        assert len(text_calls) == first_text_count
        assert badge.depth == 0
        assert badge.isVisible() is False

    def test_main_window_slot_drives_badge(self, qtbot):
        """AC #5/#6 integration: registry signal → slot → badge.

        Mirrors the ``TestSessionRegistryIndicatorWiring`` fixture pattern.
        Drives ``registry.playback_queue_depth_changed`` directly to exercise
        the full ``QueuedConnection`` chain — signal → slot →
        ``self._queue_depth_badge.set_depth`` → visible badge text.
        """
        from myvoice.ui.main_window import MainWindow
        from myvoice.services.sessions.session_registry import SessionRegistry
        registry = SessionRegistry()
        window = MainWindow(session_registry=registry)
        qtbot.addWidget(window)
        # Badge constructed during _create_ui (AC #6).
        assert window._queue_depth_badge is not None
        # AC #6: badge is added to the status_bar (not just constructed).
        assert window._queue_depth_badge.parent() is window.status_bar
        # AC #6: badge is added BEFORE service_status_bar so it lands LEFT of
        # the service indicators (QStatusBar lays permanent widgets out
        # left-to-right in insertion order; see story Tricky Bit #1). Test the
        # insertion order directly via QObject parent-child registration —
        # this is layout-free (no ``show()``/``adjustSize()`` needed) and
        # holds across Qt versions independent of geometry-computation
        # timing.
        children = window.status_bar.children()
        assert children.index(window._queue_depth_badge) < children.index(
            window.service_status_bar
        ), "badge must be added to status_bar BEFORE service_status_bar"
        # Initial state: depth = 0.
        assert window._queue_depth_badge.depth == 0
        # Emit depth=2; QueuedConnection means we must drain the event
        # loop before asserting on the slot side-effect.
        registry.playback_queue_depth_changed.emit(2)
        qtbot.waitUntil(lambda: window._queue_depth_badge.depth == 2, timeout=1000)
        assert window._queue_depth_badge.text().endswith("2")
        # Drain to 0; depth attribute reverts and digit is gone.
        registry.playback_queue_depth_changed.emit(0)
        qtbot.waitUntil(lambda: window._queue_depth_badge.depth == 0, timeout=1000)


class TestVirtualMicSetupDialog:
    """Tests for virtual microphone setup dialog."""

    def test_dialog_created(self, qtbot):
        """Dialog should be created successfully."""
        dialog = VirtualMicSetupDialog()
        qtbot.addWidget(dialog)
        assert dialog is not None

    def test_dialog_has_title(self, qtbot):
        """Dialog should have appropriate title."""
        dialog = VirtualMicSetupDialog()
        qtbot.addWidget(dialog)
        assert "Virtual Microphone" in dialog.windowTitle()

    def test_dialog_is_modal(self, qtbot):
        """Dialog should be modal."""
        dialog = VirtualMicSetupDialog()
        qtbot.addWidget(dialog)
        assert dialog.isModal() is True

    def test_dialog_has_vb_cable_section(self, qtbot):
        """Dialog should have VB-Audio Cable instructions."""
        dialog = VirtualMicSetupDialog()
        qtbot.addWidget(dialog)
        # Check that VB-Cable section exists by looking for the group box
        from PyQt6.QtWidgets import QGroupBox
        group_boxes = dialog.findChildren(QGroupBox)
        vb_cable_found = any("VB-Audio Cable" in gb.title() for gb in group_boxes)
        assert vb_cable_found is True

    def test_dialog_has_voicemeeter_section(self, qtbot):
        """Dialog should have Voicemeeter instructions."""
        dialog = VirtualMicSetupDialog()
        qtbot.addWidget(dialog)
        from PyQt6.QtWidgets import QGroupBox
        group_boxes = dialog.findChildren(QGroupBox)
        voicemeeter_found = any("Voicemeeter" in gb.title() for gb in group_boxes)
        assert voicemeeter_found is True

    def test_dialog_has_myvoice_config_section(self, qtbot):
        """Dialog should have MyVoice configuration section."""
        dialog = VirtualMicSetupDialog()
        qtbot.addWidget(dialog)
        from PyQt6.QtWidgets import QGroupBox
        group_boxes = dialog.findChildren(QGroupBox)
        config_found = any("MyVoice Configuration" in gb.title() for gb in group_boxes)
        assert config_found is True

    def test_dialog_has_close_button(self, qtbot):
        """Dialog should have a close button."""
        dialog = VirtualMicSetupDialog()
        qtbot.addWidget(dialog)
        from PyQt6.QtWidgets import QPushButton
        buttons = dialog.findChildren(QPushButton)
        close_found = any("Close" in btn.text() for btn in buttons)
        assert close_found is True

    @patch("webbrowser.open")
    def test_download_button_opens_url(self, mock_webbrowser, qtbot):
        """Download button should open URL in browser."""
        dialog = VirtualMicSetupDialog()
        qtbot.addWidget(dialog)

        from PyQt6.QtWidgets import QPushButton
        buttons = dialog.findChildren(QPushButton)
        download_buttons = [btn for btn in buttons if "Download" in btn.text()]

        # Click the first download button
        if download_buttons:
            download_buttons[0].click()
            mock_webbrowser.assert_called_once()


class TestMainWindowStatusIndicators:
    """Tests for status indicators in MainWindow.

    Story 12.1: the three pulsing tests below were migrated from driving
    the indicator via ``set_generation_pulsing`` / ``set_generation_status``
    (legacy callback chain) to driving it via ``SessionRegistry`` mutations
    (the new substate source-of-truth per AC #1). The other tests in this
    class — health/loading and virtual-mic — are unchanged because Story
    12.1 explicitly carves them out (AC #1 keeps the health path intact).
    """

    @pytest.fixture
    def main_window(self, qtbot):
        """Create a MainWindow instance (legacy path — no registry).

        Used by tests that exercise health, loading, mic, and structural
        properties, which are unaffected by Story 12.1's rewire.
        """
        from myvoice.ui.main_window import MainWindow
        window = MainWindow()
        qtbot.addWidget(window)
        return window

    @pytest.fixture
    def registry_main_window(self, qtbot):
        """Story 12.1: MainWindow wired to a real SessionRegistry.

        Used by the three migrated pulsing tests (and shared with the
        ``TestSessionRegistryIndicatorWiring`` class below). The registry
        runs on the Qt main thread — same as production — and the
        QueuedConnection between registry and window means tests must
        cycle the event loop after each mutation before asserting.
        """
        from myvoice.ui.main_window import MainWindow
        from myvoice.services.sessions.session_registry import SessionRegistry
        registry = SessionRegistry()
        window = MainWindow(session_registry=registry)
        qtbot.addWidget(window)
        return window, registry

    def test_main_window_has_status_bar(self, main_window):
        """MainWindow should have a service status bar."""
        assert main_window.service_status_bar is not None

    def test_main_window_has_tts_indicator(self, main_window):
        """MainWindow should have a TTS indicator."""
        indicator = main_window.service_status_bar.get_indicator("TTS")
        assert indicator is not None

    def test_main_window_has_mic_indicator(self, main_window):
        """MainWindow should have a Mic indicator."""
        indicator = main_window.service_status_bar.get_indicator("Mic")
        assert indicator is not None

    def test_set_tts_loading(self, main_window):
        """set_tts_loading should update TTS indicator loading state."""
        main_window.set_tts_loading(True)
        indicator = main_window.service_status_bar.get_indicator("TTS")
        assert indicator.is_loading() is True

    def test_set_generation_pulsing(self, registry_main_window, qtbot):
        """Story 12.1: pulsing is enabled when the registry's focal session
        enters GENERATING — replaces the legacy ``set_generation_pulsing``
        path which is now a no-op when a registry is wired (AC #1).
        """
        window, registry = registry_main_window
        sid = registry.create_session(
            text="Test", voice="default", model_type="qwen3"
        )
        registry.start_generation(sid)
        indicator = window.service_status_bar.get_indicator("TTS")
        qtbot.waitUntil(lambda: indicator.is_pulsing(), timeout=1000)
        assert indicator.is_pulsing() is True

    def test_generation_status_enables_pulsing(self, registry_main_window, qtbot):
        """Story 12.1: GENERATING state enables pulsing (replaces legacy
        ``set_generation_status(..., is_generating=True)`` semantics).
        """
        window, registry = registry_main_window
        sid = registry.create_session(
            text="Test", voice="default", model_type="qwen3"
        )
        registry.start_generation(sid)
        indicator = window.service_status_bar.get_indicator("TTS")
        qtbot.waitUntil(lambda: indicator.is_pulsing(), timeout=1000)
        assert indicator.is_pulsing() is True

    def test_generation_status_disables_pulsing(self, registry_main_window, qtbot):
        """Story 12.1: PLAYING+is_audible disables pulsing (replaces legacy
        ``set_generation_status(..., is_generating=False)`` semantics —
        the registry-driven equivalent is the audible-playback substate).
        """
        window, registry = registry_main_window
        sid = registry.create_session(
            text="Test", voice="default", model_type="qwen3"
        )
        registry.start_generation(sid)
        # Drive through GENERATING → READY_TO_PLAY → PLAYING → audible.
        registry.append_chunk(sid, np.zeros(1000, dtype=np.float32))
        registry.finalize(sid)
        registry.mark_playing(sid)
        registry.mark_audible(sid)
        indicator = window.service_status_bar.get_indicator("TTS")
        # Pulsing OFF once is_audible is True (AC #2's rule: PLAYING +
        # is_audible == True means "playing", not "working").
        qtbot.waitUntil(lambda: indicator.is_pulsing() is False, timeout=1000)
        assert indicator.is_pulsing() is False

    def test_update_virtual_mic_status_connected(self, main_window):
        """update_virtual_mic_status should show green when connected."""
        main_window.update_virtual_mic_status(True, "VB-Cable Input")
        indicator = main_window.service_status_bar.get_indicator("Mic")
        status_info = indicator.get_current_status()
        assert status_info.health_status == ServiceHealthStatus.HEALTHY

    def test_update_virtual_mic_status_not_connected(self, main_window):
        """update_virtual_mic_status should show warning when not connected."""
        main_window.update_virtual_mic_status(False)
        indicator = main_window.service_status_bar.get_indicator("Mic")
        status_info = indicator.get_current_status()
        assert status_info.health_status == ServiceHealthStatus.WARNING

    def test_virtual_mic_tooltip_when_connected(self, main_window):
        """Virtual mic tooltip should show device name when connected."""
        main_window.update_virtual_mic_status(True, "VB-Cable Input")
        indicator = main_window.service_status_bar.get_indicator("Mic")
        assert "VB-Cable Input" in indicator.toolTip()

    def test_virtual_mic_tooltip_when_not_connected(self, main_window):
        """Virtual mic tooltip should show setup help when not connected."""
        main_window.update_virtual_mic_status(False)
        indicator = main_window.service_status_bar.get_indicator("Mic")
        assert "setup help" in indicator.toolTip().lower() or "not detected" in indicator.toolTip().lower()


class TestSessionRegistryIndicatorWiring:
    """Story 12.1: regression tests for the three OFR-D bugs.

    Each test corresponds to one OFR-D bug report and verifies that the
    registry-driven substate path makes the bug disappear by construction
    (AC #6). The fixture wires a real registry to a real MainWindow on
    the Qt main thread and lets QueuedConnection deliver naturally —
    same conditions as production.
    """

    @pytest.fixture
    def registry_main_window(self, qtbot):
        """Build MainWindow + SessionRegistry shared across all OFR-D tests."""
        from myvoice.ui.main_window import MainWindow
        from myvoice.services.sessions.session_registry import SessionRegistry
        registry = SessionRegistry()
        window = MainWindow(session_registry=registry)
        qtbot.addWidget(window)
        return window, registry

    def test_indicator_shows_playing_until_mark_done(self, registry_main_window, qtbot):
        """OFR-D bug #1: indicator must NOT show "ready" while audio is playing.

        With the registry-driven path, pulsing is OFF once the focal
        session reaches PLAYING + is_audible == True (the indicator is
        no longer "working", it is "playing"). It stays OFF — never
        flipping to a ready/idle frame — until ``mark_done`` fires.

        The waitUntil checks gate on the EXPECTED transition (pulsing
        ON during GENERATING, then OFF once audible) so the queued
        slots actually run before assertions — otherwise a default-
        False ``is_pulsing()`` would falsely satisfy a "wait for off"
        check before any slot fired.
        """
        window, registry = registry_main_window
        indicator = window.service_status_bar.get_indicator("TTS")
        sid = registry.create_session(
            text="Test", voice="default", model_type="qwen3"
        )
        registry.start_generation(sid)
        # Pulsing must turn ON during GENERATING before we proceed —
        # this confirms the queued slot chain is delivering.
        qtbot.waitUntil(lambda: indicator.is_pulsing() is True, timeout=1000)

        registry.append_chunk(sid, np.zeros(1000, dtype=np.float32))
        registry.finalize(sid)
        registry.mark_playing(sid)
        registry.mark_audible(sid)
        # Pulsing OFF once is_audible is True (AC #2's rule: PLAYING +
        # is_audible == True means "playing", not "working").
        qtbot.waitUntil(
            lambda: indicator.is_pulsing() is False
            and "Playing" in window.status_bar.currentMessage(),
            timeout=1000,
        )
        assert indicator.is_pulsing() is False
        # The status text reflects the audible-playback substate, NOT
        # the legacy "Ready" / completed text — this is the bug fix.
        assert "Playing" in window.status_bar.currentMessage()
        # Now flip to DONE and verify no transient "ready" showed up
        # while audio was still playing.
        registry.mark_done(sid)
        qtbot.wait(50)
        # After mark_done, focal session is still terminal-within-decay,
        # so pulsing remains OFF (was OFF, stays OFF — net-zero from
        # before). DISCARD removes the entry.
        assert indicator.is_pulsing() is False
        registry.discard(sid)

    def test_error_state_does_not_show_generating(self, registry_main_window, qtbot):
        """OFR-D bug #2: indicator must NOT show "generating" after an error.

        With the registry-driven path, ``set_error`` transitions the
        focal session into ERROR. The state→pulsing map has no
        "generating" cell for ERROR; pulsing must be OFF and the status
        text must reflect the error flavor.
        """
        window, registry = registry_main_window
        sid = registry.create_session(
            text="Test", voice="default", model_type="qwen3"
        )
        registry.start_generation(sid)
        # Pulsing should be ON during GENERATING
        indicator = window.service_status_bar.get_indicator("TTS")
        qtbot.waitUntil(lambda: indicator.is_pulsing(), timeout=1000)
        # Error mid-generation
        registry.set_error(sid)
        # After ERROR transition, pulsing must be OFF
        qtbot.waitUntil(lambda: indicator.is_pulsing() is False, timeout=1000)
        assert indicator.is_pulsing() is False
        assert "failed" in window.status_bar.currentMessage().lower()

    def test_no_idle_frame_under_discard_then_start_race(self, registry_main_window, qtbot):
        """AI-Review H3 (2026-05-04): regression — when ``discard(A)`` and
        ``start_generation(B)`` are dispatched in sequence on the Qt main
        thread, no transient pulsing-OFF call must reach the status bar.

        The original Story 12.1 implementation painted idle synchronously
        from ``_on_current_session_changed(None)`` — the ``None`` emission
        from ``discard(A)`` would reach the slot before ``B``'s
        ``current_session_changed(B)`` did, briefly toggling pulsing OFF.
        That is exactly the OFR-D bug #3 the rewire is supposed to fix.

        The fix defers the idle paint via ``QTimer.singleShot(0, ...)``
        so a successor's ``GENERATING`` emission is processed first, the
        deferred redraw re-reads ``focal_session_id`` (now ``B``), and
        the off-cycle never happens.

        We spy on ``service_status_bar.set_service_pulsing("TTS", ...)``
        — called from the redraw helper directly — so the indicator
        widget's own ``set_pulsing`` early-out does not mask transient
        OFF calls upstream.
        """
        window, registry = registry_main_window
        indicator = window.service_status_bar.get_indicator("TTS")

        # Drive A through full lifecycle into DONE so A is terminal and
        # focal under tier-(c) before the race window opens.
        sid_a = registry.create_session(
            text="A", voice="default", model_type="qwen3"
        )
        registry.start_generation(sid_a)
        registry.append_chunk(sid_a, np.zeros(1000, dtype=np.float32))
        registry.finalize(sid_a)
        registry.mark_playing(sid_a)
        registry.mark_audible(sid_a)
        registry.mark_done(sid_a)
        # Settle: A is DONE/audible-was-true, pulsing should be OFF.
        qtbot.waitUntil(lambda: indicator.is_pulsing() is False, timeout=1000)

        # Spy on the indicator-driving call. Using set_service_pulsing on
        # the bar (not set_pulsing on the indicator) because the widget
        # short-circuits no-change calls — we want every upstream call.
        pulsing_calls = []
        original = window.service_status_bar.set_service_pulsing

        def hooked(name: str, enabled: bool) -> None:
            if name == "TTS":
                pulsing_calls.append(enabled)
            original(name, enabled)

        window.service_status_bar.set_service_pulsing = hooked

        # Race: discard(A) emits current_session_changed(None); start B
        # immediately so the queue holds both A's DISCARDED/None and B's
        # GENERATING/B emissions. The H3 defer must absorb the (None).
        sid_b = registry.create_session(
            text="B", voice="default", model_type="qwen3"
        )
        registry.discard(sid_a)
        registry.start_generation(sid_b)

        # Wait for B's GENERATING to settle. Pulsing must be ON.
        qtbot.waitUntil(lambda: indicator.is_pulsing() is True, timeout=1000)

        # Restore the original to keep the fixture clean for any teardown.
        window.service_status_bar.set_service_pulsing = original

        # All pulsing calls during the race window must be True. A False
        # entry signals the regression — discard's None emission painted
        # idle synchronously instead of deferring.
        assert all(p is True for p in pulsing_calls), (
            f"H3 regression: idle frame leaked during discard→start race; "
            f"pulsing calls captured: {pulsing_calls}"
        )

    def test_focal_handoff_no_idle_frame(self, registry_main_window, qtbot):
        """OFR-D bug #3: indicator must NOT show idle between rapid generations.

        When session A's generation ends and session B's generation
        begins, the focal session stays in tier-(b) (active session
        — GENERATING/READY_TO_PLAY) without ever passing through None.
        Both READY_TO_PLAY (A) and GENERATING (B) produce
        ``pulsing=True``, so the indicator never flickers to idle —
        the OFR-D bug is gone by construction.

        Story 12.2 migrated ``_last_transition_at`` from ``time.time()``
        to ``time.perf_counter()``; cross-session focal handoff is now
        deterministic on all platforms — ``sid_b`` is the unambiguous
        focal under tier-(b)'s most-recent rule. Story 12.3 (this
        revision) tightens the assertion accordingly: the relaxed
        ``in {sid_a, sid_b}`` form is no longer required, and the
        strict equality form catches a regression in the perf_counter
        substrate as a test failure rather than a silent drift.
        """
        window, registry = registry_main_window
        sid_a = registry.create_session(
            text="A", voice="default", model_type="qwen3"
        )
        registry.start_generation(sid_a)
        registry.append_chunk(sid_a, np.zeros(1000, dtype=np.float32))
        registry.finalize(sid_a)
        # A is now READY_TO_PLAY (tier-b). Start B before A plays —
        # both are in tier-b, indicator must keep pulsing.
        sid_b = registry.create_session(
            text="B", voice="default", model_type="qwen3"
        )
        registry.start_generation(sid_b)
        # Verify pulsing stays ON across the handoff. The waitUntil
        # would still succeed if the indicator briefly went off and
        # came back; the qtbot.wait pump after lets us re-check.
        indicator = window.service_status_bar.get_indicator("TTS")
        qtbot.waitUntil(lambda: indicator.is_pulsing() is True, timeout=1000)
        assert indicator.is_pulsing() is True
        # Story 12.3 tighten (was: `in {sid_a, sid_b}`): tier-(b)'s
        # "most recent" rule promotes B because its start_generation
        # happened strictly after A's finalize. With perf_counter's
        # sub-microsecond resolution there are no ties on any platform.
        assert registry.focal_session_id == sid_b

    # ----- Story 12.3: cancel-mid-generation (AC #1) ------------------- #

    def test_cancel_during_generating_transitions_to_stopped(
        self, registry_main_window, qtbot
    ):
        """Story 12.3 AC #1 (cancel-mid-gen): GENERATING → CANCELLED transitions
        the indicator to the cancelled flavor in one Qt event-loop tick.

        Scope is the *registry-driven indicator contract*: this test
        invokes ``registry.cancel(sid)`` directly (NOT the Stop-button
        handler ``_on_clear_clicked``), because the Stop-button's
        residual-audio behavior is owned by Epic 16 / Story 16.5 and is
        a separate concern from indicator state.
        """
        window, registry = registry_main_window
        indicator = window.service_status_bar.get_indicator("TTS")
        sid = registry.create_session(
            text="cancel-mid-gen", voice="default", model_type="qwen3"
        )
        registry.start_generation(sid)
        # Confirm pulsing turned ON during GENERATING — gates that the
        # queued slot chain is delivering before we exercise cancel.
        qtbot.waitUntil(lambda: indicator.is_pulsing() is True, timeout=1000)

        registry.cancel(sid)
        qtbot.waitUntil(
            lambda: indicator.is_pulsing() is False
            and "Stopped" in window.status_bar.currentMessage(),
            timeout=1000,
        )
        assert indicator.is_pulsing() is False
        assert window.status_bar.currentMessage() == "Stopped"

    def test_cancel_during_audible_playback_no_transient_working_state(
        self, registry_main_window, qtbot
    ):
        """Story 12.3 AC #1 (validation gap #3): cancel during audible
        playback transitions to "Stopped" without any transient pulsing
        flicker through a "working" frame.

        Pulsing was OFF (PLAYING + is_audible == True), must stay OFF
        through the cancel — no upstream True call may reach the bar.
        We spy on ``service_status_bar.set_service_pulsing`` (not the
        widget's own ``set_pulsing``) because the widget short-circuits
        same-value calls and would mask transient flickers.
        """
        window, registry = registry_main_window
        indicator = window.service_status_bar.get_indicator("TTS")

        # Drive to PLAYING + audible — pulsing OFF.
        sid = registry.create_session(
            text="cancel-mid-playback", voice="default", model_type="qwen3"
        )
        registry.start_generation(sid)
        registry.append_chunk(sid, np.zeros(1000, dtype=np.float32))
        registry.finalize(sid)
        registry.mark_playing(sid)
        registry.mark_audible(sid)
        qtbot.waitUntil(
            lambda: indicator.is_pulsing() is False
            and "Playing" in window.status_bar.currentMessage(),
            timeout=1000,
        )

        # Install spies AFTER reaching the audible-playing settle so that
        # only cancel-driven calls land in the capture lists. Mirrors the
        # spy pattern used by ``test_no_idle_frame_under_discard_then_start_race``.
        # AI-Review M4 (Story 12.3): the message spy is the *positive*
        # half of the assertion — without it, an empty ``pulsing_calls``
        # list would make ``all(... is False ...)`` pass vacuously even
        # if cancel never reached the redraw path.
        pulsing_calls: list[bool] = []
        message_calls: list[str] = []
        original_pulse = window.service_status_bar.set_service_pulsing
        original_show = window.status_bar.showMessage

        def hooked_pulse(name: str, enabled: bool) -> None:
            if name == "TTS":
                pulsing_calls.append(enabled)
            original_pulse(name, enabled)

        def hooked_show(text: str, *args, **kwargs) -> None:
            message_calls.append(text)
            original_show(text, *args, **kwargs)

        window.service_status_bar.set_service_pulsing = hooked_pulse
        window.status_bar.showMessage = hooked_show
        try:
            registry.cancel(sid)
            qtbot.waitUntil(
                lambda: "Stopped" in window.status_bar.currentMessage(),
                timeout=1000,
            )
        finally:
            window.service_status_bar.set_service_pulsing = original_pulse
            window.status_bar.showMessage = original_show

        # Positive half (AI-Review M4): cancel must have reached the redraw
        # path — proven by "Stopped" landing in the captured message stream.
        assert "Stopped" in message_calls, (
            f"AC #1 (cancel-during-playback): cancel did not drive a "
            f"'Stopped' message through the indicator; spy captured: "
            f"{message_calls}"
        )
        # Negative half: set_pulsing de-dups same-value calls at the widget
        # level (service_status_indicator.py:249-250) so the captured list
        # may be empty (False→False collapsed) OR all-False — both are
        # valid. A single True entry would be the OFR-D regression.
        assert all(p is False for p in pulsing_calls), (
            f"AC #1 (cancel-during-playback): transient working-state "
            f"flicker leaked through cancel; pulsing calls: {pulsing_calls}"
        )
        assert indicator.is_pulsing() is False
        assert window.status_bar.currentMessage() == "Stopped"

    # ----- Story 12.3: full lifecycle happy-path (AC #2) ---------------- #

    def test_full_lifecycle_idle_to_done_no_intermediate_ready(
        self, registry_main_window, qtbot
    ):
        """Story 12.3 AC #2: idle → generating → ready → playing → audible
        → done proceeds through the indicator without ever transiting
        through the 'Ready' / idle text. The OFR-D bug #1 negative
        assertion at lifecycle scope.

        Note: ``set_pulsing`` de-dups same-value calls at the widget level
        (``service_status_indicator.py:249-250``) so the pulsing-call
        history may collapse repeated True→True entries. The assertion
        is on the *transition pattern* (a True-then-False landmark),
        not a literal call-count or list match.
        """
        window, registry = registry_main_window
        indicator = window.service_status_bar.get_indicator("TTS")

        pulsing_calls: list[bool] = []
        message_calls: list[str] = []
        original_pulse = window.service_status_bar.set_service_pulsing
        original_show = window.status_bar.showMessage

        def hooked_pulse(name: str, enabled: bool) -> None:
            if name == "TTS":
                pulsing_calls.append(enabled)
            original_pulse(name, enabled)

        def hooked_show(text: str, *args, **kwargs) -> None:
            message_calls.append(text)
            original_show(text, *args, **kwargs)

        window.service_status_bar.set_service_pulsing = hooked_pulse
        window.status_bar.showMessage = hooked_show
        try:
            sid = registry.create_session(
                text="lifecycle", voice="default", model_type="qwen3"
            )
            registry.start_generation(sid)
            qtbot.waitUntil(
                lambda: "Generating speech..."
                in window.status_bar.currentMessage(),
                timeout=1000,
            )
            registry.append_chunk(sid, np.zeros(1000, dtype=np.float32))
            registry.finalize(sid)
            qtbot.waitUntil(
                lambda: "Audio ready" in window.status_bar.currentMessage(),
                timeout=1000,
            )
            registry.mark_playing(sid)
            qtbot.waitUntil(
                lambda: "Starting playback..."
                in window.status_bar.currentMessage(),
                timeout=1000,
            )
            registry.mark_audible(sid)
            qtbot.waitUntil(
                lambda: "Playing audio on speakers and virtual microphone"
                in window.status_bar.currentMessage(),
                timeout=1000,
            )
            registry.mark_done(sid)
            qtbot.waitUntil(
                lambda: "Audio playback completed"
                in window.status_bar.currentMessage(),
                timeout=1000,
            )
            registry.discard(sid)
            # discard's current_session_changed(None) is deferred via
            # QTimer.singleShot(0, ...); one event-loop tick suffices for
            # the deferred idle paint to land.
            qtbot.wait(50)
        finally:
            window.service_status_bar.set_service_pulsing = original_pulse
            window.status_bar.showMessage = original_show

        # Subset-in-order: every canonical _STATE_TEXT_MAP string for the
        # happy-path appears in order. Transient strings (e.g., "Ready"
        # from a None-focal redraw before discard) are checked separately
        # below; here we only require that the expected sequence appears.
        expected_sequence = [
            "Generating speech...",
            "Audio ready",
            "Starting playback...",
            "Playing audio on speakers and virtual microphone",
            "Audio playback completed",
        ]
        cursor = 0
        for msg in message_calls:
            if cursor < len(expected_sequence) and msg == expected_sequence[cursor]:
                cursor += 1
        assert cursor == len(expected_sequence), (
            f"AC #2: expected lifecycle text sequence not observed in order. "
            f"Captured messages: {message_calls}"
        )

        # Negative assertion: 'Ready' / 'TTS Unavailable' (the
        # _get_ready_message outputs) must NOT appear before
        # 'Audio playback completed'. The post-discard idle paint that
        # lands during the qtbot.wait(50) is allowed and arrives after.
        final_done_idx = next(
            (i for i, m in enumerate(message_calls)
             if m == "Audio playback completed"),
            -1,
        )
        assert final_done_idx >= 0, (
            "AC #2 setup: 'Audio playback completed' was never observed; "
            "the test's qtbot.waitUntil for it would have failed first."
        )
        pre_done = message_calls[:final_done_idx]
        assert "Ready" not in pre_done and "TTS Unavailable" not in pre_done, (
            f"AC #2: idle text leaked into the lifecycle. "
            f"Messages before 'Audio playback completed': {pre_done}"
        )

        # Pulsing transition pattern: at least one True (working frames),
        # then a False (once audible-playing). De-dup may collapse repeats
        # so we look for the first-True/first-False-after-True landmark.
        assert any(p is True for p in pulsing_calls), (
            f"AC #2: pulsing never turned ON during lifecycle; "
            f"calls: {pulsing_calls}"
        )
        first_true_idx = next(
            i for i, p in enumerate(pulsing_calls) if p is True
        )
        first_false_after_true = next(
            (i for i, p in enumerate(
                pulsing_calls[first_true_idx + 1:], start=first_true_idx + 1
            ) if p is False),
            -1,
        )
        assert first_false_after_true > first_true_idx, (
            f"AC #2: pulsing never transitioned True→False during lifecycle. "
            f"Pulsing calls: {pulsing_calls}"
        )

    # ----- Story 12.3: 5-second focal-decay timer (AC #4) -------------- #

    def test_focal_decay_timer_clears_indicator_after_5s_window(
        self, registry_main_window, qtbot, monkeypatch
    ):
        """Story 12.3 AC #4: after a focal session decays out of tier-(c)
        (terminal-within-window), the indicator paints idle.

        Acceleration: monkeypatch ``registry_module.time.perf_counter`` to
        fast-forward the registry's clock past ``_FOCAL_DECAY_SECONDS``
        instead of ``time.sleep(5)``. Pattern matches Story 12.2's
        ``TestFocalPriorityExplicit`` tests at
        ``tests/unit/services/sessions/test_session_registry.py``.

        We do NOT wait for the real 5050ms QTimer to fire — the timer is
        a *trigger*; the *behavior* under test is
        ``_redraw_tts_indicator_from_focal``'s response to a focal-decayed
        registry. Calling the redraw helper directly after monkeypatching
        the clock yields the same observable behavior in milliseconds.
        """
        window, registry = registry_main_window
        indicator = window.service_status_bar.get_indicator("TTS")

        # Drive to DONE — focal under tier-(c).
        sid = registry.create_session(
            text="decay", voice="default", model_type="qwen3"
        )
        registry.start_generation(sid)
        registry.append_chunk(sid, np.zeros(1000, dtype=np.float32))
        registry.finalize(sid)
        registry.mark_playing(sid)
        registry.mark_audible(sid)
        registry.mark_done(sid)
        qtbot.waitUntil(
            lambda: "Audio playback completed"
            in window.status_bar.currentMessage(),
            timeout=1000,
        )
        # Tier-(c) settle: focal is sid, pulsing OFF, terminal-flavor text.
        assert registry.focal_session_id == sid
        assert indicator.is_pulsing() is False

        # Pin the timer setup so a future "optimization" cannot silently
        # change the headroom and break the AC #4 contract.
        assert window._focal_decay_timer is not None
        assert window._focal_decay_timer.isActive() is True, (
            "AC #4: focal-decay timer must be active after a terminal "
            "transition (started by _on_session_state_changed at "
            "main_window.py:1777-1782)."
        )
        assert window._focal_decay_timer.interval() == _FOCAL_DECAY_TIMER_INTERVAL_MS, (
            f"AC #4: timer headroom drifted from "
            f"{_FOCAL_DECAY_TIMER_INTERVAL_MS}ms; got "
            f"{window._focal_decay_timer.interval()}ms"
        )
        assert window._focal_decay_timer.isSingleShot() is True

        # Fast-forward the registry's perf_counter past the decay window
        # so focal_session_id transitions tier-(c) → tier-(d) (None).
        # Source must match GenerationSession._last_transition_at — both
        # use registry_module.time, so this single monkeypatch is enough.
        future = (
            time.perf_counter() + registry_module._FOCAL_DECAY_SECONDS + 1.0
        )
        monkeypatch.setattr(
            registry_module.time, "perf_counter", lambda: future
        )
        # Sanity: focal returns None now under the future clock.
        assert registry.focal_session_id is None

        # Simulate the timer's timeout slot firing. We invoke the redraw
        # helper directly rather than waiting the literal 5050ms — the
        # timer is the trigger, the redraw is the behavior under test.
        window._redraw_tts_indicator_from_focal()

        # Post-decay paint: pulsing OFF (was OFF, stays OFF) and the
        # status text reflects _get_ready_message (which is "Ready" or
        # "TTS Unavailable" depending on _tts_available; either is a
        # valid idle text under AC #4).
        assert indicator.is_pulsing() is False
        assert window.status_bar.currentMessage() in {"Ready", "TTS Unavailable"}

    def test_focal_decay_timer_started_on_each_terminal_transition(
        self, registry_main_window, qtbot
    ):
        """Story 12.3 AC #4 corollary: ``_focal_decay_timer`` is *restarted*
        on every terminal transition, not only on the first one.

        Without restart, a session DONE → next session DONE would leave
        the timer in whatever state the first DONE left it (already-fired,
        already-stopped, etc.) and the indicator would never observe the
        second tier-(c) → None decay.
        """
        window, registry = registry_main_window
        indicator = window.service_status_bar.get_indicator("TTS")

        # First terminal: timer must become active.
        sid_a = registry.create_session(
            text="A", voice="default", model_type="qwen3"
        )
        registry.start_generation(sid_a)
        registry.append_chunk(sid_a, np.zeros(1000, dtype=np.float32))
        registry.finalize(sid_a)
        registry.mark_playing(sid_a)
        registry.mark_audible(sid_a)
        registry.mark_done(sid_a)
        qtbot.waitUntil(
            lambda: window._focal_decay_timer.isActive() is True,
            timeout=1000,
        )

        # Force the timer inactive (manual stop). A single-shot timer that
        # has already fired ends in the same `isActive() == False` state,
        # so this is observationally equivalent for the restart-on-each-
        # terminal contract under test (AI-Review L7, Story 12.3).
        window._focal_decay_timer.stop()
        assert window._focal_decay_timer.isActive() is False

        # Discard A so it leaves the registry; create and drive B to DONE.
        registry.discard(sid_a)
        sid_b = registry.create_session(
            text="B", voice="default", model_type="qwen3"
        )
        registry.start_generation(sid_b)
        registry.append_chunk(sid_b, np.zeros(1000, dtype=np.float32))
        registry.finalize(sid_b)
        registry.mark_playing(sid_b)
        registry.mark_audible(sid_b)
        registry.mark_done(sid_b)

        # Second terminal: timer must be restarted.
        qtbot.waitUntil(
            lambda: window._focal_decay_timer.isActive() is True,
            timeout=1000,
        )
        assert window._focal_decay_timer.isActive() is True
