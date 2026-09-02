"""Story 20.7 — the consumer half: Generate is gated while compile priming
holds the TTS request slot, and comes back afterwards.

The service-side declaration and its release-on-every-path obligation live in
``tests/unit/services/test_compile_priming_generate_gate.py``. This file covers
the two things only the UI side can answer:

* **AC #4 — the overlap.** Two independent producers can block the Generate
  button: the user's own generation (``set_generation_status``) and startup
  compile priming (``set_engine_priming``). On a real launch they overlap in
  both orders — a user who presses Generate a beat before priming ends, and
  priming that ends a beat after a generation starts. Whichever finished last
  must not decide for the one still running, so the button state is *derived*
  from both flags rather than assigned by either.
* **The wiring.** ``app.py`` must actually connect the service's declaration to
  the window, and must survive being called before the window exists and by a
  consumer that raises.
"""

from __future__ import annotations

import inspect

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


# ==========================================================================
# AC #1 — the gate disables and re-enables Generate
# ==========================================================================


def test_priming_disables_generate_and_releasing_re_enables_it(window):
    assert window.generate_button.isEnabled()

    window.set_engine_priming(True)
    assert not window.generate_button.isEnabled()

    window.set_engine_priming(False)
    assert window.generate_button.isEnabled(), (
        "the Generate button did not come back after priming released — the "
        "one failure Story 20.7 AC #2 exists to prevent"
    )


def test_set_engine_priming_is_idempotent(window):
    """The producer's release runs in a ``finally``; a duplicate False (or a
    False with no preceding True, e.g. a consumer wired mid-prime) must be a
    no-op rather than a state corruption."""
    window.set_engine_priming(False)
    assert window.generate_button.isEnabled()

    window.set_engine_priming(True)
    window.set_engine_priming(True)
    assert not window.generate_button.isEnabled()

    window.set_engine_priming(False)
    window.set_engine_priming(False)
    assert window.generate_button.isEnabled()


def test_priming_does_not_disturb_the_generation_visuals(window):
    """AC #4 — priming is not a generation. It must not steal the spinner
    icon, the "Generating speech..." tooltip, or the Clear/Stop mode."""
    tooltip_before = window.generate_button.toolTip()

    window.set_engine_priming(True)
    assert window._is_generating is False
    assert window.generate_button.toolTip() == tooltip_before

    window.set_engine_priming(False)
    assert window.generate_button.toolTip() == tooltip_before


# ==========================================================================
# AC #4 — the two producers overlap, in both orders
# ==========================================================================


def test_generation_ending_during_priming_leaves_generate_disabled(window):
    """Order A — the user presses Generate while priming is still running
    (the request queues behind the semaphore, which is the defect), and that
    generation completes first. Priming still owns the slot, so the button
    must stay disabled."""
    window.set_engine_priming(True)
    window.set_generation_status("Generating...", True)
    assert not window.generate_button.isEnabled()

    window.set_generation_status("Ready", False)
    assert not window.generate_button.isEnabled(), (
        "a finished generation re-enabled Generate while compile priming "
        "still holds the request semaphore"
    )

    window.set_engine_priming(False)
    assert window.generate_button.isEnabled()


def test_priming_ending_during_a_generation_leaves_generate_disabled(window):
    """Order B — priming releases while the user's own generation is in
    flight. The pre-existing ``is_generating`` gate must survive it."""
    window.set_generation_status("Generating...", True)
    window.set_engine_priming(True)
    assert not window.generate_button.isEnabled()

    window.set_engine_priming(False)
    assert not window.generate_button.isEnabled(), (
        "priming's release re-enabled Generate mid-generation — AC #4"
    )

    window.set_generation_status("Ready", False)
    assert window.generate_button.isEnabled()


def test_the_normal_generation_path_is_unchanged(window):
    """AC #4 — with priming never engaged, ``set_generation_status`` behaves
    exactly as before: disable + spinner + tooltip on entry, restore on
    exit."""
    window.set_generation_status("Generating...", True)
    assert not window.generate_button.isEnabled()
    assert window._is_generating is True
    assert window.generate_button.toolTip() == "Generating speech..."

    window.set_generation_status("Ready", False)
    assert window.generate_button.isEnabled()
    assert window._is_generating is False
    assert window.generate_button.toolTip() == "Generate speech (Enter)"


def test_no_producer_assigns_the_button_state_directly():
    """Source invariant for AC #4.

    ``_refresh_generate_enabled`` is the single owner of the Generate
    button's enabled state. A producer that calls ``setEnabled`` itself has
    re-opened exactly the last-writer-wins bug the two overlap tests above
    describe — and a same-order test run would not necessarily catch it.
    """
    from myvoice.ui import main_window as mw

    source = inspect.getsource(mw.MainWindow)
    lines = [
        ln.strip() for ln in source.splitlines()
        if "generate_button.setEnabled" in ln
    ]
    assert len(lines) == 1, (
        f"the Generate button's enabled state is assigned from "
        f"{len(lines)} places: {lines}. It must be derived once, in "
        f"_refresh_generate_enabled, from both _is_generating and "
        f"_is_priming."
    )
    owner = inspect.getsource(mw.MainWindow._refresh_generate_enabled)
    assert "generate_button.setEnabled" in owner
    assert "_is_generating" in owner and "_is_priming" in owner


# ==========================================================================
# The orchestrator wiring
# ==========================================================================


def test_app_wires_the_service_declaration_to_the_window():
    """The gate is inert unless ``app.py`` connects it, and the connection
    sits in the same block as the pre-existing preparing-voice wiring."""
    from myvoice import app as app_module

    source = inspect.getsource(app_module.MyVoiceApp)
    assert "set_compile_priming_callback" in source, (
        "app.py never wires the compile-priming declaration; the service "
        "would declare into the void and Generate would never be gated."
    )
    handler = inspect.getsource(
        app_module.MyVoiceApp._on_tts_compile_priming_changed
    )
    assert "set_engine_priming" in handler


def test_the_orchestrator_handler_survives_no_window_and_a_raising_window(
    qapp,
):
    """AC #2 — the producer calls this from a ``finally``. It must not raise
    when the main window does not exist yet (priming is fired from startup),
    nor when the window itself blows up."""
    from myvoice.app import MyVoiceApp

    app = MyVoiceApp(qapp)
    app._main_window = None
    app._on_tts_compile_priming_changed(True)   # must not raise
    app._on_tts_compile_priming_changed(False)

    class _Explodes:
        def set_engine_priming(self, _):
            raise RuntimeError("Qt is having a day")

    app._main_window = _Explodes()
    app._on_tts_compile_priming_changed(True)
    app._on_tts_compile_priming_changed(False)


def test_end_to_end_service_declaration_reaches_the_button(window, qapp):
    """Producer -> orchestrator -> window, through the real callback the app
    wires, with no mocks in the middle."""
    from myvoice.app import MyVoiceApp
    from myvoice.models.app_settings import AppSettings
    from myvoice.services.qwen_tts_service import QwenTTSService

    app = MyVoiceApp(qapp)
    app._main_window = window
    service = QwenTTSService(
        device="cpu", dtype="float32", app_settings=AppSettings()
    )
    service.set_compile_priming_callback(
        app._on_tts_compile_priming_changed
    )

    service._set_compile_priming_active(True)
    assert not window.generate_button.isEnabled()

    service._set_compile_priming_active(False)
    assert window.generate_button.isEnabled()


@pytest.mark.asyncio
async def test_the_task_boundary_release_is_a_second_guarantee(qapp, window):
    """AC #2, belt and braces — ``_compile_warmup_entrypoint`` releases the
    gate in its own ``finally``.

    The service already releases around each priming call. This covers what
    that cannot: something going wrong *outside* those blocks. Simulated by
    declaring the gate and then raising from the warmup coroutine itself.
    """
    from myvoice.app import MyVoiceApp

    app = MyVoiceApp(qapp)
    app._main_window = window
    app._voice_clone_prompt_hydration_task = None

    async def _declares_then_explodes():
        window.set_engine_priming(True)
        raise RuntimeError("blew up outside the service's own finally")

    with pytest.raises(RuntimeError):
        await app._compile_warmup_entrypoint(_declares_then_explodes)

    assert window.generate_button.isEnabled(), (
        "the warmup task unwound with the Generate gate still engaged"
    )


@pytest.mark.asyncio
async def test_the_task_boundary_release_survives_cancellation(qapp, window):
    """AC #2 — the warmup task is fire-and-forget and can be cancelled at
    shutdown or by the qasync scheduling path. Cancellation mid-prime must
    not be the one way the button stays dead."""
    import asyncio

    from myvoice.app import MyVoiceApp

    app = MyVoiceApp(qapp)
    app._main_window = window
    app._voice_clone_prompt_hydration_task = None
    started = asyncio.Event()

    async def _declares_then_hangs():
        window.set_engine_priming(True)
        started.set()
        await asyncio.sleep(3600)

    task = asyncio.ensure_future(
        app._compile_warmup_entrypoint(_declares_then_hangs)
    )
    await started.wait()
    assert not window.generate_button.isEnabled()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert window.generate_button.isEnabled()
