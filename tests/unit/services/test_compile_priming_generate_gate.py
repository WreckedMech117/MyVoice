"""Story 20.7 — Generate must not silently queue behind compile priming.

WHY THIS EXISTS
---------------
Compile priming (Stories 20.2/20.3) is a **real generation**. It runs through
the production dispatch chain and therefore takes ``_request_semaphore``, which
``QwenTTSService`` constructs at ``Semaphore(max_concurrent_requests)`` with a
default of **1**. A user who presses Generate during the ~4.4-4.9 s priming
window does not get an error and does not get an explanation — the request
simply waits.

Story 20.6's operator captures measured it twice, in the segment-1a dispatch
interval: 840.2 ms and 1,382.6 ms against ~1.5-3.0 ms on the sixteen clean
generations. Three orders of magnitude of silence.

THE FIX, AND THE THING THAT COULD GO WRONG
------------------------------------------
Producer-declares / consumer-acts, the Story 20.5 shape: the service declares
``_set_compile_priming_active(True/False)`` around the region where priming
holds the semaphore, and the orchestrator gates the Generate button.

The failure mode that matters is **not** the gate failing to engage — it is the
gate failing to release. Priming is explicitly non-fatal: it can raise, it can
be cancelled, it can bail out through ``CompilePrimingSkipped``, it can take an
early ``return`` out of the key/model coherence check, and it can be skipped
entirely by five separate gates. A Generate button that never comes back is
strictly worse than the silent queue this story exists to fix. Hence AC #2, and
hence the bulk of this file: every exit path gets a row, and a source invariant
pins the structural property those rows sample.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import os
import textwrap
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from myvoice.models.app_settings import AppSettings
from myvoice.observability import metrics
from myvoice.services.qwen_tts_service import (
    CompilePrimingSkipped,
    QwenTTSService,
)


# -- Harness (mirrors tests/unit/services/test_qwen_tts_service_compile_warmup.py) --


@pytest.fixture
def metric_records():
    captured: List[metrics.MetricRecord] = []

    def listener(record: metrics.MetricRecord) -> None:
        captured.append(record)

    unsub = metrics.add_listener(listener)
    try:
        yield captured
    finally:
        unsub()


@pytest.fixture
def _clean_priming_env():
    """Neither disable env var may leak in or out of these tests."""
    names = (
        "MYVOICE_DISABLE_COMPILE_WARMUP",
        "MYVOICE_DISABLE_WARM_COMPILE_PRIMING",
    )
    snapshot = {n: os.environ.get(n) for n in names}
    for n in names:
        os.environ.pop(n, None)
    yield
    for n, v in snapshot.items():
        if v is None:
            os.environ.pop(n, None)
        else:
            os.environ[n] = v


def _make_service(*, with_model_registry: bool = True,
                  with_loaded_model: bool = True) -> QwenTTSService:
    service = QwenTTSService(
        device="cpu",
        dtype="float32",
        app_settings=AppSettings(tts_compile="auto"),
    )
    if with_model_registry:
        import torch

        fake_inner = type("FakeInner", (), {})()
        fake_inner.name_or_path = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        fake_inner.dtype = torch.bfloat16
        fake_model = type("FakeModel", (), {})()
        fake_model.model = fake_inner

        registry = MagicMock(name="ModelRegistry")
        registry.get_loaded_model.return_value = (
            fake_model if with_loaded_model else None
        )
        service._model_registry = registry
    else:
        service._model_registry = None
    return service


def _patch_hardware(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "torch.cuda.get_device_capability", lambda *a, **k: (8, 9)
    )
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.is_ampere_or_newer", lambda: True
    )


def _patch_cache(monkeypatch, *, warm: bool) -> MagicMock:
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.is_warm",
        lambda key: warm,
    )
    mark_warm = MagicMock()
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.mark_warm", mark_warm
    )
    return mark_warm


class _GateRecorder:
    """Records the declaration stream and, alongside it, whether the
    "Preparing TTS engine…" indicator was up at the time — so a test can
    assert the gate and the visible reason travel together."""

    def __init__(self, service: QwenTTSService) -> None:
        self.calls: List[bool] = []
        self.messages: List[Optional[str]] = []
        self.message_when_gated: List[Optional[str]] = []
        self._service = service
        service.set_compile_priming_callback(self._on_priming)
        service.set_preparing_voice_callback(self.messages.append)

    def _on_priming(self, active: bool) -> None:
        self.calls.append(active)
        self.message_when_gated.append(
            self._service._last_preparing_voice_message
        )


async def _prime_and_record(service, monkeypatch, priming_impl):
    monkeypatch.setattr(service, "_run_compile_priming", priming_impl)
    rec = _GateRecorder(service)
    await service.warmup_compile_async()
    return rec


# ==========================================================================
# AC #1 — the gate engages while priming holds the semaphore
# ==========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
async def test_gate_engages_and_releases_around_priming(
    monkeypatch, metric_records, _clean_priming_env, warm
):
    """Both priming paths (cold-cache prime and Story 20.2's warm-cache
    prime) declare busy for exactly the duration of the priming call."""
    _patch_hardware(monkeypatch)
    _patch_cache(monkeypatch, warm=warm)
    service = _make_service()
    observed_during: List[bool] = []

    async def _priming():
        observed_during.append(service.compile_priming_active)

    rec = await _prime_and_record(service, monkeypatch, _priming)

    assert observed_during == [True], (
        "priming ran without the gate engaged — a Generate pressed here "
        "would still queue silently behind the semaphore"
    )
    assert rec.calls == [True, False]
    assert service.compile_priming_active is False


@pytest.mark.asyncio
async def test_the_visible_reason_is_the_existing_indicator_message(
    monkeypatch, metric_records, _clean_priming_env
):
    """AC #1 — the button is disabled and the reason is on screen, via the
    message the priming region already emitted. No second channel."""
    _patch_hardware(monkeypatch)
    _patch_cache(monkeypatch, warm=False)
    service = _make_service()

    async def _priming():
        return None

    rec = await _prime_and_record(service, monkeypatch, _priming)

    assert rec.calls == [True, False]
    # When the gate went up, "Preparing TTS engine…" was already showing.
    assert rec.message_when_gated[0] == service._PREPARING_TTS_ENGINE_MESSAGE
    # And the message channel itself is unchanged: one message, one clear.
    assert rec.messages == [service._PREPARING_TTS_ENGINE_MESSAGE, None]


@pytest.mark.asyncio
async def test_gate_is_not_driven_by_a_timer_or_a_duration(
    monkeypatch, metric_records, _clean_priming_env
):
    """AC #1 — the gate keys off priming actually running, not off an
    assumed duration. A priming call that takes markedly longer than the
    ~4.4-4.9 s reference stays gated for all of it."""
    _patch_hardware(monkeypatch)
    _patch_cache(monkeypatch, warm=False)
    service = _make_service()
    samples: List[bool] = []

    async def _slow_priming():
        for _ in range(5):
            await asyncio.sleep(0)
            samples.append(service.compile_priming_active)

    await _prime_and_record(service, monkeypatch, _slow_priming)

    assert samples == [True] * 5
    assert service.compile_priming_active is False


# ==========================================================================
# AC #2 — the button must always come back (load-bearing)
# ==========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
async def test_priming_raises_and_the_gate_still_releases(
    monkeypatch, metric_records, _clean_priming_env, warm
):
    """AC #2 — the named case: priming raises, ``priming_failed`` telemetry
    lands, and Generate comes back."""
    _patch_hardware(monkeypatch)
    _patch_cache(monkeypatch, warm=warm)
    service = _make_service()

    async def _raises():
        raise RuntimeError("boom")

    rec = await _prime_and_record(service, monkeypatch, _raises)

    assert rec.calls == [True, False], "a failed prime stranded the gate"
    assert service.compile_priming_active is False
    reasons = [
        r.tags["reason"] for r in metric_records
        if r.name == "tts_compile_warmup_priming"
    ]
    assert reasons == ["priming_failed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
async def test_priming_skipped_and_the_gate_still_releases(
    monkeypatch, metric_records, _clean_priming_env, warm
):
    """AC #2 — ``CompilePrimingSkipped`` (Story 20.3's resident-model bail)
    exits through a different ``except`` arm. Same obligation."""
    _patch_hardware(monkeypatch)
    _patch_cache(monkeypatch, warm=warm)
    service = _make_service()

    async def _skips():
        raise CompilePrimingSkipped("no_priming_prompt", "nothing to prime")

    rec = await _prime_and_record(service, monkeypatch, _skips)

    assert rec.calls == [True, False]
    assert service.compile_priming_active is False


@pytest.mark.asyncio
async def test_cancellation_releases_the_gate(
    monkeypatch, metric_records, _clean_priming_env
):
    """AC #2 — cancellation is not an exception the warmup catches; it
    unwinds through the ``finally``. Shutdown mid-prime must not be able to
    leave a disabled button behind for a restart-less session."""
    _patch_hardware(monkeypatch)
    _patch_cache(monkeypatch, warm=False)
    service = _make_service()
    gate_calls: List[bool] = []
    service.set_compile_priming_callback(gate_calls.append)
    started = asyncio.Event()

    async def _hangs():
        started.set()
        await asyncio.sleep(3600)

    monkeypatch.setattr(service, "_run_compile_priming", _hangs)

    task = asyncio.ensure_future(service.warmup_compile_async())
    await started.wait()
    assert service.compile_priming_active is True
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert gate_calls == [True, False], "cancellation stranded the gate"
    assert service.compile_priming_active is False


@pytest.mark.asyncio
async def test_key_model_mismatch_early_return_releases_the_gate(
    monkeypatch, metric_records, _clean_priming_env
):
    """AC #2 — the cold path carries a ``return`` from *inside* the try
    (Story 20.3 AC #3's coherence veto). An early return is the quietest way
    to skip a release; the ``finally`` has to cover it."""
    _patch_hardware(monkeypatch)
    mark_warm = _patch_cache(monkeypatch, warm=False)
    service = _make_service()
    monkeypatch.setattr(
        service,
        "_priming_matches_cache_key",
        lambda *a, **k: (False, "primed a different model"),
    )

    async def _priming():
        return None

    rec = await _prime_and_record(service, monkeypatch, _priming)

    mark_warm.assert_not_called()  # the 20.3 veto still holds
    assert rec.calls == [True, False]
    assert service.compile_priming_active is False


@pytest.mark.asyncio
async def test_a_raising_consumer_cannot_strand_the_gate(
    monkeypatch, metric_records, _clean_priming_env
):
    """AC #2 — the release runs first in the ``finally`` and swallows, so a
    UI callback that blows up neither propagates out of a warmup contracted
    never to raise, nor prevents the indicator from being cleared."""
    _patch_hardware(monkeypatch)
    _patch_cache(monkeypatch, warm=False)
    service = _make_service()
    messages: List[Optional[str]] = []
    service.set_preparing_voice_callback(messages.append)
    seen: List[bool] = []

    def _explodes(active: bool) -> None:
        seen.append(active)
        raise RuntimeError("Qt is having a day")

    service.set_compile_priming_callback(_explodes)

    async def _priming():
        return None

    monkeypatch.setattr(service, "_run_compile_priming", _priming)
    await service.warmup_compile_async()  # must not raise

    assert seen == [True, False]
    assert service.compile_priming_active is False
    assert messages == [service._PREPARING_TTS_ENGINE_MESSAGE, None], (
        "the raising gate callback skipped the indicator clear that shares "
        "its finally"
    )


@pytest.mark.asyncio
async def test_no_consumer_wired_is_not_an_error(
    monkeypatch, metric_records, _clean_priming_env
):
    """The service must run standalone (every pre-20.7 test does)."""
    _patch_hardware(monkeypatch)
    _patch_cache(monkeypatch, warm=False)
    service = _make_service()

    async def _priming():
        assert service.compile_priming_active is True

    monkeypatch.setattr(service, "_run_compile_priming", _priming)
    await service.warmup_compile_async()

    assert service.compile_priming_active is False


# --------------------------------------------------------------------------
# The source invariant behind the rows above (the Story 20.5/20.6 device).
#
# The rows sample today's exit paths. This pins the *structural* property
# that makes them exhaustive, so a sixth exit path — or a third priming call
# site — cannot be added without a release.
# --------------------------------------------------------------------------


def _attr_nodes(node, name: str):
    return [
        n for n in ast.walk(node)
        if isinstance(n, ast.Attribute) and n.attr == name
    ]


def test_every_priming_call_site_sits_in_a_try_that_releases_the_gate():
    """Source invariant — AC #2, structurally.

    Every call to ``_run_compile_priming`` must sit inside a ``try`` whose
    ``finally`` calls ``_set_compile_priming_active``. That is the only
    construct in Python that survives a raise, a ``CancelledError``, a
    ``return`` from inside the block, and a future sixth ``except`` arm
    somebody adds without reading this file.
    """
    from myvoice.services import qwen_tts_service

    source = Path(inspect.getfile(qwen_tts_service)).read_text(encoding="utf-8")
    tree = ast.parse(source)

    all_sites = {id(n) for n in _attr_nodes(tree, "_run_compile_priming")}
    assert all_sites, (
        "no call to _run_compile_priming remains; this invariant has lost "
        "its subject and must be re-derived."
    )

    guarded = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        releases = any(
            _attr_nodes(stmt, "_set_compile_priming_active")
            for stmt in node.finalbody
        )
        if not releases:
            continue
        for stmt in node.body:
            guarded.update(id(n) for n in _attr_nodes(stmt, "_run_compile_priming"))

    missing = all_sites - guarded
    assert not missing, (
        f"{len(missing)} call site(s) to _run_compile_priming are not inside "
        "a try whose finally calls _set_compile_priming_active. Priming is "
        "non-fatal by design — it raises, it is cancelled, it returns early "
        "— and any of those on an unguarded site leaves Generate disabled "
        "for the rest of the session (Story 20.7 AC #2)."
    )


def test_the_release_is_the_first_statement_of_each_finally():
    """AC #2 — ordering, not just presence.

    The ``finally`` blocks also clear the preparing-voice indicator. If a
    statement ahead of the release could raise, the release is skipped and
    the button never returns. Putting it first makes that impossible.
    """
    from myvoice.services import qwen_tts_service

    source = Path(inspect.getfile(qwen_tts_service)).read_text(encoding="utf-8")
    tree = ast.parse(source)

    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if not _attr_nodes(node, "_set_compile_priming_active"):
            continue
        if not any(_attr_nodes(s, "_run_compile_priming") for s in node.body):
            continue
        assert node.finalbody, "priming try lost its finally"
        first = node.finalbody[0]
        assert _attr_nodes(first, "_set_compile_priming_active"), (
            "the gate release is not the first statement of the finally; a "
            "raise from the statement(s) ahead of it would strand Generate."
        )
        checked += 1
    assert checked == 2, (
        f"expected the two priming paths (cache-hit and cache-miss), "
        f"found {checked}"
    )


def test_the_declaration_helper_cannot_raise():
    """AC #2 — the release is called from a ``finally``. If it could
    propagate, it would both break ``warmup_compile_async``'s never-raises
    contract and skip the indicator clear that follows it."""
    from myvoice.services import qwen_tts_service

    source = inspect.getsource(
        qwen_tts_service.QwenTTSService._set_compile_priming_active
    )
    fn = ast.parse(textwrap.dedent(source)).body[0]

    def _callback_calls(node):
        return [
            n for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_compile_priming_callback"
        ]

    all_calls = _callback_calls(fn)
    assert all_calls, "the helper no longer invokes the consumer callback"

    protected = set()
    for tryblock in [n for n in ast.walk(fn) if isinstance(n, ast.Try)]:
        catches_everything = any(
            h.type is None
            or (isinstance(h.type, ast.Name) and h.type.id in {"Exception", "BaseException"})
            for h in tryblock.handlers
        )
        if not catches_everything:
            continue
        for stmt in tryblock.body:
            protected.update(id(c) for c in _callback_calls(stmt))

    assert {id(c) for c in all_calls} <= protected, (
        "the consumer callback is invoked outside a try/except Exception. "
        "A raising consumer would propagate out of the finally that exists "
        "to guarantee Generate comes back (Story 20.7 AC #2)."
    )


# ==========================================================================
# AC #3 — don't gate what isn't blocking
# ==========================================================================


def test_the_precompute_path_does_not_touch_the_gate():
    """AC #3 — Story 17.2's voice_clone_prompt precompute emits the *same*
    advisory message but serialises on a per-voice ``asyncio.Lock``, not on
    ``_request_semaphore``. It must not drive this gate. Checked at the
    source, because the runtime path needs a model."""
    from myvoice.services import qwen_tts_service

    for name in ("prepare_voice_clone_prompt", "generate_voice_clone"):
        fn = getattr(qwen_tts_service.QwenTTSService, name, None)
        if fn is None:
            continue
        assert "_set_compile_priming_active" not in inspect.getsource(fn), (
            f"{name} drives the compile-priming gate. It holds a per-voice "
            "lock, not the request semaphore (AC #3)."
        )

    source = Path(inspect.getfile(qwen_tts_service)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    setters = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _attr_nodes(node, "_set_compile_priming_active"):
                setters.add(node.name)
    assert setters == {"warmup_compile_async"}, (
        f"the gate is declared from unexpected methods: {sorted(setters)}. "
        "Story 20.7 gates compile priming only."
    )


@pytest.mark.asyncio
async def test_the_precompute_message_still_flows_ungated(
    monkeypatch, metric_records, _clean_priming_env
):
    """AC #3 — the precompute's indicator message reaches the UI exactly as
    before and produces no gate traffic."""
    service = _make_service()
    gate_calls: List[bool] = []
    messages: List[Optional[str]] = []
    service.set_compile_priming_callback(gate_calls.append)
    service.set_preparing_voice_callback(messages.append)

    service._emit_preparing_voice("Preparing voice for streaming…")
    service._emit_preparing_voice(None)

    assert messages == ["Preparing voice for streaming…", None]
    assert gate_calls == []
    assert service.compile_priming_active is False


# ==========================================================================
# AC #5 — every skip gate leaves the gate un-engaged
# ==========================================================================


@pytest.mark.asyncio
async def test_env_disabled_warmup_never_engages_the_gate(
    monkeypatch, metric_records, _clean_priming_env
):
    os.environ["MYVOICE_DISABLE_COMPILE_WARMUP"] = "1"
    service = _make_service()
    gate_calls: List[bool] = []
    service.set_compile_priming_callback(gate_calls.append)

    await service.warmup_compile_async()

    assert gate_calls == []
    assert service.compile_priming_active is False


@pytest.mark.asyncio
async def test_tts_compile_off_never_engages_the_gate(
    monkeypatch, metric_records, _clean_priming_env
):
    service = _make_service()
    service._app_settings = AppSettings(tts_compile="off")
    gate_calls: List[bool] = []
    service.set_compile_priming_callback(gate_calls.append)

    await service.warmup_compile_async()

    assert gate_calls == []


@pytest.mark.asyncio
async def test_warm_priming_disable_env_never_engages_the_gate(
    monkeypatch, metric_records, _clean_priming_env
):
    """AC #5 — the Story 20.2 AC #6 reversibility gate returns from inside
    the cache-hit branch, *before* the try. Nothing to release."""
    _patch_hardware(monkeypatch)
    _patch_cache(monkeypatch, warm=True)
    os.environ["MYVOICE_DISABLE_WARM_COMPILE_PRIMING"] = "1"
    service = _make_service()
    gate_calls: List[bool] = []
    service.set_compile_priming_callback(gate_calls.append)

    await service.warmup_compile_async()

    assert gate_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs, patcher",
    [
        ({"with_model_registry": False}, None),
        ({"with_loaded_model": False}, None),
        ({}, "pre_ampere"),
        ({}, "probe_raises"),
    ],
    ids=["no_registry", "no_model_loaded", "pre_ampere", "probe_raises"],
)
async def test_each_early_exit_gate_leaves_the_button_enabled(
    monkeypatch, metric_records, _clean_priming_env, kwargs, patcher
):
    """AC #2/#5 — the remaining skip gates. None of them may emit a lone
    ``True``: they return before the gate is ever engaged, which is the
    strongest form of "the release cannot be skipped"."""
    _patch_hardware(monkeypatch)
    if patcher == "pre_ampere":
        monkeypatch.setattr(
            "myvoice.services.tts_streaming.is_ampere_or_newer", lambda: False
        )
    elif patcher == "probe_raises":
        def _boom():
            raise RuntimeError("no cuda")
        monkeypatch.setattr(
            "myvoice.services.tts_streaming.is_ampere_or_newer", _boom
        )
    service = _make_service(**kwargs)
    gate_calls: List[bool] = []
    service.set_compile_priming_callback(gate_calls.append)

    def _must_not_run():
        raise AssertionError("priming ran on a skip path")

    monkeypatch.setattr(service, "_run_compile_priming", _must_not_run)

    await service.warmup_compile_async()

    assert gate_calls == []
    assert service.compile_priming_active is False


@pytest.mark.asyncio
async def test_cache_key_computation_failure_never_engages_the_gate(
    monkeypatch, metric_records, _clean_priming_env
):
    """The one skip path that lands ``priming_failed`` telemetry without
    priming having run — it must not look like a gate that engaged."""
    service = _make_service()  # built BEFORE the probe is broken
    _patch_hardware(monkeypatch)
    monkeypatch.setattr(
        "torch.cuda.get_device_capability",
        MagicMock(side_effect=RuntimeError("no device")),
    )
    gate_calls: List[bool] = []
    service.set_compile_priming_callback(gate_calls.append)

    await service.warmup_compile_async()

    assert gate_calls == []
    reasons = [
        r.tags["reason"] for r in metric_records
        if r.name == "tts_compile_warmup_priming"
    ]
    assert reasons == ["priming_failed"]
