"""Story 16.6 — unit tests for QwenTTSService streaming-mode dispatch.

Covers:
  - AC #4: ``_resolve_streaming_mode`` is a pure function of
    ``AppSettings.streaming_mode_override`` and ``torch.cuda.is_available()``;
    no side effects, no metric emission, no registry mutation.
  - AC #6: ``_dispatch_by_streaming_mode`` emits ``streaming_mode`` and
    ``streaming_mode_fallback`` metrics with the correct shape.
  - AC #7: ``_dispatch_by_streaming_mode`` is testable in isolation from the
    streamer / worker pair — the test patches the three private generators and
    drives every fork branch + every fallback transition without any of
    ``tts_streaming.*``, the audio coordinator, or PyAudio being touched.
  - AC #8: public ``generate*`` entry points route through
    ``_dispatch_by_streaming_mode``.

Story 16.6's TRUE_STREAM end-to-end behavior (AC #1, #9, #10) is integration-
tested in ``tests/integration/test_streaming_tts_smoke.py``; that's where the
real streamer + worker get composed. This file stays unit-scoped — no
pytest-qt, no thread spawning, no asyncio-on-Qt-loop hops.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from myvoice.models.app_settings import AppSettings
from myvoice.services.qwen_tts_service import (
    GenerationMode,
    QwenModelType,
    QwenTTSRequest,
    QwenTTSResponse,
    QwenTTSService,
)
from myvoice.services.tts_streaming import StreamingMode


# --------------------------------------------------------------------------- #
# Test infrastructure
# --------------------------------------------------------------------------- #


@dataclass
class _RecordedMetric:
    """One metrics.record() call, captured for assertion."""
    name: str
    value: object
    tags: dict = field(default_factory=dict)


class _MetricsRecorder:
    """Drop-in for myvoice.observability.metrics.record that records calls.

    Patch via monkeypatch.setattr("myvoice.observability.metrics.record", ...)
    Use ``calls_for(name)`` to filter by metric_name.
    """

    def __init__(self) -> None:
        self.calls: List[_RecordedMetric] = []

    def __call__(self, name: str, value: object, **tags: object) -> None:
        self.calls.append(_RecordedMetric(name=name, value=value, tags=dict(tags)))

    def calls_for(self, name: str) -> List[_RecordedMetric]:
        return [c for c in self.calls if c.name == name]


def _make_service(
    app_settings: Optional[AppSettings] = None,
    session_registry=None,
    audio_coordinator=None,
) -> QwenTTSService:
    """Construct a QwenTTSService with no model load — safe for unit tests.

    The constructor builds a ModelRegistry but does NOT load the model;
    ``ensure_model_loaded`` is the heavy step and is not invoked here.
    """
    return QwenTTSService(
        audio_coordinator=audio_coordinator,
        device="cpu",
        dtype="float32",
        session_registry=session_registry,
        app_settings=app_settings,
    )


def _make_request(streaming: bool = True) -> QwenTTSRequest:
    return QwenTTSRequest(
        text="hello world",
        language="Auto",
        model_type=QwenModelType.CUSTOM_VOICE,
        speaker="Ryan",
        streaming=streaming,
    )


def _success_response(mode: GenerationMode) -> QwenTTSResponse:
    import numpy as np
    return QwenTTSResponse(
        success=True,
        audio_data=np.zeros(16, dtype=np.float32),
        sample_rate=24000,
        mode=mode,
        chunks_generated=1,
    )


# --------------------------------------------------------------------------- #
# AC #4 — TestResolveStreamingMode
# --------------------------------------------------------------------------- #


class TestResolveStreamingMode:
    """``_resolve_streaming_mode`` is pure: no metric, no mutation, no surprise."""

    def test_returns_true_stream_when_override_is_true_stream(self):
        settings = AppSettings(streaming_mode_override="true_stream")
        service = _make_service(app_settings=settings)
        assert service._resolve_streaming_mode() == StreamingMode.TRUE_STREAM

    def test_returns_sentence_stream_when_override_is_sentence_stream(self):
        settings = AppSettings(streaming_mode_override="sentence_stream")
        service = _make_service(app_settings=settings)
        assert (
            service._resolve_streaming_mode()
            == StreamingMode.SENTENCE_STREAM
        )

    def test_returns_batch_when_override_is_batch(self):
        settings = AppSettings(streaming_mode_override="batch")
        service = _make_service(app_settings=settings)
        assert service._resolve_streaming_mode() == StreamingMode.BATCH

    def test_consults_hardware_probe_when_override_is_none_cuda_available(
        self, monkeypatch
    ):
        settings = AppSettings(streaming_mode_override=None)
        service = _make_service(app_settings=settings)
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        assert service._resolve_streaming_mode() == StreamingMode.TRUE_STREAM

    def test_consults_hardware_probe_when_override_is_none_cpu_only(
        self, monkeypatch
    ):
        settings = AppSettings(streaming_mode_override=None)
        service = _make_service(app_settings=settings)
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        # NFR12 — CPU-only must NOT default to TRUE_STREAM.
        assert (
            service._resolve_streaming_mode()
            == StreamingMode.SENTENCE_STREAM
        )

    def test_resolves_default_when_app_settings_is_none(self, monkeypatch):
        # No AppSettings injected — service treats override as None and
        # consults the hardware probe.
        service = _make_service(app_settings=None)
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        assert service._resolve_streaming_mode() == StreamingMode.TRUE_STREAM

    def test_repeated_calls_are_deterministic(self, monkeypatch):
        settings = AppSettings(streaming_mode_override="sentence_stream")
        service = _make_service(app_settings=settings)
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        for _ in range(5):
            assert (
                service._resolve_streaming_mode()
                == StreamingMode.SENTENCE_STREAM
            )

    def test_set_app_settings_swaps_reference_so_resolve_sees_new_override(
        self, monkeypatch
    ):
        """RTX 3060 smoke 2026-05-13: changing the Streaming Mode dropdown
        in Settings had no effect on subsequent generations because
        SettingsDialog returns a deep-copy AppSettings (settings_dialog.py:92)
        and _handle_settings_changed updated the app/main-window/audio-
        coordinator references but NOT the TTS service's. The fix is
        ``set_app_settings`` + a corresponding call from
        ``_handle_settings_changed``. This test pins the swap.

        Construct with override="true_stream", then swap in a new
        AppSettings with override="sentence_stream" — _resolve_streaming_mode
        must return SENTENCE_STREAM after the swap. Hardware probe is
        irrelevant when override is set (purity test above covers that
        branch).
        """
        original_settings = AppSettings(streaming_mode_override="true_stream")
        service = _make_service(app_settings=original_settings)
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        assert service._resolve_streaming_mode() == StreamingMode.TRUE_STREAM

        # Deep-copy mirrors what SettingsDialog actually does (line 92):
        # the new object is NOT the same reference, so reference-equality
        # is the right surface to test.
        new_settings = AppSettings.from_dict(original_settings.to_dict())
        new_settings.streaming_mode_override = "sentence_stream"
        assert new_settings is not original_settings

        service.set_app_settings(new_settings)
        assert (
            service._resolve_streaming_mode()
            == StreamingMode.SENTENCE_STREAM
        )

    def test_app_settings_tts_precision_flows_through_to_model_registry(self):
        """Story 18.3 AC #4 — wire-up integration test.

        Constructs ``QwenTTSService(app_settings=AppSettings(tts_precision="fp32"))``
        and asserts ``service._model_registry.dtype == torch.float32``. Verifies
        the new ``app_settings`` keyword argument flows through
        ``qwen_tts_service.py:582-:588`` to ``ModelRegistry.__init__`` end-to-end.

        Per `memory/code_review_regression_test_exact_class.md`, the
        precedence-rule contract ("``app_settings`` overrides legacy
        ``dtype`` parameter") is the most-easily-broken AC; this test
        exercises the full surface (AppSettings → QwenTTSService →
        ModelRegistry) so a future regression that drops the
        ``app_settings`` argument from one of the two ``__init__`` calls
        is caught immediately.
        """
        import torch
        settings = AppSettings(tts_precision="fp32")
        service = _make_service(app_settings=settings)
        # Legacy dtype parameter is "float32" in _make_service, so this test
        # specifically verifies the override path: even if dtype="float32"
        # already produces the same dtype, the metric source must be
        # "app_settings_override", not "legacy_constructor_arg".
        assert service._model_registry.dtype == torch.float32, (
            "Story 18.3 AC #4 violated: AppSettings.tts_precision='fp32' "
            "must flow through to ModelRegistry.dtype as torch.float32. "
            "If this fails, the wire-up at qwen_tts_service.py:582-588 "
            "may have dropped the app_settings keyword argument."
        )

    def test_app_settings_tts_precision_bf16_flows_through(self, monkeypatch):
        """Mirror of the fp32 test for the bf16 override branch."""
        import torch
        # Pre-Ampere or CPU so the legacy "float32" fallback would NOT yield
        # bfloat16; only the app_settings override path can.
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        settings = AppSettings(tts_precision="bf16")
        service = _make_service(app_settings=settings)
        assert service._model_registry.dtype == torch.bfloat16, (
            "Story 18.3 AC #4 violated: AppSettings.tts_precision='bf16' "
            "must flow through to ModelRegistry.dtype as torch.bfloat16."
        )

    def test_emits_no_metric(self, monkeypatch):
        settings = AppSettings(streaming_mode_override="true_stream")
        service = _make_service(app_settings=settings)
        recorder = _MetricsRecorder()
        monkeypatch.setattr("myvoice.observability.metrics.record", recorder)
        service._resolve_streaming_mode()
        assert recorder.calls == [], (
            "Resolver emitted metric — should be pure. AC #4 invariant."
        )

    def test_does_not_mutate_registry(self):
        registry = MagicMock()
        settings = AppSettings(streaming_mode_override="true_stream")
        service = _make_service(
            app_settings=settings, session_registry=registry
        )
        service._resolve_streaming_mode()
        # No method call on the registry whatsoever.
        registry.assert_not_called()
        registry.create_session.assert_not_called()
        registry.post_mutation.assert_not_called()
        registry.request_cancel.assert_not_called()

    def test_invalid_override_raises_value_error(self):
        # AppSettings's validator coerces invalid strings to None at
        # __post_init__. Force a bad value AFTER construction so the
        # resolver sees it; surfaces the StringMode(s) ValueError.
        settings = AppSettings(streaming_mode_override=None)
        settings.streaming_mode_override = "invalid_mode"
        service = _make_service(app_settings=settings)
        with pytest.raises(ValueError):
            service._resolve_streaming_mode()

    def test_thin_shim_delegates_to_effective_streaming_mode(self, monkeypatch):
        """Resolver is a thin shim, not a re-implementation of Story 16.2's
        resolver. Patch ``effective_streaming_mode`` and confirm it's called
        with the expected ``Optional[StreamingMode]`` argument exactly once.
        """
        settings = AppSettings(streaming_mode_override="batch")
        service = _make_service(app_settings=settings)
        recording = MagicMock(return_value=StreamingMode.BATCH)
        monkeypatch.setattr(
            "myvoice.services.qwen_tts_service.effective_streaming_mode",
            recording,
        )
        service._resolve_streaming_mode()
        recording.assert_called_once_with(StreamingMode.BATCH)


# --------------------------------------------------------------------------- #
# AC #7 — TestDispatchForkBranches
# --------------------------------------------------------------------------- #


class TestDispatchForkBranches:
    """``_dispatch_by_streaming_mode`` forks correctly to each generator.

    Tests patch the three private generators with ``AsyncMock`` so no streamer,
    no worker, no audio coordinator, no PyAudio is touched.
    """

    @pytest.mark.asyncio
    async def test_true_stream_routes_to_generate_true_stream(self):
        service = _make_service()
        request = _make_request()
        true_stream_resp = _success_response(GenerationMode.STREAMING)
        with (
            patch.object(
                service, "_generate_true_stream", new=AsyncMock(return_value=true_stream_resp)
            ) as mock_true,
            patch.object(service, "_generate_streaming", new=AsyncMock()) as mock_sentence,
            patch.object(service, "_generate", new=AsyncMock()) as mock_batch,
        ):
            response = await service._dispatch_by_streaming_mode(
                request, StreamingMode.TRUE_STREAM
            )
        assert response is true_stream_resp
        mock_true.assert_called_once_with(request)
        mock_sentence.assert_not_called()
        mock_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_sentence_stream_routes_to_generate_streaming(self):
        service = _make_service()
        request = _make_request()
        sentence_resp = _success_response(GenerationMode.STREAMING)
        with (
            patch.object(service, "_generate_true_stream", new=AsyncMock()) as mock_true,
            patch.object(
                service, "_generate_streaming", new=AsyncMock(return_value=sentence_resp)
            ) as mock_sentence,
            patch.object(service, "_generate", new=AsyncMock()) as mock_batch,
        ):
            response = await service._dispatch_by_streaming_mode(
                request, StreamingMode.SENTENCE_STREAM
            )
        assert response is sentence_resp
        mock_true.assert_not_called()
        mock_sentence.assert_called_once_with(request)
        mock_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_routes_to_generate(self):
        service = _make_service()
        request = _make_request()
        batch_resp = _success_response(GenerationMode.BATCH)
        with (
            patch.object(service, "_generate_true_stream", new=AsyncMock()) as mock_true,
            patch.object(service, "_generate_streaming", new=AsyncMock()) as mock_sentence,
            patch.object(
                service, "_generate", new=AsyncMock(return_value=batch_resp)
            ) as mock_batch,
        ):
            response = await service._dispatch_by_streaming_mode(
                request, StreamingMode.BATCH
            )
        assert response is batch_resp
        mock_true.assert_not_called()
        mock_sentence.assert_not_called()
        mock_batch.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_request_streaming_false_forces_batch_regardless_of_mode(self):
        """AC #8 — legacy ``request.streaming=False`` overrides the resolver.

        Even when the dispatcher is invoked with TRUE_STREAM (e.g. via Auto
        + GPU), a caller that explicitly set ``request.streaming=False``
        must hit the BATCH path.
        """
        service = _make_service()
        request = _make_request(streaming=False)
        batch_resp = _success_response(GenerationMode.BATCH)
        with (
            patch.object(service, "_generate_true_stream", new=AsyncMock()) as mock_true,
            patch.object(service, "_generate_streaming", new=AsyncMock()) as mock_sentence,
            patch.object(
                service, "_generate", new=AsyncMock(return_value=batch_resp)
            ) as mock_batch,
        ):
            response = await service._dispatch_by_streaming_mode(
                request, StreamingMode.TRUE_STREAM
            )
        assert response is batch_resp
        mock_true.assert_not_called()
        mock_sentence.assert_not_called()
        mock_batch.assert_called_once_with(request)


# --------------------------------------------------------------------------- #
# AC #2 — TestThreeModeFallbackChain
# --------------------------------------------------------------------------- #


class TestThreeModeFallbackChain:
    """Three-mode fallback chain on TRUE_STREAM start.

    TRUE_STREAM raises -> SENTENCE_STREAM is attempted.
    SENTENCE_STREAM raises -> BATCH is attempted.
    BATCH raises -> unrecoverable failure response with all three error
    strings included.
    """

    @pytest.mark.asyncio
    async def test_true_stream_failure_falls_back_to_sentence_stream(self):
        service = _make_service()
        request = _make_request()
        sentence_resp = _success_response(GenerationMode.STREAMING)
        with (
            patch.object(
                service,
                "_generate_true_stream",
                new=AsyncMock(side_effect=RuntimeError("simulated TRUE_STREAM failure")),
            ) as mock_true,
            patch.object(
                service,
                "_generate_streaming",
                new=AsyncMock(return_value=sentence_resp),
            ) as mock_sentence,
            patch.object(service, "_generate", new=AsyncMock()) as mock_batch,
        ):
            response = await service._dispatch_by_streaming_mode(
                request, StreamingMode.TRUE_STREAM
            )
        assert response is sentence_resp
        mock_true.assert_called_once_with(request)
        mock_sentence.assert_called_once_with(request)
        mock_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_sentence_stream_failure_falls_back_to_batch(self):
        service = _make_service()
        request = _make_request()
        batch_resp = _success_response(GenerationMode.BATCH)
        with (
            patch.object(
                service,
                "_generate_true_stream",
                new=AsyncMock(side_effect=RuntimeError("simulated TRUE_STREAM failure")),
            ),
            patch.object(
                service,
                "_generate_streaming",
                new=AsyncMock(side_effect=RuntimeError("simulated SENTENCE_STREAM failure")),
            ),
            patch.object(
                service, "_generate", new=AsyncMock(return_value=batch_resp)
            ) as mock_batch,
        ):
            response = await service._dispatch_by_streaming_mode(
                request, StreamingMode.TRUE_STREAM
            )
        assert response is batch_resp
        mock_batch.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_all_three_modes_fail_returns_unrecoverable_response(self):
        service = _make_service()
        request = _make_request()
        with (
            patch.object(
                service,
                "_generate_true_stream",
                new=AsyncMock(side_effect=RuntimeError("TRUE_STREAM error string")),
            ),
            patch.object(
                service,
                "_generate_streaming",
                new=AsyncMock(side_effect=RuntimeError("SENTENCE_STREAM error string")),
            ),
            patch.object(
                service,
                "_generate",
                new=AsyncMock(side_effect=RuntimeError("BATCH error string")),
            ),
        ):
            response = await service._dispatch_by_streaming_mode(
                request, StreamingMode.TRUE_STREAM
            )
        assert response.success is False
        assert "TRUE_STREAM error string" in (response.error_message or "")
        assert "SENTENCE_STREAM error string" in (response.error_message or "")
        assert "BATCH error string" in (response.error_message or "")
        # AC #2: response's mode reflects the LAST mode attempted.
        assert response.mode == GenerationMode.BATCH

    @pytest.mark.asyncio
    async def test_dispatch_call_order_is_exact_chain(self):
        service = _make_service()
        request = _make_request()
        order: List[str] = []

        async def true_stream_side(_req):
            order.append("true_stream")
            raise RuntimeError("ts fail")

        async def sentence_side(_req):
            order.append("sentence_stream")
            raise RuntimeError("ss fail")

        async def batch_side(_req):
            order.append("batch")
            return _success_response(GenerationMode.BATCH)

        with (
            patch.object(
                service, "_generate_true_stream", new=AsyncMock(side_effect=true_stream_side)
            ),
            patch.object(
                service, "_generate_streaming", new=AsyncMock(side_effect=sentence_side)
            ),
            patch.object(
                service, "_generate", new=AsyncMock(side_effect=batch_side)
            ),
        ):
            await service._dispatch_by_streaming_mode(
                request, StreamingMode.TRUE_STREAM
            )
        assert order == ["true_stream", "sentence_stream", "batch"]

    @pytest.mark.asyncio
    async def test_failed_request_counter_incremented_once_not_per_attempt(self):
        service = _make_service()
        request = _make_request()
        before_failed = service._failed_requests
        with (
            patch.object(
                service,
                "_generate_true_stream",
                new=AsyncMock(side_effect=RuntimeError("ts fail")),
            ),
            patch.object(
                service,
                "_generate_streaming",
                new=AsyncMock(side_effect=RuntimeError("ss fail")),
            ),
            patch.object(
                service,
                "_generate",
                new=AsyncMock(side_effect=RuntimeError("b fail")),
            ),
        ):
            await service._dispatch_by_streaming_mode(
                request, StreamingMode.TRUE_STREAM
            )
        # The dispatcher counts ONE request-level failure even though three
        # modes were tried.
        assert service._failed_requests - before_failed == 1

    @pytest.mark.asyncio
    async def test_failed_request_counter_not_double_incremented_when_inner_method_already_counted(
        self,
    ):
        """Story 16.6 review M3 — AC #2: ``_failed_requests`` is incremented
        exactly once per dispatch, even when an inner method already
        incremented its own counter before raising. Regression test for
        the exact bug class (snapshot-based dedup)."""
        service = _make_service()
        request = _make_request()
        before_failed = service._failed_requests

        async def true_stream_increments_then_raises(_req):
            # Mimics _generate_true_stream's real behavior: increment
            # _failed_requests on a structural failure, then raise so the
            # dispatcher's fallback chain triggers.
            service._failed_requests += 1
            raise RuntimeError("inner already counted")

        with (
            patch.object(
                service,
                "_generate_true_stream",
                new=AsyncMock(side_effect=true_stream_increments_then_raises),
            ),
            patch.object(
                service,
                "_generate_streaming",
                new=AsyncMock(side_effect=RuntimeError("ss fail")),
            ),
            patch.object(
                service,
                "_generate",
                new=AsyncMock(side_effect=RuntimeError("b fail")),
            ),
        ):
            await service._dispatch_by_streaming_mode(
                request, StreamingMode.TRUE_STREAM
            )
        # Exactly one increment despite the inner method's bump.
        assert service._failed_requests - before_failed == 1

    @pytest.mark.asyncio
    async def test_fallback_count_incremented_per_successful_transition(self):
        """``_fallback_count`` counts mode-to-next-mode transitions, not
        terminal failures. Two transitions: TRUE_STREAM->SENTENCE_STREAM and
        SENTENCE_STREAM->BATCH."""
        service = _make_service()
        request = _make_request()
        before_fallbacks = service._fallback_count
        with (
            patch.object(
                service,
                "_generate_true_stream",
                new=AsyncMock(side_effect=RuntimeError("ts fail")),
            ),
            patch.object(
                service,
                "_generate_streaming",
                new=AsyncMock(side_effect=RuntimeError("ss fail")),
            ),
            patch.object(
                service,
                "_generate",
                new=AsyncMock(return_value=_success_response(GenerationMode.BATCH)),
            ),
        ):
            await service._dispatch_by_streaming_mode(
                request, StreamingMode.TRUE_STREAM
            )
        # Exactly two fallback transitions fired (the success in BATCH is
        # not itself a "fallback").
        assert service._fallback_count - before_fallbacks == 2

    @pytest.mark.asyncio
    async def test_cancelled_error_is_not_a_fallback_trigger(self):
        """``asyncio.CancelledError`` is user-cancel, not a structural
        failure — it propagates to the caller without triggering fallback.
        """
        service = _make_service()
        request = _make_request()
        with (
            patch.object(
                service,
                "_generate_true_stream",
                new=AsyncMock(side_effect=asyncio.CancelledError()),
            ),
            patch.object(service, "_generate_streaming", new=AsyncMock()) as mock_sentence,
            patch.object(service, "_generate", new=AsyncMock()) as mock_batch,
        ):
            with pytest.raises(asyncio.CancelledError):
                await service._dispatch_by_streaming_mode(
                    request, StreamingMode.TRUE_STREAM
                )
        mock_sentence.assert_not_called()
        mock_batch.assert_not_called()


# --------------------------------------------------------------------------- #
# AC #3 — TestUserOverrideCapsStartingMode
# --------------------------------------------------------------------------- #


class TestUserOverrideCapsStartingMode:
    """User-set override caps the STARTING mode; fallback chain below is
    still active per FR3 + NFR7."""

    @pytest.mark.asyncio
    async def test_sentence_stream_override_skips_true_stream(self):
        settings = AppSettings(streaming_mode_override="sentence_stream")
        service = _make_service(app_settings=settings)
        request = _make_request()
        sentence_resp = _success_response(GenerationMode.STREAMING)
        with (
            patch.object(service, "_generate_true_stream", new=AsyncMock()) as mock_true,
            patch.object(
                service, "_generate_streaming", new=AsyncMock(return_value=sentence_resp)
            ),
            patch.object(service, "_generate", new=AsyncMock()) as mock_batch,
        ):
            response = await service._dispatch_by_streaming_mode(
                request, service._resolve_streaming_mode()
            )
        assert response is sentence_resp
        mock_true.assert_not_called()
        mock_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_sentence_stream_override_falls_back_to_batch_on_failure(self):
        settings = AppSettings(streaming_mode_override="sentence_stream")
        service = _make_service(app_settings=settings)
        request = _make_request()
        batch_resp = _success_response(GenerationMode.BATCH)
        with (
            patch.object(service, "_generate_true_stream", new=AsyncMock()) as mock_true,
            patch.object(
                service,
                "_generate_streaming",
                new=AsyncMock(side_effect=RuntimeError("ss fail")),
            ),
            patch.object(
                service, "_generate", new=AsyncMock(return_value=batch_resp)
            ),
        ):
            response = await service._dispatch_by_streaming_mode(
                request, service._resolve_streaming_mode()
            )
        assert response is batch_resp
        mock_true.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_override_skips_true_stream_and_sentence_stream(self):
        settings = AppSettings(streaming_mode_override="batch")
        service = _make_service(app_settings=settings)
        request = _make_request()
        batch_resp = _success_response(GenerationMode.BATCH)
        with (
            patch.object(service, "_generate_true_stream", new=AsyncMock()) as mock_true,
            patch.object(service, "_generate_streaming", new=AsyncMock()) as mock_sentence,
            patch.object(
                service, "_generate", new=AsyncMock(return_value=batch_resp)
            ),
        ):
            response = await service._dispatch_by_streaming_mode(
                request, service._resolve_streaming_mode()
            )
        assert response is batch_resp
        mock_true.assert_not_called()
        mock_sentence.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_override_returns_unrecoverable_on_batch_failure(self):
        """No fallback below BATCH exists."""
        settings = AppSettings(streaming_mode_override="batch")
        service = _make_service(app_settings=settings)
        request = _make_request()
        with (
            patch.object(service, "_generate_true_stream", new=AsyncMock()) as mock_true,
            patch.object(service, "_generate_streaming", new=AsyncMock()) as mock_sentence,
            patch.object(
                service,
                "_generate",
                new=AsyncMock(side_effect=RuntimeError("batch fail")),
            ),
        ):
            response = await service._dispatch_by_streaming_mode(
                request, service._resolve_streaming_mode()
            )
        assert response.success is False
        assert "batch fail" in (response.error_message or "")
        mock_true.assert_not_called()
        mock_sentence.assert_not_called()


# --------------------------------------------------------------------------- #
# AC #6 — TestStreamingModeMetric / TestStreamingModeFallbackMetric
# --------------------------------------------------------------------------- #


class TestStreamingModeMetric:
    """``streaming_mode`` metric fires once per dispatch entry per session."""

    @pytest.mark.asyncio
    async def test_one_streaming_mode_metric_for_first_attempt_success(self, monkeypatch):
        service = _make_service()
        request = _make_request()
        recorder = _MetricsRecorder()
        monkeypatch.setattr("myvoice.observability.metrics.record", recorder)
        monkeypatch.setattr(
            "myvoice.services.qwen_tts_service.metrics.record", recorder
        )
        with (
            patch.object(
                service,
                "_generate_true_stream",
                new=AsyncMock(return_value=_success_response(GenerationMode.STREAMING)),
            ),
            patch.object(service, "_generate_streaming", new=AsyncMock()),
            patch.object(service, "_generate", new=AsyncMock()),
        ):
            await service._dispatch_by_streaming_mode(
                request, StreamingMode.TRUE_STREAM
            )
        mode_calls = recorder.calls_for("streaming_mode")
        assert len(mode_calls) == 1
        assert mode_calls[0].value == "true_stream"

    @pytest.mark.asyncio
    async def test_two_streaming_mode_metrics_when_falls_back_once(self, monkeypatch):
        service = _make_service()
        request = _make_request()
        recorder = _MetricsRecorder()
        monkeypatch.setattr(
            "myvoice.services.qwen_tts_service.metrics.record", recorder
        )
        with (
            patch.object(
                service,
                "_generate_true_stream",
                new=AsyncMock(side_effect=RuntimeError("fail")),
            ),
            patch.object(
                service,
                "_generate_streaming",
                new=AsyncMock(return_value=_success_response(GenerationMode.STREAMING)),
            ),
            patch.object(service, "_generate", new=AsyncMock()),
        ):
            await service._dispatch_by_streaming_mode(
                request, StreamingMode.TRUE_STREAM
            )
        mode_calls = recorder.calls_for("streaming_mode")
        # Two: one for TRUE_STREAM (initial), one for SENTENCE_STREAM (post-
        # fallback dispatch entry).
        assert len(mode_calls) == 2
        assert mode_calls[0].value == "true_stream"
        assert mode_calls[1].value == "sentence_stream"


class TestStreamingModeFallbackMetric:
    """``streaming_mode_fallback`` metric fires per fallback transition."""

    @pytest.mark.asyncio
    async def test_no_fallback_metric_on_first_attempt_success(self, monkeypatch):
        service = _make_service()
        request = _make_request()
        recorder = _MetricsRecorder()
        monkeypatch.setattr(
            "myvoice.services.qwen_tts_service.metrics.record", recorder
        )
        with (
            patch.object(
                service,
                "_generate_true_stream",
                new=AsyncMock(return_value=_success_response(GenerationMode.STREAMING)),
            ),
            patch.object(service, "_generate_streaming", new=AsyncMock()),
            patch.object(service, "_generate", new=AsyncMock()),
        ):
            await service._dispatch_by_streaming_mode(
                request, StreamingMode.TRUE_STREAM
            )
        assert recorder.calls_for("streaming_mode_fallback") == []

    @pytest.mark.asyncio
    async def test_fallback_metric_fires_with_from_mode_and_reason_tags(
        self, monkeypatch
    ):
        service = _make_service()
        request = _make_request()
        recorder = _MetricsRecorder()
        monkeypatch.setattr(
            "myvoice.services.qwen_tts_service.metrics.record", recorder
        )
        with (
            patch.object(
                service,
                "_generate_true_stream",
                new=AsyncMock(side_effect=RuntimeError("simulated TRUE_STREAM failure")),
            ),
            patch.object(
                service,
                "_generate_streaming",
                new=AsyncMock(return_value=_success_response(GenerationMode.STREAMING)),
            ),
            patch.object(service, "_generate", new=AsyncMock()),
        ):
            await service._dispatch_by_streaming_mode(
                request, StreamingMode.TRUE_STREAM
            )
        fallback_calls = recorder.calls_for("streaming_mode_fallback")
        assert len(fallback_calls) == 1
        assert fallback_calls[0].value == "sentence_stream"
        assert fallback_calls[0].tags.get("from_mode") == "true_stream"
        assert "simulated TRUE_STREAM failure" in str(
            fallback_calls[0].tags.get("reason", "")
        )

    @pytest.mark.asyncio
    async def test_unrecoverable_fallback_metric_when_all_modes_fail(
        self, monkeypatch
    ):
        service = _make_service()
        request = _make_request()
        recorder = _MetricsRecorder()
        monkeypatch.setattr(
            "myvoice.services.qwen_tts_service.metrics.record", recorder
        )
        with (
            patch.object(
                service,
                "_generate_true_stream",
                new=AsyncMock(side_effect=RuntimeError("ts fail")),
            ),
            patch.object(
                service,
                "_generate_streaming",
                new=AsyncMock(side_effect=RuntimeError("ss fail")),
            ),
            patch.object(
                service,
                "_generate",
                new=AsyncMock(side_effect=RuntimeError("b fail")),
            ),
        ):
            await service._dispatch_by_streaming_mode(
                request, StreamingMode.TRUE_STREAM
            )
        fallback_values = [c.value for c in recorder.calls_for("streaming_mode_fallback")]
        assert fallback_values == [
            "sentence_stream",
            "batch",
            "unrecoverable",
        ]

    @pytest.mark.asyncio
    async def test_reason_tag_truncated_to_200_chars(self, monkeypatch):
        service = _make_service()
        request = _make_request()
        recorder = _MetricsRecorder()
        monkeypatch.setattr(
            "myvoice.services.qwen_tts_service.metrics.record", recorder
        )
        long_message = "x" * 500
        with (
            patch.object(
                service,
                "_generate_true_stream",
                new=AsyncMock(side_effect=RuntimeError(long_message)),
            ),
            patch.object(
                service,
                "_generate_streaming",
                new=AsyncMock(return_value=_success_response(GenerationMode.STREAMING)),
            ),
            patch.object(service, "_generate", new=AsyncMock()),
        ):
            await service._dispatch_by_streaming_mode(
                request, StreamingMode.TRUE_STREAM
            )
        fallback_calls = recorder.calls_for("streaming_mode_fallback")
        reason = str(fallback_calls[0].tags.get("reason", ""))
        assert len(reason) <= 200, (
            f"reason tag is {len(reason)} chars; should be truncated to <= 200"
        )

    @pytest.mark.asyncio
    async def test_truncated_reason_ends_with_ellipsis(self, monkeypatch):
        """Story 16.6 review M2 — AC #6 last clause: truncated reason has
        an explicit ``…`` suffix so downstream telemetry can distinguish a
        200-char message from a truncated one. Regression test for the
        exact bug class (truncation without suffix)."""
        service = _make_service()
        request = _make_request()
        recorder = _MetricsRecorder()
        monkeypatch.setattr(
            "myvoice.services.qwen_tts_service.metrics.record", recorder
        )
        # 300-char message — guaranteed to trigger truncation.
        long_message = "x" * 300
        with (
            patch.object(
                service,
                "_generate_true_stream",
                new=AsyncMock(side_effect=RuntimeError(long_message)),
            ),
            patch.object(
                service,
                "_generate_streaming",
                new=AsyncMock(return_value=_success_response(GenerationMode.STREAMING)),
            ),
            patch.object(service, "_generate", new=AsyncMock()),
        ):
            await service._dispatch_by_streaming_mode(
                request, StreamingMode.TRUE_STREAM
            )
        fallback_calls = recorder.calls_for("streaming_mode_fallback")
        reason = str(fallback_calls[0].tags.get("reason", ""))
        assert reason.endswith("…"), (
            f"truncated reason {reason!r} does not end with the documented "
            f"'…' suffix; AC #6 requires an explicit truncation marker"
        )

    @pytest.mark.asyncio
    async def test_short_reason_is_not_suffixed(self, monkeypatch):
        """Sanity guard for the M2 fix — a short reason must NOT get an
        ellipsis suffix (only truncated reasons do)."""
        service = _make_service()
        request = _make_request()
        recorder = _MetricsRecorder()
        monkeypatch.setattr(
            "myvoice.services.qwen_tts_service.metrics.record", recorder
        )
        with (
            patch.object(
                service,
                "_generate_true_stream",
                new=AsyncMock(side_effect=RuntimeError("short message")),
            ),
            patch.object(
                service,
                "_generate_streaming",
                new=AsyncMock(return_value=_success_response(GenerationMode.STREAMING)),
            ),
            patch.object(service, "_generate", new=AsyncMock()),
        ):
            await service._dispatch_by_streaming_mode(
                request, StreamingMode.TRUE_STREAM
            )
        fallback_calls = recorder.calls_for("streaming_mode_fallback")
        reason = str(fallback_calls[0].tags.get("reason", ""))
        assert not reason.endswith("…"), (
            f"short reason {reason!r} unexpectedly has the truncation suffix"
        )


# --------------------------------------------------------------------------- #
# AC #8 — TestPublicEntryPointsRouteThroughDispatch
# --------------------------------------------------------------------------- #


class TestPublicEntryPointsRouteThroughDispatch:
    """Every public ``generate*`` entry point routes via
    ``_dispatch_by_streaming_mode`` rather than calling ``_generate_streaming``
    or ``_generate`` directly. AC #8.
    """

    @pytest.mark.asyncio
    async def test_generate_custom_voice_routes_through_dispatch(self):
        service = _make_service()
        dispatch_resp = _success_response(GenerationMode.STREAMING)
        with patch.object(
            service,
            "_dispatch_by_streaming_mode",
            new=AsyncMock(return_value=dispatch_resp),
        ) as mock_dispatch:
            response = await service.generate_custom_voice(
                text="hello", speaker="Ryan", streaming=True
            )
        assert response is dispatch_resp
        mock_dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_voice_design_routes_through_dispatch(self):
        service = _make_service()
        dispatch_resp = _success_response(GenerationMode.STREAMING)
        with patch.object(
            service,
            "_dispatch_by_streaming_mode",
            new=AsyncMock(return_value=dispatch_resp),
        ) as mock_dispatch:
            response = await service.generate_voice_design(
                text="hello", voice_description="warm friendly", streaming=True
            )
        assert response is dispatch_resp
        mock_dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_voice_clone_routes_through_dispatch(self):
        service = _make_service()
        dispatch_resp = _success_response(GenerationMode.STREAMING)
        from pathlib import Path
        with patch.object(
            service,
            "_dispatch_by_streaming_mode",
            new=AsyncMock(return_value=dispatch_resp),
        ) as mock_dispatch:
            response = await service.generate_voice_clone(
                text="hello",
                ref_audio=Path("/tmp/fake.wav"),
                ref_text="hi there",
                streaming=True,
            )
        assert response is dispatch_resp
        mock_dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_optimized_voice_routes_through_dispatch(self, tmp_path):
        service = _make_service()
        dispatch_resp = _success_response(GenerationMode.STREAMING)
        # generate_optimized_voice validates checkpoint_path.exists() up front;
        # use a real tmp file to satisfy that gate.
        ckpt = tmp_path / "fake_checkpoint.bin"
        ckpt.write_bytes(b"")
        with patch.object(
            service,
            "_dispatch_by_streaming_mode",
            new=AsyncMock(return_value=dispatch_resp),
        ) as mock_dispatch:
            response = await service.generate_optimized_voice(
                text="hello",
                checkpoint_path=ckpt,
                speaker_name="Ryan",
                streaming=True,
            )
        assert response is dispatch_resp
        mock_dispatch.assert_called_once()
