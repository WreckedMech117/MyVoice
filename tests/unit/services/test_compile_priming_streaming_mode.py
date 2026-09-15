"""Story 20.11 AC #3 — compile priming dispatches through the HARDWARE-DEFAULT
streaming mode, regardless of the user's ``streaming_mode_override``.

**The exact defect class this file guards.** ``_run_compile_priming`` used to
resolve its dispatch mode through ``_resolve_streaming_mode()`` — the same
override-aware resolver the public generators use. With the sentence-stream
override on (the RTX 3060 operator had left it on from an earlier test,
2026-09-14 20:17), priming primed the batch decode path. The first TRUE_STREAM
generation after switching back still paid the ``CodecStateCache`` self-test
(~1.5 s on a 3060) and the streaming-geometry compile that priming exists to
absorb: 3.08 s to first chunk against 1.86 s warm (Story 20.10 evidence §5).

TRUE_STREAM is the shipping path and the one whose geometry
(``decode_window_frames``) is a compile-cache-key dimension, so it is what
priming must exercise. ``effective_streaming_mode(None)`` is the D-9 hardware
probe (TRUE_STREAM on CUDA, SENTENCE_STREAM otherwise), and it is what the
priming dispatch now reads.

Three rows:

  1. The load-bearing one — override SENTENCE_STREAM, CUDA available: the
     priming dispatch sees TRUE_STREAM while a user generation, in the same
     service with the same settings, sees SENTENCE_STREAM. Reverting the fix
     (``_resolve_streaming_mode()`` in ``_run_compile_priming``) turns this
     row red; nothing else in the suite does.
  2. Override TRUE_STREAM (or no override) on CUDA — still TRUE_STREAM, so
     the common configuration is unchanged.
  3. CPU-only host — the hardware default is what it always was
     (SENTENCE_STREAM). Priming never reaches the dispatch there in the
     shipped app (the Ampere gate exits first); the row pins that the
     resolver is the D-9 probe and not a hard-coded TRUE_STREAM.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from myvoice.models.app_settings import AppSettings
from myvoice.models.service_enums import ModelQualityTier, QwenModelType
from myvoice.services.qwen_tts_service import QwenTTSService
from myvoice.services.tts_streaming.streaming_mode import StreamingMode


class _FakeRegistry:
    """The attributes ``_build_compile_priming_request`` reads (mirrors
    ``test_compile_priming_resident_model.py``)."""

    def __init__(self, model_type: QwenModelType) -> None:
        self.current_model_type = model_type
        self.current_checkpoint_path = None
        self.quality_tier = ModelQualityTier.QUALITY
        self.device = "cuda"

    def get_loaded_model(self) -> Any:
        return object()


def _make_service(override: Optional[str]) -> QwenTTSService:
    service = QwenTTSService(
        device="cpu",
        dtype="float32",
        app_settings=AppSettings(
            tts_compile="auto", streaming_mode_override=override
        ),
    )
    service._model_registry = _FakeRegistry(QwenModelType.CUSTOM_VOICE)
    return service


def _capture_dispatch(service: QwenTTSService) -> List[Tuple[Any, StreamingMode]]:
    seen: List[Tuple[Any, StreamingMode]] = []

    async def _capture(request, mode):
        seen.append((request, mode))
        return MagicMock(success=True)

    service._dispatch_by_streaming_mode = _capture
    return seen


@pytest.mark.parametrize("cuda", [True, False], ids=["cuda", "cpu-only"])
def test_priming_ignores_the_sentence_stream_override_but_user_generations_honour_it(
    monkeypatch, cuda
):
    """AC #3, the load-bearing row.

    Same service, same ``AppSettings(streaming_mode_override="sentence_stream")``:
    the priming dispatch must see the hardware default, and a user generation
    must still see the override. On CUDA those differ (TRUE_STREAM vs
    SENTENCE_STREAM), which is the defect; on a CPU-only host they coincide,
    which pins that the priming resolver is the D-9 probe and not a
    hard-coded TRUE_STREAM.
    """
    monkeypatch.setattr("torch.cuda.is_available", lambda: cuda)
    service = _make_service("sentence_stream")
    seen = _capture_dispatch(service)

    asyncio.run(service._run_compile_priming())
    asyncio.run(service.generate_custom_voice("user text"))

    assert len(seen) == 2, seen
    (priming_request, priming_mode), (user_request, user_mode) = seen
    assert priming_request.suppress_audio_output is True
    assert user_request.suppress_audio_output is False

    expected_priming = (
        StreamingMode.TRUE_STREAM if cuda else StreamingMode.SENTENCE_STREAM
    )
    assert priming_mode is expected_priming, (
        f"priming dispatched {priming_mode}; it must read the hardware "
        f"default ({expected_priming}), not the user's override. Priming "
        "the batch decode path leaves the first TRUE_STREAM generation to "
        "pay the CodecStateCache self-test and the streaming-geometry "
        "compile (Story 20.10 §5: 3.08 s vs 1.86 s on the RTX 3060)."
    )
    assert user_mode is StreamingMode.SENTENCE_STREAM, (
        f"a user generation dispatched {user_mode}; the user's override "
        "must still win for the user's own generations"
    )


@pytest.mark.parametrize("override", [None, "true_stream"], ids=["auto", "true_stream"])
def test_priming_mode_on_cuda_is_true_stream_for_the_common_configurations(
    monkeypatch, override
):
    """AC #3 — no behaviour change where the override and the hardware
    default already agreed (every RTX launch in the evidence files)."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    service = _make_service(override)
    seen = _capture_dispatch(service)

    asyncio.run(service._run_compile_priming())

    assert [mode for _req, mode in seen] == [StreamingMode.TRUE_STREAM]


def test_priming_mode_is_resolved_at_dispatch_time_not_cached(monkeypatch):
    """The probe is read on every priming call (Story 20.11's tier-change
    re-prime runs the same body again later in the process), so a service
    constructed under one probe result must not pin the other."""
    seen_modes: List[StreamingMode] = []
    service = _make_service("sentence_stream")

    async def _capture(request, mode):
        seen_modes.append(mode)
        return MagicMock(success=True)

    service._dispatch_by_streaming_mode = _capture

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    asyncio.run(service._run_compile_priming())
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    asyncio.run(service._run_compile_priming())

    assert seen_modes == [StreamingMode.SENTENCE_STREAM, StreamingMode.TRUE_STREAM]
