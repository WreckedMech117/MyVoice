"""Story 20.4 AC #1 — the streamer's chunk geometry is the SINGLE source of
truth for the compile path's ``decode_window_frames``.

Why this file exists
--------------------
Story 20.1 §5.4 found a latent trap: ``engage_compile_optimizations``
declared ``streamer_chunk_size: int = 25, streamer_lookahead: int = 5`` as
hard-coded defaults, and the sole production call site
(``model_registry._load_model_sync``) passed **neither**, so
``decode_window_frames`` resolved to 30 regardless of what
``CodecTokenStreamer`` actually emitted. A second copy of the same literal
lived in ``QwenTTSService.warmup_compile_async``'s cache-key construction.

That made the D-25 invariant decorative and the compile-cache key blind to
the streamer geometry. Retuning ``DEFAULT_CHUNK_SIZE`` — which Story 20.4
does, 25 → 10 — would then have:

  * told the compile path a 30-frame window while the streamer emitted 15
    (a silent D-25 violation: a CUDA-graph captured at one window shape and
    replayed at another diverges without raising), and
  * computed a warm-path priming key for a window nothing uses, so Story
    20.3's startup priming would have warmed a key the engage path never
    reads — silently reverting the ~4 s first-generation win.

Per ``memory/code_review_regression_test_exact_class.md`` these rows mirror
the exact bug class (a literal that duplicates the streamer geometry), not
an adjacent one. They are deliberately split between a runtime arm (does
the value actually track the constants?) and a static arm (is the literal
gone from the source?), because either alone can be satisfied while the
trap is re-introduced through the other.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

import torch  # noqa: F401 — conftest already enforced DLL ordering

from myvoice.models.app_settings import AppSettings
from myvoice.services.qwen_tts_service import QwenTTSService
from myvoice.services.tts_streaming import codec_token_streamer, compile_cache


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src" / "myvoice"


def _live_window() -> int:
    return (
        codec_token_streamer.DEFAULT_CHUNK_SIZE
        + codec_token_streamer.DEFAULT_LOOKAHEAD
    )


# --------------------------------------------------------------------------- #
# Runtime arm — the warm-path priming key tracks the live streamer constants
# --------------------------------------------------------------------------- #


def _make_service() -> QwenTTSService:
    """A service whose registry hands back a model the key computation likes."""
    service = QwenTTSService(
        device="cpu",
        dtype="float32",
        app_settings=AppSettings(tts_compile="auto"),
    )
    fake_inner = type("FakeInner", (), {})()
    fake_inner.name_or_path = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    fake_inner.dtype = torch.bfloat16
    fake_model = type("FakeModel", (), {})()
    fake_model.model = fake_inner

    registry = MagicMock(name="ModelRegistry")
    registry.get_loaded_model.return_value = fake_model
    service._model_registry = registry
    return service


@pytest.fixture
def _restore_warm_priming_env():
    name = "MYVOICE_DISABLE_WARM_COMPILE_PRIMING"
    snapshot = os.environ.get(name)
    yield
    if snapshot is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = snapshot


async def _capture_warmup_key(monkeypatch) -> str:
    """Run ``warmup_compile_async`` far enough to observe the cache key.

    The AC #6 reversibility env gate short-circuits immediately after
    ``is_warm(key)``, so the key is observable without running a priming
    generation (which would need a real model).
    """
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(
        "myvoice.services.tts_streaming.is_ampere_or_newer", lambda: True
    )
    seen: List[str] = []

    def _is_warm(key: str) -> bool:
        seen.append(key)
        return True

    monkeypatch.setattr(
        "myvoice.services.tts_streaming.compile_cache.is_warm", _is_warm
    )
    os.environ["MYVOICE_DISABLE_WARM_COMPILE_PRIMING"] = "1"

    service = _make_service()
    await service.warmup_compile_async()
    assert len(seen) == 1, f"expected exactly one is_warm() call, got {len(seen)}"
    return seen[0]


def _expected_key(decode_window_frames: int) -> str:
    return compile_cache.compute_key(
        qwen_tts_pin_hash=QwenTTSService._QWEN_TTS_PIN_HASH,
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        precision_str="bf16",
        torch_version=torch.__version__,
        decode_window_frames=decode_window_frames,
        cuda_capability=(8, 9),
        compile_mode="reduce-overhead",
    )


@pytest.mark.asyncio
async def test_warmup_cache_key_uses_the_live_streamer_window(
    monkeypatch, _restore_warm_priming_env
):
    """The warm-path priming key is computed at the committed geometry."""
    key = await _capture_warmup_key(monkeypatch)
    assert key == _expected_key(_live_window())


@pytest.mark.asyncio
async def test_warmup_cache_key_moves_with_a_streamer_retune(
    monkeypatch, _restore_warm_priming_env
):
    """The exact bug class: retuning the streamer must move this key.

    Before Story 20.4 the key carried a hard-coded ``decode_window_frames=30``.
    With that literal back, this row fails: the key would stay pinned at the
    30-frame window while the engage path (and therefore the inductor cache
    directory that priming actually warms) moved to 20.
    """
    monkeypatch.setattr(codec_token_streamer, "DEFAULT_CHUNK_SIZE", 17)
    monkeypatch.setattr(codec_token_streamer, "DEFAULT_LOOKAHEAD", 3)
    key = await _capture_warmup_key(monkeypatch)
    assert key == _expected_key(20)
    assert key != _expected_key(30), (
        "warmup priming key is still pinned to the pre-20.4 30-frame window"
    )


@pytest.mark.asyncio
async def test_warmup_key_and_engage_key_agree(
    monkeypatch, _restore_warm_priming_env
):
    """Coherence: priming warms the key the engage path will read.

    ``warmup_compile_async`` and ``engage_compile_optimizations`` compute
    the cache key independently (the architecture's module-boundary rule
    keeps ``tts_streaming/*`` from reaching into ``services.*``). They must
    still land on the same seven dimensions, or Story 20.3's startup
    priming warms a directory nothing reads.
    """
    warm_key = await _capture_warmup_key(monkeypatch)

    engage_key = compile_cache.compute_key(
        qwen_tts_pin_hash=QwenTTSService._QWEN_TTS_PIN_HASH,
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        precision_str="bf16",
        torch_version=torch.__version__,
        # This is what engage_compile_optimizations resolves when the
        # caller passes no geometry (its documented canonical contract).
        decode_window_frames=_live_window(),
        cuda_capability=(8, 9),
        compile_mode="reduce-overhead",
    )
    assert warm_key == engage_key


# --------------------------------------------------------------------------- #
# Static arm — no literal duplicate of the geometry survives in src/
# --------------------------------------------------------------------------- #


def _module_ast(rel: str) -> ast.Module:
    return ast.parse((SRC / rel).read_text(encoding="utf-8"))


def _iter_keyword(tree: ast.AST, func_name: str, kw: str):
    """Yield the AST node for ``kw=`` in every call to ``func_name``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = (
            fn.id if isinstance(fn, ast.Name)
            else fn.attr if isinstance(fn, ast.Attribute)
            else None
        )
        if name != func_name:
            continue
        for keyword in node.keywords:
            if keyword.arg == kw:
                yield keyword.value


def test_no_source_file_passes_a_literal_decode_window_frames():
    """``decode_window_frames=<int literal>`` must not exist under src/myvoice.

    A literal here is the trap: it silently decouples the compile geometry
    (and the compile-cache key) from the streamer that actually produces
    the chunks. Every production site must derive the value.
    """
    offenders = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for value in _iter_keyword(tree, "compute_key", "decode_window_frames"):
            if isinstance(value, ast.Constant) and isinstance(value.value, int):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: compute_key")
        for value in _iter_keyword(
            tree, "enable_streaming_optimizations", "decode_window_frames"
        ):
            if isinstance(value, ast.Constant) and isinstance(value.value, int):
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}: enable_streaming_optimizations"
                )
    assert not offenders, (
        "decode_window_frames must be derived from the streamer geometry, "
        f"never a literal. Offending sites: {offenders}"
    )


def test_engage_compile_optimizations_has_no_hard_coded_geometry_defaults():
    """The function's own defaults must be ``None``, not 25/5.

    Non-None defaults are what let the sole production call site pass
    nothing and still get a plausible-looking answer.
    """
    tree = _module_ast("services/tts_streaming/torch_runtime.py")
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "engage_compile_optimizations"
    )
    kwonly = {a.arg: d for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults)}
    for arg in ("streamer_chunk_size", "streamer_lookahead"):
        assert arg in kwonly, f"{arg} is no longer a keyword-only parameter"
        default = kwonly[arg]
        assert isinstance(default, ast.Constant) and default.value is None, (
            f"{arg} must default to None so the value is resolved from "
            f"codec_token_streamer at call time; found a hard-coded default"
        )


def test_model_registry_threads_the_streamer_geometry_into_the_compile_call():
    """The call site Story 20.1 §5.4 named passes the real geometry.

    Belt-and-braces with the ``None``-resolution above: this row pins the
    call site itself, and requires the values to come from the streamer
    module rather than from freshly-typed literals.
    """
    tree = _module_ast("services/model_registry.py")
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "engage_compile_optimizations"
    ]
    assert len(calls) == 1, (
        f"expected exactly one engage_compile_optimizations call site, "
        f"found {len(calls)}"
    )
    kwargs = {k.arg: k.value for k in calls[0].keywords}
    for arg, const in (
        ("streamer_chunk_size", "DEFAULT_CHUNK_SIZE"),
        ("streamer_lookahead", "DEFAULT_LOOKAHEAD"),
    ):
        assert arg in kwargs, (
            f"model_registry must pass {arg} — the Story 20.1 §5.4 trap was "
            f"exactly that this call site passed neither"
        )
        value = kwargs[arg]
        assert isinstance(value, ast.Attribute) and value.attr == const, (
            f"{arg} must be read from codec_token_streamer.{const}, not "
            f"restated as a literal"
        )
        assert (
            isinstance(value.value, ast.Name)
            and value.value.id == "codec_token_streamer"
        )
