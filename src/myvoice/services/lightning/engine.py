"""Lightning engine: Qwen3-TTS 0.6B through qwen3-tts.cpp (GGML, 4-bit), via its C API.

The same model family as the standard PyTorch engine, at roughly 60 % of the
VRAM (1.9 GB peak against 3.0 GB) and with no torch involvement at speaking
time. Evidence and the verified configuration:
``_bmad-output/implementation-artifacts/19-0-lightning-engine-spike-evidence.md``
and ``19-1-lightning-build-recipe.md``.

This module is deliberately free of Qt, asyncio and the session registry: it
loads the DLL, owns the native handle and turns one request into PCM. The
service layer (``QwenTTSService._generate_lightning``) does the rest.

Threading: every native call is serialized by one lock. ``synthesize_*`` block
for the length of a generation and are meant to run on a worker thread; the
chunk callback is invoked on that same thread.
"""
from __future__ import annotations

import ctypes as C
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000

_DLL_NAME = "qwen3tts.dll"
# Names are fixed by the qwen3-tts.cpp loader, whatever quantization they hold.
_MODEL_FILES = ("qwen3-tts-0.6b-q8_0.gguf", "qwen3-tts-tokenizer-f16.gguf")
_RUNTIME_DLLS = (_DLL_NAME, "ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll")

# The verified configuration (build recipe, section 1). These are read by the
# DLL through getenv, once, so they must be in the environment before the first
# native call. ``setdefault`` keeps a developer's own override.
_ENV_DEFAULTS = {
    "QWEN3_TTS_BACKEND": "cuda",
    # Decoder look-back per streamed window (12.5 frames/s -> 3.8 s). The
    # upstream default. 24 saved ~130 MB of VRAM and sounded clean on short
    # lines, but on long lines the user heard an occasional mid-sentence tone
    # shift; 48 was their pick from a long-line comparison (2026-11-27).
    "QWEN3_TTS_STREAM_CONTEXT_FRAMES": "48",
    # Frames per streamed chunk (12.5 frames/s). Upstream 25 waits 2 s of audio
    # before the first delivery; 10 matches the standard engine's chunk size
    # and halves time-to-first-audio (1.9 s -> 1.0 s measured).
    "QWEN3_TTS_STREAM_CHUNK_FRAMES": "10",
    # Cut the reference echo at exactly n_ref_frames x 1920 samples. Upstream's
    # wider margin removes the first 1-3 words of the line.
    "QWEN3_TTS_ECHO_MARGIN_FRAMES": "0",
}

# Upper bound on generated codec frames (12.5 frames/s -> 41 s of audio). The
# upstream default of 4096 sizes a 450 MB attention cache.
MAX_AUDIO_FRAMES = 512


class LightningError(RuntimeError):
    """The Lightning engine could not load or could not produce audio."""


class _Params(C.Structure):
    _fields_ = [
        ("max_audio_tokens", C.c_int32),
        ("temperature", C.c_float),
        ("top_p", C.c_float),
        ("top_k", C.c_int32),
        ("n_threads", C.c_int32),
        ("repetition_penalty", C.c_float),
        ("language_id", C.c_int32),
        ("ref_code_path", C.c_char_p),
    ]


class _Audio(C.Structure):
    _fields_ = [
        ("samples", C.POINTER(C.c_float)),
        ("n_samples", C.c_int32),
        ("sample_rate", C.c_int32),
    ]


_StreamCallback = C.CFUNCTYPE(C.c_int, C.POINTER(C.c_float), C.c_int32, C.c_void_p)


def lightning_dir() -> Path:
    """Folder holding ``bin/`` (DLLs) and ``models/`` (GGUFs).

    ``MYVOICE_LIGHTNING_DIR`` overrides; otherwise ``<app root>/lightning``,
    which is the repo root in development (see ``build_tools/stage_lightning.py``)
    and the install folder in a frozen build.
    """
    override = os.environ.get("MYVOICE_LIGHTNING_DIR")
    if override:
        return Path(override)
    from myvoice.utils.portable_paths import get_app_root

    root = get_app_root()
    candidate = root / "lightning"
    if not candidate.is_dir() and getattr(sys, "frozen", False):
        # PyInstaller one-dir builds place collected data under _internal.
        internal = root / "_internal" / "lightning"
        if internal.is_dir():
            return internal
    return candidate


def _cuda_dll_dirs() -> List[Path]:
    """Folders holding cudart64_12 / cublas64_12 / cublasLt64_12.

    The engine reuses the CUDA 12.8 runtime torch already ships rather than
    bundling a second 800 MB copy.
    """
    dirs: List[Path] = []
    try:
        import importlib.util

        spec = importlib.util.find_spec("torch")
        if spec is not None and spec.origin:
            lib = Path(spec.origin).parent / "lib"
            if lib.is_dir():
                dirs.append(lib)
    except Exception:  # pragma: no cover - defensive
        pass
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        lib = Path(bundle) / "torch" / "lib"
        if lib.is_dir() and lib not in dirs:
            dirs.append(lib)
    return dirs


def availability() -> Tuple[bool, str]:
    """Whether Lightning can be offered on this machine: ``(ok, reason)``.

    ``reason`` is a short sentence fit for a tooltip when ``ok`` is False.
    Cheap: checks files and the GPU, loads nothing.
    """
    if sys.platform != "win32":
        return False, "Lightning mode is only available on Windows."
    root = lightning_dir()
    missing = [n for n in _RUNTIME_DLLS if not (root / "bin" / n).is_file()]
    missing += [n for n in _MODEL_FILES if not (root / "models" / n).is_file()]
    if missing:
        return False, "Lightning mode files are not installed."
    if not any((d / "cublas64_12.dll").is_file() for d in _cuda_dll_dirs()):
        return False, "Lightning mode needs the CUDA runtime, which was not found."
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "Lightning mode needs an NVIDIA GPU."
    except Exception:
        return False, "Lightning mode needs an NVIDIA GPU."
    return True, ""


def _path_bytes(path: Path) -> bytes:
    # The DLL opens files with the narrow C runtime, i.e. the ANSI code page.
    return str(path).encode("mbcs")


class LightningEngine:
    """One loaded qwen3-tts.cpp model. Load and unload are explicit."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self._root = root or lightning_dir()
        self._lib: Optional[C.CDLL] = None
        self._handle: Optional[int] = None
        self._lock = threading.Lock()
        self._dll_dir_handles: list = []

    @property
    def is_loaded(self) -> bool:
        return self._handle is not None

    # ----- lifecycle ---------------------------------------------------- #

    def load(self, n_threads: int = 4) -> None:
        """Load the DLLs and the model onto the GPU. Blocking (~1 s)."""
        with self._lock:
            if self._handle is not None:
                return
            for key, value in _ENV_DEFAULTS.items():
                os.environ.setdefault(key, value)
            lib = self._load_library()
            handle = lib.qwen3_tts_create(_path_bytes(self._root / "models"), n_threads)
            if not handle:
                raise LightningError("The Lightning model failed to load.")
            self._lib, self._handle, self._n_threads = lib, handle, n_threads
            logger.info("Lightning engine loaded from %s", self._root)

    def unload(self) -> None:
        """Free the model and its VRAM. Waits for a generation in flight."""
        with self._lock:
            if self._handle is None:
                return
            self._lib.qwen3_tts_destroy(self._handle)
            self._handle = None
            logger.info("Lightning engine unloaded")

    def _load_library(self) -> C.CDLL:
        if self._lib is not None:
            return self._lib
        bin_dir = self._root / "bin"
        for d in [bin_dir, *_cuda_dll_dirs()]:
            self._dll_dir_handles.append(os.add_dll_directory(str(d)))
        try:
            lib = C.CDLL(str(bin_dir / _DLL_NAME))
        except OSError as exc:
            raise LightningError(f"The Lightning engine could not be loaded: {exc}") from exc

        lib.qwen3_tts_default_params.argtypes = [C.POINTER(_Params)]
        lib.qwen3_tts_default_params.restype = None
        lib.qwen3_tts_create.argtypes = [C.c_char_p, C.c_int32]
        lib.qwen3_tts_create.restype = C.c_void_p
        lib.qwen3_tts_destroy.argtypes = [C.c_void_p]
        lib.qwen3_tts_destroy.restype = None
        lib.qwen3_tts_get_error.argtypes = [C.c_void_p]
        lib.qwen3_tts_get_error.restype = C.c_char_p
        lib.qwen3_tts_free_audio.argtypes = [C.POINTER(_Audio)]
        lib.qwen3_tts_free_audio.restype = None
        lib.qwen3_tts_synthesize_in_context_streaming.argtypes = [
            C.c_void_p, C.c_char_p, C.c_char_p, C.c_char_p,
            C.POINTER(_Params), _StreamCallback, C.c_void_p,
        ]
        lib.qwen3_tts_synthesize_in_context_streaming.restype = C.POINTER(_Audio)
        lib.qwen3_tts_extract_embedding_file.argtypes = [
            C.c_void_p, C.c_char_p, C.POINTER(C.c_float), C.c_int32,
        ]
        lib.qwen3_tts_extract_embedding_file.restype = C.c_int32
        lib.qwen3_tts_synthesize_with_embedding.argtypes = [
            C.c_void_p, C.c_char_p, C.POINTER(C.c_float), C.c_int32, C.POINTER(_Params),
        ]
        lib.qwen3_tts_synthesize_with_embedding.restype = C.POINTER(_Audio)
        return lib

    # ----- synthesis ---------------------------------------------------- #

    def _params(self, ref_code_path: Optional[Path]) -> _Params:
        params = _Params()
        self._lib.qwen3_tts_default_params(C.byref(params))
        params.max_audio_tokens = MAX_AUDIO_FRAMES
        params.n_threads = self._n_threads
        # Kept alive by the caller holding ``params`` for the native call.
        params.ref_code_path = _path_bytes(ref_code_path) if ref_code_path else None
        return params

    def _last_error(self) -> str:
        raw = self._lib.qwen3_tts_get_error(self._handle)
        return raw.decode("utf-8", "replace") if raw else "unknown error"

    def _take_audio(self, result) -> np.ndarray:
        audio = result.contents
        pcm = np.ctypeslib.as_array(audio.samples, shape=(audio.n_samples,)).astype(
            np.float32, copy=True
        )
        self._lib.qwen3_tts_free_audio(result)
        return pcm

    def synthesize_icl_stream(
        self,
        text: str,
        ref_audio: Path,
        ref_text: str,
        on_chunk: Callable[[np.ndarray], bool],
        ref_code_path: Optional[Path] = None,
    ) -> np.ndarray:
        """Speak ``text`` in the voice and accent of the reference, streaming.

        ``on_chunk(pcm)`` receives each new stretch of 24 kHz mono float32
        audio as it is decoded and returns False to stop the generation.
        Returns everything delivered. ``ref_code_path`` is the reference's
        precomputed codec codes (see ``ref_codes``); without it the DLL encodes
        the reference itself, which costs ~200 MB of VRAM and ~0.4 s.
        """
        with self._lock:
            if self._handle is None:
                raise LightningError("The Lightning engine is not loaded.")
            callback_error: List[BaseException] = []

            def _on_native_chunk(samples, n_samples, _user) -> int:
                try:
                    pcm = np.ctypeslib.as_array(samples, shape=(n_samples,)).astype(
                        np.float32, copy=True
                    )
                    return 1 if on_chunk(pcm) else 0
                except BaseException as exc:  # never unwind into native code
                    callback_error.append(exc)
                    return 0

            callback = _StreamCallback(_on_native_chunk)
            params = self._params(ref_code_path)
            result = self._lib.qwen3_tts_synthesize_in_context_streaming(
                self._handle, text.encode("utf-8"), _path_bytes(ref_audio),
                ref_text.encode("utf-8"), C.byref(params), callback, None,
            )
            if callback_error:
                if result:
                    self._lib.qwen3_tts_free_audio(result)
                raise callback_error[0]
            if not result:
                raise LightningError(f"Lightning generation failed: {self._last_error()}")
            return self._take_audio(result)

    def synthesize_xvector(self, text: str, ref_audio: Path) -> np.ndarray:
        """Speak ``text`` in the timbre of the reference (no transcript needed).

        Not streamed: the C API has no streaming entry point for this mode, so
        the whole line comes back at once.
        """
        with self._lock:
            if self._handle is None:
                raise LightningError("The Lightning engine is not loaded.")
            embedding = (C.c_float * 4096)()
            n = self._lib.qwen3_tts_extract_embedding_file(
                self._handle, _path_bytes(ref_audio), embedding, 4096
            )
            if n <= 0:
                raise LightningError(
                    f"Lightning could not read the reference audio: {self._last_error()}"
                )
            params = self._params(None)
            result = self._lib.qwen3_tts_synthesize_with_embedding(
                self._handle, text.encode("utf-8"), embedding, n, C.byref(params)
            )
            if not result:
                raise LightningError(f"Lightning generation failed: {self._last_error()}")
            return self._take_audio(result)
