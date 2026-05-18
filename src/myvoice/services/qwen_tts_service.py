"""
Qwen3-TTS Service Implementation

This module implements the core Text-to-Speech service using embedded Qwen3-TTS,
replacing the external GPT-SoVITS backend. Supports CustomVoice, VoiceDesign,
and Base (voice clone) models with lazy loading.

Story 1.1: QwenTTSService Core Integration
Story 1.2: Streaming Audio Output
Story 1.3: Error Handling
Story 1.4: Text Validation
Story 1.5: Startup & Bundled Voices
"""

import asyncio
import json
import logging
import re
import tempfile
import threading
import time
import weakref
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Tuple, AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import soundfile as sf
import numpy as np

# Import library's VoiceClonePromptItem with alias to avoid conflict with our wrapper class
try:
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem as LibraryVoiceClonePromptItem
except ImportError:
    LibraryVoiceClonePromptItem = None


class StartupState(Enum):
    """
    State of TTS service startup/initialization.

    Used to track progress through startup phases for UI feedback.
    """
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    LOADING_MODEL = "loading_model"
    READY = "ready"
    FAILED = "failed"


class GenerationMode(Enum):
    """TTS generation mode."""
    BATCH = "batch"           # Generate all at once
    STREAMING = "streaming"   # Generate in chunks progressively


# Legacy: still consumed by get_service_metrics() and the existing UI
# status indicator. Story 12.1 (Epic 12) rewires UI subscribers to
# SessionRegistry.session_state_changed; once Epic 12 ships, this enum
# becomes a candidate for removal in a subsequent pass (D-14).
class GenerationState(Enum):
    """Current state of TTS generation."""
    IDLE = "idle"
    LOADING_MODEL = "loading_model"
    GENERATING = "generating"
    STREAMING = "streaming"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    ERROR = "error"


class TTSErrorCode(Enum):
    """
    Error codes for TTS generation failures.

    Used to categorize errors for appropriate user messaging
    and recovery suggestions.
    """
    # Recoverable errors
    OUT_OF_MEMORY = "out_of_memory"
    CUDA_ERROR = "cuda_error"
    TIMEOUT = "timeout"
    STREAMING_FAILED = "streaming_failed"

    # User action required
    EMPTY_TEXT = "empty_text"
    TEXT_TOO_LONG = "text_too_long"
    INVALID_VOICE = "invalid_voice"
    INVALID_AUDIO_FILE = "invalid_audio_file"

    # System errors
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_LOAD_FAILED = "model_load_failed"
    SERVICE_NOT_RUNNING = "service_not_running"

    # Unknown
    UNKNOWN = "unknown"


@dataclass
class TTSError:
    """
    Structured TTS error with user-friendly messaging.

    Attributes:
        code: Error code for categorization
        user_message: User-friendly error message (what happened)
        recovery_suggestion: Actionable suggestion for the user (what to do)
        technical_details: Technical error details for logging
        is_recoverable: Whether the user can retry
        used_fallback: Whether batch fallback was attempted
    """
    code: TTSErrorCode
    user_message: str
    recovery_suggestion: str
    technical_details: Optional[str] = None
    is_recoverable: bool = True
    used_fallback: bool = False

    def __str__(self) -> str:
        """Format as user-friendly message with suggestion."""
        return f"{self.user_message} {self.recovery_suggestion}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "code": self.code.value,
            "user_message": self.user_message,
            "recovery_suggestion": self.recovery_suggestion,
            "technical_details": self.technical_details,
            "is_recoverable": self.is_recoverable,
            "used_fallback": self.used_fallback,
        }


class TextValidationStatus(Enum):
    """Status of text validation."""
    VALID = "valid"
    EMPTY = "empty"
    WHITESPACE_ONLY = "whitespace_only"
    TOO_LONG = "too_long"  # Warning, not error


@dataclass
class TextValidationResult:
    """
    Result of text validation for TTS generation.

    Attributes:
        is_valid: Whether text can be used for generation
        status: Validation status code
        message: User-facing message (if any)
        can_proceed: Whether generation can proceed (may be True with warnings)
        warning: Warning message if text is valid but has issues
        character_count: Number of characters in text
    """
    is_valid: bool
    status: TextValidationStatus
    message: Optional[str] = None
    can_proceed: bool = True
    warning: Optional[str] = None
    character_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_valid": self.is_valid,
            "status": self.status.value,
            "message": self.message,
            "can_proceed": self.can_proceed,
            "warning": self.warning,
            "character_count": self.character_count,
        }


@dataclass
class VoiceClonePromptItem:
    """
    Normalized voice clone prompt data structure.

    Wraps the result from Qwen3-TTS create_voice_clone_prompt() to ensure
    consistent access regardless of library version changes.

    This class is compatible with the Qwen3-TTS library's VoiceClonePromptItem
    and includes all required fields for the library's _prompt_items_to_voice_clone_prompt()
    method.

    Attributes:
        ref_code: Reference audio code tensor (may be None)
        ref_spk_embedding: Reference speaker embedding tensor
        x_vector_only_mode: Whether to use x-vector only mode (no ICL)
        icl_mode: Whether ICL mode is enabled
        ref_text: Reference text for ICL mode (may be None)
    """
    ref_code: Any = None
    ref_spk_embedding: Any = None
    x_vector_only_mode: bool = False
    icl_mode: bool = True
    ref_text: Optional[str] = None


@dataclass
class BundledVoiceConfig:
    """
    Configuration for bundled voice defaults.

    Used during startup to initialize TTS with sensible defaults
    for immediate use without additional configuration.

    Attributes:
        speaker: Default speaker for CustomVoice model
        language: Default language setting
        emotion_preset: Default emotion preset
        preload_on_startup: Whether to preload model at startup
        startup_timeout_seconds: Maximum time to wait for model loading
    """
    speaker: str = "Ryan"  # English native, dynamic male
    language: str = "Auto"
    emotion_preset: str = "neutral"
    preload_on_startup: bool = True
    startup_timeout_seconds: float = 30.0  # NFR2: Model loads within 30 seconds


@dataclass
class StartupProgress:
    """
    Progress information during TTS startup.

    Emitted via callback to allow UI to show loading indicator.
    """
    state: StartupState
    progress_percent: float = 0.0
    message: str = ""
    is_complete: bool = False
    is_ready: bool = False


from myvoice.services.core.base_service import BaseService, ServiceStatus
from myvoice.services.model_registry import ModelRegistry, ModelLoadProgress
from myvoice.models.service_enums import ModelState, QwenModelType
from myvoice.models.error import MyVoiceError, ErrorSeverity
from myvoice.models.ui_state import ServiceHealthStatus
from myvoice.observability import metrics, MetricRecord
# Story 11.4: Phase 1 D-20 — wire SessionRegistry into the generation flow.
from myvoice.services.sessions.session_registry import SessionRegistry
from myvoice.services.sessions.generation_session import SessionSource
# Story 16.6: TRUE_STREAM dispatch + three-mode fallback chain (Phase ⊥).
from myvoice.services.tts_streaming import (
    StreamingMode,
    effective_streaming_mode,
    CodecTokenStreamer,
    StreamingDecoderWorker,
    END_OF_STREAM,
)
from myvoice.models.app_settings import AppSettings


@dataclass
class AudioChunk:
    """
    A chunk of generated audio for streaming playback.

    Attributes:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        chunk_index: Index of this chunk (0-based)
        is_final: Whether this is the last chunk
        text_segment: The text that was synthesized for this chunk
        session_id: Registry-issued session id for the generation that
            emitted this chunk; None for legacy callers that bypass the
            SessionRegistry. Story 18.1 code-review pass M1 added this
            so the consumer-side instrumentation metrics
            (``progressive_chunk_playback_arrival_ms`` /
            ``progressive_chunk_audio_duration_ms``) carry the same
            session_id the producer-side ``progressive_chunk_emit_ms``
            does — keeping the Task 1.4 CSV joinable across multiple
            generations captured in one run.
    """
    audio_data: np.ndarray
    sample_rate: int
    chunk_index: int
    is_final: bool = False
    text_segment: str = ""
    session_id: Optional[str] = None


@dataclass
class QwenTTSRequest:
    """
    Request model for Qwen3-TTS generation.

    Attributes:
        text: Text to convert to speech
        language: Language code (English, Chinese, Japanese, etc. or "Auto")
        model_type: Which Qwen3-TTS model to use
        speaker: Speaker name for CustomVoice model
        instruct: Emotion/style instruction for CustomVoice/VoiceDesign
        ref_audio: Reference audio path for voice cloning (Base model)
        ref_text: Transcript of reference audio for cloning (ICL mode)
        x_vector_only_mode: If True, use x-vector mode (no ref_text needed); if False, use ICL mode (ref_text required)
        voice_description: Text description for VoiceDesign model
        streaming: Whether to use streaming mode (progressive chunk generation)
        checkpoint_path: Path to fine-tuned checkpoint (OPTIMIZED voices only)
        voice_clone_prompt: Pre-computed voice clone prompt tensor for embedding-based generation (QA5)
    """
    text: str
    language: str = "Auto"
    model_type: QwenModelType = QwenModelType.CUSTOM_VOICE
    speaker: str = "Ryan"
    instruct: Optional[str] = None
    ref_audio: Optional[Path] = None
    ref_text: Optional[str] = None
    x_vector_only_mode: bool = False  # False=ICL mode (needs ref_text), True=x-vector mode (no ref_text needed)
    voice_description: Optional[str] = None
    streaming: bool = True  # Default to streaming mode
    checkpoint_path: Optional[Path] = None  # Fine-tuned checkpoint path for OPTIMIZED voices
    voice_clone_prompt: Optional[Any] = None  # QA5: Pre-computed voice clone prompt for embedding voices


@dataclass
class QwenTTSResponse:
    """
    Response model for Qwen3-TTS generation.

    Attributes:
        success: Whether generation was successful
        audio_data: Generated audio as numpy array (complete audio)
        sample_rate: Audio sample rate (typically 24000)
        audio_file_path: Path to saved audio file
        error_message: Error description if failed
        generation_time_seconds: Time taken for generation
        mode: Generation mode used (batch or streaming)
        chunks_generated: Number of chunks generated (streaming mode)
        first_chunk_latency: Time to first audio chunk in seconds (streaming mode)
        used_fallback: True if a lower-priority mode in the three-mode
            fallback chain handled the request (Story 16.6). Independent of
            ``success`` — a successful BATCH fallback after TRUE_STREAM
            failed sets both ``success=True`` and ``used_fallback=True``.
    """
    success: bool = False
    audio_data: Optional[np.ndarray] = None
    sample_rate: int = 24000
    audio_file_path: Optional[Path] = None
    error_message: Optional[str] = None
    generation_time_seconds: Optional[float] = None
    mode: GenerationMode = GenerationMode.BATCH
    chunks_generated: int = 0
    first_chunk_latency: Optional[float] = None
    used_fallback: bool = False


class _FirstChunkLatencyAggregator:
    """
    Derives ``_avg_first_chunk_latency`` from the metrics stream (Story 11.3).

    The architecture (P-9) mandates that running averages over telemetry
    aggregate from the metric stream, not from inline arithmetic at the call
    site. This aggregator subscribes to ``metrics.record(...)`` and updates
    the service's ``_avg_first_chunk_latency`` field whenever a
    ``first_chunk_latency_ms`` record fires.

    Note on units (AC #13): the metric is emitted in **milliseconds**
    (architecture's chosen unit) but ``_avg_first_chunk_latency`` is held in
    **seconds** for backward-compat with the public ``get_service_metrics()``
    surface (key ``"avg_first_chunk_latency"``). The aggregator divides by
    1000 before applying the running-average update.

    Note on the denominator (AC #15): ``_streaming_requests`` is incremented
    at request entry (``try:`` block at the top of the streaming generation
    path), not at successful completion. The aggregator reads the live
    counter from the service rather than maintaining its own — this
    preserves the pre-migration semantics ("requests attempted" not
    "requests completed") that the inline math relied on. Migrating the
    counter into the metric stream would shift increment timing on
    early-failure paths, which is a behavior change AC #18 forbids.

    Thread safety: ``__call__`` guards its read-then-write of
    ``_avg_first_chunk_latency`` with a per-aggregator ``threading.Lock``
    so two listener invocations cannot interleave their running-average
    update. The increment of ``_streaming_requests`` itself is NOT under
    this lock — it lives on the streaming generation path and is part of
    the service's pre-existing single-threaded streaming contract; the
    pre-migration inline math had the same boundary.
    """

    def __init__(self, service: "QwenTTSService") -> None:
        self._service = service
        # Guard the read-modify-write of ``_avg_first_chunk_latency`` so
        # concurrent listener invocations cannot lose updates. Cheap;
        # uncontended in the current single-threaded streaming path.
        self._lock = threading.Lock()
        # Register synchronously so any record() emitted between this line
        # and a later cleanup is captured. The unsubscribe must be invoked
        # in the service's stop() path (or wherever shutdown teardown runs).
        self.unsubscribe: Callable[[], None] = metrics.add_listener(self)

    def __call__(self, record: MetricRecord) -> None:
        if record.name != "first_chunk_latency_ms":
            return
        # ms → s. The roundtrip * 1000 / 1000 introduces float noise on the
        # order of 1e-13 — well within AC #13's 1e-9 tolerance.
        value_seconds = record.value / 1000.0
        with self._lock:
            n = self._service._streaming_requests
            if n <= 0:
                # Defensive — the metric is only emitted from the streaming
                # path AFTER ``_streaming_requests += 1``, so n must be >= 1.
                # Returning silently keeps a misuse from corrupting the
                # running average.
                return
            prior = self._service._avg_first_chunk_latency
            self._service._avg_first_chunk_latency = (
                prior * (n - 1) + value_seconds
            ) / n


class QwenTTSService(BaseService):
    """
    Core Text-to-Speech service using embedded Qwen3-TTS.

    This service provides the main interface for TTS generation using Qwen3-TTS
    models locally. It replaces the external GPT-SoVITS backend with embedded
    inference for offline operation.

    Features:
    - Lazy model loading with ModelRegistry
    - CustomVoice generation with bundled speakers
    - Emotion/style control via instruct parameter
    - Voice Design from text descriptions
    - Voice Cloning from audio samples
    - Streaming mode with progressive chunk generation (NFR1: <2s first chunk)
    - Async operation with non-blocking generation
    - PyQt6 signal integration for UI updates

    Callbacks (to be connected by UI layer):
    - generation_started: Emitted when generation begins (for visual indicator)
    - generation_complete: Emitted when generation finishes successfully
    - generation_failed: Emitted when generation fails
    - audio_chunk_ready: Emitted for each audio chunk in streaming mode
    - model_loading: Emitted when a model starts loading
    - model_ready: Emitted when a model finishes loading
    """

    # Default emotion instructions for presets
    EMOTION_PRESETS = {
        "neutral": None,  # No instruct for neutral
        "happy": "Speak happily and cheerfully",
        "sad": "Speak sadly and melancholically",
        "angry": "Speak angrily and forcefully",
        "flirtatious": "Speak flirtatiously and playfully",
    }

    # Sentence splitting pattern for streaming mode
    # Splits on sentence-ending punctuation while preserving the punctuation
    SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?。！？])\s*')

    # Minimum chunk length to avoid very short generations
    MIN_CHUNK_LENGTH = 10

    # Text length limits (FR5)
    MAX_TEXT_LENGTH_WARNING = 5000  # Warn but allow
    MAX_TEXT_LENGTH_HARD = 50000    # Hard limit to prevent OOM

    # ----- Story 11.4: session-label resolvers (AC #12) ----------------- #

    @staticmethod
    def _resolve_voice_label(request: "QwenTTSRequest") -> str:
        """Map a request's identity fields to a single human-readable voice
        label for SessionRegistry.create_session(voice=...) (Story 11.4).

        Resolution priority (revised 2026-05-06 after Epic 14 smoke test):

          1. ``request.speaker`` — but ONLY if it is non-default (i.e. the
             user explicitly set it). The dataclass default is the literal
             ``"Ryan"``; treat that as a sentinel meaning "not explicitly
             chosen" so the request paths that leave ``speaker`` untouched
             (``generate_voice_design``, ``generate_voice_clone``,
             ``generate_voice_clone_with_embedding``) fall through to
             their own identifiers instead of mislabeling every save as
             "Ryan".
          2. ``request.voice_description`` — voice-design requests.
          3. ``Path(request.checkpoint_path).stem`` — optimized fine-tuned
             voice (e.g. "Sarira").
          4. ``Path(request.ref_audio).stem`` — voice-clone (BASE) requests.
          5. ``request.speaker`` (default ``"Ryan"``) — fallback when the
             user actually picked Ryan via the voice selector AND no
             other identifier was set. Also catches the still-broken
             embedding-only path (no speaker, no description, no
             checkpoint, no ref_audio — only ``voice_clone_prompt``
             tensor); a follow-up should add an explicit ``voice_label``
             field to ``QwenTTSRequest`` for that case.
          6. ``"unknown"`` — true fallback for empty speaker.
        """
        # Sentinel-aware first pass: prefer a speaker that the caller
        # explicitly overrode away from the dataclass default.
        if request.speaker and request.speaker != "Ryan":
            return request.speaker
        if request.voice_description:
            return request.voice_description
        if request.checkpoint_path:
            return Path(request.checkpoint_path).stem
        if request.ref_audio:
            return Path(request.ref_audio).stem
        if request.speaker:
            # User explicitly picked "Ryan", or the dataclass default
            # leaked through an embedding-only request — either way,
            # this is the most-informative label available.
            return request.speaker
        return "unknown"

    @staticmethod
    def _resolve_model_type_label(request: "QwenTTSRequest") -> str:
        """Resolve ``request.model_type.display_name`` (Story 11.3 metric tag
        convention) for SessionRegistry.create_session(model_type=...).

        Defensive ``"default"`` fallback if model_type is unexpectedly None.
        """
        if request.model_type is None:
            return "default"
        return request.model_type.display_name

    def __init__(
        self,
        audio_coordinator: Optional['AudioCoordinator'] = None,
        device: str = "auto",
        dtype: str = "bfloat16",
        models_path: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        max_concurrent_requests: int = 1,
        quality_tier: str = "quality",
        *,
        session_registry: Optional[SessionRegistry] = None,
        app_settings: Optional[AppSettings] = None,
    ):
        """
        Initialize the Qwen TTS service.

        Args:
            audio_coordinator: AudioCoordinator for dual-service audio routing
            device: PyTorch device ("auto", "cuda:0", "cpu")
            dtype: PyTorch dtype ("bfloat16", "float16", "float32")
            models_path: Optional local path for model weights
            cache_dir: Directory for caching generated audio
            max_concurrent_requests: Maximum concurrent TTS requests (default 1)
            quality_tier: Model quality tier ("small" or "quality")
            session_registry: Optional SessionRegistry for Story 11.4 session
                lifecycle wiring. When None, the service runs the legacy
                code path unchanged (registry-disabled fallback for tests
                or environments where Qt isn't available).
            app_settings: Optional AppSettings carrying user preferences. Story
                16.6 reads ``streaming_mode_override`` from this object to
                resolve the streaming mode at dispatch time. When None, the
                resolver behaves as if the override is None (Auto: hardware
                probe decides) per Story 16.2's contract.
        """
        super().__init__("QwenTTSService")

        # Audio integration
        self.audio_coordinator = audio_coordinator
        if self.audio_coordinator:
            self.logger.info("QwenTTSService using AudioCoordinator")

        # Story 11.4: SessionRegistry wiring (D-20 Phase 1).
        self._session_registry: Optional[SessionRegistry] = session_registry
        if self._session_registry is not None:
            self.logger.info("QwenTTSService using SessionRegistry")

        # Story 16.6: AppSettings carries streaming_mode_override; consumed by
        # _resolve_streaming_mode at dispatch entry. None means "Auto" (the
        # hardware probe in default_streaming_mode_for_hardware decides).
        self._app_settings: Optional[AppSettings] = app_settings

        # Story 16.5: tracked for cancel_generation's request_cancel call;
        # set after each create_session in the streaming/batch dispatch
        # forks and cleared on every terminal discard post. None when no
        # generation is in flight or when no registry is wired.
        self._current_session_id: Optional[str] = None

        # Model registry for lazy loading. Story 18.3 — pass app_settings so
        # ModelRegistry can route ``tts_precision`` through the precedence
        # resolver (resolve_tts_precision honors the Ampere+ probe gate).
        # The legacy ``dtype`` string is preserved as the fallback when
        # ``app_settings`` is None or its ``tts_precision`` is None
        # (backwards-compatible with tests / non-AppSettings call sites).
        self._model_registry = ModelRegistry(
            device=device,
            dtype=dtype,
            models_path=models_path,
            progress_callback=self._on_model_progress,
            quality_tier=quality_tier,
            app_settings=self._app_settings,
        )

        # Configuration
        self._cache_dir = cache_dir or Path(tempfile.gettempdir())
        self._current_audio_cache = self._cache_dir / "myvoice_current.wav"
        self._max_concurrent = max_concurrent_requests

        # Service components
        self._executor: Optional[ThreadPoolExecutor] = None
        self._request_semaphore: Optional[asyncio.Semaphore] = None

        # Callbacks
        self._generation_started_callback: Optional[Callable[[], None]] = None
        self._generation_complete_callback: Optional[Callable[[Path], None]] = None
        self._generation_failed_callback: Optional[Callable[[str], None]] = None
        self._generation_error_callback: Optional[Callable[[TTSError], None]] = None
        self._generation_cancelled_callback: Optional[Callable[[], None]] = None
        self._text_validation_callback: Optional[Callable[[TextValidationResult], None]] = None
        self._audio_chunk_ready_callback: Optional[Callable[[AudioChunk], None]] = None
        self._model_loading_callback: Optional[Callable[[str], None]] = None
        self._model_ready_callback: Optional[Callable[[str], None]] = None
        self._health_status_callback: Optional[Callable[[ServiceHealthStatus, Optional[str]], None]] = None

        # Track last model state for replay when UI connects (Fix: startup indicator timing)
        self._last_model_loading_message: Optional[str] = None
        self._last_model_ready_name: Optional[str] = None
        self._is_model_loading: bool = False

        # Generation state
        self._generation_state = GenerationState.IDLE
        self._cancel_requested = False
        self._current_generation_task: Optional[asyncio.Task] = None
        self._last_error: Optional[TTSError] = None

        # Startup state (Story 1.5)
        self._startup_state = StartupState.NOT_STARTED
        self._bundled_config = BundledVoiceConfig()
        self._startup_progress_callback: Optional[Callable[[StartupProgress], None]] = None
        self._tts_ready_callback: Optional[Callable[[], None]] = None

        # Metrics
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._last_generation_time: Optional[float] = None
        self._streaming_requests = 0
        self._avg_first_chunk_latency: float = 0.0
        self._fallback_count = 0  # Count of streaming->batch fallbacks

        # Story 11.3: derive _avg_first_chunk_latency from the metric stream
        # via the single-chokepoint helper (architecture P-9). The aggregator
        # subscribes in its __init__; ``stop()`` calls ``unsubscribe`` to
        # drop the registration.
        self._latency_aggregator = _FirstChunkLatencyAggregator(self)

        # Voice clone prompt cache (Story 17.2 — wired into generate_voice_clone;
        # Story 17.2 review pass — H1 + H3 fixes).
        # Key: (str(ref_audio.resolve()), tier) where tier ∈ {"quality", "small"}.
        # Tier-locked because the 1.7B/0.6B Qwen3 models produce embeddings with
        # different hidden dimensions; the same .pt cannot serve both.
        # Value shape: (prompt, ref_audio_mtime, ref_audio_size, txt_mtime).
        # The mtime/size triple is re-validated on every cache hit so that
        # within-session changes to ref_audio (replace .wav) or .txt (fix
        # transcription) invalidate the in-memory cache without requiring
        # an app restart. (H1, H2 review-pass fixes.)
        # Backed by ``OrderedDict`` with LRU eviction at
        # ``_VOICE_CLONE_PROMPT_CACHE_MAX`` to bound memory growth on long-
        # running sessions with large voice libraries (M3 review-pass fix).
        self._voice_clone_prompts: "OrderedDict[Tuple[str, str], Tuple[Any, float, int, Optional[float]]]" = (
            OrderedDict()
        )
        # Per-voice asyncio.Lock registry. WeakValueDictionary so locks for
        # deleted/unloaded voices are GC'd; the registry mutation itself is
        # guarded by `_voice_clone_prompt_locks_guard` (lazy-init under
        # asyncio because Lock requires a running event loop).
        self._voice_clone_prompt_locks: "weakref.WeakValueDictionary[Tuple[str, str], asyncio.Lock]" = (
            weakref.WeakValueDictionary()
        )
        self._voice_clone_prompt_locks_guard: Optional[asyncio.Lock] = None

        # Story 17.2 — Whisper integration for lazy CLONED-voice precompute.
        # Wired post-construction by the orchestrator after on-demand
        # WhisperSubprocessService init lands (app.py:_initialize_whisper_
        # service_on_demand). When None, precompute raises so the dispatch
        # chain falls through to SENTENCE_STREAM (NFR7 preserved).
        self._whisper_service: Optional[Any] = None
        # Optional fire-and-forget callback the orchestrator wires so the
        # precompute can request on-demand Whisper init when it discovers
        # _whisper_service is None on the very first call.
        self._whisper_init_callback: Optional[Callable[[], None]] = None

        # Story 17.2 — VoiceProfileManager handle for startup hydration of
        # the cache. Wired post-construction; hydration is a separate explicit
        # call, not auto-fired in start(), because the orchestrator constructs
        # the manager AFTER tts.start() returns.
        self._voice_profile_manager: Optional[Any] = None

        # Story 17.2 — UI feedback for first-run precompute (AC #4). The
        # orchestrator wires a callback that translates the (Optional[str])
        # message into a ServiceStatusInfo update for the TTS indicator.
        self._preparing_voice_callback: Optional[Callable[[Optional[str]], None]] = None

        self.logger.info(
            f"QwenTTSService initialized: device={device}, cache_dir={self._cache_dir}"
        )

    async def start(self) -> bool:
        """
        Start the TTS service.

        Note: Model loading is deferred until first generation request (lazy loading).

        Returns:
            bool: True if service started successfully
        """
        try:
            await self._update_status(ServiceStatus.STARTING)
            self.logger.info("Starting QwenTTSService")

            # Initialize thread executor
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_concurrent,
                thread_name_prefix="QwenTTS"
            )

            # Initialize request semaphore
            self._request_semaphore = asyncio.Semaphore(self._max_concurrent)

            # Ensure cache directory exists
            self._cache_dir.mkdir(parents=True, exist_ok=True)

            # Service is ready (models will be loaded lazily on first request)
            await self._update_status(ServiceStatus.RUNNING)
            self.logger.info("QwenTTSService started (models will load on first use)")

            # Notify health status
            if self._health_status_callback:
                self._health_status_callback(ServiceHealthStatus.HEALTHY, None)

            return True

        except Exception as e:
            self.logger.exception(f"Failed to start QwenTTSService: {e}")
            await self._update_status(ServiceStatus.ERROR)
            return False

    async def initialize_with_defaults(
        self,
        config: Optional[BundledVoiceConfig] = None,
        preferred_model_type: Optional[QwenModelType] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Initialize TTS with bundled voice defaults and preload model.

        This is the recommended startup method for MyVoice. It:
        1. Starts the service if not already running
        2. Preloads the appropriate model (based on cached voice or defaults)
        3. Configures default speaker for immediate use
        4. Emits progress and ready callbacks for UI

        NFR2: Model loading completes within 30 seconds.

        Args:
            config: Optional configuration override (uses defaults if None)
            preferred_model_type: Model to preload at startup. If None, uses CUSTOM_VOICE.
                                 Pass the model type from cached active voice profile
                                 for faster first generation.

        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        import asyncio

        self._bundled_config = config or BundledVoiceConfig()
        self._startup_state = StartupState.INITIALIZING

        self._emit_startup_progress(
            StartupState.INITIALIZING,
            progress=5,
            message="Initializing TTS service..."
        )

        try:
            # Start service if not running
            if not self.is_running():
                self.logger.info("Starting TTS service for initialization")
                self._emit_startup_progress(
                    StartupState.INITIALIZING,
                    progress=10,
                    message="Starting service..."
                )

                if not await self.start():
                    self._startup_state = StartupState.FAILED
                    self._emit_startup_progress(
                        StartupState.FAILED,
                        progress=0,
                        message="Failed to start TTS service"
                    )
                    return False, "Failed to start TTS service"

            # Preload model if configured
            if self._bundled_config.preload_on_startup:
                self._startup_state = StartupState.LOADING_MODEL

                # Use preferred model type if provided, otherwise default to CUSTOM_VOICE
                model_to_load = preferred_model_type or QwenModelType.CUSTOM_VOICE

                self._emit_startup_progress(
                    StartupState.LOADING_MODEL,
                    progress=20,
                    message=f"Loading {model_to_load.display_name} model... (this may take up to {int(self._bundled_config.startup_timeout_seconds)}s)"
                )

                self.logger.info(
                    f"Preloading {model_to_load.display_name} model"
                    + (f" with speaker '{self._bundled_config.speaker}'" if model_to_load == QwenModelType.CUSTOM_VOICE else "")
                )

                # Load with timeout (NFR2)
                try:
                    success, error = await asyncio.wait_for(
                        self._model_registry.ensure_model_loaded(model_to_load),
                        timeout=self._bundled_config.startup_timeout_seconds
                    )
                except asyncio.TimeoutError:
                    self.logger.error(
                        f"Model loading timed out after {self._bundled_config.startup_timeout_seconds}s"
                    )
                    self._startup_state = StartupState.FAILED
                    self._emit_startup_progress(
                        StartupState.FAILED,
                        progress=0,
                        message="Model loading timed out. Please restart the application."
                    )
                    return False, "Model loading timed out"

                if not success:
                    self.logger.error(f"Failed to preload model: {error}")
                    self._startup_state = StartupState.FAILED
                    self._emit_startup_progress(
                        StartupState.FAILED,
                        progress=0,
                        message=f"Failed to load voice model: {error}"
                    )
                    return False, f"Failed to load model: {error}"

                self._emit_startup_progress(
                    StartupState.LOADING_MODEL,
                    progress=90,
                    message="Model loaded successfully"
                )

            # Mark as ready
            self._startup_state = StartupState.READY
            self._emit_startup_progress(
                StartupState.READY,
                progress=100,
                message="TTS Ready",
                is_complete=True,
                is_ready=True
            )

            # Emit TTS ready callback
            if self._tts_ready_callback:
                self._tts_ready_callback()

            self.logger.info(
                f"TTS initialized with defaults: speaker={self._bundled_config.speaker}, "
                f"language={self._bundled_config.language}"
            )

            return True, None

        except Exception as e:
            self.logger.exception(f"Failed to initialize TTS: {e}")
            self._startup_state = StartupState.FAILED
            self._emit_startup_progress(
                StartupState.FAILED,
                progress=0,
                message=f"Initialization failed: {str(e)}"
            )
            return False, str(e)

    def _emit_startup_progress(
        self,
        state: StartupState,
        progress: float,
        message: str,
        is_complete: bool = False,
        is_ready: bool = False
    ):
        """Emit startup progress update via callback."""
        if self._startup_progress_callback:
            try:
                progress_info = StartupProgress(
                    state=state,
                    progress_percent=progress,
                    message=message,
                    is_complete=is_complete,
                    is_ready=is_ready,
                )
                self._startup_progress_callback(progress_info)
            except Exception as e:
                self.logger.error(f"Error in startup progress callback: {e}")

    def get_startup_state(self) -> StartupState:
        """Get the current startup state."""
        return self._startup_state

    def is_tts_ready(self) -> bool:
        """
        Check if TTS is ready for generation.

        Returns True when:
        - Service is running
        - Startup completed successfully OR any model is loaded

        Note: With Qwen3-TTS, we have 3 models (CUSTOM_VOICE, VOICE_DESIGN, BASE).
        TTS is considered ready if the service is running and either startup
        completed or any model is already loaded and ready for use.
        """
        if not self.is_running():
            return False

        # Ready if startup completed successfully
        if self._startup_state == StartupState.READY:
            return True

        # Also ready if any model is currently loaded (for lazy loading scenarios)
        if self._model_registry.current_model_type is not None:
            return True

        return False

    def get_bundled_config(self) -> BundledVoiceConfig:
        """Get the current bundled voice configuration."""
        return self._bundled_config

    def get_default_speaker(self) -> str:
        """Get the configured default speaker name."""
        return self._bundled_config.speaker

    def get_tts_status(self) -> Dict[str, Any]:
        """
        Get comprehensive TTS status for UI display.

        Returns dict with:
        - is_ready: Whether TTS can generate speech
        - startup_state: Current startup state
        - status_text: Human-readable status (e.g., "TTS Ready", "Loading...")
        - status_color: Suggested color ("green", "yellow", "red")
        - current_model: Currently loaded model name (or None)
        - default_speaker: Configured default speaker
        """
        if self._startup_state == StartupState.READY and self.is_running():
            return {
                "is_ready": True,
                "startup_state": self._startup_state.value,
                "status_text": "TTS Ready",
                "status_color": "green",
                "current_model": self._model_registry.current_model_type.display_name if self._model_registry.current_model_type else None,
                "default_speaker": self._bundled_config.speaker,
            }
        elif self._startup_state == StartupState.LOADING_MODEL:
            return {
                "is_ready": False,
                "startup_state": self._startup_state.value,
                "status_text": "Loading Model...",
                "status_color": "yellow",
                "current_model": None,
                "default_speaker": self._bundled_config.speaker,
            }
        elif self._startup_state == StartupState.INITIALIZING:
            return {
                "is_ready": False,
                "startup_state": self._startup_state.value,
                "status_text": "Initializing...",
                "status_color": "yellow",
                "current_model": None,
                "default_speaker": self._bundled_config.speaker,
            }
        elif self._startup_state == StartupState.FAILED:
            return {
                "is_ready": False,
                "startup_state": self._startup_state.value,
                "status_text": "TTS Failed",
                "status_color": "red",
                "current_model": None,
                "default_speaker": self._bundled_config.speaker,
            }
        else:  # NOT_STARTED
            return {
                "is_ready": False,
                "startup_state": self._startup_state.value,
                "status_text": "TTS Not Started",
                "status_color": "gray",
                "current_model": None,
                "default_speaker": self._bundled_config.speaker,
            }

    async def stop(self) -> bool:
        """
        Stop the TTS service.

        Returns:
            bool: True if service stopped successfully
        """
        try:
            await self._update_status(ServiceStatus.STOPPING)
            self.logger.info("Stopping QwenTTSService")

            # Story 11.3: drop the metric-listener registration so this
            # service can be garbage-collected after stop().
            try:
                self._latency_aggregator.unsubscribe()
            except Exception:
                self.logger.exception("Failed to unsubscribe latency aggregator")

            # Unload all models
            await self._model_registry.unload_all()

            # Shutdown executor - QA Round 2 Item #8: Non-blocking shutdown
            if self._executor:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None

            self._request_semaphore = None

            # Shutdown model registry
            self._model_registry.shutdown()

            await self._update_status(ServiceStatus.STOPPED)
            self.logger.info("QwenTTSService stopped")
            return True

        except Exception as e:
            self.logger.exception(f"Error stopping QwenTTSService: {e}")
            await self._update_status(ServiceStatus.ERROR)
            return False

    async def health_check(self) -> Tuple[bool, Optional[MyVoiceError]]:
        """
        Check TTS service health.

        Returns:
            tuple[bool, Optional[MyVoiceError]]: (is_healthy, error_if_any)
        """
        try:
            if not self.is_running():
                return False, MyVoiceError(
                    severity=ErrorSeverity.ERROR,
                    code="SERVICE_NOT_RUNNING",
                    user_message="TTS service is not running",
                    suggested_action="Start the TTS service"
                )

            # Service is healthy if running (model loading is lazy)
            return True, None

        except Exception as e:
            self.logger.exception(f"Health check failed: {e}")
            return False, MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="HEALTH_CHECK_FAILED",
                user_message="Failed to check TTS service health",
                technical_details=str(e)
            )

    async def generate_custom_voice(
        self,
        text: str,
        speaker: str = "Ryan",
        language: str = "Auto",
        instruct: Optional[str] = None,
        emotion_preset: Optional[str] = None,
        streaming: bool = True,
    ) -> QwenTTSResponse:
        """
        Generate speech using CustomVoice model with bundled speakers.

        This is the primary method for bundled voice generation with emotion control.
        By default uses streaming mode for low-latency first audio (<2s).

        Args:
            text: Text to convert to speech
            speaker: Speaker name (Ryan, Vivian, Serena, etc.)
            language: Language code or "Auto"
            instruct: Custom emotion/style instruction
            emotion_preset: Preset emotion name (neutral, happy, sad, angry, flirtatious)
            streaming: Use streaming mode for progressive audio output

        Returns:
            QwenTTSResponse: Response with audio data or error
        """
        # Resolve emotion instruction
        if emotion_preset and emotion_preset in self.EMOTION_PRESETS:
            instruct = self.EMOTION_PRESETS[emotion_preset]

        request = QwenTTSRequest(
            text=text,
            language=language,
            model_type=QwenModelType.CUSTOM_VOICE,
            speaker=speaker,
            instruct=instruct,
            streaming=streaming,
        )

        # Story 16.6: route every public entry through the three-mode dispatch
        # fork so the streaming-mode resolver + fallback chain are the single
        # source of truth (D-9 / FR3 / NFR7). The dispatcher honors the legacy
        # ``request.streaming=False`` override by forcing BATCH.
        return await self._dispatch_by_streaming_mode(
            request, self._resolve_streaming_mode()
        )

    async def generate_voice_design(
        self,
        text: str,
        voice_description: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        streaming: bool = True,
    ) -> QwenTTSResponse:
        """
        Generate speech using VoiceDesign model with text description.

        Args:
            text: Text to convert to speech
            voice_description: Natural language description of desired voice
            language: Language code or "Auto"
            instruct: Additional style instruction
            streaming: Use streaming mode for progressive audio output

        Returns:
            QwenTTSResponse: Response with audio data or error
        """
        request = QwenTTSRequest(
            text=text,
            language=language,
            model_type=QwenModelType.VOICE_DESIGN,
            voice_description=voice_description,
            instruct=instruct,
            streaming=streaming,
        )

        # Story 16.6: route through the three-mode dispatch fork.
        return await self._dispatch_by_streaming_mode(
            request, self._resolve_streaming_mode()
        )

    # ----- Story 17.2: lazy + persistent voice_clone_prompt precompute --- #

    # qwen-tts pin per requirements.txt:23 — embedded in .pt.meta.json so a
    # pin bump invalidates cached embeddings. Story 18.4 / D-22 Branch B bumped
    # the pin from `1ab0dd75` (QwenLM/Qwen3-TTS upstream) to `3fdb4682`
    # (dffdeeq/Qwen3-TTS-streaming fork) — Story 17.2 cached `.pt` files from
    # the previous pin are invalidated on first run after this bump (Story 17.2
    # H1+H2 cache-invalidation discipline per memory/code_review_regression_test_exact_class.md).
    _QWEN_TTS_PIN_HASH = "3fdb4682"
    # Whisper retry policy for the lazy-precompute (AC #2): three attempts
    # total, with progressive backoff between. The bundled subprocess
    # Whisper path (whisper_subprocess.py) typically completes in 1-3s
    # cold, so 1s + 3s gives ~5s envelope before declaring FAILED.
    _WHISPER_RETRY_BACKOFFS_SECONDS: Tuple[float, ...] = (1.0, 3.0)
    # Story 17.2 review-pass M3 — bound the in-memory voice_clone_prompt
    # cache so long-running sessions with large voice libraries (e.g.
    # audiobook narration with N character voices) don't accumulate
    # unbounded resident embedding tensors. 64 entries is comfortably above
    # the 12 bundled CLONED voices and any realistic small/medium custom
    # library; LRU eviction kicks in at the cap.
    _VOICE_CLONE_PROMPT_CACHE_MAX = 64

    @staticmethod
    def _ref_audio_stat(ref_audio: Path) -> Tuple[float, int]:
        """Return ``(mtime, size)`` for ``ref_audio``. Raises FileNotFoundError
        if the file disappeared between caller's exists() check and now.
        """
        stat = ref_audio.stat()
        return (stat.st_mtime, stat.st_size)

    @staticmethod
    def _txt_sidecar_mtime(ref_audio: Path) -> Optional[float]:
        """Return mtime of ``ref_audio.with_suffix('.txt')`` if present, else
        None. Used by AC #3 cache-invalidation to detect user edits of the
        transcription sidecar that would otherwise leave the cached
        embedding (computed against the OLD transcription) stale.
        """
        sidecar = ref_audio.with_suffix(".txt")
        try:
            return sidecar.stat().st_mtime if sidecar.exists() else None
        except OSError:
            return None

    def _cache_lookup_validated(
        self,
        cache_key: Tuple[str, str],
        ref_audio: Path,
    ) -> Optional[Any]:
        """Story 17.2 review-pass H1 + H2 — in-memory cache hit gated on
        re-validation of ``ref_audio`` mtime/size AND ``.txt`` sidecar
        mtime against the cached fingerprint. On mismatch, evict and
        return None (caller treats as miss; precompute recomputes against
        the current state).

        Returns the prompt on hit, None on miss-or-stale.
        Side-effects: marks the entry as recently-used on hit; evicts on
        stale-detection.
        """
        entry = self._voice_clone_prompts.get(cache_key)
        if entry is None:
            return None
        prompt, cached_mtime, cached_size, cached_txt_mtime = entry
        try:
            current_mtime, current_size = self._ref_audio_stat(ref_audio)
        except (OSError, FileNotFoundError):
            # ref_audio went missing — drop the stale entry.
            self._voice_clone_prompts.pop(cache_key, None)
            return None
        # mtime tolerance matches _voice_clone_prompt_meta_is_valid (1ms).
        if abs(current_mtime - cached_mtime) > 1e-3 or current_size != cached_size:
            self._voice_clone_prompts.pop(cache_key, None)
            self.logger.info(
                f"Voice clone prompt in-memory cache invalidated for "
                f"{cache_key[0]}: ref_audio mtime/size changed"
            )
            return None
        # .txt sidecar mtime check (sidecar may appear / disappear / change).
        current_txt_mtime = self._txt_sidecar_mtime(ref_audio)
        if current_txt_mtime != cached_txt_mtime:
            self._voice_clone_prompts.pop(cache_key, None)
            self.logger.info(
                f"Voice clone prompt in-memory cache invalidated for "
                f"{cache_key[0]}: transcription sidecar changed"
            )
            return None
        # Mark as recently-used for LRU eviction order.
        self._voice_clone_prompts.move_to_end(cache_key)
        return prompt

    def _cache_store(
        self,
        cache_key: Tuple[str, str],
        prompt: Any,
        ref_audio: Path,
    ) -> None:
        """Store ``prompt`` in the cache with the current mtime/size/txt_mtime
        fingerprint, evicting the oldest entry when at the LRU cap (M3)."""
        try:
            mtime, size = self._ref_audio_stat(ref_audio)
        except (OSError, FileNotFoundError):
            # If the file disappeared between compute and store, don't cache —
            # the next request will recompute against the new state.
            return
        txt_mtime = self._txt_sidecar_mtime(ref_audio)
        self._voice_clone_prompts[cache_key] = (prompt, mtime, size, txt_mtime)
        self._voice_clone_prompts.move_to_end(cache_key)
        # LRU eviction at the cap (M3 review-pass fix).
        while len(self._voice_clone_prompts) > self._VOICE_CLONE_PROMPT_CACHE_MAX:
            evicted_key, _ = self._voice_clone_prompts.popitem(last=False)
            self.logger.debug(
                f"Voice clone prompt cache LRU-evicted {evicted_key[0]} "
                f"(cache at {self._VOICE_CLONE_PROMPT_CACHE_MAX} cap)"
            )

    def set_whisper_service(self, whisper_service: Any) -> None:
        """Story 17.2 AC #2 — orchestrator-injected WhisperSubprocessService.

        Wired by ``app.py:_initialize_whisper_service_on_demand`` after the
        on-demand init flow lands. When None, the lazy CLONED-voice
        precompute raises ``RuntimeError`` so ``_dispatch_by_streaming_mode``
        falls through to SENTENCE_STREAM (NFR7 preserved).
        """
        self._whisper_service = whisper_service
        self.logger.info("WhisperSubprocessService injected into QwenTTSService")

    def set_whisper_init_callback(
        self, callback: Optional[Callable[[], None]]
    ) -> None:
        """Story 17.2 — fire-and-forget hook the orchestrator wires so the
        first cache-miss precompute can request on-demand Whisper init.

        The callback is invoked synchronously from the precompute when
        ``_whisper_service is None``; the precompute then raises so the
        dispatch chain falls through to SENTENCE_STREAM. The next generation
        on the same voice (after init lands) hits the populated cache.
        """
        self._whisper_init_callback = callback

    def set_voice_profile_manager(self, voice_profile_manager: Any) -> None:
        """Story 17.2 AC #3 — orchestrator-injected VoiceProfileManager.

        Wired by ``app.py`` after ``await self._voice_manager.start()``.
        Used by ``hydrate_voice_clone_prompt_cache()`` to enumerate CLONED
        voices and pre-load any persisted ``.pt`` files into the in-memory
        cache. ``generate_voice_clone`` does NOT require this manager — it
        derives ``ref_audio.resolve()`` from its parameter directly — so a
        None manager only disables startup hydration (lazy fallback works).
        """
        self._voice_profile_manager = voice_profile_manager

    def set_preparing_voice_callback(
        self, callback: Optional[Callable[[Optional[str]], None]]
    ) -> None:
        """Story 17.2 AC #4 — orchestrator-wired UI feedback for first-run
        precompute. Invoked with a string message on entry (e.g. "Preparing
        voice for streaming…") and with ``None`` on exit (success or
        failure). Cache hits do NOT invoke this callback — only misses.
        """
        self._preparing_voice_callback = callback

    def set_app_settings(self, app_settings: AppSettings) -> None:
        """Replace the AppSettings reference consulted at dispatch time.

        ``SettingsDialog`` constructs a deep-copy ``AppSettings`` on open
        (``settings_dialog.py:92`` — ``AppSettings.from_dict(settings.to_dict())``)
        and returns the mutated copy on OK. ``_handle_settings_changed``
        in ``app.py`` swaps the orchestrator + main-window + audio-
        coordinator references but historically did NOT swap the TTS
        service's — leaving ``_resolve_streaming_mode`` reading the
        constructor-time settings forever. Symptom: changing the
        ``Streaming Mode`` dropdown in Settings had no observable effect
        on subsequent generations (RTX 3060 smoke 2026-05-13: dropdown
        set to ``sentence_stream``; next generation still ran TRUE_STREAM).

        Only ``streaming_mode_override`` is read at runtime
        (``_resolve_streaming_mode``); ``tts_precision`` and
        ``tts_compile`` are load-time fields that the ``ModelRegistry``
        already snapshotted, so a runtime swap does NOT retroactively
        change the loaded model's precision or compile state. Those
        settings still require a restart to take effect. The setter
        documents the bound — callers should not expect compile/precision
        to flip mid-session.
        """
        self._app_settings = app_settings

    async def _get_voice_clone_prompt_lock(
        self, cache_key: Tuple[str, str]
    ) -> asyncio.Lock:
        """Lazy-allocate a per-voice asyncio.Lock keyed by ``cache_key``.

        Concurrent same-key calls return the same Lock instance so the
        precompute serializes per-voice; different keys proceed in parallel.
        Locks for unreferenced keys are GC'd via WeakValueDictionary, so
        long-running services don't accumulate stale Lock objects.

        The registry mutation itself is guarded by a single asyncio.Lock
        (``_voice_clone_prompt_locks_guard``) lazy-initialized here because
        ``asyncio.Lock`` requires a running event loop. The lock instance
        is held in a local ``hold`` variable so the WeakValueDictionary's
        weak reference does not GC it before we return — the caller's
        ``async with`` then takes the strong reference.
        """
        if self._voice_clone_prompt_locks_guard is None:
            self._voice_clone_prompt_locks_guard = asyncio.Lock()
        async with self._voice_clone_prompt_locks_guard:
            existing = self._voice_clone_prompt_locks.get(cache_key)
            if existing is not None:
                return existing
            new_lock = asyncio.Lock()
            self._voice_clone_prompt_locks[cache_key] = new_lock
            return new_lock

    async def _ensure_transcription_for_clone_voice(
        self,
        voice_profile: Optional[Any],
        ref_audio: Path,
    ) -> str:
        """Story 17.2 AC #2 — resolve a transcription for ``ref_audio``.

        Resolution priority:
          1. ``voice_profile.transcription`` if non-empty (in-memory).
          2. ``<ref_audio>.with_suffix('.txt')`` sidecar (mirrors existing
             auto-detect at voice_profile.py:348-355).
          3. WhisperSubprocessService.transcribe_file with retry+backoff.

        On Whisper success the result is written to the .txt sidecar AND
        ``voice_profile.transcription`` is updated in-memory. On exhausted
        retries the profile is marked FAILED and the helper raises so the
        dispatch chain falls through to SENTENCE_STREAM (NFR7).

        ``voice_profile`` may be None when only the audio path is known
        (lock-only call sites); status transitions are skipped in that case
        but the .txt sidecar is still consulted and (on Whisper success)
        written.
        """
        # Priority 1: in-memory transcription
        if voice_profile is not None:
            existing = (voice_profile.transcription or "").strip()
            if existing:
                return existing

        # Priority 2: .txt sidecar
        sidecar = ref_audio.with_suffix(".txt")
        if sidecar.exists():
            try:
                text = sidecar.read_text(encoding="utf-8").strip()
                if text:
                    if voice_profile is not None:
                        voice_profile.transcription = text
                        # Mark COMPLETED via the canonical helper so the rest
                        # of the system observes a coherent transcription
                        # status (avoids leaving FAILED stuck if a previous
                        # attempt lost — sidecar wins).
                        voice_profile.set_transcription_result(
                            text, confidence=1.0, model_name="sidecar"
                        )
                    return text
            except Exception as exc:
                self.logger.warning(
                    f"Failed reading transcription sidecar {sidecar}: {exc}"
                )

        # Priority 3: Whisper. Lazy-fail-safe: if no service, request init
        # (fire-and-forget) and raise — dispatch chain falls through to
        # SENTENCE_STREAM (NFR7); subsequent calls (post-init) hit cache.
        if self._whisper_service is None:
            if self._whisper_init_callback is not None:
                try:
                    self._whisper_init_callback()
                except Exception as exc:
                    self.logger.warning(
                        f"Whisper init callback raised: {exc}"
                    )
            raise RuntimeError(
                "WhisperSubprocessService is not initialized; cannot "
                "compute transcription for TRUE_STREAM voice clone "
                "precompute. Falling through to SENTENCE_STREAM."
            )

        # Status transitions for diagnostics. QUEUED on entry; PROCESSING
        # before the first await; FAILED on exhausted retries; COMPLETED
        # on success (via set_transcription_result).
        from myvoice.models.voice_profile import TranscriptionStatus

        if voice_profile is not None:
            voice_profile.update_transcription_status(
                TranscriptionStatus.QUEUED
            )

        last_error: Optional[BaseException] = None
        for attempt_index in range(len(self._WHISPER_RETRY_BACKOFFS_SECONDS) + 1):
            if voice_profile is not None:
                voice_profile.update_transcription_status(
                    TranscriptionStatus.PROCESSING
                )
            try:
                self.logger.info(
                    f"Whisper transcription started for {ref_audio.name} "
                    f"(attempt {attempt_index + 1})"
                )
                result = await self._whisper_service.transcribe_file(ref_audio)
                text = (result.text or "").strip()
                if not text:
                    raise RuntimeError(
                        "Whisper returned empty transcription"
                    )
                # Persist sidecar (UTF-8, no BOM) before updating in-memory
                # state so a crash mid-update can still recover.
                try:
                    sidecar.write_text(text, encoding="utf-8")
                except Exception as exc:
                    self.logger.warning(
                        f"Failed writing transcription sidecar {sidecar}: {exc}"
                    )
                if voice_profile is not None:
                    voice_profile.set_transcription_result(
                        text,
                        confidence=getattr(result, "confidence", 0.9),
                        model_name="whisper-base",
                    )
                self.logger.info(
                    f"Whisper transcription completed for {ref_audio.name}"
                )
                return text
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    f"Whisper attempt {attempt_index + 1} failed for "
                    f"{ref_audio.name}: {exc}"
                )
                # If a backoff is configured for THIS attempt, sleep and
                # retry; otherwise drop out to FAILED handling below.
                if attempt_index < len(self._WHISPER_RETRY_BACKOFFS_SECONDS):
                    backoff = self._WHISPER_RETRY_BACKOFFS_SECONDS[
                        attempt_index
                    ]
                    await asyncio.sleep(backoff)
                    continue

        # Retries exhausted.
        error_str = str(last_error) if last_error else "unknown"
        if voice_profile is not None:
            voice_profile.mark_transcription_failed(error_str)
        raise RuntimeError(
            f"Whisper transcription failed after retries: {error_str}"
        )

    def _voice_clone_prompt_persist_paths(
        self, ref_audio: Path, tier: str
    ) -> Tuple[Path, Path]:
        """Return (pt_path, meta_path) for persisting an embedding next to
        ``ref_audio``. Tier-locked naming: ``<voice>.<tier>.pt`` plus
        ``<voice>.<tier>.pt.meta.json``. Spaces / parens in stem (e.g.
        ``Base (Clone)``) are accepted by both Windows and Linux filesystems.
        """
        pt_path = ref_audio.with_name(f"{ref_audio.stem}.{tier}.pt")
        meta_path = ref_audio.with_name(
            f"{ref_audio.stem}.{tier}.pt.meta.json"
        )
        return pt_path, meta_path

    def _voice_clone_prompt_meta_is_valid(
        self,
        meta_path: Path,
        ref_audio: Path,
        tier: str,
    ) -> bool:
        """Return True if ``meta_path`` matches the current ref_audio
        (mtime + size), the current ``.txt`` sidecar mtime (H2 review-pass
        fix), the current qwen-tts pin, AND the current tier. On any
        mismatch (or unparseable meta), return False — caller deletes both
        files and treats as miss.
        """
        if not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.logger.warning(
                f"Voice clone prompt meta unparseable {meta_path}: {exc}"
            )
            return False
        try:
            stat = ref_audio.stat()
        except FileNotFoundError:
            return False
        if meta.get("tier") != tier:
            return False
        if meta.get("qwen_tts_pin") != self._QWEN_TTS_PIN_HASH:
            return False
        # mtime is a float; allow exact equality (filesystem timestamps round-
        # trip exactly on the platforms MyVoice ships to) but tolerate ~1ms
        # of float drift defensively.
        meta_mtime = meta.get("ref_audio_mtime")
        if not isinstance(meta_mtime, (int, float)):
            return False
        if abs(float(meta_mtime) - stat.st_mtime) > 1e-3:
            return False
        if meta.get("ref_audio_size") != stat.st_size:
            return False
        # H2 review-pass fix — ``.txt`` sidecar invalidation. The cached
        # embedding was computed against a specific transcription (whether
        # from a previous Whisper run, a sidecar at hydration time, or an
        # in-memory profile.transcription that was persisted). If the user
        # has since edited the .txt to fix a transcription error, the
        # cached embedding no longer matches the user's intended input —
        # invalidate so the next generation recomputes against the
        # corrected transcription. ``txt_mtime`` may be absent in legacy
        # meta files (pre-review-pass); treat absent + sidecar-now-present
        # as a mismatch, absent + sidecar-still-absent as a match.
        meta_txt_mtime = meta.get("txt_mtime")
        current_txt_mtime = self._txt_sidecar_mtime(ref_audio)
        if meta_txt_mtime is None and current_txt_mtime is None:
            pass  # neither side has a sidecar — match
        elif (
            meta_txt_mtime is None
            or current_txt_mtime is None
            or abs(float(meta_txt_mtime) - float(current_txt_mtime)) > 1e-3
        ):
            return False
        return True

    def _delete_voice_clone_prompt_files(
        self, pt_path: Path, meta_path: Path
    ) -> None:
        """Best-effort delete of (pt_path, meta_path) — both ignored if
        already absent. Logged at DEBUG; failures swallowed (cache miss
        treats them as ephemeral)."""
        for path in (pt_path, meta_path):
            try:
                if path.exists():
                    path.unlink()
                    self.logger.debug(f"Deleted stale cache file: {path}")
            except Exception as exc:
                self.logger.warning(
                    f"Failed deleting stale cache file {path}: {exc}"
                )

    async def _ensure_voice_clone_prompt_for_voice(
        self,
        voice_profile: Optional[Any],
        ref_audio: Path,
        transcription: str,
        tier: str,
    ) -> Any:
        """Story 17.2 AC #3 — compute or load a tier-locked voice clone
        prompt for ``ref_audio`` and return the in-memory tensor.

        Order of operations:
          1. If a valid persisted ``.pt`` (per meta) exists, load + normalize
             it and return — no network or compute.
          2. Otherwise, await ``create_voice_clone_prompt_for_tier``;
             move tensors to CPU; ``torch.save`` to ``.pt``; write meta JSON;
             verify by re-loading; on verification failure delete both files
             and raise.

        Caller (``generate_voice_clone``) holds the per-voice asyncio.Lock,
        so a concurrent same-voice call sees the cache hit on its turn.
        """
        pt_path, meta_path = self._voice_clone_prompt_persist_paths(
            ref_audio, tier
        )

        # Fast path: persisted .pt exists with valid meta — load + normalize.
        if pt_path.exists() and self._voice_clone_prompt_meta_is_valid(
            meta_path, ref_audio, tier
        ):
            try:
                import torch  # local to keep the cold-start surface unchanged
                # weights_only=False is required for VoiceClonePromptItem
                # deserialization in PyTorch 2.6+; mirrors voice_design_
                # studio_dialog.py:1172 and scripts/validate_embedding_api.py:219.
                loaded = torch.load(
                    str(pt_path), map_location="cpu", weights_only=False
                )
                normalized = self._normalize_voice_clone_prompt(loaded)
                self.logger.info(
                    f"Voice clone prompt cache hit on disk: {pt_path}"
                )
                return normalized
            except Exception as exc:
                self.logger.warning(
                    f"Voice clone prompt reload failed for {pt_path}: {exc}; "
                    "treating as miss"
                )
                self._delete_voice_clone_prompt_files(pt_path, meta_path)
        elif pt_path.exists():
            # Stale .pt (mtime / size / pin / tier mismatch) — purge + miss.
            self._delete_voice_clone_prompt_files(pt_path, meta_path)

        # Compute path: call the tier-locked extractor (which loads the Base
        # model under tier override). The shared sync helper at line 1286
        # is what tests mock for fast unit coverage.
        prompt = await self.create_voice_clone_prompt_for_tier(
            ref_audio=ref_audio,
            ref_text=transcription,
            tier=tier,
        )
        if prompt is None:
            raise RuntimeError(
                "create_voice_clone_prompt_for_tier returned None"
            )

        # Move tensors to CPU before persistence (mirrors
        # voice_design_studio_dialog.py:1154-1158).
        try:
            if getattr(prompt, "ref_code", None) is not None:
                prompt.ref_code = prompt.ref_code.cpu()
            if getattr(prompt, "ref_spk_embedding", None) is not None:
                prompt.ref_spk_embedding = prompt.ref_spk_embedding.cpu()
        except Exception as exc:
            # Tensors may be plain torch.Tensor without .cpu() in mocked
            # tests; only log and continue — the in-memory cache still
            # works, only persistence may be skipped.
            self.logger.warning(
                f"CPU-move on prompt tensors failed: {exc}"
            )

        try:
            stat = ref_audio.stat()
            # Schema version bumped to 1.1 in the Story 17.2 review pass:
            # adds ``txt_mtime`` so the persisted cache invalidates when
            # the user edits the transcription sidecar (H2 review-pass
            # fix). Older 1.0 meta files lack this field; the validator
            # in _voice_clone_prompt_meta_is_valid treats absent
            # txt_mtime as "no sidecar at compute time" and matches only
            # if the current sidecar is also absent — otherwise stale.
            meta = {
                "schema_version": "1.1",
                "ref_audio_mtime": stat.st_mtime,
                "ref_audio_size": stat.st_size,
                "txt_mtime": self._txt_sidecar_mtime(ref_audio),
                "tier": tier,
                "qwen_tts_pin": self._QWEN_TTS_PIN_HASH,
            }
        except FileNotFoundError:
            # Caller's ref_audio went missing between dispatch entry and
            # save — surface the underlying error instead of silently
            # writing an unverifiable cache file.
            raise

        # Persist + verify. Delete both files on verification failure so a
        # later attempt re-computes cleanly (no half-written state).
        try:
            import torch  # local for the same reason as above
            torch.save(prompt, str(pt_path))
            meta_path.write_text(
                json.dumps(meta), encoding="utf-8"
            )
            try:
                verify = torch.load(
                    str(pt_path), map_location="cpu", weights_only=False
                )
                # Accept either the wrapper-class shape or anything
                # _normalize_voice_clone_prompt can produce — the
                # invariant is that ref_spk_embedding is recoverable.
                verify_norm = self._normalize_voice_clone_prompt(verify)
                if getattr(verify_norm, "ref_spk_embedding", None) is None:
                    raise RuntimeError(
                        "Verification reload produced empty embedding"
                    )
            except Exception as verify_exc:
                self._delete_voice_clone_prompt_files(pt_path, meta_path)
                raise RuntimeError(
                    f"Voice clone prompt verification failed: "
                    f"{verify_exc}"
                ) from verify_exc
            self.logger.info(
                f"Voice clone prompt persisted to {pt_path}"
            )
        except Exception:
            # Re-raise with context already logged; the outer dispatch
            # chain catches and falls through to SENTENCE_STREAM.
            raise

        return prompt

    async def prepare_voice_clone_prompt(
        self, voice_profile: Any
    ) -> Tuple[bool, Optional[str]]:
        """Eagerly precompute + persist the voice_clone_prompt for a CLONED
        voice profile so a subsequent TRUE_STREAM generation hits the cache
        and starts producing audio immediately.

        Mirrors the cache-check / per-voice-lock / disk-cache / in-memory-
        store discipline of the lazy gate in ``generate_voice_clone`` (the
        Story 17.2 AC #1 four-condition block at :2146-2238) but is callable
        from outside the generation request flow — used by the orchestrator
        to warm the cache when the user switches to a freshly-transcribed
        voice, before they hit Generate.

        Args:
            voice_profile: VoiceProfile of a CLONED voice with a populated
                ``transcription`` (.txt sidecar or in-memory). Non-CLONED
                voices and CLONED voices without transcription are treated
                as no-ops and return (False, reason).

        Returns:
            (success, error_message). On cache hit (in-memory or disk),
            returns (True, None) without recomputing. On miss, runs the
            Base-model precompute and persists ``<stem>.<tier>.pt`` next
            to the .wav.
        """
        from myvoice.models.voice_profile import VoiceType  # local — avoid cycle

        if voice_profile is None:
            return (False, "voice_profile is None")
        if getattr(voice_profile, "voice_type", None) != VoiceType.CLONED:
            return (False, "Not a CLONED voice; skipping precompute")
        transcription = getattr(voice_profile, "transcription", None)
        if not transcription:
            return (
                False,
                "No transcription available; cannot precompute the prompt "
                "(use Whisper to transcribe first)",
            )
        ref_audio = getattr(voice_profile, "file_path", None)
        if ref_audio is None or not ref_audio.exists():
            return (False, f"Reference audio not found: {ref_audio}")

        try:
            tier = self._model_registry.quality_tier.value
            cache_key = (str(ref_audio.resolve()), tier)
        except Exception as exc:
            return (False, f"Cache-key derivation failed: {exc}")

        # In-memory cache hit (mtime/size/txt-mtime validated) — nothing to do.
        if self._cache_lookup_validated(cache_key, ref_audio) is not None:
            self.logger.debug(
                f"prepare_voice_clone_prompt: in-memory cache hit for "
                f"{voice_profile.name} @ {tier}; skipping precompute"
            )
            return (True, None)

        # Per-voice lock to serialize against any concurrent generation
        # request on the same voice (double-checked locking).
        lock = await self._get_voice_clone_prompt_lock(cache_key)
        self._emit_preparing_voice("Preparing voice for streaming…")
        try:
            async with lock:
                cached = self._cache_lookup_validated(cache_key, ref_audio)
                if cached is not None:
                    return (True, None)
                self.logger.info(
                    f"prepare_voice_clone_prompt: cache miss for "
                    f"{voice_profile.name} ({cache_key[0]}) @ tier={tier}; "
                    f"computing"
                )
                prompt = await self._ensure_voice_clone_prompt_for_voice(
                    voice_profile, ref_audio, transcription, tier
                )
                self._cache_store(cache_key, prompt, ref_audio)
                self.logger.info(
                    f"prepare_voice_clone_prompt: cached prompt for "
                    f"{voice_profile.name} @ tier={tier}"
                )
                return (True, None)
        except Exception as exc:
            self.logger.warning(
                f"prepare_voice_clone_prompt failed for "
                f"{voice_profile.name}: {exc}",
                exc_info=True,
            )
            return (False, str(exc))
        finally:
            self._emit_preparing_voice(None)

    async def hydrate_voice_clone_prompt_cache(self) -> Tuple[int, int]:
        """Story 17.2 AC #3 — startup hydration of the in-memory cache.

        Iterates CLONED voices in the wired ``_voice_profile_manager``;
        for each one, attempts to load ``<voice>.<tier>.pt`` for the
        currently-loaded tier per the same invalidation rules as the lazy
        path. Hits populate ``_voice_clone_prompts[(resolved_path, tier)]``;
        misses are silent (lazy precompute will fill them on first use).

        Returns ``(hits, total)`` for caller-side telemetry/logging. Does
        not raise — bad meta on one voice does not abort the scan.

        Idempotent: re-running it is a no-op for entries already cached
        (the on-disk fast-path in ``_ensure_voice_clone_prompt_for_voice``
        is bypassed because the in-memory check at the gate fires first).
        """
        if self._voice_profile_manager is None:
            self.logger.debug(
                "Voice clone prompt cache hydration skipped: "
                "no VoiceProfileManager wired"
            )
            return (0, 0)
        try:
            from myvoice.models.voice_profile import VoiceType
            profiles = self._voice_profile_manager.get_profiles()
        except Exception as exc:
            self.logger.warning(
                f"Voice clone prompt cache hydration aborted: {exc}"
            )
            return (0, 0)

        tier = self._model_registry.quality_tier.value
        hits = 0
        total = 0
        for name, profile in profiles.items():
            if getattr(profile, "voice_type", None) != VoiceType.CLONED:
                continue
            ref_audio = getattr(profile, "file_path", None)
            if ref_audio is None or not isinstance(ref_audio, Path):
                continue
            if not ref_audio.exists():
                continue
            total += 1
            pt_path, meta_path = self._voice_clone_prompt_persist_paths(
                ref_audio, tier
            )
            if not pt_path.exists():
                continue
            if not self._voice_clone_prompt_meta_is_valid(
                meta_path, ref_audio, tier
            ):
                # Stale — let lazy path purge & recompute.
                continue
            try:
                import torch
                loaded = torch.load(
                    str(pt_path), map_location="cpu", weights_only=False
                )
                normalized = self._normalize_voice_clone_prompt(loaded)
                cache_key = (str(ref_audio.resolve()), tier)
                # Use _cache_store so the in-memory entry carries the
                # mtime/size/txt-mtime fingerprint and participates in
                # H1/H2 invalidation + M3 LRU eviction.
                self._cache_store(cache_key, normalized, ref_audio)
                hits += 1
            except Exception as exc:
                self.logger.warning(
                    f"Voice clone prompt hydration failed for {name}: "
                    f"{exc}",
                    exc_info=True,
                )
        self.logger.info(
            f"Voice clone prompt cache: hydrated {hits}/{total} CLONED "
            f"voices for tier {tier} from disk"
        )
        return (hits, total)

    def _emit_preparing_voice(self, message: Optional[str]) -> None:
        """Best-effort preparing-voice indicator update; failures swallow
        (UI feedback is not on the critical generation path)."""
        if self._preparing_voice_callback is None:
            return
        try:
            self._preparing_voice_callback(message)
        except Exception as exc:
            self.logger.warning(
                f"preparing_voice callback raised: {exc}"
            )

    # ------------------------------------------------------------------ #
    # Story 18.4 — torch.compile warmup worker (D-23)
    # ------------------------------------------------------------------ #

    _COMPILE_PRIMING_TEXT = "Hello world."
    _COMPILE_WARMUP_DISABLE_ENV = "MYVOICE_DISABLE_COMPILE_WARMUP"

    async def warmup_compile_async(self) -> None:
        """Story 18.4 / D-23 — background warmup for the torch.compile cache.

        Fire-and-forget worker that runs ONCE per process startup. Decides
        between two paths based on ``compile_cache.is_warm(key)``:

          * Cache HIT — log the steady-state breadcrumb + emit a
            ``cache_hit`` telemetry record; do NOT trigger a priming
            generation (the inductor cache reloads from disk lazily on
            first user-facing generation).
          * Cache MISS — emit the "Preparing TTS engine…" indicator, run
            one short synthetic priming generation through the production
            dispatch chain (which exercises ``torch.compile``'s graph
            capture and writes the inductor cache to the per-key directory),
            then call ``compile_cache.mark_warm(key)`` on success and clear
            the indicator. On priming failure, the cache stays cold (next
            run retries) and a WARNING is logged.

        Gates (fast-exit; fail-closed):
          1. ``MYVOICE_DISABLE_COMPILE_WARMUP=1`` env var → skip entirely
             (test surface; CI envs that should not trigger compile).
          2. ``is_ampere_or_newer()`` False → skip (CPU / pre-Ampere hosts
             never engage compile per D-9 / NFR12).
          3. ``self._model_registry`` not wired → skip (no model means no
             compile state to prime).

        Telemetry contract: ``metrics.record("tts_compile_warmup_priming",
        value, reason=..., duration_ms=...)``. The eight possible
        ``reason`` values mirror the engage path: ``"cache_hit"`` (skip),
        ``"primed_cold"`` (priming succeeded), ``"priming_failed"``
        (priming raised), ``"cuda_unavailable"`` (D-9 skip),
        ``"pre_ampere"`` (D-9 skip), ``"user_disabled"`` (env-var skip),
        ``"no_model_registry"`` (precondition not met), and
        ``"no_model_loaded"`` (model not yet loaded; defer to first
        user-facing generation).

        Never raises — every failure path lands in a WARNING log + a
        telemetry record. The fire-and-forget caller in ``app.py`` does
        not need a try/except wrapper (mirrors Story 17.2's
        ``hydrate_voice_clone_prompt_cache`` discipline).
        """
        import os
        import time

        from myvoice.observability import metrics

        start_ms = time.monotonic()

        # Gate 1: env-var-gated skip (production behavior unchanged when
        # env var unset). Mirrors Story 18.3's ``MYVOICE_AUTO_QUIT_ON_CLOSE``
        # env-var precedent at memory/main_window_close_confirm_dialog_in_tests.md.
        if os.environ.get(self._COMPILE_WARMUP_DISABLE_ENV) == "1":
            self.logger.info(
                "torch.compile warmup skipped: %s=1 (test mode)",
                self._COMPILE_WARMUP_DISABLE_ENV,
            )
            metrics.record(
                "tts_compile_warmup_priming",
                0.0,
                reason="user_disabled",
                duration_ms=int((time.monotonic() - start_ms) * 1000),
            )
            return

        # Gate 1b: AppSettings.tts_compile gate. If the user (or the
        # bundled-smoke default flipped to "off" per Fix #4) has disabled
        # compile, the warmup must NOT run a priming generation — the
        # priming dispatches a real TRUE_STREAM utterance whose audio
        # chunks reach the wired audio_chunk_ready_callback (the user's
        # speakers). Without this gate, a first-launch on Ampere+ CUDA
        # with tts_compile="off" produces audible "Hello world." spurious
        # audio AND writes a meaningless meta.json sidecar (engage stayed
        # eager so no inductor artifacts exist). Mirrors engage_compile_
        # optimizations' tts_compile="off" → reason="user_disabled" branch.
        tts_compile = getattr(self._app_settings, "tts_compile", None)
        if tts_compile == "off":
            self.logger.info(
                "torch.compile warmup skipped: tts_compile='off' "
                "(NFR7 fallback — eager-mode generation remains)"
            )
            metrics.record(
                "tts_compile_warmup_priming",
                0.0,
                reason="user_disabled",
                duration_ms=int((time.monotonic() - start_ms) * 1000),
            )
            return

        # Gate 2: D-9 hardware probe.
        try:
            from myvoice.services.tts_streaming import is_ampere_or_newer
            ampere_ok = is_ampere_or_newer()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "torch.compile warmup skipped: hardware probe raised (%s: %s)",
                type(exc).__name__,
                exc,
            )
            metrics.record(
                "tts_compile_warmup_priming",
                0.0,
                reason="cuda_unavailable",
                duration_ms=int((time.monotonic() - start_ms) * 1000),
            )
            return
        if not ampere_ok:
            self.logger.info(
                "torch.compile warmup skipped: pre-Ampere or no CUDA"
            )
            metrics.record(
                "tts_compile_warmup_priming",
                0.0,
                reason="pre_ampere",
                duration_ms=int((time.monotonic() - start_ms) * 1000),
            )
            return

        # Gate 3: model_registry must be wired.
        if self._model_registry is None:
            self.logger.debug(
                "torch.compile warmup skipped: no model_registry wired"
            )
            metrics.record(
                "tts_compile_warmup_priming",
                0.0,
                reason="no_model_registry",
                duration_ms=int((time.monotonic() - start_ms) * 1000),
            )
            return

        # Cache-key computation (mirrors engage_compile_optimizations'
        # construction so the warmup and the engage path share state).
        # If the model isn't loaded yet, defer — the first user-facing
        # generation will trigger engage + compile + lazy cache write.
        loaded_model = self._model_registry.get_loaded_model()
        if loaded_model is None:
            self.logger.debug(
                "torch.compile warmup skipped: no model loaded yet "
                "(lazy fallback path — first generation will prime the cache)"
            )
            metrics.record(
                "tts_compile_warmup_priming",
                0.0,
                reason="no_model_loaded",
                duration_ms=int((time.monotonic() - start_ms) * 1000),
            )
            return

        from myvoice.services.tts_streaming import compile_cache
        import torch
        try:
            capability = torch.cuda.get_device_capability()
            model_id = (
                getattr(getattr(loaded_model, "model", None), "name_or_path", None)
                or getattr(loaded_model, "name_or_path", None)
                or "unknown"
            )
            model_dtype = getattr(getattr(loaded_model, "model", None), "dtype", None)
            precision_str = "bf16" if model_dtype == torch.bfloat16 else "fp32"
            key = compile_cache.compute_key(
                qwen_tts_pin_hash=self._QWEN_TTS_PIN_HASH,
                model_id=model_id,
                precision_str=precision_str,
                torch_version=torch.__version__,
                decode_window_frames=30,
                cuda_capability=capability,
                compile_mode="reduce-overhead",
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "torch.compile warmup skipped: cache-key computation raised "
                "(%s: %s)",
                type(exc).__name__,
                exc,
            )
            metrics.record(
                "tts_compile_warmup_priming",
                0.0,
                reason="priming_failed",
                duration_ms=int((time.monotonic() - start_ms) * 1000),
                error=type(exc).__name__,
            )
            return

        # Cache-hit path: short-circuit; no priming needed.
        if compile_cache.is_warm(key):
            self.logger.info(
                "Compile cache hit; skipping warmup priming"
            )
            metrics.record(
                "tts_compile_warmup_priming",
                0.0,
                reason="cache_hit",
                duration_ms=int((time.monotonic() - start_ms) * 1000),
            )
            return

        # Cache-miss path: emit indicator + run priming + mark_warm on success.
        self._emit_preparing_voice("Preparing TTS engine…")
        try:
            await self._run_compile_priming()
            compile_cache.mark_warm(key)
            duration_ms = int((time.monotonic() - start_ms) * 1000)
            self.logger.info(
                "Compile warmup primed cache successfully (duration=%dms)",
                duration_ms,
            )
            metrics.record(
                "tts_compile_warmup_priming",
                1.0,
                reason="primed_cold",
                duration_ms=duration_ms,
            )
        except Exception as exc:  # noqa: BLE001
            # Cache stays cold (mark_warm not called); next run retries.
            duration_ms = int((time.monotonic() - start_ms) * 1000)
            self.logger.warning(
                "Compile warmup priming failed (%s: %s); cache stays cold, "
                "next run will retry. Eager-mode generation remains available.",
                type(exc).__name__,
                exc,
            )
            metrics.record(
                "tts_compile_warmup_priming",
                0.0,
                reason="priming_failed",
                duration_ms=duration_ms,
                error=type(exc).__name__,
            )
        finally:
            # Always clear the indicator — even on priming failure the
            # user should not be left looking at a stuck "Preparing TTS
            # engine…" message.
            self._emit_preparing_voice(None)

    async def _run_compile_priming(self) -> None:
        """Story 18.4 — run one synthetic priming generation.

        Separated from ``warmup_compile_async`` so the gating, cache
        check, indicator emission, and ``mark_warm`` logic stay independently
        testable (tests monkeypatch this method to succeed/fail without
        spinning up a real model or audio output).

        The priming flows through ``generate_custom_voice`` with a short
        text + the canonical default speaker ("Ryan", matching the
        ``QwenTTSCustomVoiceRequest.speaker`` default at line 219) so the
        talker's first forward pass triggers ``torch.compile``'s graph
        capture and the inductor compile that PyTorch's per-key cache
        directory absorbs.
        No audio output reaches the user (the priming runs before the
        ``set_audio_chunk_ready_callback`` wires consumers up — and even
        if a consumer is wired, the generation is short enough that any
        audible artifact is bounded).

        The architecture's D-23 acceptance: "warm = compile-priming
        generation completed without error". The caller's ``mark_warm``
        call lands on the success path. Any raised exception bubbles up
        and the caller's ``except`` lands the ``priming_failed`` telemetry.
        """
        # Use a bundled CustomVoice speaker for priming. The default
        # speaker ("Ryan") is canonical; if it's unavailable, the priming
        # generation raises and the caller routes to priming_failed (NOT
        # a fatal — the next run will retry; eager-mode generation stays
        # available). The custom-voice API does NOT require a voice file;
        # it dispatches via the same TRUE_STREAM path the user-facing
        # generation will take, so the talker's first forward pass
        # triggers torch.compile's graph capture.
        await self.generate_custom_voice(
            text=self._COMPILE_PRIMING_TEXT,
            speaker="Ryan",
            language="English",
        )

    async def generate_voice_clone(
        self,
        text: str,
        ref_audio: Path,
        ref_text: str,
        language: str = "Auto",
        streaming: bool = True,
        x_vector_only_mode: bool = False,
    ) -> QwenTTSResponse:
        """
        Generate speech using Base model with voice cloning.

        Note: Voice cloning does NOT support emotion control.

        Story 17.2: When the four-condition gate passes (streaming AND
        TRUE_STREAM resolved AND BASE model AND not x_vector_only_mode),
        the precompute pipeline lazily produces / hydrates a voice_clone_
        prompt and sets it on the request BEFORE dispatch. On gate-fail
        the dispatch is unchanged from Story 16.6 behavior.

        Args:
            text: Text to convert to speech
            ref_audio: Path to reference audio file (3+ seconds)
            ref_text: Transcript of reference audio (required for ICL mode)
            language: Language code or "Auto"
            streaming: Use streaming mode for progressive audio output
            x_vector_only_mode: If True, use x-vector mode (extracts voice timbre only, no ref_text needed).
                               If False (default), use ICL mode (higher quality, requires ref_text).

        Returns:
            QwenTTSResponse: Response with audio data or error
        """
        request = QwenTTSRequest(
            text=text,
            language=language,
            model_type=QwenModelType.BASE,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
            streaming=streaming,
        )

        resolved_mode = self._resolve_streaming_mode()

        # x_vector_only_mode is structurally incompatible with TRUE_STREAM:
        # TRUE_STREAM's BASE-model dispatch at :3961 demands a
        # voice_clone_prompt, but voice_clone_prompt comes from
        # create_voice_clone_prompt(ref_audio, ref_text=...), and x_vector
        # mode means "no ref_text is available." Without this downgrade the
        # request enters TRUE_STREAM, the talker raises immediately, an empty
        # session.finalize() raises a SECOND ValueError that surfaces as a
        # user-visible error dialog, and then the fallback chain produces the
        # audio anyway. Routing to SENTENCE_STREAM up front avoids the noisy
        # double-error round trip for cloned voices without a .txt sidecar.
        if request.x_vector_only_mode and resolved_mode == StreamingMode.TRUE_STREAM:
            self.logger.info(
                "Downgrading TRUE_STREAM -> SENTENCE_STREAM for "
                "x_vector_only_mode voice clone (no ref_text means no "
                "voice_clone_prompt; TRUE_STREAM would raise and fall back)"
            )
            resolved_mode = StreamingMode.SENTENCE_STREAM

        # Story 17.2 AC #1 — four-condition gate. All four must hold;
        # otherwise skip precompute and let the dispatch chain handle it
        # (BATCH-force for streaming=False, SENTENCE_STREAM for non-GPU,
        # x-vector path that doesn't need a prompt at all).
        gate_pass = (
            streaming is True
            and resolved_mode == StreamingMode.TRUE_STREAM
            and request.model_type == QwenModelType.BASE
            and request.x_vector_only_mode is False
        )

        if gate_pass:
            try:
                tier = self._model_registry.quality_tier.value
                cache_key = (str(ref_audio.resolve()), tier)
            except Exception as exc:
                # Path resolve failure (extremely rare) — log and skip
                # precompute; the dispatch chain still handles the
                # request via fallback.
                self.logger.warning(
                    f"Voice clone prompt cache key derivation failed: "
                    f"{exc}; skipping precompute"
                )
                cache_key = None

            if cache_key is not None:
                # Cache hit (validated against current mtime/size/txt-mtime —
                # H1 + H2 review-pass fixes prevent stale prompts from being
                # used after the user replaces ref_audio or fixes the .txt).
                # IMPORTANT: wrap in a list — the qwen-tts library at
                # `qwen_tts/inference/qwen3_tts_model.py:584-586` only
                # converts to its dict-form (`_prompt_items_to_voice_clone_
                # prompt`) when `voice_clone_prompt` is a `list`; a bare
                # VoiceClonePromptItem falls into the else branch and is
                # passed straight to `model.generate(...)` which crashes
                # on `voice_clone_prompt['ref_spk_embedding']`. Mirrors
                # the canonical pattern at qwen_tts_service.py:2254 used
                # by `generate_with_embedding`.
                cached = self._cache_lookup_validated(cache_key, ref_audio)
                if cached is not None:
                    request.voice_clone_prompt = [cached]
                    self.logger.debug(
                        f"Voice clone prompt cache hit for {cache_key[0]} "
                        f"(tier={tier})"
                    )
                else:
                    # Cache miss — serialize per-voice via DCL.
                    voice_profile = self._lookup_voice_profile(ref_audio)
                    lock = await self._get_voice_clone_prompt_lock(cache_key)
                    self._emit_preparing_voice(
                        "Preparing voice for streaming…"
                    )
                    try:
                        async with lock:
                            # Re-check inside the critical section
                            # (double-checked locking).
                            cached2 = self._cache_lookup_validated(
                                cache_key, ref_audio
                            )
                            if cached2 is None:
                                self.logger.info(
                                    "Voice clone prompt cache miss for "
                                    f"{cache_key[0]} (tier={tier}); "
                                    "computing"
                                )
                                transcription = (
                                    await self._ensure_transcription_for_clone_voice(
                                        voice_profile, ref_audio
                                    )
                                )
                                prompt = (
                                    await self._ensure_voice_clone_prompt_for_voice(
                                        voice_profile,
                                        ref_audio,
                                        transcription,
                                        tier,
                                    )
                                )
                                self._cache_store(cache_key, prompt, ref_audio)
                                cached2 = prompt
                            # Same list-wrapping discipline as the cache-
                            # hit branch above.
                            request.voice_clone_prompt = [cached2]
                    except Exception as exc:
                        # Precompute failed; let the dispatch chain catch
                        # so the SENTENCE_STREAM fallback runs (NFR7).
                        # Do NOT swallow — this is an exception bubble,
                        # not a recovery. The indicator is cleared in
                        # the finally block below.
                        # M2 review-pass fix: include exc_info=True so
                        # production logs carry the full traceback;
                        # debugging mysterious fall-throughs no longer
                        # requires reproducing locally.
                        self.logger.warning(
                            "Voice clone prompt precompute failed for "
                            f"{cache_key[0]}: {exc}; falling through "
                            "to dispatch (NFR7 graceful degradation)",
                            exc_info=True,
                        )
                        # Return the request as-is (no voice_clone_prompt);
                        # dispatch chain will fail TRUE_STREAM at the
                        # voice_clone_prompt-None check and fall back.
                    finally:
                        self._emit_preparing_voice(None)

        # Story 16.6: route through the three-mode dispatch fork.
        return await self._dispatch_by_streaming_mode(
            request, resolved_mode
        )

    def _lookup_voice_profile(self, ref_audio: Path) -> Optional[Any]:
        """Best-effort VoiceProfile lookup by ref_audio path. Returns None
        when no profile manager is wired or no profile matches — callers
        gracefully degrade (transcription status updates skipped, but the
        precompute itself still runs).

        Story 17.2 review-pass M1 — avoid an O(N) syscall storm on each
        cache miss. The orchestrator calls ``generate_voice_clone(ref_audio
        =active_profile.file_path, ...)`` with the SAME Path object that's
        stored on the profile, so a string-equality fast-path resolves
        almost all real call sites without touching the filesystem.
        Only fall back to the per-profile ``.resolve()`` syscall when the
        cheap compare misses (e.g. relative-vs-absolute, symlink, or a
        different Path instance with the same content).
        """
        if self._voice_profile_manager is None:
            return None
        try:
            profiles = self._voice_profile_manager.get_profiles()
        except Exception:
            return None
        # Cheap path: compare against the input path string directly. The
        # typical caller (app.py) hands us active_profile.file_path
        # verbatim, so this hits without any filesystem syscall.
        target_str = str(ref_audio)
        for profile in profiles.values():
            try:
                if str(profile.file_path) == target_str:
                    return profile
            except Exception:
                continue
        # Fallback: resolve target and per-profile paths to handle the
        # corner cases (relative paths, symlinks, alternate path forms).
        try:
            target_resolved = str(ref_audio.resolve())
        except Exception:
            return None
        for profile in profiles.values():
            try:
                if str(profile.file_path.resolve()) == target_resolved:
                    return profile
            except Exception:
                continue
        return None

    async def generate_optimized_voice(
        self,
        text: str,
        checkpoint_path: Path,
        speaker_name: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        emotion_preset: Optional[str] = None,
        streaming: bool = True,
    ) -> QwenTTSResponse:
        """
        Generate speech using a fine-tuned optimized voice checkpoint.

        Optimized voices are fine-tuned from the Base model and support
        emotional presets just like bundled voices.

        Args:
            text: Text to convert to speech
            checkpoint_path: Path to the fine-tuned model checkpoint
            speaker_name: Speaker name registered during fine-tuning
            language: Language code or "Auto"
            instruct: Custom emotion/style instruction
            emotion_preset: Preset emotion name (neutral, happy, sad, angry, flirtatious)
            streaming: Use streaming mode for progressive audio output

        Returns:
            QwenTTSResponse: Response with audio data or error
        """
        # Validate checkpoint path
        if not checkpoint_path.exists():
            return QwenTTSResponse(
                success=False,
                error_message=f"Optimized voice checkpoint not found: {checkpoint_path}"
            )

        # Resolve emotion instruction
        if emotion_preset and emotion_preset in self.EMOTION_PRESETS:
            instruct = self.EMOTION_PRESETS[emotion_preset]

        request = QwenTTSRequest(
            text=text,
            language=language,
            model_type=QwenModelType.CUSTOM_VOICE,  # Fine-tuned uses custom_voice generation
            speaker=speaker_name,
            instruct=instruct,
            streaming=streaming,
            checkpoint_path=checkpoint_path,  # Custom checkpoint for optimized voice
        )

        self.logger.info(f"Generating with optimized voice: {speaker_name} from {checkpoint_path}")

        # Story 16.6: route through the three-mode dispatch fork.
        return await self._dispatch_by_streaming_mode(
            request, self._resolve_streaming_mode()
        )

    async def create_voice_clone_prompt(
        self,
        ref_audio: Path,
        ref_text: str,
    ) -> Any:
        """
        Create a reusable voice clone prompt from reference audio (QA5/QA6).

        This extracts the voice characteristics from reference audio and creates
        a prompt tensor that can be saved and reused for consistent voice generation.

        Args:
            ref_audio: Path to reference audio file (3+ seconds recommended)
            ref_text: Transcript of reference audio (required for high-quality prompt)

        Returns:
            Voice clone prompt tensor

        Raises:
            RuntimeError: If model loading fails or prompt creation fails
        """
        self.logger.info(f"Creating voice clone prompt from {ref_audio}")

        # Ensure Base model is loaded (it has the create_voice_clone_prompt method)
        async with self._request_semaphore:
            success, error = await self._model_registry.ensure_model_loaded(
                QwenModelType.BASE
            )
            if not success:
                raise RuntimeError(f"Failed to load Base model: {error}")

            # Get the model
            model = self._model_registry.get_loaded_model()
            if model is None:
                raise RuntimeError("Base model not available after loading")

            # Create voice clone prompt in thread pool
            loop = asyncio.get_event_loop()
            prompt = await loop.run_in_executor(
                self._executor,
                self._create_voice_clone_prompt_sync,
                model,
                ref_audio,
                ref_text
            )

            if prompt is None:
                raise RuntimeError("Voice clone prompt creation returned None")

            return prompt

    async def create_voice_clone_prompt_for_tier(
        self,
        ref_audio: Path,
        ref_text: str,
        tier: str,
    ) -> Any:
        """
        Create a reusable voice clone prompt for a specific model tier.

        This is used by Voice Design Studio to extract embeddings for both
        1.7B and 0.6B model tiers, regardless of the current tier setting.

        Args:
            ref_audio: Path to reference audio file (3+ seconds recommended)
            ref_text: Transcript of reference audio
            tier: Target tier - "quality" (1.7B) or "small" (0.6B)

        Returns:
            Voice clone prompt tensor with correct dimensions for the specified tier

        Raises:
            RuntimeError: If model loading fails or prompt creation fails
        """
        tier_display = "1.7B" if tier == "quality" else "0.6B"
        self.logger.info(f"Creating voice clone prompt for {tier_display} tier from {ref_audio}")

        # Load Base model with tier override
        async with self._request_semaphore:
            success, error = await self._model_registry.ensure_model_loaded(
                QwenModelType.BASE,
                tier_override=tier
            )
            if not success:
                raise RuntimeError(f"Failed to load {tier_display} Base model: {error}")

            # Get the model
            model = self._model_registry.get_loaded_model()
            if model is None:
                raise RuntimeError(f"{tier_display} Base model not available after loading")

            # Create voice clone prompt in thread pool
            loop = asyncio.get_event_loop()
            prompt = await loop.run_in_executor(
                self._executor,
                self._create_voice_clone_prompt_sync,
                model,
                ref_audio,
                ref_text
            )

            if prompt is None:
                raise RuntimeError("Voice clone prompt creation returned None")

            self.logger.info(f"Created {tier_display} voice clone prompt successfully")
            return prompt

    def _create_voice_clone_prompt_sync(
        self,
        model: Any,
        ref_audio: Path,
        ref_text: str
    ) -> Optional[VoiceClonePromptItem]:
        """
        Synchronous implementation of voice clone prompt creation.

        Args:
            model: Qwen3-TTS Base model instance
            ref_audio: Path to reference audio
            ref_text: Transcript text

        Returns:
            VoiceClonePromptItem with ref_code and ref_spk_embedding, or None if creation failed

        Raises:
            RuntimeError: If model doesn't support create_voice_clone_prompt
            Exception: Re-raises any other errors for better debugging
        """
        # Check model has the method
        if not hasattr(model, 'create_voice_clone_prompt'):
            raise RuntimeError(
                "Base model does not have create_voice_clone_prompt method. "
                "Please ensure you have the latest Qwen3-TTS version."
            )

        # Validate audio file exists
        if not ref_audio.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

        self.logger.info(f"Creating voice clone prompt from {ref_audio} with transcript: {ref_text[:50]}...")

        # QA6: Try passing file path directly first (Qwen3-TTS supports both)
        # ref_audio can be: (audio_data, sample_rate) OR "path/to/audio.wav"
        try:
            prompt = model.create_voice_clone_prompt(
                ref_audio=str(ref_audio),  # Pass file path as string
                ref_text=ref_text,
            )
            self.logger.info("Voice clone prompt created successfully")
            return self._normalize_voice_clone_prompt(prompt)
        except TypeError as e:
            # If string path doesn't work, try with loaded audio data
            self.logger.warning(f"File path method failed ({e}), trying with audio data...")
            import soundfile as sf
            audio_data, sample_rate = sf.read(str(ref_audio))
            prompt = model.create_voice_clone_prompt(
                ref_audio=(audio_data, sample_rate),
                ref_text=ref_text,
            )
            self.logger.info("Voice clone prompt created successfully (audio data method)")
            return self._normalize_voice_clone_prompt(prompt)

    def _normalize_voice_clone_prompt(self, prompt: Any) -> Any:
        """
        Normalize voice clone prompt to the LIBRARY's VoiceClonePromptItem.

        CRITICAL: The Qwen3-TTS library expects its own VoiceClonePromptItem class,
        not our wrapper class. This method converts any format (including our saved
        custom class) to the library's native class.

        Handles API changes in Qwen3-TTS library where create_voice_clone_prompt()
        may return either:
        - Object with .ref_code and .ref_spk_embedding attributes (older versions)
        - List/tuple of [ref_code, ref_spk_embedding] (newer versions)

        Args:
            prompt: Raw result from model.create_voice_clone_prompt() or torch.load()

        Returns:
            Library's VoiceClonePromptItem (qwen_tts.inference.qwen3_tts_model.VoiceClonePromptItem)
        """
        # Helper to check if item is the library's native class
        def is_library_class(item: Any) -> bool:
            return (type(item).__module__ == 'qwen_tts.inference.qwen3_tts_model' and
                    type(item).__name__ == 'VoiceClonePromptItem')

        # Helper to convert any object with the right attributes to library's class
        def to_library_class(item: Any) -> Any:
            if LibraryVoiceClonePromptItem is None:
                raise ImportError("qwen_tts library not available")
            ref_text = getattr(item, 'ref_text', None)
            # CRITICAL: If ref_text is None or empty, we MUST set icl_mode=False
            # Otherwise the library will try to tokenize ref_text and crash with
            # "'NoneType' object is not subscriptable" at ref_ids[index][:, 3:-2]
            icl_mode = getattr(item, 'icl_mode', True)
            if ref_text is None or ref_text == "":
                icl_mode = False
                self.logger.info("Setting icl_mode=False because ref_text is None/empty")
            return LibraryVoiceClonePromptItem(
                ref_code=getattr(item, 'ref_code', None),
                ref_spk_embedding=getattr(item, 'ref_spk_embedding', None),
                x_vector_only_mode=getattr(item, 'x_vector_only_mode', False),
                icl_mode=icl_mode,
                ref_text=ref_text,
            )

        # Case 1: Already the library's VoiceClonePromptItem - return as-is
        if is_library_class(prompt):
            self.logger.info("Prompt is already library VoiceClonePromptItem, returning as-is")
            return prompt

        # Case 2: Our custom VoiceClonePromptItem (from saved embeddings) - convert to library's
        if isinstance(prompt, VoiceClonePromptItem):
            self.logger.info("Converting our custom VoiceClonePromptItem to library's class")
            return to_library_class(prompt)

        # Case 3: Dictionary - some formats use dict with 'ref_code' and 'ref_spk_embedding' keys
        if isinstance(prompt, dict):
            self.logger.info(f"Normalizing dict result (keys={list(prompt.keys())})")
            if LibraryVoiceClonePromptItem is None:
                raise ImportError("qwen_tts library not available")
            ref_text = prompt.get('ref_text')
            icl_mode = prompt.get('icl_mode', True)
            if ref_text is None or ref_text == "":
                icl_mode = False
                self.logger.info("Dict: Setting icl_mode=False because ref_text is None/empty")
            return LibraryVoiceClonePromptItem(
                ref_code=prompt.get('ref_code'),
                ref_spk_embedding=prompt.get('ref_spk_embedding'),
                x_vector_only_mode=prompt.get('x_vector_only_mode', False),
                icl_mode=icl_mode,
                ref_text=ref_text,
            )

        # Case 4: List or tuple - Qwen3-TTS returns [VoiceClonePromptItem] (list of one)
        if isinstance(prompt, (list, tuple)):
            self.logger.info(f"Normalizing list result (len={len(prompt)}, types={[type(x).__name__ for x in prompt]})")
            if len(prompt) >= 2:
                # Might be [ref_code, ref_spk_embedding] or list of VoiceClonePromptItems
                if hasattr(prompt[0], 'ref_spk_embedding'):
                    # It's a list of VoiceClonePromptItems - convert first one
                    self.logger.info("First element has ref_spk_embedding, converting to library class")
                    if is_library_class(prompt[0]):
                        return prompt[0]
                    return to_library_class(prompt[0])
                else:
                    # It's [ref_code, ref_spk_embedding] - no ref_text, so icl_mode must be False
                    if LibraryVoiceClonePromptItem is None:
                        raise ImportError("qwen_tts library not available")
                    self.logger.info("Tuple format: Setting icl_mode=False (no ref_text available)")
                    return LibraryVoiceClonePromptItem(
                        ref_code=prompt[0],
                        ref_spk_embedding=prompt[1],
                        x_vector_only_mode=False,
                        icl_mode=False,  # Must be False when ref_text is None
                        ref_text=None,
                    )
            elif len(prompt) == 1:
                # Single element - recurse to handle it
                single_item = prompt[0]
                self.logger.info(f"Single element type: {type(single_item).__name__}, module: {type(single_item).__module__}")

                # Check if it's a dict
                if isinstance(single_item, dict):
                    self.logger.info("Single element is dict, converting to library class")
                    if LibraryVoiceClonePromptItem is None:
                        raise ImportError("qwen_tts library not available")
                    ref_text = single_item.get('ref_text')
                    icl_mode = single_item.get('icl_mode', True)
                    if ref_text is None or ref_text == "":
                        icl_mode = False
                        self.logger.info("Single dict: Setting icl_mode=False because ref_text is None/empty")
                    return LibraryVoiceClonePromptItem(
                        ref_code=single_item.get('ref_code'),
                        ref_spk_embedding=single_item.get('ref_spk_embedding'),
                        x_vector_only_mode=single_item.get('x_vector_only_mode', False),
                        icl_mode=icl_mode,
                        ref_text=ref_text,
                    )
                # Check if it's the library's class
                elif is_library_class(single_item):
                    self.logger.info("Single element is library VoiceClonePromptItem, returning as-is")
                    return single_item
                # Check if it has ref_spk_embedding attribute (our custom class or similar)
                elif hasattr(single_item, 'ref_spk_embedding'):
                    self.logger.info("Single element has ref_spk_embedding, converting to library class")
                    return to_library_class(single_item)
                else:
                    # Assume it's a raw tensor (the embedding itself) - no ref_text, so icl_mode=False
                    self.logger.info("Single element assumed to be raw tensor, icl_mode=False")
                    if LibraryVoiceClonePromptItem is None:
                        raise ImportError("qwen_tts library not available")
                    return LibraryVoiceClonePromptItem(
                        ref_code=None,
                        ref_spk_embedding=single_item,
                        x_vector_only_mode=False,
                        icl_mode=False,  # Must be False when ref_text is None
                        ref_text=None,
                    )
            else:
                raise ValueError(f"Unexpected empty list from create_voice_clone_prompt")

        # Case 5: Object with attributes (library's or custom VoiceClonePromptItem)
        if hasattr(prompt, 'ref_spk_embedding'):
            self.logger.info(f"Object with ref_spk_embedding attribute (type={type(prompt).__name__})")
            if is_library_class(prompt):
                return prompt
            return to_library_class(prompt)

        # Case 6: Single tensor - assume it's the embedding - no ref_text, so icl_mode=False
        self.logger.warning(f"Unknown prompt type {type(prompt)}, treating as single embedding tensor, icl_mode=False")
        if LibraryVoiceClonePromptItem is None:
            raise ImportError("qwen_tts library not available")
        return LibraryVoiceClonePromptItem(
            ref_code=None,
            ref_spk_embedding=prompt,
            x_vector_only_mode=False,
            icl_mode=False,  # Must be False when ref_text is None
            ref_text=None,
        )

    async def generate_with_embedding(
        self,
        text: str,
        embedding_path: Path,
        language: str = "Auto",
        instruct: Optional[str] = None,
        emotion_preset: Optional[str] = None,
        emotion: Optional[str] = None,
        streaming: bool = True,
    ) -> QwenTTSResponse:
        """
        Generate speech using a saved voice embedding (QA5).

        This loads a pre-computed voice clone prompt from disk and uses it
        for consistent voice generation with emotion support.

        Emotion Variants: If `emotion` is specified, attempts to load the
        emotion-specific embedding from {base_dir}/{emotion}/embedding.pt.
        Falls back to neutral, then legacy root embedding.pt.

        Args:
            text: Text to convert to speech
            embedding_path: Path to saved .pt embedding file OR base voice directory
                           (for emotion-specific resolution)
            language: Language code or "Auto"
            instruct: Custom emotion/style instruction
            emotion_preset: Preset emotion name (neutral, happy, sad, angry, flirtatious)
                           Note: For EMBEDDING voices, use `emotion` instead
            emotion: Emotion Variants: Specific emotion to use (neutral, happy, sad, angry, flirtatious)
                    If specified, resolves embedding path to {base_dir}/{emotion}/embedding.pt
            streaming: Use streaming mode for progressive audio output

        Returns:
            QwenTTSResponse: Response with audio data or error
        """
        import torch

        # Emotion Variants: Resolve emotion-specific embedding path
        self.logger.info(f"[DEBUG] generate_with_embedding called: embedding_path={embedding_path}, emotion={emotion}")
        resolved_path = self._resolve_emotion_embedding_path(embedding_path, emotion)
        self.logger.info(f"[DEBUG] Resolved emotion path: {resolved_path}")

        # Validate embedding path
        if not resolved_path.exists():
            return QwenTTSResponse(
                success=False,
                error_message=f"Embedding file not found: {resolved_path}"
            )

        try:
            # Load the voice clone prompt from disk
            # weights_only=False required for VoiceClonePromptItem dataclass (PyTorch 2.6+)
            # Safe because embeddings are generated by this application
            # Always load to CPU first to handle cross-device compatibility (CUDA->CPU, different GPUs)
            self.logger.info(f"[DEBUG] Loading embedding from: {resolved_path}")
            raw_prompt = torch.load(
                str(resolved_path),
                map_location='cpu',
                weights_only=False
            )

            # Normalize loaded embedding to VoiceClonePromptItem (handles legacy formats)
            voice_clone_prompt = self._normalize_voice_clone_prompt(raw_prompt)

            # Move tensors to target device if not already there
            target_device = self._model_registry.device
            if target_device != 'cpu':
                if voice_clone_prompt.ref_code is not None:
                    voice_clone_prompt.ref_code = voice_clone_prompt.ref_code.to(target_device)
                if voice_clone_prompt.ref_spk_embedding is not None:
                    voice_clone_prompt.ref_spk_embedding = voice_clone_prompt.ref_spk_embedding.to(target_device)

            self.logger.info(f"[DEBUG] Loaded embedding: type={type(voice_clone_prompt).__name__}, module={type(voice_clone_prompt).__module__}")
            self.logger.info(f"[DEBUG] ref_text={voice_clone_prompt.ref_text!r}")
            self.logger.debug(f"Loaded voice embedding from {resolved_path}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding: {e}")
            return QwenTTSResponse(
                success=False,
                error_message=f"Failed to load embedding: {e}"
            )

        # Resolve emotion instruction
        # Note: For EMBEDDING voices with Emotion Variants, we use different
        # embeddings per emotion rather than instruct parameter
        if emotion_preset and emotion_preset in self.EMOTION_PRESETS:
            instruct = self.EMOTION_PRESETS[emotion_preset]

        # Create request with the loaded voice clone prompt
        # IMPORTANT: Pass as a list - the library expects List[VoiceClonePromptItem]
        # and converts it to a dict with lists of values via _prompt_items_to_voice_clone_prompt()
        request = QwenTTSRequest(
            text=text,
            language=language,
            model_type=QwenModelType.BASE,  # Use Base model for voice clone generation
            instruct=instruct,
            streaming=streaming,
            voice_clone_prompt=[voice_clone_prompt],  # Wrap in list
        )

        emotion_info = f" (emotion: {emotion})" if emotion else ""
        self.logger.info(f"Generating with saved embedding: {resolved_path.name}{emotion_info}")

        # Try generation with embedding, fall back to source_audio.wav on tier mismatch
        try:
            # Story 16.6: route through the three-mode dispatch fork.
            response = await self._dispatch_by_streaming_mode(
                request, self._resolve_streaming_mode()
            )

            # Check if generation failed due to embedding dimension mismatch
            if not response.success and self._is_embedding_tier_mismatch_error(response.error_message):
                return await self._fallback_to_source_audio(
                    resolved_path, text, language, streaming,
                    voice_clone_prompt.ref_text if voice_clone_prompt else None
                )

            return response

        except RuntimeError as e:
            # Catch tensor dimension mismatch errors directly
            if self._is_embedding_tier_mismatch_error(str(e)):
                return await self._fallback_to_source_audio(
                    resolved_path, text, language, streaming,
                    voice_clone_prompt.ref_text if voice_clone_prompt else None
                )
            raise

    def _resolve_emotion_embedding_path(
        self,
        embedding_path: Path,
        emotion: Optional[str] = None
    ) -> Path:
        """
        Resolve the actual embedding file path based on emotion and current model tier.

        Supports multi-tier structure (1.7/, 0.6/ subfolders) and legacy single-file structures.

        Path resolution order (for each emotion, tries current tier first):
        1. If embedding_path is a file that exists, use it directly
        2. If emotion specified:
           a. Try {base_dir}/{emotion}/{tier}/embedding.pt (tier-specific)
           b. Try {base_dir}/{emotion}/embedding.pt (legacy, assumed 1.7B)
        3. Fall back to neutral with same tier preference
        4. Fall back to {base_dir}/embedding.pt (legacy root)
        5. Return original path if nothing found (will fail validation later)

        Args:
            embedding_path: Base path (file or directory)
            emotion: Target emotion (optional)

        Returns:
            Path: Resolved path to the embedding file
        """
        # If path is already a file that exists, use it directly
        if embedding_path.is_file():
            return embedding_path

        # Determine base directory
        if embedding_path.is_dir():
            base_dir = embedding_path
        else:
            # Could be a non-existent file path - get parent
            base_dir = embedding_path.parent

        # Get current model tier for tier-aware resolution
        current_tier = self._model_registry.quality_tier
        tier_folder = "1.7" if current_tier.value == "quality" else "0.6"

        # Valid emotions for Emotion Variants
        valid_emotions = ["neutral", "happy", "sad", "angry", "flirtatious"]

        # Try emotion-specific path first
        if emotion and emotion in valid_emotions:
            # Try tier-specific path first: {emotion}/{tier}/embedding.pt
            tier_path = base_dir / emotion / tier_folder / "embedding.pt"
            if tier_path.exists():
                self.logger.debug(f"Using tier-specific embedding: {emotion}/{tier_folder}")
                return tier_path

            # Fall back to legacy emotion path (assumed 1.7B): {emotion}/embedding.pt
            emotion_path = base_dir / emotion / "embedding.pt"
            if emotion_path.exists():
                self.logger.debug(f"Using legacy emotion embedding: {emotion} (assuming 1.7B)")
                return emotion_path

        # Fall back to neutral with tier preference
        if emotion != "neutral":  # Avoid double check
            # Try tier-specific neutral
            neutral_tier_path = base_dir / "neutral" / tier_folder / "embedding.pt"
            if neutral_tier_path.exists():
                self.logger.debug(f"Falling back to tier-specific neutral: {tier_folder}")
                return neutral_tier_path

            # Try legacy neutral
            neutral_path = base_dir / "neutral" / "embedding.pt"
            if neutral_path.exists():
                self.logger.debug(f"Falling back to legacy neutral embedding")
                return neutral_path

        # Legacy fallback: root embedding.pt
        legacy_path = base_dir / "embedding.pt"
        if legacy_path.exists():
            self.logger.debug(f"Using legacy root embedding")
            return legacy_path

        # Return original path - will fail validation later with clear error
        self.logger.warning(f"No embedding found at {base_dir}")
        return embedding_path

    def _is_embedding_tier_mismatch_error(self, error_message: Optional[str]) -> bool:
        """
        Check if an error is caused by embedding dimension mismatch between tiers.

        1.7B models use 2048-dimensional embeddings, 0.6B models use 1024-dimensional.
        When an embedding from one tier is used with the other, torch.cat fails with
        a tensor size mismatch error.

        Args:
            error_message: The error message to check

        Returns:
            bool: True if this is a tier mismatch error
        """
        if not error_message:
            return False

        # Check for the specific tensor size mismatch pattern
        # "Expected size 1024 but got size 2048" or vice versa
        error_lower = error_message.lower()
        return (
            "sizes of tensors must match" in error_lower or
            ("expected size 1024" in error_lower and "2048" in error_lower) or
            ("expected size 2048" in error_lower and "1024" in error_lower)
        )

    async def _fallback_to_source_audio(
        self,
        embedding_path: Path,
        text: str,
        language: str,
        streaming: bool,
        ref_text: Optional[str] = None
    ) -> QwenTTSResponse:
        """
        Fall back to real-time voice cloning using source_audio.wav.

        When an embedding is incompatible with the current model tier (dimension mismatch),
        we fall back to using the source audio file that was saved alongside the embedding.

        Args:
            embedding_path: Path to the embedding file (used to find source_audio.wav)
            text: Text to convert to speech
            language: Language code
            streaming: Whether to use streaming mode
            ref_text: Reference text for the source audio (optional, uses x-vector mode if not provided)

        Returns:
            QwenTTSResponse: Response from voice cloning fallback
        """
        # Find source_audio.wav - it should be in the same emotion folder as the embedding
        # Structure: VoiceName/emotion/embedding.pt and VoiceName/emotion/source_audio.wav
        emotion_dir = embedding_path.parent
        source_audio_path = emotion_dir / "source_audio.wav"

        if not source_audio_path.exists():
            # Try parent directory (in case embedding was in a tier subfolder)
            source_audio_path = emotion_dir.parent / "source_audio.wav"

        if not source_audio_path.exists():
            current_tier = self._model_registry.quality_tier.display_name
            self.logger.error(
                f"Cannot fall back to source audio - file not found: {source_audio_path}. "
                f"Embedding is incompatible with {current_tier} tier."
            )
            return QwenTTSResponse(
                success=False,
                error_message=(
                    f"Voice embedding incompatible with {current_tier} model tier. "
                    f"No source audio available for fallback. "
                    f"Re-extract embeddings for this tier or switch to Quality tier."
                )
            )

        current_tier = self._model_registry.quality_tier.display_name
        self.logger.warning(
            f"Embedding incompatible with {current_tier} tier - "
            f"falling back to real-time voice cloning from {source_audio_path.name}"
        )

        # Use x_vector_only_mode if no ref_text available (slightly lower quality but works)
        use_xvector = not ref_text

        if use_xvector:
            self.logger.info("Using x-vector mode for fallback (no reference text available)")

        # Call generate_voice_clone with the source audio
        return await self.generate_voice_clone(
            text=text,
            ref_audio=source_audio_path,
            ref_text=ref_text or "",
            language=language,
            streaming=streaming,
            x_vector_only_mode=use_xvector
        )

    async def _generate(self, request: QwenTTSRequest) -> QwenTTSResponse:
        """
        Internal batch generation method that handles all model types.

        Args:
            request: Qwen TTS request

        Returns:
            QwenTTSResponse: Response with audio data or error
        """
        self._total_requests += 1
        start_time = time.time()
        self._generation_state = GenerationState.IDLE

        # Story 11.4: create the session at the very top of the entry point
        # for telemetry symmetry — even early-rejected requests get a
        # PENDING → ERROR → DISCARDED state trail. Local `sid` is None when
        # registry is absent (legacy code path runs unchanged).
        sid: Optional[str] = None
        if self._session_registry is not None:
            sid = self._session_registry.create_session(
                text=request.text,
                voice=self._resolve_voice_label(request),
                model_type=self._resolve_model_type_label(request),
                source=SessionSource.GENERATED,
            )
            # Story 16.5: publish the active session id so cancel_generation
            # can request_cancel(sid). Cleared in the outer finally below.
            self._current_session_id = sid

        # Story 11.4 review fix (F1): publish the running asyncio task so
        # cancel_generation() can call task.cancel() — the only mechanism
        # that gets CancelledError into the batch path (which never polls
        # _cancel_requested). Cleared in the outer finally below.
        self._current_generation_task = asyncio.current_task()

        try:
            # Validate text input (FR5)
            validation = self.validate_text(request.text)
            if not validation.can_proceed:
                self._failed_requests += 1
                error_code = (TTSErrorCode.EMPTY_TEXT if validation.status in
                             (TextValidationStatus.EMPTY, TextValidationStatus.WHITESPACE_ONLY)
                             else TTSErrorCode.TEXT_TOO_LONG)
                tts_error = TTSError(
                    code=error_code,
                    user_message=validation.message or "Invalid text input.",
                    recovery_suggestion="Enter text to speak." if error_code == TTSErrorCode.EMPTY_TEXT
                                       else "Try with shorter text.",
                    is_recoverable=True,
                )
                self._last_error = tts_error
                # Story 11.4: clean up the PENDING session — telemetry trail.
                if sid is not None and self._session_registry is not None:
                    self._session_registry.post_mutation('set_error', sid)
                    self._session_registry.post_mutation('discard', sid)
                return QwenTTSResponse(
                    success=False,
                    error_message=str(tts_error),
                )

            # Log warning for long text (but proceed)
            if validation.warning:
                self.logger.warning(f"Text validation warning: {validation.warning}")

            # Check service is running
            if not self.is_running():
                self._failed_requests += 1
                tts_error = TTSError(
                    code=TTSErrorCode.SERVICE_NOT_RUNNING,
                    user_message="TTS service is not running.",
                    recovery_suggestion="Please wait for the service to start.",
                    is_recoverable=True,
                )
                self._last_error = tts_error
                # Story 11.4: clean up the PENDING session — telemetry trail.
                if sid is not None and self._session_registry is not None:
                    self._session_registry.post_mutation('set_error', sid)
                    self._session_registry.post_mutation('discard', sid)
                return QwenTTSResponse(
                    success=False,
                    error_message=str(tts_error),
                )

            # Log with checkpoint info if present
            checkpoint_info = f", checkpoint={request.checkpoint_path}" if request.checkpoint_path else ""
            self.logger.info(
                f"Starting TTS generation (batch): model={request.model_type.display_name}{checkpoint_info}, "
                f"text='{request.text[:50]}...' ({validation.character_count} chars)"
            )

            # Ensure model is loaded (lazy loading)
            async with self._request_semaphore:
                self._generation_state = GenerationState.LOADING_MODEL

                # Notify model loading
                load_source = str(request.checkpoint_path) if request.checkpoint_path else request.model_type.display_name
                if self._model_loading_callback:
                    self._model_loading_callback(f"Loading {load_source}...")

                success, error = await self._model_registry.ensure_model_loaded(
                    request.model_type,
                    checkpoint_path=str(request.checkpoint_path) if request.checkpoint_path else None
                )

                if not success:
                    self._failed_requests += 1
                    self._generation_state = GenerationState.ERROR
                    # Story 11.4: model-load-failed → ERROR → DISCARDED.
                    if sid is not None and self._session_registry is not None:
                        self._session_registry.post_mutation('set_error', sid)
                        self._session_registry.post_mutation('discard', sid)
                    if self._generation_failed_callback:
                        self._generation_failed_callback(f"Failed to load model: {error}")
                    return QwenTTSResponse(
                        success=False,
                        error_message=f"Failed to load model: {error}"
                    )

                # Notify model ready
                if self._model_ready_callback:
                    self._model_ready_callback(request.model_type.display_name)

                # Notify generation started
                self._generation_state = GenerationState.GENERATING
                # Story 11.4: PENDING → GENERATING in parallel to legacy state.
                if sid is not None and self._session_registry is not None:
                    self._session_registry.post_mutation('start_generation', sid)
                if self._generation_started_callback:
                    self._generation_started_callback()

                # Execute generation in thread pool
                loop = asyncio.get_event_loop()
                audio_data, sample_rate = await loop.run_in_executor(
                    self._executor,
                    self._generate_sync,
                    request
                )

            # Save to cache file
            audio_file = self._save_audio_to_cache(audio_data, sample_rate)

            generation_time = time.time() - start_time
            self._successful_requests += 1
            self._last_generation_time = generation_time
            self._generation_state = GenerationState.COMPLETE

            # Story 11.4: append the entire batch buffer as one chunk, then
            # finalize. The registry needs at least one chunk for finalize
            # to succeed. Note: post_mutation() is QueuedConnection — the
            # legacy completion callback below fires synchronously while
            # these slot calls sit in the Qt event queue, so a subscriber
            # connected to both observes the callback first and the
            # session_state_changed(READY_TO_PLAY) emission on the next
            # event-loop iteration. Documented in Dev Notes ("QueuedConnection
            # ordering implications") and accepted by Story 12.1's
            # 5-second focal-decay window.
            if sid is not None and self._session_registry is not None:
                self._session_registry.post_mutation('append_chunk', sid, audio_data)
                self._session_registry.post_mutation('finalize', sid)

            self.logger.info(
                f"TTS generation complete (batch): {len(audio_data)} samples, "
                f"{generation_time:.2f}s"
            )

            # Notify completion
            if self._generation_complete_callback and audio_file:
                self._generation_complete_callback(audio_file)

            return QwenTTSResponse(
                success=True,
                audio_data=audio_data,
                sample_rate=sample_rate,
                audio_file_path=audio_file,
                generation_time_seconds=generation_time,
                mode=GenerationMode.BATCH,
            )

        except asyncio.CancelledError:
            self.logger.info("TTS generation cancelled")
            self._generation_state = GenerationState.CANCELLED
            # Note: text input is retained - only generation is aborted
            # Story 11.4: cancel → discard one-tick chain (P-7).
            if sid is not None and self._session_registry is not None:
                self._session_registry.post_mutation('cancel', sid)
                self._session_registry.post_mutation('discard', sid)
            if self._generation_cancelled_callback:
                self._generation_cancelled_callback()
            return QwenTTSResponse(
                success=False,
                error_message="Generation was cancelled"
            )

        except Exception as e:
            self.logger.exception(f"TTS generation failed: {e}")
            self._failed_requests += 1
            self._generation_state = GenerationState.ERROR
            # Story 11.4: set_error → discard one-tick chain.
            if sid is not None and self._session_registry is not None:
                self._session_registry.post_mutation('set_error', sid)
                self._session_registry.post_mutation('discard', sid)

            tts_error = self._handle_generation_error(e, used_fallback=False)

            return QwenTTSResponse(
                success=False,
                error_message=str(tts_error),
            )
        finally:
            # Story 11.4 review fix (F1): drop the task reference so a
            # later cancel_generation() doesn't try to cancel a finished
            # task. Set unconditionally — applies to every return path
            # above, including handler returns.
            self._current_generation_task = None
            # Story 16.5: drop the session id alongside the task so a later
            # cancel_generation() finds no in-flight session and the new
            # request_cancel call short-circuits to a quiet no-op. The
            # registry has already cleared the cancel hook via its own
            # terminal-state cleanup (cancel/discard slots).
            self._current_session_id = None

    async def _generate_streaming(self, request: QwenTTSRequest) -> QwenTTSResponse:
        """
        Internal streaming generation method - generates audio in progressive chunks.

        Splits text into sentences/phrases and generates each chunk sequentially,
        emitting audio_chunk_ready callback for each chunk to enable immediate
        playback while subsequent chunks are being generated.

        This achieves the NFR1 requirement of <2s first audio chunk latency.

        Args:
            request: Qwen TTS request

        Returns:
            QwenTTSResponse: Response with complete concatenated audio
        """
        self._total_requests += 1
        self._streaming_requests += 1
        start_time = time.time()
        first_chunk_time: Optional[float] = None
        self._cancel_requested = False
        self._generation_state = GenerationState.IDLE

        all_chunks: List[np.ndarray] = []
        sample_rate = 24000
        chunk_count = 0

        # Story 11.4: create the session at the very top of the entry point.
        # Local `sid` is None when registry is absent (legacy code path).
        sid: Optional[str] = None
        if self._session_registry is not None:
            sid = self._session_registry.create_session(
                text=request.text,
                voice=self._resolve_voice_label(request),
                model_type=self._resolve_model_type_label(request),
                source=SessionSource.GENERATED,
            )
            # Story 16.5: publish the active session id so cancel_generation
            # can request_cancel(sid). Cleared in the outer finally below.
            self._current_session_id = sid

        # Story 11.4 review fix (F1): publish the running asyncio task so
        # cancel_generation() can call task.cancel(). Streaming has the
        # _cancel_requested poll as a fallback path, but task.cancel() is
        # the canonical mechanism — and the only one that interrupts an
        # in-flight `await loop.run_in_executor(...)` chunk generation.
        self._current_generation_task = asyncio.current_task()

        try:
            # Validate text input (FR5)
            validation = self.validate_text(request.text)
            if not validation.can_proceed:
                self._failed_requests += 1
                error_code = (TTSErrorCode.EMPTY_TEXT if validation.status in
                             (TextValidationStatus.EMPTY, TextValidationStatus.WHITESPACE_ONLY)
                             else TTSErrorCode.TEXT_TOO_LONG)
                tts_error = TTSError(
                    code=error_code,
                    user_message=validation.message or "Invalid text input.",
                    recovery_suggestion="Enter text to speak." if error_code == TTSErrorCode.EMPTY_TEXT
                                       else "Try with shorter text.",
                    is_recoverable=True,
                )
                self._last_error = tts_error
                # Story 11.4: clean up the PENDING session — telemetry trail.
                if sid is not None and self._session_registry is not None:
                    self._session_registry.post_mutation('set_error', sid)
                    self._session_registry.post_mutation('discard', sid)
                return QwenTTSResponse(
                    success=False,
                    error_message=str(tts_error),
                    mode=GenerationMode.STREAMING,
                )

            # Log warning for long text (but proceed)
            if validation.warning:
                self.logger.warning(f"Text validation warning: {validation.warning}")

            # Check service is running
            if not self.is_running():
                self._failed_requests += 1
                tts_error = TTSError(
                    code=TTSErrorCode.SERVICE_NOT_RUNNING,
                    user_message="TTS service is not running.",
                    recovery_suggestion="Please wait for the service to start.",
                    is_recoverable=True,
                )
                self._last_error = tts_error
                # Story 11.4: clean up the PENDING session — telemetry trail.
                if sid is not None and self._session_registry is not None:
                    self._session_registry.post_mutation('set_error', sid)
                    self._session_registry.post_mutation('discard', sid)
                return QwenTTSResponse(
                    success=False,
                    error_message=str(tts_error),
                    mode=GenerationMode.STREAMING,
                )

            # Split text into chunks for progressive generation
            text_chunks = self._split_text_for_streaming(request.text)
            checkpoint_info = f", checkpoint={request.checkpoint_path}" if request.checkpoint_path else ""
            self.logger.info(
                f"Starting TTS generation (streaming): model={request.model_type.display_name}{checkpoint_info}, "
                f"chunks={len(text_chunks)}, text='{request.text[:50]}...'"
            )

            # Ensure model is loaded (lazy loading)
            async with self._request_semaphore:
                self._generation_state = GenerationState.LOADING_MODEL

                # Notify model loading
                load_source = str(request.checkpoint_path) if request.checkpoint_path else request.model_type.display_name
                if self._model_loading_callback:
                    self._model_loading_callback(f"Loading {load_source}...")

                success, error = await self._model_registry.ensure_model_loaded(
                    request.model_type,
                    checkpoint_path=str(request.checkpoint_path) if request.checkpoint_path else None
                )

                if not success:
                    self._failed_requests += 1
                    self._generation_state = GenerationState.ERROR
                    # Story 11.4: model-load-failed → ERROR → DISCARDED.
                    if sid is not None and self._session_registry is not None:
                        self._session_registry.post_mutation('set_error', sid)
                        self._session_registry.post_mutation('discard', sid)
                    if self._generation_failed_callback:
                        self._generation_failed_callback(f"Failed to load model: {error}")
                    return QwenTTSResponse(
                        success=False,
                        error_message=f"Failed to load model: {error}"
                    )

                # Notify model ready
                if self._model_ready_callback:
                    self._model_ready_callback(load_source)

                # Notify generation started
                self._generation_state = GenerationState.STREAMING
                # Story 11.4: legacy STREAMING maps to session GENERATING
                # (registry has no STREAMING substate per AC #2 table).
                if sid is not None and self._session_registry is not None:
                    self._session_registry.post_mutation('start_generation', sid)
                if self._generation_started_callback:
                    self._generation_started_callback()

                # Generate each chunk
                loop = asyncio.get_event_loop()

                for i, text_chunk in enumerate(text_chunks):
                    # Check for cancellation
                    if self._cancel_requested:
                        self.logger.info("Streaming generation cancelled by user")
                        self._generation_state = GenerationState.CANCELLED
                        # Story 11.4: cancel/discard handled in the outer
                        # except asyncio.CancelledError handler (one site).
                        raise asyncio.CancelledError()

                    # Skip empty chunks
                    if not text_chunk.strip():
                        continue

                    self.logger.debug(f"Generating chunk {i+1}/{len(text_chunks)}: '{text_chunk[:30]}...'")

                    # Create chunk request
                    chunk_request = QwenTTSRequest(
                        text=text_chunk,
                        language=request.language,
                        model_type=request.model_type,
                        speaker=request.speaker,
                        instruct=request.instruct,
                        ref_audio=request.ref_audio,
                        ref_text=request.ref_text,
                        x_vector_only_mode=request.x_vector_only_mode,
                        voice_description=request.voice_description,
                        streaming=False,  # Individual chunks use batch
                        checkpoint_path=request.checkpoint_path,
                        voice_clone_prompt=request.voice_clone_prompt,
                    )

                    # Generate chunk in thread pool
                    audio_data, sr = await loop.run_in_executor(
                        self._executor,
                        self._generate_sync,
                        chunk_request
                    )

                    sample_rate = sr
                    all_chunks.append(audio_data)
                    # Story 11.4: registry append after local accumulator.
                    if sid is not None and self._session_registry is not None:
                        self._session_registry.post_mutation('append_chunk', sid, audio_data)
                    chunk_count += 1

                    # Track first chunk latency
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - start_time
                        self.logger.info(f"First chunk latency: {first_chunk_time:.2f}s")

                    # Emit chunk ready callback
                    is_final = (i == len(text_chunks) - 1)
                    audio_chunk = AudioChunk(
                        audio_data=audio_data,
                        sample_rate=sr,
                        chunk_index=chunk_count - 1,
                        is_final=is_final,
                        text_segment=text_chunk,
                        session_id=sid,
                    )

                    if self._audio_chunk_ready_callback:
                        self._audio_chunk_ready_callback(audio_chunk)

                    # Allow other tasks to run
                    await asyncio.sleep(0)

            # Concatenate all chunks
            if all_chunks:
                complete_audio = np.concatenate(all_chunks)
            else:
                complete_audio = np.array([], dtype=np.float32)
            # Story 11.4: D-7 memory hygiene — clear the local accumulator
            # immediately after concat so only `complete_audio` remains.
            # Applies even in legacy mode (registry-less). The registry
            # session's own `chunks` list is cleared inside `finalize()`.
            all_chunks.clear()
            if sid is not None and self._session_registry is not None:
                if chunk_count > 0:
                    self._session_registry.post_mutation('finalize', sid)
                else:
                    # Degenerate case (empty text_chunks): the session is
                    # in GENERATING with no chunks; finalize would raise.
                    # Clean up via set_error → discard for a bounded
                    # registry collection.
                    self._session_registry.post_mutation('set_error', sid)
                    self._session_registry.post_mutation('discard', sid)

            # Save to cache file
            audio_file = self._save_audio_to_cache(complete_audio, sample_rate)

            generation_time = time.time() - start_time
            self._successful_requests += 1
            self._last_generation_time = generation_time
            self._generation_state = GenerationState.COMPLETE

            # Story 11.3: emit through the single-chokepoint helper (P-9).
            # The _FirstChunkLatencyAggregator subscribes in __init__ and
            # updates ``_avg_first_chunk_latency`` synchronously inside
            # record(), so the field is already current by the time control
            # returns here. Story 11.4 wires the registry-issued session_id.
            if first_chunk_time is not None:
                metrics.record(
                    "first_chunk_latency_ms",
                    first_chunk_time * 1000.0,
                    session_id=sid,  # Story 11.4: registry-issued session id
                    model_type=(
                        request.model_type.display_name
                        if request.model_type is not None
                        else "default"
                    ),
                    hardware=(
                        "gpu"
                        if "cuda" in str(self._model_registry.device).lower()
                        else "cpu"
                    ),
                )

            self.logger.info(
                f"TTS generation complete (streaming): {len(complete_audio)} samples, "
                f"{chunk_count} chunks, {generation_time:.2f}s total, "
                f"{first_chunk_time:.2f}s first chunk"
            )

            # Notify completion
            if self._generation_complete_callback and audio_file:
                self._generation_complete_callback(audio_file)

            return QwenTTSResponse(
                success=True,
                audio_data=complete_audio,
                sample_rate=sample_rate,
                audio_file_path=audio_file,
                generation_time_seconds=generation_time,
                mode=GenerationMode.STREAMING,
                chunks_generated=chunk_count,
                first_chunk_latency=first_chunk_time,
            )

        except asyncio.CancelledError:
            self.logger.info("Streaming generation cancelled")
            self._generation_state = GenerationState.CANCELLED
            # Note: text input is retained - only generation is aborted
            # Story 11.4: cancel → discard one-tick chain (P-7). Defensive
            # `sid is not None` because cancellation can fire before
            # create_session if the asyncio task is cancelled extremely
            # early; in that case the legacy code path runs unchanged.
            if sid is not None and self._session_registry is not None:
                self._session_registry.post_mutation('cancel', sid)
                self._session_registry.post_mutation('discard', sid)
            if self._generation_cancelled_callback:
                self._generation_cancelled_callback()
            return QwenTTSResponse(
                success=False,
                error_message="Generation was cancelled",
                mode=GenerationMode.STREAMING,
                chunks_generated=chunk_count,
            )

        except Exception as e:
            self.logger.exception(f"Streaming generation failed: {e}")
            # Story 11.4: the streaming session is doomed; mark + discard
            # NOW so the recursive batch fallback below creates a fresh
            # session via `_generate(request)` (Task 3 wiring) rather than
            # racing with this one in the registry.
            if sid is not None and self._session_registry is not None:
                self._session_registry.post_mutation('set_error', sid)
                self._session_registry.post_mutation('discard', sid)

            # Try batch fallback (FR3, NFR7: graceful degradation)
            self.logger.warning("[QwenTTS] Streaming failed, falling back to batch")
            self._fallback_count += 1

            try:
                # Reset state for batch attempt
                self._generation_state = GenerationState.GENERATING
                request.streaming = False

                # Attempt batch generation
                batch_response = await self._generate(request)

                # If batch succeeded, mark the response as having used fallback
                if batch_response.success:
                    self.logger.info("[QwenTTS] Batch fallback succeeded")
                    # Note: batch_response is already complete, user experience is seamless
                    return batch_response
                else:
                    # Batch also failed - this is an unrecoverable failure
                    self.logger.error("[QwenTTS] Batch fallback also failed")
                    self._failed_requests += 1
                    self._generation_state = GenerationState.ERROR

                    # Include the actual error in user message for better debugging
                    error_detail = str(e)
                    if len(error_detail) > 100:
                        error_detail = error_detail[:100] + "..."

                    tts_error = TTSError(
                        code=TTSErrorCode.UNKNOWN,
                        user_message=f"Speech generation failed: {error_detail}",
                        recovery_suggestion="Check logs for details.",
                        technical_details=f"Streaming error: {e}; Batch error: {batch_response.error_message}",
                        is_recoverable=True,
                        used_fallback=True,
                    )
                    self._last_error = tts_error

                    if self._generation_error_callback:
                        self._generation_error_callback(tts_error)
                    if self._generation_failed_callback:
                        self._generation_failed_callback(str(tts_error))

                    return QwenTTSResponse(
                        success=False,
                        error_message=str(tts_error),
                        mode=GenerationMode.STREAMING,
                        chunks_generated=chunk_count,
                    )

            except Exception as fallback_error:
                # Both streaming and batch failed
                self.logger.exception(f"[QwenTTS] Batch fallback failed: {fallback_error}")
                self._failed_requests += 1
                self._generation_state = GenerationState.ERROR

                tts_error = self._handle_generation_error(fallback_error, used_fallback=True)

                return QwenTTSResponse(
                    success=False,
                    error_message=str(tts_error),
                    mode=GenerationMode.STREAMING,
                    chunks_generated=chunk_count,
                )

        finally:
            # Story 11.4 review fix (F1): drop the task reference so a
            # later cancel_generation() doesn't try to cancel a finished
            # task. The recursive batch-fallback _generate() also sets and
            # clears this field, but the outer clear is idempotent.
            self._current_generation_task = None
            # Story 16.5: drop the session id alongside the task. Recursive
            # batch-fallback _generate() set its own _current_session_id (a
            # different sid for the fallback session) and cleared it in its
            # own finally; the outer clear here covers the streaming sid.
            self._current_session_id = None

    def _split_text_for_streaming(self, text: str) -> List[str]:
        """
        Split text into chunks suitable for streaming generation.

        Splits on sentence boundaries (. ! ? 。 ！ ？) and merges
        very short chunks to avoid generating tiny audio fragments.

        Args:
            text: Full text to split

        Returns:
            List of text chunks
        """
        # Split on sentence boundaries
        sentences = self.SENTENCE_SPLIT_PATTERN.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [text.strip()] if text.strip() else []

        # Merge short sentences
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if not current_chunk:
                current_chunk = sentence
            elif len(current_chunk) < self.MIN_CHUNK_LENGTH:
                # Merge with current chunk
                current_chunk = f"{current_chunk} {sentence}"
            else:
                chunks.append(current_chunk)
                current_chunk = sentence

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        # If we ended up with just one chunk, return it
        if len(chunks) == 1:
            return chunks

        # If text is short overall, just return as single chunk
        if len(text) < self.MIN_CHUNK_LENGTH * 2:
            return [text.strip()]

        return chunks

    def _build_true_stream_decode_fn(
        self, model: Any
    ) -> Callable[[Any], np.ndarray]:
        """Story 16.8 — adapter wrapping the qwen-tts speech tokenizer's decode.

        Returned callable conforms to ``StreamingDecoderWorker``'s P-6
        contract: input is one chunk produced by the talker forward-hook
        (a ``torch.Tensor`` of shape ``(N_steps, num_code_groups)``); output
        is a float32 PCM sample array.

        Two Story-16.6 bugs uncovered and fixed in 16.8 against real
        qwen-tts 0.0.4 (RTX 5090, see Story 16.8 Change Log entry #5):

          1. ``model.speech_tokenizer`` does not exist — the speech
             tokenizer lives on the inner ``Qwen3TTSForConditionalGeneration``
             at ``model.model.speech_tokenizer`` (set during the inner
             wrapper's ``from_pretrained`` at modeling_qwen3_tts.py:1920).
          2. The 12Hz tokenizer's ``decode`` expects shape
             ``(batch_size, codes_length, num_quantizers)`` (see
             ``modeling_qwen3_tts_tokenizer_v2.py:1004``). Passing a flat
             tensor of single-codebook tokens silently misinterprets
             ``codes_length`` as ``num_quantizers``. The fix is to receive
             multi-codebook ``codec_ids`` from the forward-hook and wrap
             in a per-sample dict ``[{"audio_codes": chunk}]`` so the
             outer ``Qwen3TTSTokenizer.decode`` normalizer adds the batch
             dimension correctly.

        The trip-wire test at ``tests/test_qwen_tts_internals.py`` covers
        the ``Qwen3TTSTokenizerV1Model.decode`` symbol (Story 16.4 added
        it; Story 16.8 left it unchanged). 12Hz models use V2 not V1, so
        the 12Hz path is not currently pinned — opportunity for a future
        trip-wire extension if 12Hz becomes critical.
        """
        def _decode(chunk: Any) -> np.ndarray:
            import torch as _torch  # lazy: keeps test envs without
            # CUDA-bound torch DLLs viable until decode actually fires
            # Forward-hook produces ``(N_steps, num_code_groups)`` tensors
            # already; defensive coerce keeps us robust to upstream shape
            # tweaks and to the residual-flush path (where the tensor may
            # have come from torch.cat).
            if not isinstance(chunk, _torch.Tensor):
                chunk = _torch.as_tensor(chunk, dtype=_torch.long)
            if chunk.dim() == 1:
                # Defensive: a 1-D chunk would be single-codebook only;
                # treat the lone dimension as ``codes_length`` and
                # ``num_quantizers=1``. Decode will likely fail or
                # produce wrong audio — the talker forward-hook should
                # have produced 2-D — but this prevents a crash so the
                # empty-chunks guard fires instead of an uncaught
                # exception killing the worker thread.
                chunk = chunk.unsqueeze(-1)
            # ``Qwen3TTSTokenizer.decode`` accepts a list of dicts per
            # qwen3_tts_tokenizer.py:307-311. Each dict's ``audio_codes``
            # is the per-sample tensor of shape ``(N_steps, num_quantizers)``.
            result = model.model.speech_tokenizer.decode(
                [{"audio_codes": chunk}]
            )
            # ``decode`` returns ``(wavs: List[np.ndarray], sample_rate: int)``
            # per qwen3_tts_tokenizer.py:281-283. We only have one sample
            # in the per-call list above, so wavs[0] is our PCM.
            if isinstance(result, tuple):
                wavs = result[0]
            else:
                wavs = result
            if isinstance(wavs, list) and wavs:
                audio = wavs[0]
            else:
                audio = wavs
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            return np.asarray(audio, dtype=np.float32).flatten()
        return _decode

    def _build_true_stream_talker(
        self,
        model: Any,
        request: "QwenTTSRequest",
        streamer: CodecTokenStreamer,
    ) -> Callable[[], None]:
        """Story 16.8 — talker callable using Path A (talker-patch variant).

        Replaces Story 16.6's literal ``model.model.generate(streamer=streamer)``
        wire-up — which Story 16.7 §3.1 measured as silently broken (50/50
        utterances on the maintainer's RTX 5090 produced empty chunks
        because the call passed no conditioning args) — with a patch-based
        approach that reaches the canonical Path A target
        (``self.talker.generate(inputs_embeds=..., streamer=streamer, ...)``)
        without replicating the wrapper's ~250 lines of preprocessing.

        Why a talker-patch rather than full preprocessing replication. The
        Story 16.8 Dev Notes describe two paths:

          - Path B: rely on the wrapper's ``**kwargs`` to forward
            ``streamer`` through to the inner ``self.talker.generate``.
            Source-read of ``modeling_qwen3_tts.py:2042-2278`` and the
            ``probe_qwen_tts_streamer.py`` empirical run both confirm the
            wrapper drops ``streamer`` before reaching the inner call — Path
            B is structurally broken at qwen-tts 0.0.4.
          - Path A (literal "replicate preprocessing"): hand-roll the input
            embedding tower (~150-250 lines of codec prefill, language
            token resolution, speaker embedding construction, role prefix,
            attention mask, trailing-text-hidden, batch left-padding).
            Brittle to upstream qwen-tts changes; large maintenance
            surface; duplicate of the wrapper's body.

        The talker-patch variant lands the same destination as literal
        Path A — ``self.talker.generate(streamer=streamer, ...)`` is
        invoked with the wrapper's correctly-constructed ``inputs_embeds``,
        ``attention_mask``, ``trailing_text_hidden``, and ``tts_pad_embed`` —
        but *does not* duplicate the preprocessing in MyVoice. The trade-off:
        it depends on the wrapper calling ``self.talker.generate(...)``
        exactly once during ``model.generate_*(...)``, which the trip-wire
        extension (``tests/test_qwen_tts_internals.py``) pins via attribute
        and call-site assertions so a future qwen-tts version that fans out
        to a different talker entrypoint will fail CI before silently
        regressing streaming.

        Concurrency: the patch is installed on the shared
        ``model.model.talker`` instance for the duration of one
        ``model.generate_*(...)`` call. Story 16.6's session-registry P-7
        invariant (one in-flight TRUE_STREAM dispatch per service instance)
        ensures the patch is single-threaded with respect to itself; the
        outer ``try/finally`` restores the bound method on every exit
        (success, exception, sentinel).

        Cancellation invariant (D-11): the talker thread MUST NOT raise on
        cancel. The patch is transparent to HF GenerationMixin's
        cancellation flow — when ``streamer._cancel_event`` flips,
        ``streamer.put`` becomes a no-op, HF iterates a few more times
        producing tokens we drop, then completes cleanly and calls
        ``streamer.end()``. No exceptions raised through HF internals;
        CUDA state stays clean.

        Short-circuit sentinel: HF's ``GenerationMixin.generate`` calls
        ``streamer.end()`` when token generation completes; the wrapper
        then runs a residual non-streaming ``speech_tokenizer.decode`` to
        build the full waveform return tuple. Since the streaming worker
        has already decoded chunk-by-chunk, that residual decode is
        wasted GPU compute. Raising ``_TalkerStreamComplete`` from the
        injecting wrapper short-circuits past it, saving ~the same time
        as one non-streaming generation per dispatch.
        """
        from myvoice.models.service_enums import QwenModelType

        # Local sentinel so it does not leak past this dispatch.
        class _TalkerStreamComplete(Exception):
            """Raised by the streamer-injecting wrapper after
            ``self.talker.generate`` returns, to skip the public wrapper's
            residual non-streaming decode work (which the streaming worker
            has already produced chunk-by-chunk)."""
            pass

        def _run_talker() -> None:
            real_talker_generate = model.model.talker.generate
            real_talker_forward = model.model.talker.forward
            forward_invocations_box = [0]
            chunks_pushed_box = [0]
            # Per-step buffer of multi-codebook codec_ids tensors. Each
            # entry is shape ``(batch=1, num_code_groups)``. We push to
            # ``streamer.queue`` directly (not via ``streamer.put``) so
            # the buffer is in STEPS not flat ints — required because
            # the qwen-tts 12Hz tokenizer's ``decode`` expects
            # ``(N_steps, num_code_groups)`` per sample.
            step_buffer: list = []
            chunk_size = streamer.chunk_size
            lookahead = streamer.lookahead
            chunk_with_lookahead = chunk_size + lookahead

            # Preserve the original forward's signature on our wrapper so
            # HF GenerationMixin's ``_validate_model_kwargs`` introspection
            # at ``transformers/generation/utils.py:1562-1566`` sees the
            # full parameter list (``trailing_text_hidden``, ``tts_pad_embed``,
            # ``subtalker_*``, etc.) rather than the wrapper's bare
            # ``*args, **kwargs``. Without this, HF raises
            # ``ValueError("The following model_kwargs are not used by the
            # model")`` when the wrapper passes the talker's custom kwargs
            # at modeling_qwen3_tts.py:2272-2278 — empirically observed in
            # Story 16.8's RTX 5090 harness re-run.
            import inspect as _inspect_local
            try:
                _real_forward_sig = _inspect_local.signature(real_talker_forward)
            except (ValueError, TypeError):  # pragma: no cover (defensive)
                _real_forward_sig = None

            def _streaming_forward(*args: Any, **kwargs: Any) -> Any:
                """Forward-hook that captures multi-codebook ``codec_ids``
                from each generation step and pushes chunks to the
                streamer queue when ``chunk_size + lookahead`` STEPS have
                accumulated.

                Story 16.8 finding: HF ``GenerationMixin._sample`` calls
                ``streamer.put(next_tokens)`` with the codec_head's
                MAIN-codebook sample only — the other codebooks are
                produced inside ``Qwen3TTSTalkerForConditionalGeneration.forward``
                via ``code_predictor.generate`` (modeling_qwen3_tts.py:1671)
                and returned in ``Qwen3TTSTalkerOutputWithPast.hidden_states[1]``
                (line 1738). We bypass HF's per-token streamer protocol
                entirely and capture from the forward output instead.
                """
                forward_invocations_box[0] += 1
                output = real_talker_forward(*args, **kwargs)
                # Prefill (inputs_embeds.shape[1] > 1): codec_ids is None
                # per modeling_qwen3_tts.py:1665-1667. Skip — no codec
                # tokens generated this step.
                hidden_states = getattr(output, "hidden_states", None)
                if hidden_states is None or len(hidden_states) < 2:
                    return output
                codec_ids = hidden_states[1]
                if codec_ids is None:
                    return output
                # Cooperative cancel (D-11): stop accumulating but let
                # HF iterate cleanly. The wrapper post-processing won't
                # observe inconsistency because we short-circuit via
                # _TalkerStreamComplete after generate returns.
                if streamer._cancel_event.is_set():
                    return output
                step_buffer.append(codec_ids)
                # Flush full chunks while threshold reached. ``while``
                # because a single forward call shouldn't produce >1
                # chunk's worth of steps, but defensive against future
                # batch-wise generation patterns.
                while len(step_buffer) >= chunk_with_lookahead:
                    if streamer._cancel_event.is_set():
                        break
                    # Stack the first ``chunk_with_lookahead`` steps into
                    # a (chunk_with_lookahead, num_code_groups) tensor.
                    # Each step is shape ``(1, num_code_groups)``; cat
                    # along dim=0 then squeeze the leading 1-batch dim.
                    chunk_tensor = _torch_local.cat(
                        step_buffer[:chunk_with_lookahead], dim=0
                    )
                    # Push to streamer.queue directly (bypass put/buffer).
                    # Backpressure: queue.put blocks if maxsize reached;
                    # HF generate yields the GIL between steps so the
                    # decoder worker drains naturally.
                    streamer.queue.put(chunk_tensor)
                    chunks_pushed_box[0] += 1
                    # Slide forward by chunk_size; keep the trailing
                    # ``lookahead`` steps as left-context for next chunk.
                    del step_buffer[:chunk_size]
                return output

            # Apply the captured signature so HF's introspection works.
            if _real_forward_sig is not None:
                _streaming_forward.__signature__ = _real_forward_sig
            try:
                _streaming_forward.__wrapped__ = real_talker_forward
            except (AttributeError, TypeError):  # pragma: no cover (defensive)
                pass

            def _patched_talker_generate(*args: Any, **kwargs: Any) -> Any:
                """Run the real generate (with forward-hook installed),
                then short-circuit the wrapper's residual non-streaming
                decode via _TalkerStreamComplete.

                Do NOT inject ``streamer`` kwarg — HF's per-token streamer
                protocol is incompatible with the qwen-tts multi-codebook
                architecture (HF's ``streamer.put`` fires only with the
                main codec_head's sample, but the speech_tokenizer's
                ``decode`` requires all ``num_code_groups`` codebooks per
                step). The forward-hook captures multi-codebook codec_ids
                from the talker's per-step output instead.
                """
                real_talker_generate(*args, **kwargs)
                raise _TalkerStreamComplete()

            # Lazy-import torch here so the module-level ``import torch``
            # in qwen_tts_service.py doesn't change. Avoids growing the
            # import surface and keeps test environments without
            # CUDA-bound torch DLLs viable.
            import torch as _torch_local

            def _flush_residual_and_eos(buf: list, strm: Any, torch_mod: Any) -> None:
                """Push residual ``step_buffer`` as one final chunk (if any)
                then push ``END_OF_STREAM`` so the worker exits cleanly. Both
                are direct ``streamer.queue.put`` calls; the streamer's
                int-buffer ``put()/end()`` mechanism is bypassed because
                Story 16.8 streams whole multi-codebook tensors per step.

                Logs (rather than silently swallows) failures from
                ``torch.cat`` or ``queue.put`` so a future codebook-shape
                regression or a closed-queue case is diagnosable post-mortem
                instead of presenting as a 60s join-timeout stall.
                """
                if buf:
                    try:
                        residual_tensor = torch_mod.cat(buf, dim=0)
                        strm.queue.put(residual_tensor)
                    except Exception:
                        self.logger.exception(
                            "[QwenTTS] TRUE_STREAM residual flush failed; "
                            "the final chunk will be dropped"
                        )
                    buf.clear()
                try:
                    strm.queue.put(END_OF_STREAM)
                except Exception:
                    self.logger.exception(
                        "[QwenTTS] TRUE_STREAM failed to push END_OF_STREAM; "
                        "the decoder worker may hang until the dispatch "
                        "join-timeout fires"
                    )

            try:
                model.model.talker.generate = _patched_talker_generate
                model.model.talker.forward = _streaming_forward
                try:
                    if request.model_type == QwenModelType.CUSTOM_VOICE:
                        model.generate_custom_voice(
                            text=request.text,
                            speaker=request.speaker or "",
                            language=request.language or "Auto",
                            instruct=request.instruct,
                            non_streaming_mode=False,
                        )
                    elif request.model_type == QwenModelType.VOICE_DESIGN:
                        model.generate_voice_design(
                            text=request.text,
                            instruct=request.instruct or "",
                            language=request.language or "Auto",
                            non_streaming_mode=False,
                        )
                    elif request.model_type == QwenModelType.BASE:
                        if request.voice_clone_prompt is None:
                            raise ValueError(
                                "TRUE_STREAM voice-clone path requires "
                                "request.voice_clone_prompt"
                            )
                        model.generate_voice_clone(
                            text=request.text,
                            language=request.language or "Auto",
                            voice_clone_prompt=request.voice_clone_prompt,
                            non_streaming_mode=False,
                        )
                    else:
                        raise NotImplementedError(
                            f"TRUE_STREAM does not support model_type "
                            f"{request.model_type!r}"
                        )
                finally:
                    # Always restore — patches must not leak past this
                    # dispatch. Restoration order doesn't matter; both
                    # are independent.
                    model.model.talker.generate = real_talker_generate
                    model.model.talker.forward = real_talker_forward
            except _TalkerStreamComplete:
                # Expected success path: forward-hook captured codec_ids per
                # step and pushed chunks. Now flush residual step_buffer as
                # the final chunk and push END_OF_STREAM to signal the worker.
                _flush_residual_and_eos(step_buffer, streamer, _torch_local)
                return
            except Exception as exc:
                self.logger.exception(
                    f"[QwenTTS] TRUE_STREAM talker error: {exc}"
                )
                # Do NOT set ``streamer._cancel_event`` here. Setting it
                # would conflate "talker raised a structural error" with
                # "user requested cancel" — the worker's drain-on-cancel
                # logic in ``StreamingDecoderWorker`` would then post the
                # canonical ``('cancel', sid)`` registry transition rather
                # than the error transition, polluting session-state
                # telemetry. The empty-chunks guard inside
                # ``_generate_true_stream`` (the ``if not accumulated_chunks``
                # check) catches the empty queue and routes to the fallback
                # chain; that is the canonical error-recovery path. Drop
                # the residual buffer and push END_OF_STREAM so the worker
                # exits its loop cleanly.
                step_buffer.clear()
                try:
                    streamer.queue.put(END_OF_STREAM)
                except Exception:
                    self.logger.exception(
                        "[QwenTTS] TRUE_STREAM error-path failed to push "
                        "END_OF_STREAM; worker may hang until join-timeout"
                    )
                return

            # The wrapper completed without firing the talker.generate
            # patch — i.e., it returned before reaching
            # ``self.talker.generate(...)``. Possible causes: an early-return
            # in the wrapper, a validation short-circuit, or a qwen-tts
            # version where the talker is invoked through a different
            # entrypoint. Either way no codec_ids were captured, so signal
            # end-of-stream and let the empty-chunks guard inside
            # ``_generate_true_stream`` (the ``if not accumulated_chunks``
            # check) route to the fallback chain.
            self.logger.warning(
                "[QwenTTS] TRUE_STREAM wrapper completed but "
                "talker.generate was never invoked; the empty-chunks guard "
                "will route to fallback"
            )
            step_buffer.clear()
            try:
                streamer.queue.put(END_OF_STREAM)
            except Exception:
                self.logger.exception(
                    "[QwenTTS] TRUE_STREAM wrapper-empty path failed to "
                    "push END_OF_STREAM; worker may hang until join-timeout"
                )

        return _run_talker

    async def _generate_true_stream(
        self, request: "QwenTTSRequest"
    ) -> "QwenTTSResponse":
        """Story 16.6 Task 3 — TRUE_STREAM dispatch path (P-5/P-6/P-7/P-8/D-9).

        Composes Stories 16.1–16.5's surfaces into the production TRUE_STREAM
        dispatch:

        - 16.2 ``effective_streaming_mode`` determined this path was reached
        - 16.3 ``CodecTokenStreamer`` is the producer plug-compatible with
          HF's ``BaseStreamer``; the talker thread runs
          ``model.model.generate(streamer=streamer)``
        - 16.4 ``StreamingDecoderWorker`` is the consumer; it pulls from the
          streamer's queue, decodes via ``decode_fn``, and posts
          ``('append_chunk', sid, pcm)`` and ``('finalize', sid)`` to the
          registry through the ``post_mutation`` callable
        - 16.5 ``register_cancel_hook`` + ``request_cancel`` flip
          ``streamer._cancel_event`` AND call
          ``audio_coordinator.cancel_playback(sid)`` on user-cancel; the
          worker's drain-on-cancel posts the canonical ``('cancel', sid)``

        Per P-7, this method's ``asyncio.CancelledError`` handler must NOT
        post ``('cancel', sid)`` itself — the worker is the canonical source
        of the CANCELLED transition, and a double-post would violate Story
        16.5 AC #1's "exactly one cancel post" assertion.
        """
        self._total_requests += 1
        self._streaming_requests += 1
        start_time = time.time()
        first_chunk_time: Optional[float] = None
        self._cancel_requested = False
        self._generation_state = GenerationState.IDLE

        accumulated_chunks: List[np.ndarray] = []
        # Hard-coded for Qwen3-TTS (the only model that flows through
        # TRUE_STREAM dispatch today). If a future model variant uses a
        # different rate, derive from `self._model_registry` and update
        # both this binding and the `sample_rate=` field in the
        # `_wrapped_post` AudioChunk emissions below — the consumer
        # (app.py:_handle_progressive_chunk_async) opens the audio
        # device with whatever rate the chunk reports, so a mismatch
        # here would surface as silent corruption (chipmunk/slow audio),
        # not a crash.
        sample_rate = 24000
        chunk_count_box: List[int] = [0]

        sid: Optional[str] = None
        streamer: Optional[CodecTokenStreamer] = None
        worker: Optional[StreamingDecoderWorker] = None
        talker_thread: Optional[threading.Thread] = None

        if self._session_registry is not None:
            sid = self._session_registry.create_session(
                text=request.text,
                voice=self._resolve_voice_label(request),
                model_type=self._resolve_model_type_label(request),
                source=SessionSource.GENERATED,
            )
            # Story 16.5: publish the active session id so cancel_generation
            # can request_cancel(sid) -> hook -> streamer._cancel_event.set().
            self._current_session_id = sid

        self._current_generation_task = asyncio.current_task()

        try:
            # Validate text input (FR5).
            validation = self.validate_text(request.text)
            if not validation.can_proceed:
                self._failed_requests += 1
                error_code = (
                    TTSErrorCode.EMPTY_TEXT
                    if validation.status in (
                        TextValidationStatus.EMPTY,
                        TextValidationStatus.WHITESPACE_ONLY,
                    )
                    else TTSErrorCode.TEXT_TOO_LONG
                )
                tts_error = TTSError(
                    code=error_code,
                    user_message=validation.message or "Invalid text input.",
                    recovery_suggestion=(
                        "Enter text to speak."
                        if error_code == TTSErrorCode.EMPTY_TEXT
                        else "Try with shorter text."
                    ),
                    is_recoverable=True,
                )
                self._last_error = tts_error
                if sid is not None and self._session_registry is not None:
                    self._session_registry.post_mutation('set_error', sid)
                    self._session_registry.post_mutation('discard', sid)
                return QwenTTSResponse(
                    success=False,
                    error_message=str(tts_error),
                    mode=GenerationMode.STREAMING,
                )

            if validation.warning:
                self.logger.warning(
                    f"Text validation warning: {validation.warning}"
                )

            if not self.is_running():
                self._failed_requests += 1
                tts_error = TTSError(
                    code=TTSErrorCode.SERVICE_NOT_RUNNING,
                    user_message="TTS service is not running.",
                    recovery_suggestion="Please wait for the service to start.",
                    is_recoverable=True,
                )
                self._last_error = tts_error
                if sid is not None and self._session_registry is not None:
                    self._session_registry.post_mutation('set_error', sid)
                    self._session_registry.post_mutation('discard', sid)
                return QwenTTSResponse(
                    success=False,
                    error_message=str(tts_error),
                    mode=GenerationMode.STREAMING,
                )

            self.logger.info(
                f"Starting TTS generation (TRUE_STREAM): "
                f"model={request.model_type.display_name}, "
                f"text='{request.text[:50]}...'"
            )

            # Per AC #10: the model-load + streamer-construction critical
            # section is guarded by the existing semaphore so concurrent
                # TRUE_STREAM dispatches serialize.
            async with self._request_semaphore:
                self._generation_state = GenerationState.LOADING_MODEL

                if self._model_loading_callback:
                    self._model_loading_callback(
                        f"Loading {request.model_type.display_name}..."
                    )

                success, error = await self._model_registry.ensure_model_loaded(
                    request.model_type,
                    checkpoint_path=(
                        str(request.checkpoint_path)
                        if request.checkpoint_path
                        else None
                    ),
                )
                if not success:
                    self._failed_requests += 1
                    self._generation_state = GenerationState.ERROR
                    if sid is not None and self._session_registry is not None:
                        self._session_registry.post_mutation('set_error', sid)
                        self._session_registry.post_mutation('discard', sid)
                    if self._generation_failed_callback:
                        self._generation_failed_callback(
                            f"Failed to load model: {error}"
                        )
                    return QwenTTSResponse(
                        success=False,
                        error_message=f"Failed to load model: {error}",
                        mode=GenerationMode.STREAMING,
                    )

                if self._model_ready_callback:
                    self._model_ready_callback(request.model_type.display_name)

                self._generation_state = GenerationState.STREAMING
                if sid is not None and self._session_registry is not None:
                    self._session_registry.post_mutation('start_generation', sid)
                if self._generation_started_callback:
                    self._generation_started_callback()

                # Build streamer + worker pair (Story 16.3 + 16.4).
                streamer = CodecTokenStreamer()
                model = self._model_registry.get_loaded_model()
                decode_fn = self._build_true_stream_decode_fn(model)

                hardware_label = (
                    "gpu"
                    if "cuda" in str(self._model_registry.device).lower()
                    else "cpu"
                )

                # Wrap the registry's post_mutation so the dispatch path can
                # observe append_chunk / finalize timing without a separate
                # subscription. The registry still receives every call.
                registry_post = (
                    self._session_registry.post_mutation
                    if self._session_registry is not None
                    else None
                )

                def _wrapped_post(*args: Any, **kwargs: Any) -> None:
                    if registry_post is not None:
                        registry_post(*args, **kwargs)
                    if args and args[0] == 'append_chunk' and len(args) >= 3:
                        nonlocal first_chunk_time
                        if first_chunk_time is None:
                            first_chunk_time = time.time() - start_time
                        chunk_data = np.asarray(args[2])
                        accumulated_chunks.append(chunk_data)
                        chunk_index = chunk_count_box[0]
                        chunk_count_box[0] += 1
                        # Story 18.1 Task 1.1: per-chunk emit timestamp
                        # (wall-clock ms — joinable with the consumer-
                        # side ``progressive_chunk_playback_arrival_ms``
                        # by (session_id, chunk_index)). Overhead
                        # validated ≤ 100 µs/call in evidence file §1.
                        metrics.record(
                            "progressive_chunk_emit_ms",
                            time.time() * 1000.0,
                            session_id=sid,
                            chunk_index=chunk_index,
                        )
                        # Story 17.3: emit progressive-playback callback. Parallels
                        # SENTENCE_STREAM at qwen_tts_service.py:3071-3082. Additive to
                        # the accumulator/counter; never replaces them. Wrapped so a
                        # buggy consumer cannot break the producer thread.
                        if self._audio_chunk_ready_callback is not None:
                            try:
                                self._audio_chunk_ready_callback(
                                    AudioChunk(
                                        audio_data=chunk_data,
                                        sample_rate=sample_rate,
                                        chunk_index=chunk_index,
                                        is_final=False,
                                        text_segment="",
                                        session_id=sid,
                                    )
                                )
                            except Exception:
                                self.logger.exception(
                                    "[QwenTTS] TRUE_STREAM "
                                    "_audio_chunk_ready_callback raised on "
                                    "append_chunk; swallowing"
                                )
                    elif args and args[0] == 'finalize':
                        # Story 17.3: synthetic terminal AudioChunk lets the
                        # progressive-playback consumer close its open audio session
                        # without needing a separate "stream done" channel.
                        # Zero-length payload — consumer must skip play_audio_chunk
                        # if audio_data.size == 0.
                        if self._audio_chunk_ready_callback is not None:
                            try:
                                self._audio_chunk_ready_callback(
                                    AudioChunk(
                                        audio_data=np.zeros(0, dtype=np.float32),
                                        sample_rate=sample_rate,
                                        chunk_index=chunk_count_box[0],
                                        is_final=True,
                                        text_segment="",
                                        session_id=sid,
                                    )
                                )
                            except Exception:
                                self.logger.exception(
                                    "[QwenTTS] TRUE_STREAM terminal "
                                    "_audio_chunk_ready_callback raised on "
                                    "finalize; swallowing"
                                )

                worker = StreamingDecoderWorker(
                    streamer=streamer,
                    decode_fn=decode_fn,
                    post_mutation=_wrapped_post,
                    session_id=sid or "no-registry",
                    model_type=self._resolve_model_type_label(request),
                    hardware=hardware_label,
                )

                # Story 16.5: register the cancel hook BEFORE starting the
                # threads so a cancel that arrives during spawn fires.
                # ``get_running_loop()`` is the correct API inside an async
                # def — ``get_event_loop()`` is deprecated in Py 3.10+ and
                # has surprise semantics on Py 3.12+ (Story 16.6 review H3).
                event_loop = asyncio.get_running_loop()
                hook_session_id = sid

                def _cancel_hook() -> None:
                    if streamer is not None:
                        streamer._cancel_event.set()
                    if (
                        self.audio_coordinator is not None
                        and hook_session_id is not None
                    ):
                        try:
                            asyncio.run_coroutine_threadsafe(
                                self.audio_coordinator.cancel_playback(
                                    hook_session_id
                                ),
                                event_loop,
                            )
                        except RuntimeError:
                            # Event loop may be closed if cancel arrives
                            # extremely late; ignore.
                            pass

                if self._session_registry is not None and sid is not None:
                    self._session_registry.register_cancel_hook(
                        sid, _cancel_hook
                    )

                # Spawn talker + start worker.
                talker_fn = self._build_true_stream_talker(
                    model, request, streamer
                )
                talker_thread = threading.Thread(
                    target=talker_fn,
                    name=f"TrueStreamTalker-{(sid or 'no-sid')[:8]}",
                    daemon=True,
                )
                worker.start()
                talker_thread.start()

                # Wait for the first chunk to land (P-8 streaming exception).
                # The dispatcher polls the locally-tracked first_chunk_time
                # rather than subscribing to a Qt signal — keeps the dispatch
                # path Qt-independent. Wallclock-based polling so the timeout
                # is meaningful regardless of asyncio.sleep granularity.
                first_chunk_timeout_s = 30.0
                poll_s = 0.005
                first_chunk_deadline = time.perf_counter() + first_chunk_timeout_s
                while (
                    first_chunk_time is None
                    and time.perf_counter() < first_chunk_deadline
                ):
                    if self._cancel_requested:
                        break
                    if not worker.is_alive() and not talker_thread.is_alive():
                        break
                    await asyncio.sleep(poll_s)

                # Kick playback once the first chunk is in. P-8 streaming
                # exception: session is in GENERATING when play_dual_stream
                # is called; mark_playing/mark_audible posts run from the
                # coordinator's existing flow.
                if (
                    accumulated_chunks
                    and self.audio_coordinator is not None
                    and sid is not None
                    and not self._cancel_requested
                ):
                    initial_audio = (
                        accumulated_chunks[0]
                        if len(accumulated_chunks) == 1
                        else np.concatenate(accumulated_chunks)
                    )
                    try:
                        await self.audio_coordinator.play_dual_stream(
                            audio_data=initial_audio,
                            session_id=sid,
                        )
                    except Exception as play_err:
                        # Playback dispatch failed; treat as a structural
                        # failure so the dispatcher falls back.
                        self.logger.exception(
                            f"[QwenTTS] play_dual_stream failed: {play_err}"
                        )
                        raise

                # Wait for the worker + talker to finish. Wallclock-based
                # deadline so asyncio.sleep granularity (~15ms on Windows)
                # doesn't blow past a nominal short timeout.
                join_timeout_s = 60.0
                join_poll_s = 0.01
                join_deadline = time.perf_counter() + join_timeout_s
                while (
                    (worker.is_alive() or talker_thread.is_alive())
                    and time.perf_counter() < join_deadline
                ):
                    if self._cancel_requested:
                        break
                    await asyncio.sleep(join_poll_s)

                if worker.is_alive() or talker_thread.is_alive():
                    raise RuntimeError(
                        "TRUE_STREAM talker/worker did not complete within "
                        f"{join_timeout_s}s"
                    )

            # Story 16.7 empirical finding: when the talker thread silently
            # fails (its except branch in ``_build_true_stream_talker`` swallows
            # all exceptions and just calls ``streamer.end()``), the worker
            # drains an empty queue and ``accumulated_chunks`` stays empty.
            # Without this guard, the dispatch returns ``success=True`` with
            # zero-sample audio, the fallback chain never fires, and the user
            # hears silence on the production CUDA path. Raising here lets
            # ``_dispatch_by_streaming_mode`` route to SENTENCE_STREAM per
            # NFR7's graceful-degradation contract.
            if not accumulated_chunks and not self._cancel_requested:
                raise RuntimeError(
                    "TRUE_STREAM produced 0 audio chunks — talker thread "
                    "likely raised (see prior log). Routing to fallback chain."
                )

            # Build the complete audio array from accumulated chunks.
            if accumulated_chunks:
                complete_audio = np.concatenate(accumulated_chunks)
            else:
                complete_audio = np.array([], dtype=np.float32)
            accumulated_chunks.clear()

            audio_file = self._save_audio_to_cache(complete_audio, sample_rate)

            generation_time = time.time() - start_time
            self._successful_requests += 1
            self._last_generation_time = generation_time
            self._generation_state = GenerationState.COMPLETE

            if first_chunk_time is not None:
                metrics.record(
                    "first_chunk_latency_ms",
                    first_chunk_time * 1000.0,
                    session_id=sid,
                    model_type=(
                        request.model_type.display_name
                        if request.model_type is not None
                        else "default"
                    ),
                    hardware=hardware_label,
                )

            self.logger.info(
                f"TTS generation complete (TRUE_STREAM): "
                f"{len(complete_audio)} samples, "
                f"{chunk_count_box[0]} chunks, "
                f"{generation_time:.2f}s total, "
                f"{first_chunk_time:.2f}s first chunk"
                if first_chunk_time is not None
                else (
                    f"TTS generation complete (TRUE_STREAM): "
                    f"{len(complete_audio)} samples, "
                    f"{chunk_count_box[0]} chunks, "
                    f"{generation_time:.2f}s total"
                )
            )

            if self._generation_complete_callback and audio_file:
                self._generation_complete_callback(audio_file)

            return QwenTTSResponse(
                success=True,
                audio_data=complete_audio,
                sample_rate=sample_rate,
                audio_file_path=audio_file,
                generation_time_seconds=generation_time,
                mode=GenerationMode.STREAMING,
                chunks_generated=chunk_count_box[0],
                first_chunk_latency=first_chunk_time,
            )

        except asyncio.CancelledError:
            self.logger.info("TRUE_STREAM generation cancelled")
            self._generation_state = GenerationState.CANCELLED
            # P-7 invariant: do NOT post ('cancel', sid) here. The worker's
            # drain-on-cancel posts the canonical CANCELLED transition.
            if streamer is not None:
                streamer._cancel_event.set()
                try:
                    streamer.end()
                except Exception:  # pragma: no cover (defensive)
                    pass
            if worker is not None and worker.is_alive():
                worker.join(timeout=2.0)
            if talker_thread is not None and talker_thread.is_alive():
                talker_thread.join(timeout=2.0)
            if self._generation_cancelled_callback:
                self._generation_cancelled_callback()
            return QwenTTSResponse(
                success=False,
                error_message="Generation was cancelled",
                mode=GenerationMode.STREAMING,
                chunks_generated=chunk_count_box[0],
            )

        except Exception:
            self.logger.exception("[QwenTTS] TRUE_STREAM dispatch failed")
            if streamer is not None:
                streamer._cancel_event.set()
                try:
                    streamer.end()
                except Exception:  # pragma: no cover
                    pass
            if worker is not None and worker.is_alive():
                worker.join(timeout=2.0)
            if talker_thread is not None and talker_thread.is_alive():
                talker_thread.join(timeout=2.0)
            if sid is not None and self._session_registry is not None:
                # try_set_error rather than set_error: setting the
                # streamer cancel_event above causes the worker's drain-
                # on-cancel logic to post ('cancel', sid) before this
                # error-cleanup mutation reaches the Qt event loop. The
                # session is already CANCELLED by the time set_error
                # fires, which strict set_error would surface as an
                # "Unexpected Error" dialog via the global exception
                # handler — even though the dispatcher's fallback chain
                # has already routed to SENTENCE_STREAM and the user
                # hears audio. try_set_error absorbs the race.
                self._session_registry.post_mutation('try_set_error', sid)
                self._session_registry.post_mutation('discard', sid)
            # Surface to the dispatcher's fallback chain. NOT
            # asyncio.CancelledError (handled above).
            raise

        finally:
            self._current_generation_task = None
            self._current_session_id = None

    @staticmethod
    def _fallback_chain_from(mode: StreamingMode) -> List[StreamingMode]:
        """Story 16.6 — order of modes to attempt starting from ``mode``.

        TRUE_STREAM caps the chain at all three modes; SENTENCE_STREAM caps
        at two; BATCH has no further fallback. The chain is the
        single-source-of-truth for FR3 + NFR7 graceful-degradation order;
        do NOT reimplement this fork in callers.
        """
        if mode == StreamingMode.TRUE_STREAM:
            return [
                StreamingMode.TRUE_STREAM,
                StreamingMode.SENTENCE_STREAM,
                StreamingMode.BATCH,
            ]
        if mode == StreamingMode.SENTENCE_STREAM:
            return [StreamingMode.SENTENCE_STREAM, StreamingMode.BATCH]
        return [StreamingMode.BATCH]

    async def _dispatch_by_streaming_mode(
        self,
        request: QwenTTSRequest,
        mode: StreamingMode,
    ) -> QwenTTSResponse:
        """Story 16.6 Task 2 — three-mode fallback dispatch.

        Forks on the resolved mode, calls one of three private generators
        (_generate_true_stream / _generate_streaming / _generate), catches
        every non-CancelledError exception, emits a
        ``streaming_mode_fallback`` metric, and recurses into the next-lower
        mode in the chain. On all-three-modes-failed, returns a synthetic
        ``QwenTTSResponse(success=False, used_fallback=True,
        mode=BATCH, error_message=<all three>)``.

        Honors the legacy ``request.streaming=False`` override per AC #8 —
        a caller that explicitly disabled streaming gets the BATCH path
        regardless of the resolver's pick.

        Per-dispatch-entry ``streaming_mode`` metric (D-19 / P-9) fires once
        per attempt — including each fallback attempt — so Story 16.7's
        empirical-validation harness can correlate fallback rates with
        hardware/model-type tags.
        """
        # AC #8 second clause: legacy override forces BATCH.
        if request.streaming is False:
            mode = StreamingMode.BATCH

        chain = self._fallback_chain_from(mode)
        # Defensive ``getattr`` against ``QwenTTSService.__new__`` callers
        # — partial-init test fixtures bypass ``__init__`` and may not set
        # all instance attributes.
        model_registry = getattr(self, "_model_registry", None)
        hardware = (
            "gpu"
            if model_registry is not None
            and "cuda" in str(getattr(model_registry, "device", "")).lower()
            else "cpu"
        )
        model_type = (
            request.model_type.display_name
            if request.model_type is not None
            else "default"
        )

        failures: List[Tuple[StreamingMode, BaseException]] = []
        # Snapshot ``_failed_requests`` before the chain so the dispatcher's
        # terminal-failure branch only increments if no inner method already
        # did (Story 16.6 review M3 — AC #2: "incremented exactly once").
        failed_requests_snapshot = getattr(self, "_failed_requests", 0)

        for i, current_mode in enumerate(chain):
            # Per-dispatch-entry metric (D-19 / P-9). session_id is the
            # currently-active session if a generator has set it; for the
            # initial call before any generator runs it's None.
            metrics.record(
                "streaming_mode",
                current_mode.value,
                session_id=getattr(self, "_current_session_id", None),
                model_type=model_type,
                hardware=hardware,
            )
            try:
                if current_mode == StreamingMode.TRUE_STREAM:
                    response = await self._generate_true_stream(request)
                elif current_mode == StreamingMode.SENTENCE_STREAM:
                    response = await self._generate_streaming(request)
                else:  # BATCH
                    response = await self._generate(request)
                return response
            except asyncio.CancelledError:
                # User cancel — not a fallback trigger. Propagate.
                raise
            except Exception as exc:
                failures.append((current_mode, exc))
                reason = repr(exc)
                # AC #6 last clause — truncate to 200 chars with an explicit
                # ellipsis suffix so downstream telemetry can distinguish a
                # 200-char message from a truncated one (Story 16.6 review
                # M2). Total bounded length = 199 + len('…') = 200 chars.
                if len(reason) > 200:
                    reason = reason[:199] + "…"
                if i + 1 < len(chain):
                    next_mode = chain[i + 1]
                    # ``getattr`` defensive against partial-init instances.
                    self._fallback_count = (
                        getattr(self, "_fallback_count", 0) + 1
                    )
                    metrics.record(
                        "streaming_mode_fallback",
                        next_mode.value,
                        session_id=getattr(self, "_current_session_id", None),
                        from_mode=current_mode.value,
                        reason=reason,
                        model_type=model_type,
                        hardware=hardware,
                    )
                else:
                    # Terminal failure — emit "unrecoverable" but do NOT
                    # increment _fallback_count (no successful transition).
                    metrics.record(
                        "streaming_mode_fallback",
                        "unrecoverable",
                        session_id=getattr(self, "_current_session_id", None),
                        from_mode=current_mode.value,
                        reason=reason,
                        model_type=model_type,
                        hardware=hardware,
                    )

        # All modes failed — synthesize unrecoverable response. Only count
        # this dispatch as a failure if no inner method already incremented
        # the counter (Story 16.6 review M3 — AC #2 "exactly once"). When an
        # inner method raises after its own ``_failed_requests += 1`` (real
        # production paths), the snapshot diff is non-zero and we skip.
        if getattr(self, "_failed_requests", 0) == failed_requests_snapshot:
            self._failed_requests = failed_requests_snapshot + 1
        error_lines = "; ".join(f"{m.value}: {e}" for m, e in failures)
        return QwenTTSResponse(
            success=False,
            error_message=f"All streaming modes failed: {error_lines}",
            # AC #2: response's mode reflects the LAST mode attempted.
            mode=GenerationMode.BATCH,
            used_fallback=True,
        )

    def _resolve_streaming_mode(self) -> StreamingMode:
        """Story 16.6 Task 1 — resolve the streaming mode for the next dispatch.

        Pure function of ``self._app_settings.streaming_mode_override`` and
        ``torch.cuda.is_available()``. No side effects, no metric emission, no
        registry mutation — Story 16.7's empirical-validation harness calls
        this repeatedly without polluting metrics. The per-mode
        ``streaming_mode`` metric (D-19 / P-9) fires from
        ``_dispatch_by_streaming_mode`` after this resolver returns.

        Reads the optional string field, converts it via
        ``StreamingMode(value)`` (raising ``ValueError`` on bad data so any
        ``AppSettings.from_dict`` regression surfaces loudly), and delegates
        to Story 16.2's ``effective_streaming_mode`` so this method remains a
        thin shim — the two surfaces stay aligned.

        Returns the resolved mode. Hardware probe (CUDA -> TRUE_STREAM, else
        SENTENCE_STREAM, NFR12 protection) only runs when the override is
        ``None``.
        """
        # ``getattr`` defensive against ``QwenTTSService.__new__`` callers
        # (existing test fixtures construct partial instances that bypass
        # ``__init__``); when neither AppSettings nor the attribute exist,
        # the resolver behaves as if override is None (Auto).
        app_settings = getattr(self, "_app_settings", None)
        override_str: Optional[str] = (
            app_settings.streaming_mode_override
            if app_settings is not None
            else None
        )
        override: Optional[StreamingMode] = (
            StreamingMode(override_str) if override_str is not None else None
        )
        return effective_streaming_mode(override)

    async def cancel_generation(self) -> bool:
        """
        Cancel the current generation.

        The text input is retained after cancellation - only the generation
        is aborted. The UI should return to ready state.

        Returns:
            bool: True if cancellation was initiated
        """
        if self._generation_state in (GenerationState.GENERATING, GenerationState.STREAMING):
            self.logger.info("Cancellation requested")
            self._cancel_requested = True
            self._generation_state = GenerationState.CANCELLED

            # Story 11.4 review fix (F1): propagate the cancel into the
            # running asyncio task so the batch path actually receives
            # CancelledError. Without this, _generate's
            # `await loop.run_in_executor(...)` runs to completion, the
            # session reaches READY_TO_PLAY in the registry, and the
            # legacy/registry states diverge. Streaming has the
            # _cancel_requested poll as a separate bail-out path; both
            # converge on the existing except-CancelledError handlers
            # which post the registry cancel/discard mutations.
            task = self._current_generation_task
            if task is not None and not task.done():
                task.cancel()

            # Story 16.5: trigger the cooperative cancel chain. Quiet no-op
            # if no registry is wired (legacy mode) or no session is in
            # flight or no hook was registered (today's batch + sentence-
            # stream sessions have no streamer event to flip; Story 16.6's
            # TRUE_STREAM dispatch path is what registers a hook). Under
            # that path, this call flips the streamer's _cancel_event AND
            # asks the audio coordinator to stop playback for the session.
            if self._session_registry is not None and self._current_session_id is not None:
                self._session_registry.request_cancel(self._current_session_id)

            # Notify cancellation callback
            if self._generation_cancelled_callback:
                self._generation_cancelled_callback()

            return True
        return False

    def validate_text(self, text: str) -> TextValidationResult:
        """
        Validate text input before generation.

        Call this before triggering generation to check for:
        - Empty text (blocks generation)
        - Whitespace-only text (blocks generation)
        - Very long text (warns but allows generation)

        Args:
            text: Text to validate

        Returns:
            TextValidationResult with validation status and messages
        """
        char_count = len(text) if text else 0

        # Check for None or empty
        if not text:
            result = TextValidationResult(
                is_valid=False,
                status=TextValidationStatus.EMPTY,
                message="Enter text to speak",
                can_proceed=False,
                character_count=0,
            )
            if self._text_validation_callback:
                self._text_validation_callback(result)
            return result

        # Check for whitespace-only
        stripped = text.strip()
        if not stripped:
            result = TextValidationResult(
                is_valid=False,
                status=TextValidationStatus.WHITESPACE_ONLY,
                message="Enter text to speak",
                can_proceed=False,
                character_count=char_count,
            )
            if self._text_validation_callback:
                self._text_validation_callback(result)
            return result

        # Check for very long text (warning, not error)
        if char_count > self.MAX_TEXT_LENGTH_HARD:
            result = TextValidationResult(
                is_valid=False,
                status=TextValidationStatus.TOO_LONG,
                message=f"Text is too long ({char_count:,} characters). Maximum is {self.MAX_TEXT_LENGTH_HARD:,}.",
                can_proceed=False,
                character_count=char_count,
            )
            if self._text_validation_callback:
                self._text_validation_callback(result)
            return result

        if char_count > self.MAX_TEXT_LENGTH_WARNING:
            result = TextValidationResult(
                is_valid=True,
                status=TextValidationStatus.TOO_LONG,
                message=None,
                can_proceed=True,
                warning=f"Text is very long ({char_count:,} characters). Consider splitting into smaller messages.",
                character_count=char_count,
            )
            if self._text_validation_callback:
                self._text_validation_callback(result)
            return result

        # Valid text
        result = TextValidationResult(
            is_valid=True,
            status=TextValidationStatus.VALID,
            message=None,
            can_proceed=True,
            warning=None,
            character_count=char_count,
        )
        if self._text_validation_callback:
            self._text_validation_callback(result)
        return result

    def get_generation_state(self) -> GenerationState:
        """Get the current generation state."""
        return self._generation_state

    def is_generating(self) -> bool:
        """Check if generation is currently in progress."""
        return self._generation_state in (
            GenerationState.LOADING_MODEL,
            GenerationState.GENERATING,
            GenerationState.STREAMING
        )

    def _generate_sync(self, request: QwenTTSRequest) -> Tuple[np.ndarray, int]:
        """
        Synchronous generation method (runs in thread pool).

        Args:
            request: Qwen TTS request

        Returns:
            Tuple[np.ndarray, int]: (audio_data, sample_rate)
        """
        model = self._model_registry.get_loaded_model()
        if model is None:
            raise RuntimeError("No model loaded")

        # Verify the loaded model matches the requested type
        current_model_type = self._model_registry.current_model_type
        if current_model_type != request.model_type:
            self.logger.error(
                f"Model type mismatch! Requested: {request.model_type.display_name}, "
                f"Loaded: {current_model_type.display_name if current_model_type else 'None'}"
            )
            raise RuntimeError(
                f"Model type mismatch: requested {request.model_type.display_name} "
                f"but {current_model_type.display_name if current_model_type else 'None'} is loaded"
            )

        # Generate based on model type
        if request.model_type == QwenModelType.CUSTOM_VOICE:
            self.logger.debug(f"Generating with CUSTOM_VOICE: speaker={request.speaker}")

            # Check if current tier supports instruct parameter
            # 0.6B models don't support emotion/style instructions
            current_tier = self._model_registry.quality_tier
            effective_instruct = request.instruct
            if not request.model_type.supports_instruct_in_tier(current_tier):
                if request.instruct:
                    self.logger.info(
                        f"Ignoring instruct parameter '{request.instruct[:30]}...' - "
                        f"not supported in {current_tier.display_name} tier"
                    )
                effective_instruct = None

            wavs, sr = model.generate_custom_voice(
                text=request.text,
                language=request.language,
                speaker=request.speaker,
                instruct=effective_instruct,
            )
        elif request.model_type == QwenModelType.VOICE_DESIGN:
            self.logger.debug(f"Generating with VOICE_DESIGN: description={request.voice_description[:50] if request.voice_description else 'None'}...")
            wavs, sr = model.generate_voice_design(
                text=request.text,
                language=request.language,
                instruct=request.voice_description,
            )
        elif request.model_type == QwenModelType.BASE:
            # QA5: Check if we have a pre-computed voice clone prompt (embedding)
            if request.voice_clone_prompt is not None:
                self.logger.info(f"[DEBUG] BASE model with voice_clone_prompt: type={type(request.voice_clone_prompt)}, len={len(request.voice_clone_prompt) if hasattr(request.voice_clone_prompt, '__len__') else 'N/A'}")
                self.logger.debug("Generating with BASE (embedding): using pre-computed voice_clone_prompt")
                wavs, sr = model.generate_voice_clone(
                    text=request.text,
                    language=request.language,
                    voice_clone_prompt=request.voice_clone_prompt,
                )
            else:
                # Traditional voice cloning from reference audio
                # Validate ref_audio for voice cloning
                if not request.ref_audio:
                    raise ValueError("Voice cloning requires ref_audio path or voice_clone_prompt")
                ref_audio_path = Path(request.ref_audio) if isinstance(request.ref_audio, str) else request.ref_audio
                if not ref_audio_path.exists():
                    raise FileNotFoundError(f"Reference audio file not found: {ref_audio_path}")

                # Determine cloning mode: ICL (with transcript) or x-vector (voice timbre only)
                use_xvector = request.x_vector_only_mode
                ref_text = request.ref_text or ""

                # Auto-enable x_vector mode if no ref_text provided and not explicitly set
                if not ref_text and not use_xvector:
                    self.logger.warning("No ref_text provided, automatically enabling x_vector_only_mode")
                    use_xvector = True

                mode_name = "x-vector" if use_xvector else "ICL"
                self.logger.debug(f"Generating with BASE (clone): ref_audio={request.ref_audio}, mode={mode_name}")

                wavs, sr = model.generate_voice_clone(
                    text=request.text,
                    language=request.language,
                    ref_audio=str(request.ref_audio),
                    ref_text=ref_text,
                    x_vector_only_mode=use_xvector,
                )
        else:
            raise ValueError(f"Unknown model type: {request.model_type}")

        # wavs is a list, get the first (only) result
        audio_data = wavs[0] if isinstance(wavs, list) else wavs

        return audio_data, sr

    def _save_audio_to_cache(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Optional[Path]:
        """
        Save audio data to cache file.

        Args:
            audio_data: Audio numpy array
            sample_rate: Sample rate

        Returns:
            Path to saved file or None on error
        """
        try:
            sf.write(str(self._current_audio_cache), audio_data, sample_rate)
            self.logger.debug(f"Audio cached to: {self._current_audio_cache}")
            return self._current_audio_cache
        except Exception as e:
            self.logger.error(f"Failed to cache audio: {e}")
            return None

    def _get_user_friendly_error(self, error: Exception, used_fallback: bool = False) -> TTSError:
        """
        Convert exception to structured user-friendly error.

        Args:
            error: The exception that occurred
            used_fallback: Whether batch fallback was attempted

        Returns:
            TTSError with user message and recovery suggestion
        """
        error_str = str(error).lower()
        error_type = type(error).__name__

        # Log technical details
        self.logger.error(f"TTS Error [{error_type}]: {error}")

        # Out of memory errors
        if "memory" in error_str or "oom" in error_str or "out of memory" in error_str:
            return TTSError(
                code=TTSErrorCode.OUT_OF_MEMORY,
                user_message="Not enough memory to generate speech.",
                recovery_suggestion="Try closing other applications and try again.",
                technical_details=str(error),
                is_recoverable=True,
                used_fallback=used_fallback,
            )

        # CUDA/GPU errors
        if "cuda" in error_str or "gpu" in error_str or "device" in error_str:
            return TTSError(
                code=TTSErrorCode.CUDA_ERROR,
                user_message="GPU error occurred.",
                recovery_suggestion="The application will try using CPU instead. Please try again.",
                technical_details=str(error),
                is_recoverable=True,
                used_fallback=used_fallback,
            )

        # Model not found
        if ("model" in error_str and "not found" in error_str) or "no such file" in error_str:
            return TTSError(
                code=TTSErrorCode.MODEL_NOT_FOUND,
                user_message="Voice model not found.",
                recovery_suggestion="Please reinstall the application or check your installation.",
                technical_details=str(error),
                is_recoverable=False,
                used_fallback=used_fallback,
            )

        # Model loading failures
        if "load" in error_str and ("fail" in error_str or "error" in error_str):
            return TTSError(
                code=TTSErrorCode.MODEL_LOAD_FAILED,
                user_message="Failed to load the voice model.",
                recovery_suggestion="Try restarting the application. If the problem persists, reinstall.",
                technical_details=str(error),
                is_recoverable=True,
                used_fallback=used_fallback,
            )

        # Timeout errors
        if "timeout" in error_str or "timed out" in error_str:
            return TTSError(
                code=TTSErrorCode.TIMEOUT,
                user_message="Speech generation took too long.",
                recovery_suggestion="Try with shorter text or try again.",
                technical_details=str(error),
                is_recoverable=True,
                used_fallback=used_fallback,
            )

        # Audio file errors (for voice cloning)
        if "audio" in error_str and ("invalid" in error_str or "corrupt" in error_str or "format" in error_str):
            return TTSError(
                code=TTSErrorCode.INVALID_AUDIO_FILE,
                user_message="The audio file could not be processed.",
                recovery_suggestion="Please use a valid WAV, MP3, or M4A file with clear speech.",
                technical_details=str(error),
                is_recoverable=True,
                used_fallback=used_fallback,
            )

        # Connection/network errors (shouldn't happen for local, but just in case)
        if "connection" in error_str or "network" in error_str:
            return TTSError(
                code=TTSErrorCode.UNKNOWN,
                user_message="A connection error occurred.",
                recovery_suggestion="Please check your system and try again.",
                technical_details=str(error),
                is_recoverable=True,
                used_fallback=used_fallback,
            )

        # Default: unknown error - include actual error for better debugging
        error_detail = str(error)
        if len(error_detail) > 100:
            error_detail = error_detail[:100] + "..."

        return TTSError(
            code=TTSErrorCode.UNKNOWN,
            user_message=f"Speech generation failed: {error_detail}",
            recovery_suggestion="Check logs for details.",
            technical_details=str(error),
            is_recoverable=True,
            used_fallback=used_fallback,
        )

    def _handle_generation_error(self, error: Exception, used_fallback: bool = False) -> TTSError:
        """
        Handle a generation error - create structured error and notify callbacks.

        Args:
            error: The exception that occurred
            used_fallback: Whether batch fallback was attempted

        Returns:
            TTSError object
        """
        tts_error = self._get_user_friendly_error(error, used_fallback)
        self._last_error = tts_error

        # Notify error callback with full error object
        if self._generation_error_callback:
            self._generation_error_callback(tts_error)

        # Notify simple failure callback with just the message
        if self._generation_failed_callback:
            self._generation_failed_callback(str(tts_error))

        return tts_error

    def _on_model_progress(self, progress: ModelLoadProgress):
        """Handle model loading progress updates."""
        self.logger.debug(
            f"Model progress: {progress.model_type.display_name} "
            f"[{progress.state.value}] {progress.progress_percent:.0f}% - {progress.message}"
        )

        # Store state for replay when UI connects (Fix: startup indicator timing)
        if progress.state == ModelState.LOADING:
            self._is_model_loading = True
            self._last_model_loading_message = progress.message
            self._last_model_ready_name = None
            if self._model_loading_callback:
                self._model_loading_callback(progress.message)
        elif progress.state == ModelState.READY:
            self._is_model_loading = False
            self._last_model_loading_message = None
            self._last_model_ready_name = progress.model_type.display_name
            if self._model_ready_callback:
                self._model_ready_callback(progress.model_type.display_name)
        elif progress.state == ModelState.ERROR:
            self._is_model_loading = False
            self._last_model_loading_message = None

    # Callback setters

    def set_generation_started_callback(self, callback: Callable[[], None]):
        """Set callback for generation start (for visual indicator)."""
        self._generation_started_callback = callback

    def set_generation_complete_callback(self, callback: Callable[[Path], None]):
        """Set callback for generation completion (emits audio file path)."""
        self._generation_complete_callback = callback

    def set_generation_failed_callback(self, callback: Callable[[str], None]):
        """Set callback for generation failure (emits error message string)."""
        self._generation_failed_callback = callback

    def set_generation_error_callback(self, callback: Callable[[TTSError], None]):
        """
        Set callback for detailed generation errors.

        The callback receives a TTSError object with:
        - code: Error category (TTSErrorCode enum)
        - user_message: What happened
        - recovery_suggestion: What to do
        - is_recoverable: Whether retry is possible
        - used_fallback: Whether batch fallback was attempted
        """
        self._generation_error_callback = callback

    def set_generation_cancelled_callback(self, callback: Callable[[], None]):
        """
        Set callback for when generation is cancelled.

        Called when user cancels via cancel_generation() or Escape key.
        The text input should be retained - only the generation is aborted.
        """
        self._generation_cancelled_callback = callback

    def set_text_validation_callback(self, callback: Callable[[TextValidationResult], None]):
        """
        Set callback for text validation results.

        Called during validate_text() with validation status and any warnings.
        UI can use this to show inline validation messages.
        """
        self._text_validation_callback = callback

    def set_audio_chunk_ready_callback(self, callback: Callable[[AudioChunk], None]):
        """
        Set callback for audio chunk ready (streaming mode).

        The callback receives an AudioChunk for each generated chunk,
        enabling immediate playback while subsequent chunks generate.
        """
        self._audio_chunk_ready_callback = callback

    def set_model_loading_callback(self, callback: Callable[[str], None]):
        """
        Set callback for model loading start (emits message).

        If model is currently loading (e.g., during startup before UI connected),
        the callback is immediately invoked with the current loading message.
        """
        self._model_loading_callback = callback
        # Replay current state if model is loading (Fix: startup indicator timing)
        if self._is_model_loading and self._last_model_loading_message and callback:
            self.logger.debug(f"Replaying model loading state to newly connected callback: {self._last_model_loading_message}")
            callback(self._last_model_loading_message)

    def set_model_ready_callback(self, callback: Callable[[str], None]):
        """
        Set callback for model ready (emits model name).

        If model was loaded before UI connected (e.g., during startup),
        the callback is immediately invoked with the loaded model name.
        """
        self._model_ready_callback = callback
        # Replay ready state if model was loaded during startup (Fix: startup indicator timing)
        if not self._is_model_loading and self._last_model_ready_name and callback:
            self.logger.debug(f"Replaying model ready state to newly connected callback: {self._last_model_ready_name}")
            callback(self._last_model_ready_name)

    def set_health_status_callback(
        self,
        callback: Callable[[ServiceHealthStatus, Optional[str]], None]
    ):
        """Set callback for health status changes."""
        self._health_status_callback = callback

    def set_startup_progress_callback(
        self,
        callback: Callable[[StartupProgress], None]
    ):
        """
        Set callback for startup progress updates.

        Called during initialize_with_defaults() with StartupProgress containing:
        - state: Current startup state (INITIALIZING, LOADING_MODEL, READY, FAILED)
        - progress_percent: 0-100 progress
        - message: Human-readable status message
        - is_complete: Whether startup is complete
        - is_ready: Whether TTS is ready for generation
        """
        self._startup_progress_callback = callback

    def set_tts_ready_callback(self, callback: Callable[[], None]):
        """
        Set callback for TTS ready notification.

        Called once when TTS has finished initialization and is ready
        for speech generation. UI should show green "TTS Ready" indicator.
        """
        self._tts_ready_callback = callback

    # Utility methods

    def get_supported_speakers(self) -> List[str]:
        """Get list of supported speakers for CustomVoice model."""
        return ModelRegistry.CUSTOM_VOICE_SPEAKERS.copy()

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return ModelRegistry.SUPPORTED_LANGUAGES.copy()

    def get_emotion_presets(self) -> Dict[str, Optional[str]]:
        """Get available emotion presets."""
        return self.EMOTION_PRESETS.copy()

    def is_model_loaded(self, model_type: QwenModelType) -> bool:
        """Check if a specific model is loaded."""
        return self._model_registry.is_model_ready(model_type)

    def get_current_model_type(self) -> Optional[QwenModelType]:
        """Get the currently loaded model type."""
        return self._model_registry.current_model_type

    async def set_quality_tier(self, tier: str) -> bool:
        """
        Set the model quality tier dynamically.

        Unloads the current model if loaded, so the next generation
        request will load the correct tier's model (lazy loading).

        Args:
            tier: Quality tier ("small" or "quality")

        Returns:
            bool: True if tier was changed, False if already set
        """
        changed = await self._model_registry.set_quality_tier(tier)
        if changed:
            self.logger.info(f"TTS quality tier set to '{tier}' - will take effect on next generation")
        return changed

    def get_quality_tier(self) -> str:
        """Get the current quality tier."""
        return self._model_registry.quality_tier.value

    def get_cached_audio_path(self) -> Optional[Path]:
        """Get path to the current cached audio file."""
        if self._current_audio_cache.exists():
            return self._current_audio_cache
        return None

    def get_service_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics."""
        # Local var name avoids shadowing the imported ``metrics`` module
        # (line 237) — Story 11.4 may add ``metrics.record(...)`` calls in
        # this method body.
        status = self.get_status_info()
        status.update({
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "streaming_requests": self._streaming_requests,
            "fallback_count": self._fallback_count,
            "success_rate": (
                self._successful_requests / self._total_requests * 100
                if self._total_requests > 0 else 0
            ),
            "last_generation_time": self._last_generation_time,
            "avg_first_chunk_latency": self._avg_first_chunk_latency,
            "generation_state": self._generation_state.value,
            "last_error": self._last_error.to_dict() if self._last_error else None,
            "registry_status": self._model_registry.get_registry_status(),
            # Story 11.4 AC #17: operational visibility into the in-flight
            # session set. Reaches into ``_sessions`` deliberately — adding
            # ``__len__`` to SessionRegistry is acceptable but out of scope
            # for this pass.
            "session_registry_in_flight": (
                len(self._session_registry._sessions)
                if self._session_registry is not None
                else 0
            ),
        })
        return status

    def get_last_error(self) -> Optional[TTSError]:
        """Get the last error that occurred."""
        return self._last_error

    async def preload_model(
        self,
        model_type: QwenModelType = QwenModelType.CUSTOM_VOICE
    ) -> Tuple[bool, Optional[str]]:
        """
        Preload a model (for startup optimization).

        Args:
            model_type: Model type to preload

        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        return await self._model_registry.ensure_model_loaded(model_type)
