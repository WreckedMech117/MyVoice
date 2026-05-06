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
import logging
import re
import tempfile
import threading
import time
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
    """
    audio_data: np.ndarray
    sample_rate: int
    chunk_index: int
    is_final: bool = False
    text_segment: str = ""


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

        # Model registry for lazy loading
        self._model_registry = ModelRegistry(
            device=device,
            dtype=dtype,
            models_path=models_path,
            progress_callback=self._on_model_progress,
            quality_tier=quality_tier
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

        # Voice clone prompt cache (for reusing voice clone prompts)
        self._voice_clone_prompts: Dict[str, Any] = {}

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

        if streaming:
            return await self._generate_streaming(request)
        return await self._generate(request)

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

        if streaming:
            return await self._generate_streaming(request)
        return await self._generate(request)

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

        if streaming:
            return await self._generate_streaming(request)
        return await self._generate(request)

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

        if streaming:
            return await self._generate_streaming(request)
        return await self._generate(request)

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
            if streaming:
                response = await self._generate_streaming(request)
            else:
                response = await self._generate(request)

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
