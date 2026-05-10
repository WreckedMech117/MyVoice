"""
Model Registry for Qwen3-TTS Models

This module implements lazy loading and lifecycle management for Qwen3-TTS models.
Only one model is loaded at a time to respect memory constraints (~3.4GB per model).

State Machine:
    UNLOADED -> LOADING -> READY -> UNLOADING -> UNLOADED
"""

import asyncio
import logging
import gc
import os
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

import torch

# HuggingFace Hub for cache detection
try:
    from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    _CACHED_NO_EXIST = None

from myvoice.models.service_enums import ModelState, QwenModelType
from myvoice.observability import metrics

# Type hint for Qwen3TTSModel - actual import happens at runtime
try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    Qwen3TTSModel = None


@dataclass
class ModelInfo:
    """Information about a registered model."""
    model_type: QwenModelType
    state: ModelState = ModelState.UNLOADED
    model_instance: Optional[Any] = None
    load_error: Optional[str] = None
    load_time_seconds: Optional[float] = None
    checkpoint_path: Optional[str] = None  # Custom checkpoint path for fine-tuned models


@dataclass
class ModelLoadProgress:
    """Progress information during model loading."""
    model_type: QwenModelType
    state: ModelState
    progress_percent: float = 0.0
    message: str = ""


class ModelRegistry:
    """
    Registry for managing Qwen3-TTS model lifecycle.

    Implements lazy loading with only one model loaded at a time.
    Provides state machine management and PyQt6 signal integration.

    Attributes:
        current_model_type: The currently loaded model type (or None)
        device: PyTorch device for model loading ("cuda:0" or "cpu")
        dtype: PyTorch dtype for model weights
    """

    # Built-in speakers for CustomVoice model
    CUSTOM_VOICE_SPEAKERS = [
        "Vivian",    # Bright, slightly edgy young female (Chinese native)
        "Serena",    # Warm, gentle young female (Chinese native)
        "Uncle_Fu",  # Seasoned male, low/mellow (Chinese native)
        "Dylan",     # Youthful Beijing male (Chinese/Beijing dialect)
        "Eric",      # Lively Chengdu male (Chinese/Sichuan dialect)
        "Ryan",      # Dynamic male, strong rhythm (English native)
        "Aiden",     # Sunny American male (English native)
        "Ono_Anna",  # Playful Japanese female (Japanese native)
        "Sohee",     # Warm Korean female (Korean native)
    ]

    # Supported languages
    SUPPORTED_LANGUAGES = [
        "Chinese", "English", "Japanese", "Korean",
        "German", "French", "Russian", "Portuguese",
        "Spanish", "Italian", "Auto"
    ]

    def __init__(
        self,
        device: str = "auto",
        dtype: str = "bfloat16",
        models_path: Optional[str] = None,
        progress_callback: Optional[Callable[[ModelLoadProgress], None]] = None,
        quality_tier: str = "quality",
        app_settings: Optional[Any] = None,
    ):
        """
        Initialize the ModelRegistry.

        Args:
            device: PyTorch device ("auto", "cuda:0", "cpu")
            dtype: PyTorch dtype ("bfloat16", "float16", "float32") —
                preserved as the legacy call surface. When ``app_settings``
                is None (or its ``tts_precision`` is None), this string
                parameter is the dtype source.
            models_path: Optional local path for model weights
            progress_callback: Callback for loading progress updates
            quality_tier: Model quality tier ("small" or "quality")
            app_settings: Optional AppSettings carrying user preferences.
                Story 18.3 reads ``tts_precision`` from this object and
                routes through ``resolve_tts_precision`` which honors the
                Ampere+ probe gate (D-9 / NFR12). Precedence: when
                ``app_settings.tts_precision`` is set, the resolver wins;
                otherwise the ``dtype`` string parameter is used (legacy
                / test compatibility).
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Device configuration
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Story 18.3 — dtype precedence resolver. When AppSettings carries an
        # explicit tts_precision, the resolver wins (and honors the Ampere+
        # probe gate via "auto"); otherwise we fall back to the legacy
        # dtype-string parameter mapping. The four `precision_source` labels
        # surface the chosen path verbatim in the INFO log + telemetry tag
        # so Commander can confirm at runtime which branch engaged.
        precision_source: str
        if app_settings is not None and getattr(app_settings, "tts_precision", None) is not None:
            from myvoice.services.tts_streaming import resolve_tts_precision
            override = app_settings.tts_precision
            self.dtype = resolve_tts_precision(override)
            if override in ("bf16", "fp32"):
                precision_source = "app_settings_override"
            elif self.dtype == torch.bfloat16:
                precision_source = "app_settings_auto_ampere"
            else:
                precision_source = "app_settings_auto_fallback"
        else:
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            self.dtype = dtype_map.get(dtype, torch.bfloat16)
            precision_source = "legacy_constructor_arg"

        self.models_path = models_path
        self._progress_callback = progress_callback

        # Quality tier configuration (Small 0.6B vs Quality 1.7B)
        from myvoice.models.service_enums import ModelQualityTier
        self._quality_tier = ModelQualityTier.SMALL if quality_tier == "small" else ModelQualityTier.QUALITY

        # Registry state
        self._models: Dict[QwenModelType, ModelInfo] = {}
        self._current_model_type: Optional[QwenModelType] = None
        self._current_checkpoint_path: Optional[str] = None  # Track loaded checkpoint for optimized voices
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ModelLoader")
        self._lock = asyncio.Lock()

        # Initialize model registry entries
        for model_type in QwenModelType:
            self._models[model_type] = ModelInfo(model_type=model_type)

        # Story 18.3 — precision-engagement telemetry. Mirrors the Story 18.2
        # tag schema verbatim: device_capability is the string form of the
        # CUDA compute capability with the "none" sentinel for CPU per
        # Story 18.2 OQ #2. Single record at startup; not a per-chunk metric.
        cap_str: str
        if torch.cuda.is_available():
            try:
                cap = torch.cuda.get_device_capability()
                cap_str = f"{cap[0]}.{cap[1]}"
            except Exception:
                cap_str = "none"
        else:
            cap_str = "none"
        dtype_str = "bfloat16" if self.dtype == torch.bfloat16 else (
            "float16" if self.dtype == torch.float16 else "float32"
        )
        metrics.record(
            "tts_precision_resolved",
            1.0 if self.dtype == torch.bfloat16 else 0.0,
            source=precision_source,
            dtype=dtype_str,
            device_capability=cap_str,
        )

        self.logger.info(
            f"ModelRegistry initialized: device={self.device}, dtype={self.dtype}, "
            f"precision_source='{precision_source}', quality_tier={self._quality_tier.value}"
        )

    @property
    def current_model_type(self) -> Optional[QwenModelType]:
        """Get the currently loaded model type."""
        return self._current_model_type

    @property
    def quality_tier(self) -> "ModelQualityTier":
        """Get the current quality tier."""
        return self._quality_tier

    async def set_quality_tier(self, tier: str) -> bool:
        """
        Set the model quality tier dynamically.

        If a model is currently loaded, it will be unloaded so the next
        generation request loads the correct tier's model.

        Args:
            tier: Quality tier ("small" or "quality")

        Returns:
            bool: True if tier was changed, False if already set to this tier
        """
        from myvoice.models.service_enums import ModelQualityTier
        new_tier = ModelQualityTier.SMALL if tier == "small" else ModelQualityTier.QUALITY

        if self._quality_tier == new_tier:
            self.logger.debug(f"Quality tier already set to {tier}")
            return False

        old_tier = self._quality_tier
        self._quality_tier = new_tier
        self.logger.info(f"Quality tier changed from {old_tier.value} to {new_tier.value}")

        # Unload current model so next request loads the new tier
        async with self._lock:
            if self._current_model_type is not None:
                self.logger.info(f"Unloading model to apply tier change")
                await self._unload_model(self._current_model_type)

        return True

    def get_model_state(self, model_type: QwenModelType) -> ModelState:
        """Get the state of a specific model."""
        return self._models[model_type].state

    def get_loaded_model(self) -> Optional[Any]:
        """Get the currently loaded model instance."""
        if self._current_model_type is None:
            return None
        return self._models[self._current_model_type].model_instance

    def is_model_ready(self, model_type: QwenModelType) -> bool:
        """Check if a specific model is loaded and ready."""
        return self._models[model_type].state == ModelState.READY

    def is_model_cached(self, model_type: QwenModelType) -> bool:
        """
        Check if a model is cached locally (no download needed).

        Uses HuggingFace Hub cache detection to determine if the model
        files are already downloaded.

        Args:
            model_type: The model type to check

        Returns:
            bool: True if model is cached, False if download is needed
        """
        if not HF_HUB_AVAILABLE:
            # Can't detect, assume not cached to show download message
            return False

        # Check if using local models path
        if self.models_path:
            model_name = model_type.value.split("/")[-1]
            local_path = Path(self.models_path) / model_name
            if local_path.exists():
                return True

        # Check HuggingFace cache for key model files
        repo_id = model_type.value
        try:
            # Check for a key file that indicates model is cached
            # config.json is always present and downloaded first
            cached_path = try_to_load_from_cache(repo_id, "config.json")
            if cached_path is None or cached_path is _CACHED_NO_EXIST:
                return False
            return True
        except Exception as e:
            self.logger.debug(f"Cache check failed for {repo_id}: {e}")
            return False

    async def ensure_model_loaded(
        self,
        model_type: QwenModelType,
        force_reload: bool = False,
        checkpoint_path: Optional[str] = None,
        tier_override: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Ensure the requested model is loaded, loading it if necessary.

        If a different model is currently loaded, it will be unloaded first.
        This is the primary interface for lazy loading.

        Args:
            model_type: The model type to load
            force_reload: Force reload even if already loaded
            checkpoint_path: Optional custom checkpoint path for fine-tuned models.
                           If provided, loads from this path instead of HuggingFace.
            tier_override: Optional tier to use instead of the configured tier.
                          Use "quality" or "small" for embedding extraction.
                          Does NOT change the persistent tier setting.

        Returns:
            tuple[bool, Optional[str]]: (success, error_message)
        """
        async with self._lock:
            # Check if already loaded with same checkpoint
            same_checkpoint = (
                checkpoint_path == self._current_checkpoint_path or
                (checkpoint_path is None and self._current_checkpoint_path is None)
            )

            # If tier override specified, force reload to ensure correct tier model
            if tier_override:
                force_reload = True

            if (self._current_model_type == model_type and
                self._models[model_type].state == ModelState.READY and
                same_checkpoint and
                not force_reload):
                self.logger.debug(f"Model {model_type.display_name} already loaded" +
                                 (f" from {checkpoint_path}" if checkpoint_path else ""))
                return True, None

            # Unload current model if different model type or different checkpoint
            if self._current_model_type is not None:
                await self._unload_model(self._current_model_type)

            # Load the requested model
            return await self._load_model(model_type, checkpoint_path=checkpoint_path, tier_override=tier_override)

    async def _load_model(
        self,
        model_type: QwenModelType,
        checkpoint_path: Optional[str] = None,
        tier_override: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Load a model (internal method, must be called with lock held).

        Args:
            model_type: The model type to load
            checkpoint_path: Optional custom checkpoint path for fine-tuned models
            tier_override: Optional tier to use instead of configured tier ("quality" or "small")

        Returns:
            tuple[bool, Optional[str]]: (success, error_message)
        """
        import time
        from functools import partial

        model_info = self._models[model_type]

        # Update state to LOADING
        model_info.state = ModelState.LOADING
        model_info.load_error = None
        model_info.checkpoint_path = checkpoint_path

        load_source = checkpoint_path if checkpoint_path else model_type.display_name

        # Check if model is cached (no download needed)
        is_cached = checkpoint_path is not None or self.is_model_cached(model_type)

        if is_cached:
            self._emit_progress(model_type, ModelState.LOADING, 0, f"Loading {load_source}...")
        else:
            # Model needs to be downloaded - show clear download message
            self._emit_progress(model_type, ModelState.LOADING, 0, f"Downloading {load_source} (~3.4 GB)...")
            self.logger.info(f"Model not cached, downloading: {load_source}")

        start_time = time.time()

        try:
            self.logger.info(f"Loading model: {load_source}")

            # Update progress message based on cache status
            if is_cached:
                self._emit_progress(model_type, ModelState.LOADING, 10, f"Loading {load_source} from cache...")
            else:
                self._emit_progress(model_type, ModelState.LOADING, 10, f"Downloading {load_source}... (this may take several minutes)")

            # Check if qwen_tts is available
            if Qwen3TTSModel is None:
                raise ImportError(
                    "qwen-tts package not installed. Install with: pip install qwen-tts"
                )

            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            load_func = partial(self._load_model_sync, model_type, checkpoint_path, tier_override)
            model = await loop.run_in_executor(
                self._executor,
                load_func
            )

            # Update state to READY
            load_time = time.time() - start_time
            model_info.model_instance = model
            model_info.state = ModelState.READY
            model_info.load_time_seconds = load_time
            self._current_model_type = model_type
            self._current_checkpoint_path = checkpoint_path

            self._emit_progress(model_type, ModelState.READY, 100, "Model ready")
            self.logger.info(
                f"Model {load_source} loaded successfully in {load_time:.2f}s"
            )

            return True, None

        except Exception as e:
            error_msg = str(e)
            model_info.state = ModelState.ERROR
            model_info.load_error = error_msg
            model_info.model_instance = None
            model_info.checkpoint_path = None

            self._emit_progress(model_type, ModelState.ERROR, 0, f"Load failed: {error_msg}")
            self.logger.exception(f"Failed to load model {load_source}: {e}")

            return False, error_msg

    def _load_model_sync(
        self,
        model_type: QwenModelType,
        checkpoint_path: Optional[str] = None,
        tier_override: Optional[str] = None
    ) -> Any:
        """
        Synchronously load a model (runs in thread pool).

        Args:
            model_type: The model type to load
            checkpoint_path: Optional custom checkpoint path for fine-tuned models.
                           If provided, loads from this path instead of HuggingFace model.
            tier_override: Optional tier to use instead of configured tier ("quality" or "small")

        Returns:
            The loaded Qwen3TTSModel instance
        """
        from myvoice.models.service_enums import ModelQualityTier

        # Determine which tier to use
        if tier_override:
            effective_tier = ModelQualityTier.SMALL if tier_override == "small" else ModelQualityTier.QUALITY
            self.logger.info(f"Using tier override: {tier_override}")
        else:
            effective_tier = self._quality_tier

        # Determine model source
        if checkpoint_path:
            # Load from custom fine-tuned checkpoint
            model_id = checkpoint_path
            self.logger.info(f"Loading fine-tuned checkpoint: {checkpoint_path}")
        else:
            # Load standard model from HuggingFace or local cache
            # Use quality tier to determine model size (0.6B or 1.7B)
            try:
                model_id = model_type.get_model_id(effective_tier)
            except ValueError as e:
                # Model not available in this tier (e.g., VoiceDesign in Small tier)
                self.logger.warning(f"{e}, falling back to Quality tier")
                model_id = model_type.get_model_id(ModelQualityTier.QUALITY)

            if self.models_path:
                # Use local path if provided
                model_path = os.path.join(self.models_path, model_id.split("/")[-1])
                model_id = model_path

        self.logger.debug(f"Loading from: {model_id} (tier: {effective_tier.value})")

        # Determine attention implementation
        attn_impl = None
        if self.device.startswith("cuda"):
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
                self.logger.debug("Using FlashAttention 2")
            except ImportError:
                self.logger.debug("FlashAttention not available, using default attention")

            # Optimize matmul precision for Ampere+ GPUs (RTX 30/40/50 series)
            # This enables TensorFloat-32 (TF32) which provides significant speedup
            # See: https://github.com/QwenLM/Qwen3-TTS/issues/89
            try:
                torch.set_float32_matmul_precision('high')
                self.logger.debug("Set float32 matmul precision to 'high' for TF32 acceleration")
            except Exception as e:
                self.logger.debug(f"Could not set matmul precision: {e}")

        # Load the model
        load_kwargs = {
            "device_map": self.device,
            "torch_dtype": self.dtype,
        }
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl

        model = Qwen3TTSModel.from_pretrained(model_id, **load_kwargs)

        # Story 18.3 Task 1 — env-var-gated dtype audit. When
        # MYVOICE_DTYPE_AUDIT=1 is set, walk the loaded model and log every
        # relevant dtype attribute, then attach one-shot forward hooks on
        # the talker + speech_tokenizer that log input/output dtypes on
        # first call and detach themselves. Disabled in production (env
        # var unset → zero overhead). Single capture per launch — the hook
        # lifetime is bounded by the first generation.
        if os.environ.get("MYVOICE_DTYPE_AUDIT") == "1":
            self._instrument_dtype_audit(model)

        return model

    def _instrument_dtype_audit(self, model: Any) -> None:
        """Story 18.3 Task 1 — one-shot dtype audit for the loaded model.

        Logs every relevant dtype attribute at INFO level (so it lands in
        myvoice.log alongside the existing ModelRegistry breadcrumbs), then
        attaches one-shot forward hooks on the talker + speech_tokenizer
        that log on first call and detach themselves. The audit lives
        here (model_registry.py) rather than in qwen_tts_service.py
        because (a) the model is already in scope, (b) the hooks need to
        be attached BEFORE any generation runs, and (c) keeping it in one
        place makes the env-var-gated code easy to lift out post-Story-18.3
        if a future cleanup wants to remove the audit infrastructure.

        Walks defensively: every attribute access wrapped in try/except,
        every getattr with a fallback. The audit is informational, not a
        gate — if a model's internal naming has shifted under a
        qwen-tts pin bump, the audit logs what it can and skips the rest
        rather than raising.
        """
        self.logger.info("=" * 70)
        self.logger.info("STORY 18.3 TASK 1 — DTYPE AUDIT (one-shot)")
        self.logger.info("=" * 70)

        # 1. Attribute-walk audit (Task 1.1)
        for attr_path in (
            "dtype",
            "model.dtype",
            "model.talker.dtype",
            "model.model.dtype",
            "model.model.talker.dtype",
        ):
            try:
                obj: Any = model
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                self.logger.info("[DTYPE_AUDIT] model.%s = %s", attr_path, obj)
            except AttributeError:
                self.logger.info("[DTYPE_AUDIT] model.%s = <attribute not present>", attr_path)
            except Exception as e:
                self.logger.info("[DTYPE_AUDIT] model.%s = <error: %s>", attr_path, e)

        # 2. speech_tokenizer walk (Task 1.1) — note that the speech_tokenizer
        # is a Qwen3TTSTokenizer wrapper (HuggingFace, not nn.Module). The
        # actual codec/vocoder lives inside it. Walk for the inner Module.
        st: Any = None
        st_path_used: str = ""
        for st_path in ("model.model.speech_tokenizer", "model.speech_tokenizer"):
            try:
                obj = model
                for part in st_path.split("."):
                    obj = getattr(obj, part)
                st = obj
                st_path_used = st_path
                self.logger.info("[DTYPE_AUDIT] speech_tokenizer found at: model.%s", st_path)
                self.logger.info("[DTYPE_AUDIT] speech_tokenizer type: %s", type(st).__name__)
                break
            except AttributeError:
                continue

        # The speech_tokenizer wrapper holds the inner module under attribute
        # names like ``codec_model`` / ``model`` / ``vocoder`` depending on the
        # qwen-tts version. Walk a small set of likely names; whichever exists
        # is the actual nn.Module that performs the GPU-side decode.
        inner_decoder: Any = None
        inner_decoder_path: str = ""
        if st is not None:
            for inner_attr in ("codec_model", "model", "vocoder", "tokenizer", "_model"):
                cand = getattr(st, inner_attr, None)
                if cand is not None and hasattr(cand, "register_forward_hook"):
                    inner_decoder = cand
                    inner_decoder_path = f"{st_path_used}.{inner_attr}"
                    self.logger.info(
                        "[DTYPE_AUDIT] speech_tokenizer inner Module: %s (type=%s)",
                        inner_decoder_path, type(cand).__name__,
                    )
                    break

        # Sample inner-decoder parameter dtypes (the wrapper itself usually has none).
        def _sample_params(label: str, mod: Any) -> None:
            try:
                if not hasattr(mod, "named_parameters"):
                    self.logger.info("[DTYPE_AUDIT] %s has no named_parameters", label)
                    return
                sampled = 0
                for name, p in mod.named_parameters():
                    self.logger.info("[DTYPE_AUDIT] %s.%s.dtype = %s", label, name, p.dtype)
                    sampled += 1
                    if sampled >= 5:
                        self.logger.info("[DTYPE_AUDIT] %s (truncated to first 5 params)", label)
                        break
                if sampled == 0:
                    self.logger.info("[DTYPE_AUDIT] %s has zero named_parameters", label)
            except Exception as e:
                self.logger.info("[DTYPE_AUDIT] %s param walk error: %s", label, e)

        if st is not None:
            _sample_params("speech_tokenizer", st)
        if inner_decoder is not None:
            _sample_params(f"speech_tokenizer.{inner_decoder_path.split('.')[-1]}", inner_decoder)
        if st is None:
            self.logger.info("[DTYPE_AUDIT] speech_tokenizer NOT found at any expected path")

        # 3. One-shot forward hooks (Task 1.2 + 1.3) — robust against
        # kwargs-only signatures and structured output objects.
        def _describe(t: Any) -> str:
            """One-line dtype/type description for a hook arg or output."""
            if t is None:
                return "None"
            if hasattr(t, "dtype"):
                shape = tuple(t.shape) if hasattr(t, "shape") else None
                return f"<Tensor dtype={t.dtype} shape={shape}>"
            if isinstance(t, (tuple, list)):
                if not t:
                    return f"{type(t).__name__}[]"
                inner = ",".join(_describe(x) for x in t[:3])
                more = ",..." if len(t) > 3 else ""
                return f"{type(t).__name__}[{inner}{more}]"
            if isinstance(t, dict):
                if not t:
                    return "dict{}"
                items = ",".join(f"{k}={_describe(v)}" for k, v in list(t.items())[:5])
                return f"dict{{{items}}}"
            # Structured output objects (e.g., Qwen3TTSTalkerOutputWithPast):
            # walk their public attributes for any tensor-like fields.
            tensor_fields = []
            for attr in dir(t):
                if attr.startswith("_"):
                    continue
                try:
                    v = getattr(t, attr)
                except Exception:
                    continue
                if hasattr(v, "dtype") and hasattr(v, "shape"):
                    tensor_fields.append(f"{attr}=<dtype={v.dtype} shape={tuple(v.shape)}>")
                if len(tensor_fields) >= 5:
                    break
            if tensor_fields:
                return f"{type(t).__name__}({'; '.join(tensor_fields)})"
            return f"<{type(t).__name__}>"

        def _make_hook(label: str, handle_box: list):
            def _hook(module, inputs, output, *extra, **hook_kwargs):
                # Newer torch invokes the hook with kwargs as a third positional arg
                # when register_forward_hook(..., with_kwargs=True). Without that
                # flag, kwargs are NOT visible to the hook — but since talker.forward
                # is kwargs-only, we capture whatever positional `inputs` we got AND
                # any kwargs passed via with_kwargs=True (best-effort).
                fwd_kwargs = extra[0] if extra and isinstance(extra[0], dict) else hook_kwargs
                try:
                    in_desc = _describe(inputs) if inputs else "()"
                    kw_desc = _describe(fwd_kwargs) if fwd_kwargs else "{}"
                    out_desc = _describe(output)
                    self.logger.info(
                        "[DTYPE_AUDIT_FWD] %s args=%s kwargs=%s out=%s",
                        label, in_desc, kw_desc, out_desc,
                    )
                except Exception as e:
                    self.logger.info("[DTYPE_AUDIT_FWD] %s hook error: %s", label, e)
                finally:
                    # One-shot: detach self after first invocation.
                    if handle_box and handle_box[0] is not None:
                        try:
                            handle_box[0].remove()
                        except Exception:
                            pass
                        handle_box[0] = None
            return _hook

        def _attach(label: str, mod: Any) -> None:
            if mod is None or not hasattr(mod, "register_forward_hook"):
                self.logger.info("[DTYPE_AUDIT] %s not hookable — skipping forward hook", label)
                return
            box: list = [None]
            try:
                # Try with_kwargs=True first (torch >= 2.0); fall back to legacy signature.
                try:
                    box[0] = mod.register_forward_hook(_make_hook(label, box), with_kwargs=True)
                except TypeError:
                    box[0] = mod.register_forward_hook(_make_hook(label, box))
                self.logger.info("[DTYPE_AUDIT] %s forward hook attached (one-shot)", label)
            except Exception as e:
                self.logger.info("[DTYPE_AUDIT] %s hook attach error: %s", label, e)

        try:
            talker = getattr(getattr(model, "model", None), "talker", None)
            if talker is None:
                # Try the alternative path (some qwen-tts versions expose it differently)
                talker = getattr(model, "talker", None)
            _attach("talker", talker)
        except Exception as e:
            self.logger.info("[DTYPE_AUDIT] talker resolve error: %s", e)

        # Hook the inner decoder Module if found; the wrapper itself is HF-only.
        _attach("speech_tokenizer.inner", inner_decoder)

        self.logger.info(
            "[DTYPE_AUDIT] audit setup complete — next generation will trigger one-shot forward-hook captures"
        )
        self.logger.info("=" * 70)

    async def _unload_model(self, model_type: QwenModelType) -> bool:
        """
        Unload a model (internal method, must be called with lock held).

        Args:
            model_type: The model type to unload

        Returns:
            bool: True if unloaded successfully
        """
        model_info = self._models[model_type]

        if model_info.state not in (ModelState.READY, ModelState.ERROR):
            self.logger.debug(f"Model {model_type.display_name} not loaded, skip unload")
            return True

        self.logger.info(f"Unloading model: {model_type.display_name}")
        model_info.state = ModelState.UNLOADING
        self._emit_progress(model_type, ModelState.UNLOADING, 50, "Unloading model...")

        try:
            # Clear model instance
            if model_info.model_instance is not None:
                del model_info.model_instance
                model_info.model_instance = None

            # Force garbage collection to free GPU memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Update state
            model_info.state = ModelState.UNLOADED
            model_info.load_error = None
            model_info.load_time_seconds = None

            if self._current_model_type == model_type:
                self._current_model_type = None
                self._current_checkpoint_path = None

            self._emit_progress(model_type, ModelState.UNLOADED, 0, "Model unloaded")
            self.logger.info(f"Model {model_type.display_name} unloaded")

            return True

        except Exception as e:
            self.logger.exception(f"Error unloading model: {e}")
            model_info.state = ModelState.ERROR
            return False

    async def unload_all(self) -> bool:
        """Unload all loaded models."""
        async with self._lock:
            success = True
            for model_type in QwenModelType:
                if self._models[model_type].state == ModelState.READY:
                    if not await self._unload_model(model_type):
                        success = False
            return success

    def get_registry_status(self) -> Dict[str, Any]:
        """Get comprehensive registry status."""
        return {
            "current_model": self._current_model_type.display_name if self._current_model_type else None,
            "current_checkpoint": self._current_checkpoint_path,
            "device": self.device,
            "dtype": str(self.dtype),
            "models": {
                model_type.display_name: {
                    "state": model_info.state.value,
                    "load_time": model_info.load_time_seconds,
                    "error": model_info.load_error,
                    "checkpoint_path": model_info.checkpoint_path,
                }
                for model_type, model_info in self._models.items()
            }
        }

    @property
    def current_checkpoint_path(self) -> Optional[str]:
        """Get the currently loaded checkpoint path (for fine-tuned models)."""
        return self._current_checkpoint_path

    def set_progress_callback(
        self,
        callback: Optional[Callable[[ModelLoadProgress], None]]
    ):
        """Set the progress callback for loading updates."""
        self._progress_callback = callback

    def _emit_progress(
        self,
        model_type: QwenModelType,
        state: ModelState,
        percent: float,
        message: str
    ):
        """Emit progress update via callback."""
        if self._progress_callback:
            try:
                progress = ModelLoadProgress(
                    model_type=model_type,
                    state=state,
                    progress_percent=percent,
                    message=message
                )
                self._progress_callback(progress)
            except Exception as e:
                self.logger.error(f"Error in progress callback: {e}")

    def shutdown(self):
        """Shutdown the model registry and release resources."""
        self.logger.info("Shutting down ModelRegistry")

        # Unload all models synchronously
        for model_type in QwenModelType:
            model_info = self._models[model_type]
            if model_info.model_instance is not None:
                del model_info.model_instance
                model_info.model_instance = None
            model_info.state = ModelState.UNLOADED

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Shutdown executor - QA Round 2 Item #8: Non-blocking shutdown
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._current_model_type = None

        self.logger.info("ModelRegistry shutdown complete")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.shutdown()
        except Exception:
            pass
