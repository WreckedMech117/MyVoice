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
        quality_tier: str = "quality"
    ):
        """
        Initialize the ModelRegistry.

        Args:
            device: PyTorch device ("auto", "cuda:0", "cpu")
            dtype: PyTorch dtype ("bfloat16", "float16", "float32")
            models_path: Optional local path for model weights
            progress_callback: Callback for loading progress updates
            quality_tier: Model quality tier ("small" or "quality")
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Device configuration
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Dtype configuration
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(dtype, torch.bfloat16)

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

        self.logger.info(
            f"ModelRegistry initialized: device={self.device}, dtype={dtype}, quality_tier={self._quality_tier.value}"
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

        return model

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
