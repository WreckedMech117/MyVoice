"""
MyVoice Application Controller

This module contains the main application controller that manages the overall
application lifecycle, initialization, and coordination between services and UI.
"""

import gc
import logging
import sys
import time  # Story 18.1: progressive-playback consumer-side wall-clock metric
import uuid
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional
from datetime import datetime

import numpy as np

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QMetaObject, QObject, Qt, QTimer, pyqtSlot

from myvoice.models.ui_state import ServiceStatusInfo, ServiceHealthStatus
from myvoice.models.service_enums import ServiceStatus, QwenModelType
from myvoice.observability import metrics  # Story 18.1: progressive-playback instrumentation
from myvoice.ui.dialogs.save_dialog import SaveAudioDialog  # Story 14.3


class _BoundedDedupSet:
    """Story 13.2 follow-up: LRU-evicting dedup set with O(1) membership.

    Backs ``MyVoiceApp._closed_session_ids`` and
    ``_advanced_replay_tokens`` so the dual-fire callback dedup memory
    stays bounded over the lifetime of a long-running app. Default cap
    (256) gives a comfortable buffer for the dual-fire window (typically
    milliseconds between the monitor and virtual-mic callbacks) even at
    sustained 1Hz generation. Eviction is FIFO: the oldest entry is
    dropped when a new one would exceed the cap.

    Provides only the surface ``MyVoiceApp`` consumes (``add``,
    ``__contains__``, ``__len__``) — drop-in replacement for ``set[str]``
    at those call sites.
    """

    __slots__ = ("_max", "_items")

    def __init__(self, max_size: int = 256) -> None:
        self._max = max_size
        self._items: "OrderedDict[str, None]" = OrderedDict()

    def add(self, key: str) -> None:
        if key in self._items:
            return
        if len(self._items) >= self._max:
            self._items.popitem(last=False)
        self._items[key] = None

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class _PendingDispatch:
    """Story 13.2: parked dispatch context for a session that was enqueued
    but cannot dispatch yet because the queue head is a different session.

    The playback-complete callback chain pulls this entry from
    ``MyVoiceApp._pending_dispatches`` once the queue advances to this
    session. Keys are the queue token (the session id, or a synthetic
    ``replay-<uuid>`` token for the registry-less replay path).

    ``queue_token`` is preserved separately from ``session_id`` so the
    re-entry path through ``_play_generated_audio`` reuses the original
    token (registry-tracked sessions: token == session_id; replay path:
    token is a synthetic ``replay-<uuid>``). Without preserving it, the
    re-entry would mint a fresh synthetic token for replay and the
    re-entry guard ``queue_token == _dispatching_session_id`` would fail.
    """

    audio_data: bytes
    session_id: Optional[str]
    queue_token: str


class _ClearCommsResolveError(Exception):
    """Story 15.2: raised by ``MyVoiceApp._resolve_clear_comms_wav_bytes``
    when the configured Clear Comms source cannot be turned into WAV bytes.

    Lives at module scope (above ``MyVoiceApp``) per Python convention for
    module-private companion classes. Carries a short user-facing
    ``user_message`` (≤ 80 chars, ASCII-clean) that the click-handler
    routes verbatim to ``MainWindow.set_generation_status``. Wraps the
    loader-side ``PreloadedAudioLoadError`` for the file branch — its
    message survives the wrap unchanged so the toast surface is
    consistent with the file-picker error wording the user has already
    seen.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.user_message: str = message


class MyVoiceApp(QObject):
    """
    Main application controller for MyVoice.

    This class manages the application lifecycle, service initialization,
    and coordinates between the UI and business logic layers.

    Attributes:
        qt_app (QApplication): The PyQt6 application instance
        logger (logging.Logger): Application logger
    """

    def __init__(self, qt_app: QApplication):
        """
        Initialize the MyVoice application controller.

        Args:
            qt_app (QApplication): The PyQt6 application instance
        """
        super().__init__()
        self.qt_app = qt_app
        self.logger = logging.getLogger(self.__class__.__name__)

        # Application state
        self._initialized = False
        self._main_window = None
        self._services = {}

        # Story 12.1: bookkeeping for the playback-side registry close path
        # (Task 3.5). The audio services fire playback-complete once for
        # the monitor task and once for the virtual-mic task — we only
        # want to post mark_done+discard once per session. The set
        # short-circuits the second call; the dict maps task ids to
        # session ids so the callback can resolve session id from task id
        # without changing the audio-service callback signature.
        self._task_to_session: dict[str, str] = {}
        # Story 13.2 follow-up: bounded dedup set (FIFO-evicting at 256
        # entries) prevents unbounded growth across long-running sessions.
        # The dual-fire callback window is milliseconds; 256 entries is
        # ample headroom even at sustained generation rates.
        self._closed_session_ids: _BoundedDedupSet = _BoundedDedupSet()

        # Story 13.2 — Phase 3 of D-20 (OFR-C): PlaybackQueue integration
        # state. The queue itself is constructed in
        # _initialize_services_async (it requires QApplication.instance() to
        # be alive, which is asserted by PlaybackQueue.__init__).
        #
        # _pending_dispatches: when a session arrives at _play_generated_audio
        # while another session is currently dispatching, we park the audio
        # bytes here keyed by the queue token. _dispatch_next_pending pulls
        # from this map when the queue advances.
        #
        # _dispatching_session_id: the session whose play_dual_stream is in
        # flight (or about to be invoked). Mirrors the queue head while the
        # head is actively playing; None when no playback is active. The
        # _play_generated_audio re-entry guard reads this to avoid double-
        # enqueueing on the deferred-dispatch re-entry path.
        self._playback_queue = None  # type: Optional[QObject]
        self._pending_dispatches: Dict[str, _PendingDispatch] = {}
        self._dispatching_session_id: Optional[str] = None
        # Story 13.2: parallel to _task_to_session for the replay path
        # (session_id is None, but we still need to advance the queue
        # exactly once per replay despite the dual-fire callback). Maps
        # task_id → synthetic "replay-XXXXXXXX" token. _advanced_replay_tokens
        # is the dedup set: the second dual-fire finds the token already
        # present and skips the queue advance.
        self._task_to_replay_token: Dict[str, str] = {}
        # Story 13.2 follow-up: bounded dedup (see _closed_session_ids
        # above) prevents the replay-token dedup memory from growing
        # forever across many Replay clicks.
        self._advanced_replay_tokens: _BoundedDedupSet = _BoundedDedupSet()

        # Story 17.3 — Phase ⊥-Polish: progressive-playback consumer state.
        # _progressive_playback_active is latched True from chunk 0 of a
        # streaming generation and consumed (cleared) by either the dispatch
        # skip-branch (normal path) or the cancel handler (interrupt path).
        # The audio device session is closed on the terminal AudioChunk
        # (is_final=True) but the flag stays True so _play_generated_audio
        # knows to skip its play_dual_stream call — clearing the flag at
        # is_final would race the dispatch on the asyncio loop ordering.
        self._progressive_playback_active: bool = False
        self._progressive_playback_sample_rate: int = 0
        # Lazy-initialized: asyncio.Lock requires a running event loop to
        # construct, so first use creates it inside the handler coro.
        self._progressive_playback_lock: Optional[asyncio.Lock] = None
        # Cancel-vs-chunk race guard. Incremented by the cancel handler so
        # any chunk that was queued via run_coroutine_threadsafe before the
        # cancel sees a stale captured value under the handler lock and
        # drops itself instead of opening a fresh session post-cancel.
        self._progressive_playback_epoch: int = 0

        # Story 18.1 Task 1.4: env-var-gated CSV capture for the three new
        # progressive-playback metrics. Disabled by default (None) — engages
        # only when ``MYVOICE_PROGRESSIVE_PLAYBACK_CSV`` is set, returning a
        # ``stop`` callable that ``_on_about_to_quit`` invokes for a clean
        # close. Resolution lives in
        # ``myvoice.observability.progressive_playback_csv_capture`` so the
        # subpackage owns the file/listener lifecycle and app.py stays slim.
        self._progressive_metric_capture_stop: Optional[
            Callable[[], None]
        ] = None
        try:
            from myvoice.observability.progressive_playback_csv_capture import (
                maybe_enable_from_env,
            )
            from myvoice.utils.portable_paths import get_logs_path

            self._progressive_metric_capture_stop = maybe_enable_from_env(
                get_logs_path()
            )
        except Exception:
            self.logger.exception(
                "Story 18.1: progressive-playback CSV capture wiring "
                "failed (non-fatal; instrumentation metrics still emit "
                "via myvoice.metrics logger)"
            )

        # Connect application signals
        self.qt_app.aboutToQuit.connect(self._on_about_to_quit)

        self.logger.debug("MyVoiceApp controller initialized")

    def initialize(self) -> bool:
        """
        DEPRECATED: Use initialize_async() instead.

        This synchronous initialization method is kept for backward compatibility
        but should not be used. The qasync migration requires async initialization.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        self.logger.warning("initialize() is deprecated - use initialize_async() instead")
        return False

    async def initialize_async(self) -> bool:
        """
        Initialize the application components asynchronously.

        This replaces the synchronous initialize() method and uses
        a shared qasync event loop for all async operations.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing MyVoice application components asynchronously")

            # Get shared event loop (set by qasync in main.py)
            self.loop = asyncio.get_event_loop()
            self.logger.debug(f"Using shared event loop: {self.loop}")

            # Initialize configuration (NOW ASYNC)
            if not await self._initialize_configuration_async():
                self.logger.error("Failed to initialize configuration")
                return False

            # Initialize services (NOW ASYNC)
            if not await self._initialize_services_async():
                self.logger.error("Failed to initialize services")
                return False

            # Initialize UI (CAN STAY SYNC - Qt is sync)
            if not self._initialize_ui():
                self.logger.error("Failed to initialize UI")
                return False

            self._initialized = True

            # Initialize mic mixing if enabled in settings
            await self._setup_mic_mixing_from_settings()

            self.logger.info("MyVoice application initialization completed successfully")
            return True

        except Exception as e:
            self.logger.exception(f"Error during application initialization: {e}")
            self._show_error_dialog("Initialization Error",
                                   f"Failed to initialize MyVoice application:\n{str(e)}")
            return False


    def _initialize_configuration(self) -> bool:
        """
        DEPRECATED: Use _initialize_configuration_async() instead.

        This is kept for backward compatibility but should not be used.
        """
        self.logger.warning("_initialize_configuration() is deprecated - use _initialize_configuration_async() instead")
        return False

    def _initialize_services(self) -> bool:
        """
        DEPRECATED: Use _initialize_services_async() instead.

        This is kept for backward compatibility but should not be used.
        """
        self.logger.warning("_initialize_services() is deprecated - use _initialize_services_async() instead")
        return False

    async def _initialize_configuration_async(self) -> bool:
        """
        Initialize application configuration asynchronously.

        This replaces _initialize_configuration() and uses direct async/await
        instead of AsyncTaskManager.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.debug("Initializing configuration asynchronously")

            # Create necessary directories (using portable paths)
            from myvoice.utils.portable_paths import ensure_portable_compatibility, get_config_file_path
            ensure_portable_compatibility()

            # Story 4.2: Clean up orphaned Voice Design Studio sessions (>24h old)
            from myvoice.utils.session_manager import SessionManager
            cleaned, preserved = SessionManager.cleanup_orphan_sessions()
            if cleaned > 0:
                self.logger.info(f"Startup cleanup: removed {cleaned} orphan sessions, preserved {preserved} recent")

            # Initialize Configuration Service with PORTABLE path
            from myvoice.services.configuration_service import ConfigurationManager
            config_path = get_config_file_path()
            self.logger.info(f"Using portable config path: {config_path}")

            self._config_manager = ConfigurationManager(config_file=config_path)
            self.register_service("config", self._config_manager)

            # Start configuration service (DIRECT AWAIT - SAME LOOP)
            await self._config_manager.start()
            self.logger.info("Configuration service started successfully")

            # Load application settings after configuration service is ready
            self._app_settings = await self._load_app_settings_on_startup()
            self._on_settings_loaded(self._app_settings)

            self.logger.debug("Configuration initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(f"Error initializing configuration: {e}")
            return False

    async def _initialize_services_async(self) -> bool:
        """
        Initialize application services asynchronously.

        This replaces _initialize_services() and uses direct async/await
        instead of AsyncTaskManager.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.debug("Initializing services asynchronously")

            # Story 11.4: Construct SessionRegistry on the Qt main thread
            # (its __init__ enforces this) before any consumer that holds a
            # reference to it. The registry is the in-flight session
            # collection (D-1) and the sole signal emitter for session
            # lifecycle (D-2). Story 11.4 wires QwenTTSService to populate
            # it; later stories (12.1, 13.1, 14.1, 15.x) subscribe to its
            # signals. AI-Review H1 (2026-05-04): construction MUST precede
            # AudioCoordinator and QwenTTSService — both consume it via
            # constructor injection, and the previous ordering passed `None`
            # to AudioCoordinator, silently disabling its registry posts in
            # the dual-stream production path.
            from myvoice.services.sessions import SessionRegistry
            self._session_registry = SessionRegistry(parent=self)
            self.register_service("session_registry", self._session_registry)
            self.logger.info("SessionRegistry constructed on Qt main thread")

            # Story 13.2 — Phase 3 of D-20 (OFR-C): construct the PlaybackQueue
            # on the Qt main thread, after SessionRegistry (so we can forward
            # the depth signal) and before AudioCoordinator (so the playback-
            # complete callback wired below can advance the queue). Story 13.1
            # sealed the queue's invariants in isolation; this story activates
            # them in the dispatch path. Construction asserts QApplication is
            # alive and currentThread() is the Qt main thread; both are
            # satisfied here because _initialize_services_async runs through
            # qasync's asyncSlot (Qt main thread) after QApplication is up.
            from myvoice.services.sessions import PlaybackQueue
            self._playback_queue = PlaybackQueue(parent=self)
            self.register_service("playback_queue", self._playback_queue)
            self.logger.info("PlaybackQueue constructed on Qt main thread")

            # Story 13.2 AC #7: forward queue.playback_queue_depth_changed
            # to registry.playback_queue_depth_changed so the existing
            # MainWindow slot wired in Story 12.1 (main_window.py:1741-1745,
            # slot at 1826-1835) receives depth events without rewiring.
            # Both queue and registry live on the Qt main thread, so
            # AutoConnection collapses to DirectConnection — synchronous,
            # zero-loss re-emission. The registry's signal at
            # session_registry.py:152 was declared inert in Story 11.2
            # ("13.1 will emit from PlaybackQueue") — 13.2 closes that loop.
            self._playback_queue.playback_queue_depth_changed.connect(
                self._session_registry.playback_queue_depth_changed.emit
            )

            # Initialize Audio Coordinator (dual-service architecture).
            # Story 12.1: thread the SessionRegistry into the coordinator so
            # play_dual_stream can post mark_playing/mark_audible mutations
            # alongside the existing dispatch path (D-20 Phase 2).
            from myvoice.services.audio_coordinator import AudioCoordinator
            self._audio_coordinator = AudioCoordinator(
                app_settings=getattr(self, '_app_settings', None),
                session_registry=self._session_registry,
            )
            self.register_service("audio_coordinator", self._audio_coordinator)

            # Start audio coordinator (DIRECT AWAIT)
            await self._audio_coordinator.start()
            self.logger.info("Audio coordinator started successfully")
            self._on_audio_coordinator_started(None)

            # Story 11.4 follow-up: hook the real playback-complete signal
            # so the Stop button stays visible for the entire duration of
            # actual audio playback. Without this, the only "playback
            # done" signal app.py receives is the start-task's on_success
            # callback, which fires the moment ``play_dual_stream``
            # returns — long before the audio finishes.
            self._audio_coordinator.set_playback_complete_callback(
                self._on_playback_complete
            )

            # Auto-detect and configure VB-Cable on first boot if no virtual device configured
            await self._auto_detect_and_configure_vb_cable()

            # Initialize TTS Service (Qwen3-TTS)
            # Use quality tier from settings (Small 0.6B vs Quality 1.7B)
            from myvoice.services.qwen_tts_service import QwenTTSService
            quality_tier = getattr(self._app_settings, 'model_quality_tier', 'quality') if self._app_settings else 'quality'
            self.logger.info(f"Initializing TTS service with quality tier: {quality_tier}")
            self._tts_service = QwenTTSService(
                quality_tier=quality_tier,
                session_registry=self._session_registry,  # Story 11.4
                app_settings=self._app_settings,  # Story 18.3 — flow tts_precision into ModelRegistry resolver
            )
            self.register_service("tts", self._tts_service)

            # Set up health status callback BEFORE starting service
            self._tts_service.set_health_status_callback(self._on_tts_health_status_changed)

            # Story 17.2 — wire UI feedback for the lazy CLONED-voice
            # voice_clone_prompt precompute (cache miss surfaces a transient
            # "Preparing voice for streaming…" message in the TTS indicator).
            self._tts_service.set_preparing_voice_callback(
                self._on_tts_preparing_voice_message
            )
            # Story 17.2 — orchestrator-driven on-demand Whisper init.
            # When the precompute hits a cache miss with no Whisper service
            # wired (e.g. user has not opened Voice Design Studio yet), the
            # callback fires init in the background; the precompute itself
            # raises so the dispatch chain falls through to SENTENCE_STREAM
            # for this attempt — next attempt (post-init) hits the cache.
            self._tts_service.set_whisper_init_callback(
                self._on_whisper_init_requested
            )

            # Story 17.3 — wire the progressive-playback consumer so the
            # SENTENCE_STREAM and TRUE_STREAM chunk-emit callbacks (per
            # qwen_tts_service.py:3071-3082 and :3897-3947) reach the
            # AudioCoordinator's start_streaming_session/play_audio_chunk
            # /stop_streaming_session triplet. Closes the user-perceived
            # progressive-playback gap surfaced by Story 17.2's smoke.
            self._tts_service.set_audio_chunk_ready_callback(
                self._on_audio_chunk_ready
            )

            # Start TTS service (DIRECT AWAIT)
            await self._tts_service.start()
            self.logger.info("TTS service started successfully")
            self._on_tts_service_started(None)

            # Initialize Voice Profile Service
            # Use portable paths to get the correct voice_files directory
            from myvoice.services.voice_profile_service import VoiceProfileManager
            from myvoice.utils.portable_paths import get_voice_files_path

            # Get voice directory - prefer portable path for bundled voices
            voice_directory = get_voice_files_path()
            self.logger.info(f"Using voice directory: {voice_directory}")

            # If settings has a custom path, use it instead (for user customization)
            if hasattr(self, '_app_settings') and self._app_settings:
                voice_dir_str = self._app_settings.voice_files_directory
                # Only use custom path if it's different from default and exists
                if voice_dir_str and voice_dir_str != "voice_files":
                    custom_path = Path(voice_dir_str)
                    if custom_path.is_absolute() and custom_path.exists():
                        voice_directory = custom_path
                        self.logger.info(f"Using custom voice directory from settings: {voice_directory}")

            self._voice_manager = VoiceProfileManager(voice_directory=voice_directory)
            self.register_service("voice_profiles", self._voice_manager)

            # Start voice profile service (DIRECT AWAIT)
            await self._voice_manager.start()
            self.logger.info("Voice profile service started successfully")
            self._on_voice_service_started(None)

            # Story 17.2 — wire VoiceProfileManager into TTS service so the
            # CLONED-voice voice_clone_prompt cache can hydrate from disk
            # at startup (and resolve VoiceProfile objects on cache miss
            # for transcription-status updates). Hydration runs as a
            # fire-and-forget background task because (i) it scans disk
            # for .pt files and (ii) it must not block the rest of startup.
            try:
                self._tts_service.set_voice_profile_manager(self._voice_manager)
                self._run_async_task(
                    self._tts_service.hydrate_voice_clone_prompt_cache(),
                    on_success=lambda result: self.logger.info(
                        f"Voice clone prompt cache hydration: {result}"
                    ),
                    on_error=lambda error: self.logger.warning(
                        f"Voice clone prompt cache hydration failed: {error}"
                    ),
                )
            except Exception as exc:
                self.logger.warning(
                    f"Voice clone prompt cache wiring failed: {exc}"
                )

            # Preload the appropriate model based on cached active voice profile
            # This ensures fast first generation without model switching delay
            preferred_model = self._voice_manager.get_active_profile_model_type()
            if preferred_model:
                self.logger.info(f"Preloading model for cached active profile: {preferred_model.display_name}")
                try:
                    success, error = await self._tts_service.preload_model(preferred_model)
                    if success:
                        self.logger.info(f"Model {preferred_model.display_name} preloaded successfully")
                    else:
                        self.logger.warning(f"Failed to preload model {preferred_model.display_name}: {error}")
                except Exception as e:
                    self.logger.warning(f"Error preloading model: {e}")
            else:
                # No cached active profile, preload default CustomVoice model
                self.logger.info("No cached active profile, preloading default CustomVoice model")
                try:
                    success, error = await self._tts_service.preload_model(QwenModelType.CUSTOM_VOICE)
                    if success:
                        self.logger.info("CustomVoice model preloaded successfully")
                    else:
                        self.logger.warning(f"Failed to preload CustomVoice model: {error}")
                except Exception as e:
                    self.logger.warning(f"Error preloading default model: {e}")

            # Note: Whisper Service will be initialized on-demand due to DLL conflicts with PyQt6
            # See _initialize_whisper_service_on_demand method
            self._whisper_service = None

            self.logger.debug("Services initialization completed successfully")
            return True

        except Exception as e:
            self.logger.exception(f"Error initializing services: {e}")
            return False

    def _initialize_ui(self) -> bool:
        """
        Initialize the user interface.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.debug("Initializing UI")

            # Create and show main window
            # Story 12.1: thread the SessionRegistry into MainWindow so the
            # TTS indicator's substate is driven by registry signals (D-20
            # Phase 2). The registry was constructed earlier on the Qt main
            # thread; passing it through __init__ keeps the wiring explicit
            # and matches the injection pattern Story 11.4 used for QwenTTSService.
            from myvoice.ui.main_window import MainWindow
            self._main_window = MainWindow(
                session_registry=getattr(self, '_session_registry', None)
            )

            # Connect voice manager if already available (after service startup)
            if hasattr(self, '_voice_manager'):
                self._main_window.set_voice_manager(self._voice_manager)

            # Connect app settings if already loaded (from configuration init)
            if hasattr(self, '_app_settings') and self._app_settings:
                self._main_window.set_app_settings(self._app_settings)
                self.logger.debug("Connected app settings to main window during UI init")

            # Connect audio coordinator if already available
            if hasattr(self, '_audio_coordinator') and self._audio_coordinator:
                self._main_window.set_audio_coordinator(self._audio_coordinator)
                self.logger.debug("Connected audio coordinator to main window during UI init")


            # Connect main window signals to application handlers
            self._main_window.text_generate_requested.connect(self._on_text_generate_requested)
            self._main_window.voice_changed.connect(self._on_voice_changed)
            self._main_window.settings_requested.connect(self._on_settings_requested)
            self._main_window.settings_changed.connect(self._on_settings_changed)
            self._main_window.audio_device_refresh_requested.connect(self._on_device_refresh_requested)
            self._main_window.audio_device_test_requested.connect(self._on_device_test_requested)
            self._main_window.virtual_device_test_requested.connect(self._on_virtual_device_test_requested)
            self._main_window.voice_directory_changed.connect(self._on_voice_directory_changed)
            self._main_window.voice_refresh_requested.connect(self._on_voice_refresh_requested)
            self._main_window.voice_transcription_requested.connect(self._on_voice_transcription_requested)
            self._main_window.replay_last_requested.connect(self._on_replay_last_requested)  # Story 2.4
            self._main_window.save_requested.connect(self._on_save_requested)  # Story 14.3
            self._main_window.clear_comms_requested.connect(self._on_clear_comms_requested)  # Story 15.2
            self._main_window.clear_comms_test_playback_requested.connect(
                self._on_clear_comms_test_playback_requested
            )  # Story 15.3
            self._main_window.cancel_generation_requested.connect(self._on_cancel_generation_requested)  # Story 11.4 follow-up
            self._main_window.whisper_init_requested.connect(self._on_whisper_init_requested)  # QA4
            self._main_window.mic_device_refresh_requested.connect(self._on_mic_device_refresh_requested)
            self._main_window.mic_monitor_toggled.connect(self._on_mic_monitor_toggled)

            # Connect TTS service to main window and update status
            # This must happen here since _on_tts_service_started's 100ms timer fires
            # before _main_window exists (services init completes before UI init)
            if hasattr(self, '_tts_service') and self._tts_service:
                self._main_window.add_service_monitoring("TTS")
                self._main_window.set_tts_service(self._tts_service)

                # Update TTS health status in UI
                from myvoice.models.ui_state import ServiceHealthStatus
                health_status = ServiceHealthStatus.HEALTHY if self._tts_service.is_running() else ServiceHealthStatus.ERROR
                self._on_tts_health_status_changed(health_status, None)
                self.logger.info(f"TTS status initialized in UI: {health_status.value}")

            # Show the main window
            self._main_window.show_and_raise()

            self.logger.debug("UI initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(f"Error initializing UI: {e}")
            return False

    def _ensure_directories(self):
        """
        DEPRECATED: Use portable_paths.ensure_portable_compatibility() instead.

        This method is kept for backward compatibility but delegates to the
        portable paths utility which handles directory creation properly.
        """
        from myvoice.utils.portable_paths import initialize_portable_directories
        initialize_portable_directories()
        self.logger.debug("Portable directories initialized")

    def _show_error_dialog(self, title: str, message: str):
        """
        Show an error dialog to the user.

        Args:
            title (str): Dialog title
            message (str): Error message
        """
        try:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.setWindowTitle(title)
            msg_box.setText(message)
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()
        except Exception as e:
            # Fallback to console if dialog fails
            self.logger.error(f"Failed to show error dialog: {e}")
            print(f"ERROR - {title}: {message}", file=sys.stderr)

    def _on_about_to_quit(self):
        """Handle application quit signal."""
        self.logger.info("Application shutting down")

        # Story 18.1 Task 1.4: flush + close the progressive-playback CSV
        # capture FIRST (before service cleanup) so the last metric records
        # land on disk even if a downstream cleanup step raises. The stop
        # callable is idempotent (see progressive_playback_csv_capture.py)
        # so a redundant call from a reentry path is safe.
        if self._progressive_metric_capture_stop is not None:
            try:
                self._progressive_metric_capture_stop()
            except Exception:
                self.logger.exception(
                    "Story 18.1: progressive-playback CSV capture stop "
                    "raised (non-fatal; partial CSV may already be on disk)"
                )
            self._progressive_metric_capture_stop = None

        try:
            # Clean up services
            self._cleanup_services()

            # Clean up UI
            if self._main_window:
                self._main_window.close()

        except Exception as e:
            self.logger.exception(f"Error during application cleanup: {e}")

    async def cleanup_async(self):
        """
        Clean up application resources asynchronously.

        This replaces _cleanup_services() and ensures services are stopped
        in the SAME event loop they were started in (CRITICAL FIX).

        QA5 Enhancement: Added aggressive memory cleanup to prevent process
        persistence after taskbar close.
        """
        self.logger.info("Starting async cleanup - stopping services...")

        try:
            # Stop services in SAME loop they were started in
            if hasattr(self, '_tts_service') and self._tts_service:
                try:
                    await self._tts_service.stop()
                    self.logger.info("TTS service stopped")
                except Exception as e:
                    self.logger.error(f"Error stopping TTS service: {e}")

            if hasattr(self, '_config_manager') and self._config_manager:
                try:
                    await self._config_manager.stop()
                    self.logger.debug("Configuration service stopped")
                except Exception as e:
                    self.logger.error(f"Error stopping configuration service: {e}")

            if hasattr(self, '_voice_manager') and self._voice_manager:
                try:
                    await self._voice_manager.stop()
                    self.logger.debug("Voice profile service stopped")
                except Exception as e:
                    self.logger.error(f"Error stopping voice manager: {e}")

            if hasattr(self, '_audio_coordinator') and self._audio_coordinator:
                try:
                    await self._audio_coordinator.stop()
                    self.logger.debug("Audio coordinator stopped")
                except Exception as e:
                    self.logger.error(f"Error stopping audio coordinator: {e}")

            if hasattr(self, '_whisper_service') and self._whisper_service is not None:
                try:
                    await self._whisper_service.stop()
                    self.logger.debug("Whisper service stopped")
                except Exception as e:
                    self.logger.error(f"Error stopping Whisper service: {e}")

            # Cleanup other services
            for service_name, service in list(self._services.items()):
                try:
                    if hasattr(service, 'cleanup'):
                        service.cleanup()
                        self.logger.debug(f"Cleaned up service: {service_name}")
                except Exception as e:
                    self.logger.error(f"Error cleaning up service {service_name}: {e}")

            # Close main window
            if self._main_window:
                self._main_window.close()

            # QA5: Aggressive memory cleanup to prevent process persistence
            self.logger.info("Releasing service references...")

            # Clear service references to break potential circular refs
            if hasattr(self, '_tts_service'):
                self._tts_service = None
            if hasattr(self, '_config_manager'):
                self._config_manager = None
            if hasattr(self, '_voice_manager'):
                self._voice_manager = None
            if hasattr(self, '_audio_coordinator'):
                self._audio_coordinator = None
            if hasattr(self, '_whisper_service'):
                self._whisper_service = None
            if hasattr(self, '_main_window'):
                self._main_window = None

            # Clear services dict
            self._services.clear()

            # Force garbage collection multiple times to handle circular refs
            self.logger.info("Running garbage collection...")
            gc.collect()
            gc.collect()
            gc.collect()

            # Release CUDA memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    self.logger.info("CUDA cache cleared")
            except ImportError:
                pass
            except Exception as e:
                self.logger.warning(f"Error clearing CUDA cache: {e}")

            self.logger.info("Async cleanup complete")

        except Exception as e:
            self.logger.exception(f"Error during async cleanup: {e}")

    def _cleanup_services(self):
        """
        DEPRECATED: Use cleanup_async() instead.

        This synchronous cleanup creates a new event loop (the root cause of the bug).
        Use cleanup_async() which uses the same qasync loop that started the services.
        """
        self.logger.warning("_cleanup_services() is deprecated - use cleanup_async() instead")
        # Don't do anything - cleanup should happen via cleanup_async()

    @property
    def is_initialized(self) -> bool:
        """Check if the application is initialized."""
        return self._initialized

    def get_service(self, service_name: str) -> Optional[object]:
        """
        Get a service by name.

        Args:
            service_name (str): Name of the service

        Returns:
            Optional[object]: Service instance or None if not found
        """
        return self._services.get(service_name)

    def register_service(self, service_name: str, service: object):
        """
        Register a service with the application.

        Args:
            service_name (str): Name of the service
            service (object): Service instance
        """
        self._services[service_name] = service
        self.logger.debug(f"Registered service: {service_name}")

    def _run_async_task(self, coro, on_success=None, on_error=None):
        """
        Helper to run async tasks from sync Qt signal handlers.

        This replaces _async_manager.start_task() calls with direct asyncio.create_task().
        Uses the shared qasync event loop.

        NOTE: This should only be called from synchronous Qt signal handlers.
        From async contexts, use direct await or asyncio.create_task() instead.

        Args:
            coro: Coroutine to execute
            on_success: Optional callback for successful completion
            on_error: Optional callback for errors
        """
        async def _handle_task():
            try:
                result = await coro
                if on_success:
                    on_success(result)
            except Exception as e:
                self.logger.exception(f"Error in async task: {e}")
                if on_error:
                    on_error(e)

        # Create task in shared qasync loop
        # Use ensure_future which works both from sync and async contexts
        asyncio.ensure_future(_handle_task())

    def _on_text_generate_requested(self, text: str):
        """
        Handle text generation request from the main window.

        Args:
            text (str): Text to convert to speech
        """
        self.logger.info(f"TTS generation requested for text: {text[:50]}...")

        try:
            # Import VoiceType early for use throughout this method
            from myvoice.models.voice_profile import VoiceType

            # Update main window status
            if self._main_window:
                self._main_window.set_generation_status("Generating speech...", True)

            # Check if TTS service is available
            if not hasattr(self, '_tts_service') or not self._tts_service.is_running():
                if self._main_window:
                    self._main_window.set_generation_status("TTS service not available", False)
                return

            # Get active voice profile with transcription
            active_profile = None
            if hasattr(self, '_voice_manager'):
                try:
                    # Get the currently active voice profile
                    active_profile = self._voice_manager.get_active_profile()
                    if active_profile:
                        # Bundled and Embedding voices use virtual paths - check by voice_type or path prefix
                        # Note: Path("bundled://X") becomes "bundled:\X" on Windows
                        # Note: Path("embedding://X") becomes "embedding:\X" on Windows
                        is_virtual_voice = (
                            active_profile.voice_type == VoiceType.BUNDLED or
                            active_profile.voice_type == VoiceType.EMBEDDING or
                            str(active_profile.file_path).startswith("bundled:") or
                            str(active_profile.file_path).startswith("embedding:")
                        )
                        if not is_virtual_voice and (not active_profile.file_path or not active_profile.file_path.exists()):
                            self.logger.warning(f"Voice file does not exist: {active_profile.file_path}")
                            active_profile = None
                    else:
                        self.logger.info("No active voice profile selected")
                except Exception as e:
                    self.logger.error(f"Error getting active voice profile: {e}")

            # Fallback: Look for any voice profile in voice manager
            if not active_profile and hasattr(self, '_voice_manager'):
                profiles = self._voice_manager.get_valid_profiles()
                if profiles:
                    # Use the first available profile
                    profile_name = next(iter(profiles))
                    active_profile = profiles[profile_name]
                    self.logger.info(f"Using fallback voice profile: {profile_name}")

            if not active_profile:
                if self._main_window:
                    self._main_window.set_generation_status("No voice profile available", False)
                return

            # Get emotion instruct from UI (Story 3.2: FR8, Story 5.3: FR35)
            # EmotionButtonGroup provides instruct string for Qwen3-TTS
            # Cloned voices do not support emotion control
            emotion_instruct = None
            if self._main_window:
                try:
                    # Story 5.3: Check if voice type supports emotion before getting instruct
                    if active_profile.voice_type and active_profile.voice_type.supports_emotion:
                        emotion_instruct = self._main_window.get_emotion_instruct()
                        if emotion_instruct:
                            self.logger.debug(f"Using emotion instruct from UI: {emotion_instruct}")
                        else:
                            self.logger.debug("Using neutral emotion (no instruct)")
                    else:
                        # Cloned voice - no emotion support (Story 5.3: FR35)
                        self.logger.info(f"Voice type '{active_profile.voice_type}' does not support emotion, skipping instruct")
                except Exception as e:
                    self.logger.warning(f"Error getting emotion instruct from UI: {e}, using neutral")

            # Log the voice profile being used
            self.logger.info(f"TTS generation using voice profile: {active_profile.name}")
            if active_profile.transcription:
                self.logger.debug(f"Using transcription: {active_profile.transcription[:50]}...")
            else:
                self.logger.warning(f"No transcription available for voice profile: {active_profile.name}")

            # Log emotion instruct (Story 3.2: FR8, Story 5.3: FR35)
            # Note: Voice cloning does NOT support emotion control in Qwen3-TTS
            if emotion_instruct:
                self.logger.info(f"Emotion instruct requested: '{emotion_instruct}' (ignored for voice clone)")
            elif active_profile.voice_type and not active_profile.voice_type.supports_emotion:
                self.logger.info(f"TTS request for cloned voice '{active_profile.name}' (no emotion support)")
            else:
                self.logger.info("TTS request using neutral emotion")

            # Start TTS generation using appropriate model based on voice type
            # QA-1: Use correct model for each voice type
            # Log voice type for debugging
            self.logger.info(f"[DEBUG] Voice type: {active_profile.voice_type} (value={active_profile.voice_type.value if active_profile.voice_type else 'None'})")
            self.logger.info(f"[DEBUG] file_path: {active_profile.file_path}")
            self.logger.info(f"[DEBUG] VoiceType.EMBEDDING = {VoiceType.EMBEDDING}, comparison = {active_profile.voice_type == VoiceType.EMBEDDING}")

            # Check for bundled voice (by type - most reliable)
            # Note: Path("bundled://X") becomes "bundled:\X" on Windows, so check type first
            is_bundled = active_profile.voice_type == VoiceType.BUNDLED

            if is_bundled:
                # BUNDLED voices use CustomVoice model with speaker timbre
                # Speaker name is stored in the profile name
                speaker_name = active_profile.name

                self.logger.info(f"Using CustomVoice model with speaker: {speaker_name}")
                self._run_async_task(
                    self._tts_service.generate_custom_voice(
                        text=text,
                        speaker=speaker_name,
                        instruct=emotion_instruct,
                    ),
                    on_success=self._on_tts_generation_complete,
                    on_error=self._on_tts_generation_failed
                )

            elif active_profile.voice_type == VoiceType.DESIGNED:
                # DESIGNED voices can be saved as Prompt (VoiceDesign) or Clone (Base)
                # Check if transcription contains a voice description (prompt voice)
                voice_description = active_profile.transcription
                if voice_description:
                    # Prompt Voice - use VoiceDesign model with description
                    self.logger.info(f"Using VoiceDesign model with description: {voice_description[:50]}...")
                    self._run_async_task(
                        self._tts_service.generate_voice_design(
                            text=text,
                            voice_description=voice_description,
                            instruct=emotion_instruct,
                        ),
                        on_success=self._on_tts_generation_complete,
                        on_error=self._on_tts_generation_failed
                    )
                else:
                    # Clone Voice (saved from design) - use Base model with x-vector mode
                    # Designed voices saved as clone don't have transcription, use x_vector_only_mode
                    self.logger.info(f"Using Base model for designed voice (clone type) with x-vector mode")
                    self._run_async_task(
                        self._tts_service.generate_voice_clone(
                            text=text,
                            ref_audio=active_profile.file_path,
                            ref_text="",
                            x_vector_only_mode=True,
                        ),
                        on_success=self._on_tts_generation_complete,
                        on_error=self._on_tts_generation_failed
                    )

            elif active_profile.voice_type == VoiceType.OPTIMIZED:
                # OPTIMIZED voices use fine-tuned checkpoint with CustomVoice generation
                # These voices support emotion presets just like bundled voices
                checkpoint_path = active_profile.checkpoint_path
                speaker_name = active_profile.speaker_name

                if not checkpoint_path:
                    self.logger.error(f"Optimized voice '{active_profile.name}' missing checkpoint path")
                    if self._main_window:
                        self._main_window.set_generation_status("Optimized voice checkpoint not configured", False)
                    return

                if not speaker_name:
                    self.logger.error(f"Optimized voice '{active_profile.name}' missing speaker name")
                    if self._main_window:
                        self._main_window.set_generation_status("Optimized voice speaker name not configured", False)
                    return

                self.logger.info(f"Using fine-tuned checkpoint: {checkpoint_path} with speaker: {speaker_name}")
                self._run_async_task(
                    self._tts_service.generate_optimized_voice(
                        text=text,
                        checkpoint_path=checkpoint_path,
                        speaker_name=speaker_name,
                        instruct=emotion_instruct,
                    ),
                    on_success=self._on_tts_generation_complete,
                    on_error=self._on_tts_generation_failed
                )

            elif active_profile.voice_type == VoiceType.EMBEDDING:
                # EMBEDDING voices use Base model with pre-computed voice clone prompt
                # Created in Voice Design Studio - Base model does NOT support emotion/instruct
                embedding_path = active_profile.get_embedding_path()

                if not embedding_path:
                    self.logger.error(f"Embedding voice '{active_profile.name}' missing embedding path")
                    if self._main_window:
                        self._main_window.set_generation_status("Embedding file not found", False)
                    return

                self.logger.info(f"[DEBUG] EMBEDDING PATH TAKEN for '{active_profile.name}'")
                self.logger.info(f"[DEBUG] embedding_path={embedding_path}, checkpoint_path={active_profile.checkpoint_path}")
                self.logger.info(f"Using embedding voice: {active_profile.name} from {embedding_path}")

                # Emotion Variants: Get current emotion for EMBEDDING voices
                # The emotion ID is used to select the correct embedding subfolder
                emotion_id = None
                if self._main_window:
                    try:
                        emotion_preset = self._main_window.get_emotion_preset()
                        emotion_id = emotion_preset.id
                        self.logger.debug(f"Using emotion variant: {emotion_id}")
                    except Exception as e:
                        self.logger.warning(f"Error getting emotion preset: {e}, using neutral")
                        emotion_id = "neutral"

                self._run_async_task(
                    self._tts_service.generate_with_embedding(
                        text=text,
                        embedding_path=embedding_path,
                        emotion=emotion_id,
                        instruct=emotion_instruct,
                    ),
                    on_success=self._on_tts_generation_complete,
                    on_error=self._on_tts_generation_failed
                )

            else:
                # CLONED voices use Base model with reference audio
                # Use stored transcription for ICL mode (better quality) or x_vector mode if none
                ref_text = active_profile.transcription or ""
                use_xvector = not bool(ref_text)

                if ref_text:
                    self.logger.info(f"Using Base model for cloned voice with ICL mode (transcription available)")
                else:
                    self.logger.info(f"Using Base model for cloned voice with x-vector mode (no transcription)")

                self._run_async_task(
                    self._tts_service.generate_voice_clone(
                        text=text,
                        ref_audio=active_profile.file_path,
                        ref_text=ref_text,
                        x_vector_only_mode=use_xvector,
                    ),
                    on_success=self._on_tts_generation_complete,
                    on_error=self._on_tts_generation_failed
                )

        except Exception as e:
            self.logger.exception(f"Error during TTS generation: {e}")
            if self._main_window:
                self._main_window.set_generation_status(f"Generation failed: {str(e)}", False)

    def _on_cancel_generation_requested(self):
        """
        Handle user-initiated Stop click — cancel generation and/or
        stop playback, whichever is in flight.

        Story 11.4 follow-up wiring: the main window's Clear button
        doubles as a Stop button while either generation or playback
        is active. This handler fires both intents unconditionally:

          * ``QwenTTSService.cancel_generation()`` — propagates
            ``task.cancel()`` into the running asyncio task and posts
            the ``cancel`` + ``discard`` registry mutations (Story 11.4
            F1 fix). No-op when the service is idle.
          * ``AudioCoordinator.stop_all_playback()`` — stops every
            active monitor + virtual playback task. No-op when nothing
            is playing.

        Both calls are idempotent under "nothing to stop", so we do
        not need to disambiguate which state the UI was actually in.

        UI-flip note: cancel-during-generation flips back to idle via
        the ``_run_async_task`` ``on_error`` chain (CancelledError →
        ``_on_tts_generation_failed`` → ``set_generation_status(...,
        False)``). Cancel-during-playback has no analogous callback
        because ``stop_playback`` marks tasks as failed and the worker
        thread exits without firing ``_playback_complete_callback``.
        We therefore flip ``set_playback_active(False)`` synchronously
        here so the button reverts immediately on click; the async
        stops complete in the background.
        """
        self.logger.info("Cancel/stop requested")
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                self.logger.error("No asyncio event loop available for cancel")
                return

            if hasattr(self, '_tts_service') and self._tts_service is not None:
                # Fire-and-forget — the signal handler is synchronous,
                # but the cancel coroutine has its own UI-flip lifecycle.
                asyncio.ensure_future(
                    self._tts_service.cancel_generation(), loop=loop
                )
            else:
                self.logger.debug("Cancel requested but TTS service not available")

            if (
                hasattr(self, '_audio_coordinator')
                and self._audio_coordinator is not None
            ):
                asyncio.ensure_future(
                    self._audio_coordinator.stop_all_playback(), loop=loop
                )
                # Story 17.3 — Phase ⊥-Polish: when a progressive-playback
                # session is open (TRUE_STREAM/SENTENCE_STREAM mid-stream),
                # stop_all_playback's batch-mode targets do not include the
                # streaming PyAudio session. Fire stop_streaming_session()
                # additively so the open stream closes within ~50ms (one
                # chunk's buffer drain). Ordered AFTER the streamer-cancel
                # chain (Story 16.5's streamer._cancel_event.set() fires
                # synchronously inside cancel_generation above) so the
                # producer stops before the consumer.
                #
                # Always bump the epoch so any chunk that the producer
                # already scheduled via run_coroutine_threadsafe before
                # the cancel-event reached it sees a stale captured value
                # under the handler lock and drops itself instead of
                # opening a fresh session post-cancel. Bumping
                # unconditionally (not just when the flag is True) covers
                # the boundary case where chunk 0 was queued but had not
                # yet executed when cancel arrived (flag still False).
                self._progressive_playback_epoch += 1
                if self._progressive_playback_active:
                    asyncio.ensure_future(
                        self._audio_coordinator.stop_streaming_session(),
                        loop=loop,
                    )
                    # Clear the flag so a subsequent generation re-opens
                    # a fresh session via the chunk-0 callback path.
                    # cancel_generation cancels the asyncio task →
                    # CancelledError propagates through _run_async_task's
                    # on_error chain → _on_audio_playback_error fires;
                    # _play_generated_audio is NOT invoked for the
                    # cancelled generation, so the assembled buffer is
                    # dropped via the CancelledError path.
                    self._progressive_playback_active = False
            else:
                self.logger.debug("Stop requested but audio coordinator not available")

            # Story 12.1 Task 3.7: cancel-during-playback registry close.
            # qwen_tts_service.cancel_generation only posts cancel+discard
            # when _generation_state is GENERATING or STREAMING. A session
            # that has reached PLAYING (generation done, audio dispatched)
            # is past that gate, so cancel_generation is a no-op and the
            # registry session would otherwise stay in PLAYING forever.
            # Post cancel+discard explicitly here for PLAYING sessions.
            # State-guarded so we don't double-post over the
            # qwen_tts_service paths (which fire for GENERATING/STREAMING
            # via the except-CancelledError handlers).
            if (
                hasattr(self, '_session_registry')
                and self._session_registry is not None
            ):
                focal_id = self._session_registry.focal_session_id
                if focal_id is not None:
                    session = self._session_registry.get(focal_id)
                    if session is not None and session.state.value == "playing":
                        self._session_registry.post_mutation('cancel', focal_id)
                        self._session_registry.post_mutation('discard', focal_id)
                        # Story 13.2 follow-up — defend against the cancel-
                        # then-natural-complete race. If the audio worker
                        # finishes the buffer before stop_all_playback's
                        # cancellation propagates, _playback_complete_callback
                        # fires on the now-discarded session and
                        # _on_playback_complete posts mark_done → KeyError on
                        # _guard_and_lookup. Adding focal_id to
                        # _closed_session_ids makes the post-cancel callback
                        # a no-op (the existing dual-fire dedup uses the same
                        # set; this just extends the same guard to the
                        # cancel-vs-natural-complete race). Latent since
                        # Story 12.1; surfaced during Story 13.2 manual
                        # acceptance because the queue makes cancel timing
                        # more deterministic.
                        self._closed_session_ids.add(focal_id)

            # Story 13.2 AC #9: cancel-during-playback does NOT fire the
            # audio coordinator's playback-complete callback (per the
            # docstring above: "stop_playback marks tasks as failed and
            # the worker thread exits without firing
            # _playback_complete_callback"). The PlaybackQueue would
            # therefore stay stuck on the cancelled session as the head.
            # Advance the queue explicitly here. We are on the Qt main
            # thread (the cancel signal is delivered via Qt slot), so we
            # can call cancel_current and _dispatch_next_pending directly
            # rather than going through QMetaObject.invokeMethod.
            if (
                self._playback_queue is not None
                and self._dispatching_session_id is not None
            ):
                cancelled_token = self._dispatching_session_id
                self._playback_queue.cancel_current()
                # Drop any pending dispatch keyed on the cancelled token
                # so it doesn't replay later (the session has been told
                # to stop — playing it would defy the user's intent).
                self._pending_dispatches.pop(cancelled_token, None)
                self._dispatching_session_id = None
                self._dispatch_next_pending()
                # Story 13.2 follow-up — extend the cancel-then-natural-
                # complete race protection (see focal-cancel block above)
                # to the queue token. _closed_session_ids covers
                # registry-tracked sessions; _advanced_replay_tokens
                # covers synthetic replay tokens. The focal-cancel block
                # already added focal_id to _closed_session_ids when
                # focal_id == _dispatching_session_id (the common case);
                # adding cancelled_token here is idempotent for that case
                # AND covers the (rare) case where the dispatching token
                # is a registry-tracked session distinct from focal, OR
                # a replay token (focal-cancel skips replay because the
                # focal isn't in PLAYING state for replay dispatches).
                if cancelled_token.startswith("replay-"):
                    self._advanced_replay_tokens.add(cancelled_token)
                else:
                    self._closed_session_ids.add(cancelled_token)

            # Synchronous UI flip — see docstring. Safe regardless of
            # which (or no) state was active; ``set_playback_active``
            # is idempotent.
            if self._main_window is not None:
                self._main_window.set_playback_active(False)
                self._main_window.set_generation_status("Stopped", False)

        except Exception as e:
            self.logger.exception(f"Error handling cancel request: {e}")

    def _on_replay_last_requested(self):
        """
        Handle replay last audio request (Story 2.4: FR28, FR29, FR31, FR32).

        Replays the last generated audio from cache, routing through both
        monitor and virtual microphone.
        """
        self.logger.info("Replay last audio requested")

        try:
            # Check if TTS service has cached audio
            if not hasattr(self, '_tts_service') or not self._tts_service:
                if self._main_window:
                    self._main_window.set_generation_status("TTS service not available", False)
                return

            cached_path = self._tts_service.get_cached_audio_path()
            if not cached_path or not cached_path.exists():
                self.logger.warning("No cached audio available for replay")
                if self._main_window:
                    self._main_window.set_generation_status("No audio to replay", False)
                return

            # Read the cached audio file
            audio_data = cached_path.read_bytes()
            self.logger.info(f"Replaying cached audio: {len(audio_data)} bytes from {cached_path}")

            # Update status
            if self._main_window:
                self._main_window.set_generation_status("Replaying last audio...", False)

            # Play using the same dual-stream playback (FR31, FR32)
            if self._audio_coordinator:
                # Story 12.1 L3 (closed by Story 12.3, 2026-05-04): replay
                # deliberately skips registry mutations — by the time replay
                # fires there is no in-flight READY_TO_PLAY session, so
                # _play_generated_audio's focal snapshot returns None and the
                # dispatch runs without registry coupling. Don't "fix" this
                # without first reading the H4 long-term follow-up in Story
                # 12.1 (thread session_id through QwenTTSResponse).
                self._run_async_task(
                    self._play_generated_audio(audio_data),
                    on_success=self._on_replay_success,
                    on_error=self._on_replay_error
                )
            else:
                self.logger.warning("Audio coordinator not available for replay")
                if self._main_window:
                    self._main_window.set_generation_status("Audio coordinator not available", False)

        except Exception as e:
            self.logger.exception(f"Error during replay: {e}")
            if self._main_window:
                self._main_window.set_generation_status(f"Replay failed: {str(e)}", False)

    def _on_save_requested(self) -> None:
        """Story 14.3: open Save dialog when MainWindow.save_requested fires.

        Each click constructs a fresh SaveAudioDialog instance; the previous
        dialog (if any) has already returned (immediate save) or is in its
        streaming-wait state and owns its own cleanup. We do NOT track or
        cancel pending dialogs — the registry's signal flow is the source
        of truth for save lifecycle.
        """
        if self._session_registry is None:
            self.logger.warning("Save requested but no session registry; ignoring")
            return
        dialog = SaveAudioDialog(
            registry=self._session_registry,
            parent=self._main_window,
        )
        dialog.run()

    def _on_replay_success(self, result):
        """Handle successful replay playback."""
        self.logger.info("Replay playback completed successfully")
        if self._main_window:
            self._main_window.set_generation_status("Replay complete", False)

    def _on_replay_error(self, error):
        """Handle replay playback failure."""
        self.logger.error(f"Replay playback failed: {error}")
        if self._main_window:
            self._main_window.set_generation_status(f"Replay failed: {error}", False)

    # ----- Story 15.2: Clear Comms click → dispatch (OFR-B) ----------- #
    # The companion exception ``_ClearCommsResolveError`` lives at module
    # scope (just above ``MyVoiceApp``), per Python convention for
    # module-private helper classes. The AC's "near the slot" wording is
    # interpreted as "in the same module", which is the closest you can
    # get without nesting the class inside ``MyVoiceApp`` (which would
    # break the integration test's ``from myvoice.app import
    # _ClearCommsResolveError`` import).

    def _on_clear_comms_requested(self) -> None:
        """Handle a click on the Clear Comms button (Story 15.2, OFR-B).

        Three-step flow per AC #7:
          1. Resolve the audio bytes from the configured source (saveable
             session's buffer or a user-uploaded WAV file via 15.1's loader).
          2. If interrupt mode (D-18 default), stop active playback and
             advance the queue without cancelling in-flight generation.
          3. Dispatch the WAV bytes through the existing replay-token path
             — same dispatch shape as Replay Last, no registry coupling.

        D-5 invariant (architecture line 247-248) is satisfied trivially:
        no PRELOADED ``GenerationSession`` is registered for the playback,
        so the saveable slot cannot be advanced and ``saveable_session_changed``
        is never re-emitted as a side effect of this click.
        """
        if self._app_settings is None:
            self.logger.warning("Clear Comms clicked but no app settings; ignoring")
            return
        settings = self._app_settings
        queue_mode = bool(getattr(settings, 'clear_comms_queue_mode', False))
        source_kind = str(getattr(settings, 'clear_comms_source_kind', 'last_generation'))

        # Step 1: resolve audio bytes (encode to WAV bytes if from numpy buffer).
        try:
            wav_bytes = self._resolve_clear_comms_wav_bytes(source_kind, settings)
        except _ClearCommsResolveError as exc:
            self.logger.warning("Clear Comms resolve failed: %s", exc.user_message)
            if self._main_window is not None:
                self._main_window.set_generation_status(exc.user_message, False)
            return
        if wav_bytes is None:
            # Defensive: the button should have been disabled in this state.
            self.logger.debug("Clear Comms resolve returned None; ignoring click")
            return

        # Step 2: interrupt active playback if interrupt mode (D-18 default).
        if not queue_mode:
            self._interrupt_active_playback_for_clear_comms()

        # Step 3: dispatch via the existing replay-token path (no registry coupling).
        self._run_async_task(
            self._play_generated_audio(wav_bytes),
            on_success=self._on_clear_comms_success,
            on_error=self._on_clear_comms_error,
        )

    def _resolve_clear_comms_wav_bytes(
        self, source_kind: str, settings
    ) -> bytes:
        """Resolve the configured Clear Comms source to WAV bytes (AC #7).

        Two branches:
          - ``"last_generation"``: read ``SessionRegistry.saveable_audio``;
            re-encode its float32 buffer to int16 PCM WAV bytes per D-16.
          - ``"file"``: load the configured path via Story 15.1's loader,
            which guarantees mono float32 + WAV-only enforcement; encode
            to bytes via the same path. We round-trip rather than calling
            ``Path.read_bytes()`` directly so the loader's mono-downmix and
            WAV-only checks are still enforced (and so PreloadedAudioLoadError
            messages reach the user via a single uniform surface).

        Raises:
            _ClearCommsResolveError: short user-facing message; the caller
                routes it to ``MainWindow.set_generation_status``.
        """
        if source_kind == "last_generation":
            registry = self._session_registry
            if registry is None:
                raise _ClearCommsResolveError("No saveable audio")
            audio = registry.saveable_audio
            if audio is None:
                # Symmetric wording with save_dialog._TOAST_NO_SAVEABLE
                # (note the em-dash, not an ASCII hyphen — they must match
                # character-for-character so reviewers grepping the codebase
                # find both surfaces).
                raise _ClearCommsResolveError(
                    "No saveable audio — generation may have been cancelled"
                )
            return self._encode_wav_bytes(audio.complete_audio, audio.sample_rate)

        if source_kind == "file":
            path_str = getattr(settings, 'clear_comms_file_path', None)
            if not path_str:
                raise _ClearCommsResolveError("No Clear Comms file configured")
            from myvoice.ui.dialogs.settings.clear_comms_settings_panel import (
                load_preloaded_audio_source,
                PreloadedAudioLoadError,
            )
            try:
                audio_array, sample_rate = load_preloaded_audio_source(Path(path_str))
            except PreloadedAudioLoadError as exc:
                raise _ClearCommsResolveError(exc.message)
            return self._encode_wav_bytes(audio_array, sample_rate)

        # Defensive default — the AppSettings validator auto-corrects unknown
        # source_kind values to "last_generation", so this branch is mostly
        # defensive against direct in-memory mutation between validate and click.
        raise _ClearCommsResolveError(f"Unknown Clear Comms source: {source_kind}")

    @staticmethod
    def _encode_wav_bytes(audio, sample_rate: int) -> bytes:
        """Encode a float32 mono numpy array to int16 PCM WAV bytes.

        D-16 (architecture line 280): "Save format = WAV PCM_16. Float32 is
        the in-memory format; saved WAVs are int16 to match the user's
        DAW import expectations and to keep file sizes reasonable." Same
        conversion math as ``save_dialog.SaveAudioDialog._write_wav`` —
        not extracted to a shared helper because the two functions write
        to different sinks (``Path`` vs ``BytesIO``), and the conversion
        body is six lines of obvious math.
        """
        import io
        import numpy as np
        import soundfile as sf

        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        sf.write(buf, audio_int16, sample_rate, format='WAV', subtype='PCM_16')
        return buf.getvalue()

    def _interrupt_active_playback_for_clear_comms(self) -> None:
        """Stop active playback + advance the queue (AC #8).

        Strict subset of ``_on_cancel_generation_requested``'s playback-
        stop block. **Divergence from the Stop button (read carefully):**

          - We do NOT call ``tts_service.cancel_generation()``. Clear
            Comms interrupts *playback*, not *generation*. If the user
            is mid-generation, the in-flight ``GENERATING`` session
            keeps generating and will play after the Clear Comms clip
            finishes. This is intentional and load-bearing — a future
            reader must NOT "fix" the asymmetry.
          - We do NOT flip ``set_playback_active(False)`` /
            ``set_generation_status("Stopped", ...)`` on the window.
            The Clear Comms clip is about to play; the window state
            advances naturally as that dispatch fires.

        Safe to call when nothing is currently playing — the inner
        ``is None`` and ``state`` guards make every branch a defensive
        no-op (tested at TestClearCommsDispatch::test_idle_no_op).
        """
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                self.logger.error("No asyncio event loop available for Clear Comms interrupt")
                return

            if (
                hasattr(self, '_audio_coordinator')
                and self._audio_coordinator is not None
            ):
                asyncio.ensure_future(
                    self._audio_coordinator.stop_all_playback(), loop=loop
                )

            # Cancel the focal session if it is in PLAYING (mirrors the
            # focal-cancel block in _on_cancel_generation_requested at
            # app.py:1065-1089). Story 15.2 review fix M5: compare against
            # the SessionState enum directly rather than the .value string,
            # so a future case-change in the enum value (drift to
            # "Playing" or similar) cannot silently break this branch.
            if (
                hasattr(self, '_session_registry')
                and self._session_registry is not None
            ):
                from myvoice.services.sessions.generation_session import SessionState
                focal_id = self._session_registry.focal_session_id
                if focal_id is not None:
                    session = self._session_registry.get(focal_id)
                    if session is not None and session.state == SessionState.PLAYING:
                        self._session_registry.post_mutation('cancel', focal_id)
                        self._session_registry.post_mutation('discard', focal_id)
                        self._closed_session_ids.add(focal_id)

            # Vacate the queue head past the cancelled session (mirrors
            # the queue-advance block at app.py:1101-1128) — but
            # **deliberately omit ``_dispatch_next_pending()``** here.
            # Rationale (Story 15.2 review fix H3): the Stop button can
            # safely advance the queue because nothing follows it in the
            # caller; Clear Comms's Step 3 is about to dispatch the
            # PRELOADED WAV through the same code path, so the freed slot
            # must be claimed by Clear Comms — not by some pre-existing
            # entry in ``_pending_dispatches``. If we called
            # ``_dispatch_next_pending()`` here, a queued session-B would
            # take the slot and Clear Comms would park behind it,
            # contradicting D-18 "interrupt by default."
            if (
                self._playback_queue is not None
                and self._dispatching_session_id is not None
            ):
                cancelled_token = self._dispatching_session_id
                self._playback_queue.cancel_current()
                self._pending_dispatches.pop(cancelled_token, None)
                self._dispatching_session_id = None
                if cancelled_token.startswith("replay-"):
                    self._advanced_replay_tokens.add(cancelled_token)
                else:
                    self._closed_session_ids.add(cancelled_token)

        except Exception as e:
            self.logger.exception(f"Error interrupting playback for Clear Comms: {e}")

    def _on_clear_comms_success(self, _result):
        """Story 15.2: Clear Comms playback completed successfully."""
        self.logger.info("Clear Comms playback completed successfully")

    def _on_clear_comms_error(self, error):
        """Story 15.2: Clear Comms playback failed."""
        self.logger.error(f"Clear Comms playback failed: {error}")
        if self._main_window:
            self._main_window.set_generation_status(
                f"Clear Comms playback failed: {error}", False
            )

    # ----- Story 15.3: Test Playback (panel-supplied config) ---------- #

    def _on_clear_comms_test_playback_requested(
        self,
        source_kind: str,
        file_path: Optional[str],
        queue_mode: bool,
    ) -> None:
        """Test Playback (Story 15.3, AC #5).

        Strict subset of ``_on_clear_comms_requested`` (Story 15.2) that
        uses the *panel-supplied* config (``source_kind``, ``file_path``,
        ``queue_mode``) instead of ``self._app_settings.*``. This is what
        lets the user preview before clicking OK in the SettingsDialog.

        Divergence from the real Clear Comms click (read carefully): we
        ALWAYS skip the interrupt step. Test Playback is a preview, not
        a callout — it must not abort an in-flight generation. The
        ``queue_mode`` parameter is plumbed for completeness (and so a
        future "test the queue path" UX has a hook) but in v1 it is
        informational; preview enqueues behind whatever is currently
        playing, full stop.
        """
        if self._app_settings is None:
            self.logger.warning(
                "Test Playback clicked but no app settings; ignoring"
            )
            return

        # Build a minimal settings shim so _resolve_clear_comms_wav_bytes
        # reads the panel-supplied values without us having to mutate
        # self._app_settings (which would leak into the persisted JSON if
        # the user then clicked OK; but we want Cancel to be lossless).
        # Inline class avoids a module-level export of a dataclass that
        # nothing else needs to know about.
        class _PanelSettingsShim:
            def __init__(self, fp: Optional[str]) -> None:
                self.clear_comms_file_path = fp

        shim = _PanelSettingsShim(file_path)

        try:
            wav_bytes = self._resolve_clear_comms_wav_bytes(source_kind, shim)
        except _ClearCommsResolveError as exc:
            self.logger.warning(
                "Test Playback resolve failed: %s", exc.user_message
            )
            if self._main_window is not None:
                self._main_window.set_generation_status(exc.user_message, False)
            return

        # Test Playback NEVER interrupts (deliberate divergence from
        # _on_clear_comms_requested). The user is configuring; they
        # should not lose their in-flight generation just to preview.

        self._run_async_task(
            self._play_generated_audio(wav_bytes),
            on_success=self._on_clear_comms_test_playback_success,
            on_error=self._on_clear_comms_test_playback_error,
        )

    def _on_clear_comms_test_playback_success(self, _result):
        """Story 15.3: Test Playback completed successfully."""
        self.logger.debug("Clear Comms Test Playback completed successfully")

    def _on_clear_comms_test_playback_error(self, error):
        """Story 15.3: Test Playback failed."""
        self.logger.debug(f"Clear Comms Test Playback failed: {error}")

    def _on_voice_changed(self, voice_name: str):
        """
        Handle voice selection change from the main window.

        Args:
            voice_name (str): Selected voice profile name
        """
        self.logger.info(f"Voice changed to: {voice_name}")

        # Update active profile in voice manager
        if hasattr(self, '_voice_manager') and voice_name:
            self._run_async_task(
                self._voice_manager.set_active_profile(voice_name),
                on_success=self._on_voice_profile_set,
                on_error=self._on_voice_profile_error
            )

        # Update configuration with new voice selection
        if hasattr(self, '_config_manager'):
            self._run_async_task(
                self._config_manager.update_voice_selection(voice_name),
                on_success=self._on_voice_selection_saved,
                on_error=self._on_voice_selection_error
            )

    def _on_voice_profile_set(self, success):
        """Callback when voice profile is set in voice manager."""
        try:
            if success:
                self.logger.info(f"Active profile set successfully")

                # Update voice label in main window
                if self._main_window and self._voice_manager:
                    active_profile = self._voice_manager.get_active_profile()
                    if active_profile:
                        self._main_window.update_voice_label(active_profile.name)

                        # Story 3.3: FR8a, Story 5.3: FR35 - Enable/disable emotion based on voice type
                        # Emotion Variants: EMBEDDING voices have per-emotion control
                        from myvoice.models.voice_profile import VoiceType

                        if active_profile.voice_type == VoiceType.EMBEDDING:
                            # Emotion Variants: Enable only available emotions
                            available_emotions = active_profile.get_available_emotions()
                            self._main_window.update_voice_emotions(
                                available_emotions,
                                active_profile.name
                            )
                            self.logger.debug(
                                f"Emotion controls updated for EMBEDDING voice: {active_profile.name} "
                                f"(available: {available_emotions})"
                            )
                        elif active_profile.voice_type and active_profile.voice_type.supports_emotion:
                            # BUNDLED/DESIGNED/OPTIMIZED: Enable all emotions
                            self._main_window.set_emotion_enabled(True)
                            self.logger.debug(
                                f"Emotion controls enabled for voice profile: {active_profile.name} "
                                f"(type: {active_profile.voice_type})"
                            )
                        else:
                            # CLONED: Disable emotion controls
                            self._main_window.set_emotion_enabled(False)
                            self.logger.debug(
                                f"Emotion controls disabled for voice profile: {active_profile.name} "
                                f"(type: {active_profile.voice_type})"
                            )

                        # Preload model if voice group changed (different model required)
                        # This ensures smoother switching between Clone/Bundled/Design voices
                        if hasattr(self, '_tts_service') and self._tts_service:
                            required_model = self._voice_manager.get_active_profile_model_type()
                            if required_model:
                                current_model = self._tts_service.get_current_model_type()
                                if current_model != required_model:
                                    self.logger.info(
                                        f"Voice group changed: {current_model.display_name if current_model else 'None'} "
                                        f"-> {required_model.display_name}, preloading model..."
                                    )
                                    # Preload in background for smoother first generation
                                    self._run_async_task(
                                        self._tts_service.preload_model(required_model),
                                        on_success=lambda s: self.logger.info(
                                            f"Model {required_model.display_name} preloaded: {'success' if s[0] else s[1]}"
                                        ) if isinstance(s, tuple) else self.logger.info(
                                            f"Model {required_model.display_name} preloaded"
                                        ),
                                        on_error=lambda e: self.logger.warning(f"Failed to preload model: {e}")
                                    )
            else:
                self.logger.warning(f"Failed to set active profile")
        except Exception as e:
            self.logger.error(f"Error in voice profile set callback: {e}")

    def _on_voice_profile_error(self, error):
        """Callback when voice profile setting fails."""
        try:
            self.logger.error(f"Error setting active profile: {error}")
        except Exception as e:
            self.logger.error(f"Error in voice profile error callback: {e}")

    def _on_voice_selection_saved(self, success):
        """Callback when voice selection is saved to config."""
        try:
            if success:
                self.logger.debug(f"Voice selection saved successfully")
            else:
                self.logger.warning(f"Failed to save voice selection")
        except Exception as e:
            self.logger.error(f"Error in voice selection saved callback: {e}")

    def _on_voice_selection_error(self, error):
        """Callback when voice selection save fails."""
        try:
            self.logger.error(f"Failed to save voice selection: {error}")
        except Exception as e:
            self.logger.error(f"Error in voice selection error callback: {e}")

    def _on_settings_requested(self):
        """Handle settings request from the main window."""
        self.logger.debug("Settings requested")

        # TODO: Open settings dialog when settings UI is implemented
        if self._main_window:
            self._main_window.set_generation_status("Settings not yet implemented", False)

    def _on_generation_complete(self, message: str):
        """
        Handle completion of TTS generation.

        Args:
            message (str): Completion message
        """
        self.logger.debug(f"Generation complete: {message}")

        if self._main_window:
            self._main_window.set_generation_status(message, False)

    def _on_tts_service_started(self, result):
        """Handle TTS service startup completion."""
        self.logger.info("TTS service started successfully")

        # Schedule deferred TTS status update using QTimer
        # This ensures the main window is fully initialized before updating
        def update_tts_status():
            if self._main_window and hasattr(self, '_tts_service') and self._tts_service:
                self._main_window.add_service_monitoring("TTS")

                # Set TTS service on main window for voice creation dialogs
                self._main_window.set_tts_service(self._tts_service)

                # Get current health status and update UI
                # QwenTTSService is running if start() succeeded, so assume HEALTHY
                from myvoice.models.ui_state import ServiceHealthStatus
                health_status = ServiceHealthStatus.HEALTHY if self._tts_service.is_running() else ServiceHealthStatus.ERROR
                self._on_tts_health_status_changed(health_status, None)
                self.logger.info(f"Updated TTS status in UI after window initialization: {health_status.value}")

        # Use QTimer to defer execution until event loop is ready and main window exists
        QTimer.singleShot(100, update_tts_status)

    def _on_tts_service_start_failed(self, error):
        """Handle TTS service startup failure."""
        self.logger.error(f"TTS service failed to start: {error}")
        if self._main_window:
            self._main_window.set_generation_status("TTS service failed to start", False)

            # Update UI with failed status
            status_info = ServiceStatusInfo(
                service_name="TTS",
                status=ServiceStatus.ERROR,
                health_status=ServiceHealthStatus.ERROR,
                last_check=datetime.now(),
                error_message=str(error)
            )
            self._main_window.update_service_status("TTS", status_info)

    def _on_tts_preparing_voice_message(self, message: Optional[str]):
        """Story 17.2 AC #4 — surface the lazy-precompute status on the TTS
        indicator. Called with a string on cache miss entry, with None on
        exit (success or failure). Cache hits never invoke this callback,
        so steady-state generations remain visually invisible.

        Implemented by re-emitting the indicator's last known status with
        the ``preparing_voice_message`` field set. Keeps the existing
        ``health_status``/``error_message`` semantics untouched (this is
        an additive UX hint, not a state mutation).
        """
        if not self._main_window:
            return
        # Reuse the indicator's current status if we have one cached;
        # otherwise synthesize HEALTHY+RUNNING (the precompute only fires
        # when the TTS service is up).
        current = None
        try:
            indicator = (
                self._main_window.get_service_indicator("TTS")
                if hasattr(self._main_window, "get_service_indicator")
                else None
            )
            if indicator is not None and hasattr(indicator, "get_current_status"):
                current = indicator.get_current_status()
        except Exception:
            current = None

        status_info = ServiceStatusInfo(
            service_name="TTS",
            status=current.status if current is not None else ServiceStatus.RUNNING,
            health_status=(
                current.health_status if current is not None
                else ServiceHealthStatus.HEALTHY
            ),
            last_check=datetime.now(),
            error_message=current.error_message if current is not None else None,
            uptime_seconds=current.uptime_seconds if current is not None else None,
            preparing_voice_message=message,
        )
        try:
            self._main_window.update_service_status("TTS", status_info)
        except Exception as exc:
            self.logger.warning(
                f"Failed updating TTS indicator with preparing-voice message: {exc}"
            )

    def _on_tts_health_status_changed(self, health_status: ServiceHealthStatus, error_message: Optional[str]):
        """
        Callback for TTS service health status changes.

        Args:
            health_status: Current health status
            error_message: Error message if unhealthy
        """
        self.logger.info(f"TTS health status callback received: {health_status.value}, main_window exists: {self._main_window is not None}")

        if not self._main_window:
            self.logger.warning("Main window not initialized yet, cannot update TTS status")
            return

        # Determine service status based on health
        if health_status == ServiceHealthStatus.HEALTHY:
            service_status = ServiceStatus.RUNNING
        elif health_status == ServiceHealthStatus.WARNING:
            service_status = ServiceStatus.DEGRADED
        else:
            service_status = ServiceStatus.ERROR

        # Create status info
        status_info = ServiceStatusInfo(
            service_name="TTS",
            status=service_status,
            health_status=health_status,
            last_check=datetime.now(),
            error_message=error_message
        )

        # Update UI
        self._main_window.update_service_status("TTS", status_info)
        self.logger.debug(f"TTS health status updated: {health_status.value}")

    async def _check_and_update_tts_health(self):
        """Perform initial TTS health check and update UI."""
        if not hasattr(self, '_tts_service') or not self._tts_service:
            return

        is_healthy, error = await self._tts_service.health_check()

        if is_healthy:
            health_status = ServiceHealthStatus.HEALTHY
            error_message = None
        else:
            health_status = ServiceHealthStatus.ERROR
            error_message = error.user_message if error else "Health check failed"

        # Trigger the callback to update UI
        self._on_tts_health_status_changed(health_status, error_message)

    def _on_config_service_started(self, result):
        """Handle configuration service startup completion."""
        self.logger.info("Configuration service started successfully")

        # Load application settings after configuration service is ready
        self._run_async_task(
            self._load_app_settings_on_startup(),
            on_success=self._on_settings_loaded,
            on_error=self._on_settings_load_failed
        )

    def _on_config_service_start_failed(self, error):
        """Handle configuration service startup failure."""
        self.logger.error(f"Configuration service failed to start: {error}")

    def _on_voice_service_started(self, result):
        """Handle voice profile service startup completion."""
        self.logger.info("Voice profile service started successfully")

        # Connect voice manager to main window if it exists
        if self._main_window and hasattr(self, '_voice_manager'):
            self._main_window.set_voice_manager(self._voice_manager)
            self.logger.debug("Connected voice manager to main window")

        # Schedule voice restoration as a coroutine to run AFTER initialization completes
        # This ensures the initialization async task is done before creating a new task
        if hasattr(self, '_config_manager') and hasattr(self, '_voice_manager'):
            # Use asyncio.create_task directly from the event loop (not from within a task)
            # Schedule it to run after a brief delay to ensure initialization is complete
            async def delayed_restore():
                await asyncio.sleep(0.5)  # Wait half second for init to complete
                try:
                    await self._restore_voice_selection_on_startup()
                    self._on_voice_restoration_complete(None)
                except Exception as e:
                    self._on_voice_restoration_failed(e)

            # Schedule the coroutine using QTimer + loop.create_task
            def schedule_restore():
                loop = asyncio.get_event_loop()
                loop.create_task(delayed_restore())
                self.logger.info("Voice restoration scheduled after initialization delay")

            QTimer.singleShot(500, schedule_restore)

    def _on_voice_service_start_failed(self, error):
        """Handle voice profile service startup failure."""
        self.logger.error(f"Voice profile service failed to start: {error}")

    def _on_whisper_service_started(self, result):
        """Handle Whisper service startup completion."""
        self.logger.info("Whisper service started successfully")

    def _on_whisper_service_start_failed(self, error):
        """Handle Whisper service startup failure."""
        self.logger.error(f"Whisper service failed to start: {error}")
        # Whisper service failure is not critical - app can continue without transcription

    def _on_whisper_init_requested(self):
        """
        Handle request to initialize whisper service on-demand (QA4).

        This is called when Voice Design Studio is opened and whisper_service
        is not yet available. Triggers async initialization.
        """
        self.logger.info("Whisper service initialization requested")

        if self._whisper_service is not None:
            self.logger.debug("Whisper service already initialized")
            return

        # Start initialization asynchronously
        self._run_async_task(
            self._initialize_whisper_service_on_demand(),
            on_success=self._on_whisper_init_completed,
            on_error=lambda error: self.logger.error(f"Whisper init failed: {error}")
        )

    def _on_whisper_init_completed(self, success: bool):
        """
        Handle whisper service initialization completion (QA4).

        Args:
            success: Whether initialization was successful
        """
        if success:
            self.logger.info("Whisper service initialized successfully for Voice Design Studio")
        else:
            self.logger.warning("Whisper service initialization failed")

    async def _initialize_whisper_service_on_demand(self):
        """
        Initialize Whisper service on-demand to avoid DLL conflicts with PyQt6.

        This method handles the import order issue where PyQt6 and LLVM libraries conflict.
        """
        if self._whisper_service is not None:
            return True  # Already initialized

        try:
            self.logger.info("Initializing Whisper service on-demand")

            # Always use WhisperSubprocessService to avoid DLL conflicts with PyQt6
            # This applies to both frozen PyInstaller apps and development environments
            # The subprocess isolation prevents Whisper's DLLs from conflicting with PyQt6
            from myvoice.services.whisper_subprocess import WhisperSubprocessService

            if getattr(sys, 'frozen', False):
                self.logger.info("Using WhisperSubprocessService (frozen app)")
            else:
                self.logger.info("Using WhisperSubprocessService (development)")

            self.logger.debug("Creating WhisperSubprocessService instance")
            self._whisper_service = WhisperSubprocessService()
            self.logger.debug(f"WhisperSubprocessService created, status: {self._whisper_service.status}")

            self.logger.debug("Registering whisper service")
            self.register_service("whisper", self._whisper_service)
            self.logger.debug("Whisper service registered")

            # Start the service
            self.logger.debug("Starting whisper service")
            await self._whisper_service.start()
            self.logger.debug(f"Whisper service started, status: {self._whisper_service.status}")

            # QA4: Propagate whisper service to MainWindow for Voice Design Studio transcription
            if self._main_window:
                self._main_window.set_whisper_service(self._whisper_service)
                self.logger.debug("Whisper service propagated to MainWindow")

            # Story 17.2: Propagate Whisper service to TTS so the lazy
            # CLONED-voice voice_clone_prompt precompute can transcribe
            # ref_audio when no .txt sidecar exists. Without this, the
            # precompute raises and the dispatch chain falls through to
            # SENTENCE_STREAM (NFR7); with it, TRUE_STREAM works first-try.
            if hasattr(self, "_tts_service") and self._tts_service:
                try:
                    self._tts_service.set_whisper_service(self._whisper_service)
                    self.logger.debug("Whisper service propagated to TTS")
                except Exception as exc:
                    self.logger.warning(
                        f"Failed propagating Whisper to TTS: {exc}"
                    )

            self.logger.info("Whisper service initialized successfully on-demand")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Whisper service on-demand: {e}", exc_info=True)
            self._whisper_service = None
            return False

    def _continue_transcription_after_init(self, voice_name: str, success: bool):
        """
        Continue transcription process after Whisper service initialization.

        Args:
            voice_name: Name of the voice to transcribe
            success: Whether initialization was successful
        """
        if not success:
            if self._main_window:
                self._main_window.show_service_notification(
                    "Transcription Unavailable",
                    "Failed to initialize transcription service",
                    "error"
                )
            return

        # Now proceed with transcription since Whisper is initialized
        try:
            self._proceed_with_transcription(voice_name)
        except Exception as e:
            self.logger.exception(f"Error proceeding with transcription: {e}")
            if self._main_window:
                self._main_window.show_service_notification(
                    "Transcription Error",
                    f"Failed to start transcription: {str(e)}",
                    "error"
                )

    def _on_whisper_init_failed(self, error):
        """Handle Whisper service initialization failure."""
        self.logger.error(f"Whisper initialization failed: {error}")
        if self._main_window:
            self._main_window.show_service_notification(
                "Transcription Unavailable",
                "Failed to initialize transcription service. This may be due to missing dependencies or system compatibility issues.",
                "error"
            )

    def _proceed_with_transcription(self, voice_name: str):
        """
        Proceed with transcription after all checks and initialization.

        Args:
            voice_name: Name of the voice profile to transcribe
        """
        try:
            # Check if voice manager is available
            if not hasattr(self, '_voice_manager'):
                if self._main_window:
                    self._main_window.show_service_notification(
                        "Transcription Unavailable",
                        "Voice manager is not available",
                        "error"
                    )
                return

            # Get the voice profile
            profiles = self._voice_manager.get_valid_profiles()
            if voice_name not in profiles:
                if self._main_window:
                    self._main_window.show_service_notification(
                        "Voice Not Found",
                        f"Voice profile '{voice_name}' not found",
                        "error"
                    )
                return

            voice_profile = profiles[voice_name]

            # Check if voice file exists
            if not voice_profile.file_path or not voice_profile.file_path.exists():
                if self._main_window:
                    self._main_window.show_service_notification(
                        "Voice File Missing",
                        f"Voice file not found for '{voice_name}'",
                        "error"
                    )
                return

            # Show status
            if self._main_window:
                self._main_window.show_service_notification(
                    "Transcription Started",
                    f"Generating transcription for '{voice_name}'...",
                    "info"
                )

            # Start transcription asynchronously
            self._run_async_task(
                self._transcribe_voice_file(voice_profile),
                on_success=lambda result: self._on_transcription_complete(voice_name, result),
                on_error=lambda error: self._on_transcription_failed(voice_name, error)
            )

        except Exception as e:
            self.logger.exception(f"Error proceeding with transcription: {e}")
            if self._main_window:
                self._main_window.show_service_notification(
                    "Transcription Error",
                    f"Failed to start transcription: {str(e)}",
                    "error"
                )

    async def _restore_voice_selection_on_startup(self):
        """Restore voice selection from configuration on application startup."""
        try:
            self.logger.info("Restoring voice selection from saved configuration")

            # Get the saved voice selection from configuration
            restored_profile = await self._config_manager.restore_voice_selection()

            if restored_profile:
                # Set the active profile in the voice manager
                success = await self._voice_manager.set_active_profile(restored_profile)
                if success:
                    self.logger.info(f"Successfully restored voice selection: {restored_profile}")
                    return restored_profile
                else:
                    self.logger.warning(f"Failed to set active profile: {restored_profile}")
                    return None
            else:
                self.logger.info("No voice profile to restore or voice file missing")
                return None

        except Exception as e:
            self.logger.exception(f"Error during voice selection restoration: {e}")
            raise

    async def _restore_voice_selection_on_startup_task(self):
        """Task wrapper for voice restoration with callbacks."""
        try:
            restored_profile = await self._restore_voice_selection_on_startup()
            self._on_voice_restoration_complete(restored_profile)
        except Exception as error:
            self._on_voice_restoration_failed(error)

    def _on_voice_restoration_complete(self, restored_profile):
        """Handle completion of voice selection restoration."""
        if restored_profile:
            self.logger.info(f"Voice selection restored successfully: {restored_profile}")
            # Update UI if available
            if self._main_window and hasattr(self._main_window, 'voice_selector'):
                # The voice selector will update automatically when voice manager's active profile changes
                pass
        else:
            self.logger.info("No voice selection to restore")

    def _on_voice_restoration_failed(self, error):
        """Handle failure of voice selection restoration."""
        self.logger.error(f"Voice selection restoration failed: {error}")
        # Continue without restored voice selection - not a critical failure

    def _on_transcription_requested(self, voice_name: str):
        """
        Handle transcription request from the voice selector.

        Args:
            voice_name: Name of the voice profile to transcribe
        """
        self.logger.info(f"Transcription requested for voice: {voice_name}")

        try:
            # Initialize Whisper service on-demand if needed
            if not self._whisper_service:
                if self._main_window:
                    self._main_window.show_service_notification(
                        "Transcription Starting",
                        "Initializing transcription service...",
                        "info"
                    )

                # Start initialization asynchronously
                self._run_async_task(
                    self._initialize_whisper_service_on_demand(),
                    on_success=lambda success: self._continue_transcription_after_init(voice_name, success),
                    on_error=lambda error: self._on_whisper_init_failed(error)
                )
                return  # Exit here and continue in callback

            # If Whisper is already initialized, proceed directly
            self._proceed_with_transcription(voice_name)

        except Exception as e:
            self.logger.exception(f"Error handling transcription request: {e}")
            if self._main_window:
                self._main_window.show_service_notification(
                    "Transcription Error",
                    f"Failed to start transcription: {str(e)}",
                    "error"
                )

    async def _transcribe_voice_file(self, voice_profile):
        """
        Transcribe a voice profile's audio file.

        Args:
            voice_profile: VoiceProfile instance to transcribe

        Returns:
            TranscriptionResult: Result of transcription
        """
        try:
            self.logger.info(f"Starting transcription for voice file: {voice_profile.file_path}")

            # Use Whisper service to transcribe the file
            result = await self._whisper_service.transcribe_file(
                file_path=voice_profile.file_path,
                language=None,  # Auto-detect language
                word_timestamps=False,
                temperature=0.0
            )

            return result

        except Exception as e:
            self.logger.error(f"Error during voice file transcription: {e}")
            raise

    def _on_transcription_complete(self, voice_name: str, result):
        """
        Handle successful transcription completion.

        Args:
            voice_name: Name of the voice that was transcribed
            result: TranscriptionResult object
        """
        try:
            self.logger.info(f"Transcription completed for '{voice_name}': {len(result.text)} characters")

            # Save transcription to file
            if hasattr(self, '_voice_manager'):
                profiles = self._voice_manager.get_valid_profiles()
                if voice_name in profiles:
                    voice_profile = profiles[voice_name]

                    # Create transcription file path
                    voice_file_path = voice_profile.file_path
                    transcription_file_path = voice_file_path.with_suffix('.txt')

                    # Save transcription
                    try:
                        with open(transcription_file_path, 'w', encoding='utf-8') as f:
                            f.write(result.text.strip())

                        self.logger.info(f"Transcription saved to: {transcription_file_path}")

                        # Refresh voice profiles to pick up the new transcription
                        self._run_async_task(
                            self._voice_manager.force_rescan(),
                            on_success=lambda _: self.logger.debug("Voice profiles refreshed after transcription"),
                            on_error=lambda error: self.logger.error(f"Failed to refresh after transcription: {error}")
                        )

                        # Show success notification
                        if self._main_window:
                            self._main_window.show_service_notification(
                                "Transcription Complete",
                                f"Successfully transcribed '{voice_name}' ({len(result.text)} characters)",
                                "info"
                            )

                    except Exception as e:
                        self.logger.error(f"Failed to save transcription: {e}")
                        if self._main_window:
                            self._main_window.show_service_notification(
                                "Transcription Save Failed",
                                f"Transcription completed but failed to save: {str(e)}",
                                "error"
                            )

        except Exception as e:
            self.logger.exception(f"Error handling transcription completion: {e}")

    def _on_transcription_failed(self, voice_name: str, error):
        """
        Handle transcription failure.

        Args:
            voice_name: Name of the voice that failed to transcribe
            error: Exception that occurred
        """
        self.logger.error(f"Transcription failed for '{voice_name}': {error}")

        if self._main_window:
            self._main_window.show_service_notification(
                "Transcription Failed",
                f"Failed to transcribe '{voice_name}': {str(error)}",
                "error"
            )

    # --------------------------------------------------------------- #
    # Story 17.3 — progressive-playback consumer (Phase ⊥-Polish)
    # --------------------------------------------------------------- #

    def _on_audio_chunk_ready(self, chunk) -> None:
        """Story 17.3 — synchronous trampoline for the TTS-service chunk
        callback.

        Fires from the StreamingDecoderWorker thread (TRUE_STREAM) or the
        SENTENCE_STREAM async generation context. Schedules
        ``_handle_progressive_chunk_async`` onto the orchestrator's
        ``self.loop`` via ``asyncio.run_coroutine_threadsafe`` so all
        AudioCoordinator calls run on the main event loop and so a slow
        consumer can never block the producer.

        Captures the current ``_progressive_playback_epoch`` at schedule
        time and threads it through to the handler so a chunk that crosses
        a cancel boundary (cancel handler increments the epoch) drops
        itself under the handler lock instead of accidentally opening a
        fresh session post-cancel.
        """
        try:
            loop = getattr(self, "loop", None)
            if loop is None or loop.is_closed():
                return
            captured_epoch = self._progressive_playback_epoch
            asyncio.run_coroutine_threadsafe(
                self._handle_progressive_chunk_async(chunk, captured_epoch),
                loop,
            )
        except Exception:
            self.logger.exception(
                "Failed to schedule progressive-playback chunk handler"
            )

    async def _handle_progressive_chunk_async(
        self, chunk, epoch: Optional[int] = None
    ) -> None:
        """Story 17.3 — process one progressive-playback ``AudioChunk``.

        State machine:
          - chunk 0 (or first chunk after a stale state) opens the audio
            device session via ``start_streaming_session(sample_rate=...)``
            and inspects the returned ``{"monitor", "virtual"}`` dict; if
            BOTH services failed to open (real PyAudio error — the
            coordinator catches the exception internally and returns
            None values, audio_coordinator.py:1070-1072), the flag stays
            False so ``_play_generated_audio`` falls through to the batch
            path (NFR7-style graceful degradation).
          - chunks 1..N convert PCM float32 → int16 bytes and call
            ``play_audio_chunk(bytes, is_final=chunk.is_final)``.
            SENTENCE_STREAM emits its last data chunk with ``is_final=True``
            AND real audio in ``audio_data`` (qwen_tts_service.py:3071-3082);
            TRUE_STREAM emits a synthetic terminal chunk with
            ``audio_data.size == 0`` (qwen_tts_service.py:3929-3951). Both
            paths are handled by writing ``audio_data`` first when present
            and then closing the session on ``is_final``.

        The flag is intentionally NOT cleared on ``is_final`` — clearing
        here would race the dispatch path on the asyncio loop ordering.
        ``_play_generated_audio`` consumes (clears) it inside the dispatch
        skip-branch so the two paths are unambiguously sequenced.

        Cancel-vs-chunk race: ``epoch`` is captured at trampoline-schedule
        time; the cancel handler increments
        ``self._progressive_playback_epoch`` so any chunk queued before
        the cancel sees a mismatch under the lock and is dropped — this
        prevents a stale chunk from opening a fresh session after a
        cancel. ``epoch=None`` (legacy direct-test callers) skips the
        check.
        """
        if self._progressive_playback_lock is None:
            self._progressive_playback_lock = asyncio.Lock()

        async with self._progressive_playback_lock:
            if (
                epoch is not None
                and epoch != self._progressive_playback_epoch
            ):
                # Stale chunk — queued before a cancel fired. Drop.
                return

            # Story 17.3 AC #5 — NFR7 fallback continuity: a chunk_index=0
            # while ``_progressive_playback_active`` is already True means
            # a fresh stream is starting on top of a stale session — the
            # canonical case is TRUE_STREAM raising mid-stream and
            # SENTENCE_STREAM taking over (variant (b): clean cut + fresh
            # restart). Close the stale session FIRST so the new session
            # can open cleanly. Audible effect: ~50-100ms gap between the
            # partial TRUE_STREAM audio and the SENTENCE_STREAM audio.
            if (
                chunk.chunk_index == 0
                and self._progressive_playback_active
                and self._audio_coordinator is not None
            ):
                try:
                    await self._audio_coordinator.stop_streaming_session()
                except Exception:
                    self.logger.warning(
                        "Stale progressive playback session close failed "
                        "during fallback restart (non-fatal)",
                        exc_info=True,
                    )
                self._progressive_playback_active = False

            if not self._progressive_playback_active:
                # Build #11 regression: only chunk_index == 0 legitimately
                # opens a session. A non-zero chunk arriving with the flag
                # cleared is stale — most commonly TRUE_STREAM's synthetic
                # terminal AudioChunk racing with ``_play_generated_audio``'s
                # skip-branch (the trampoline's chunk-handler future is
                # chained via ``run_coroutine_threadsafe`` from the worker
                # thread, while ``_play_generated_audio`` is scheduled via
                # ``ensure_future`` on the main thread; the chained future
                # ordering means dispatch can run before the terminal
                # handler). Opening a fresh session for that stale chunk
                # implicitly closes (via ``MonitorAudioService.start_
                # streaming_session``'s "close existing" prelude) the
                # PyAudio stream that may still be playing the user's
                # audio — observed on Win11 MME as audible chunk repeats
                # in the 11:08 / 11:10 / 11:11 entries of the bundled-
                # smoke log. Drop the stale chunk; if it's the terminal,
                # best-effort close any session that may still be open
                # (no-op if already closed).
                if chunk.chunk_index != 0:
                    if (
                        chunk.is_final
                        and chunk.audio_data.size > 0
                    ):
                        # SENTENCE_STREAM-shape stale terminal: the last
                        # chunk carries real audio AND is_final=True, and
                        # it arrived after the session was closed. Audio
                        # is lost. SENTENCE_STREAM normally cannot race
                        # because each chunk's callback is followed by
                        # ``await asyncio.sleep(0)`` in the producer's
                        # for-loop — flag this loudly if it ever happens
                        # so the race becomes visible.
                        self.logger.warning(
                            "Stale terminal AudioChunk with non-empty "
                            f"audio (chunk_index={chunk.chunk_index}); "
                            "audio dropped because the progressive "
                            "session was closed before this chunk "
                            "arrived. If observed in production, the "
                            "skip-branch <-> chunk-handler race needs "
                            "lock-serialization."
                        )
                    if (
                        chunk.is_final
                        and self._audio_coordinator is not None
                    ):
                        try:
                            # wait_for_drain=True — even on the stale-terminal
                            # branch the buffer should drain so any audio that
                            # got through plays out (Story 17.3 finalization-
                            # drain follow-up).
                            await (
                                self._audio_coordinator
                                .stop_streaming_session(wait_for_drain=True)
                            )
                        except Exception:
                            self.logger.warning(
                                "Stale terminal-chunk session close failed "
                                "(non-fatal)",
                                exc_info=True,
                            )
                    return
                if self._audio_coordinator is None:
                    return
                try:
                    open_result = (
                        await self._audio_coordinator.start_streaming_session(
                            sample_rate=chunk.sample_rate,
                            channels=1,
                            sample_width=2,
                        )
                    )
                except Exception:
                    self.logger.warning(
                        "Progressive playback session open raised; "
                        "falling back to batch playback",
                        exc_info=True,
                    )
                    self._progressive_playback_active = False
                    return

                # AudioCoordinator.start_streaming_session catches all
                # exceptions internally (audio_coordinator.py:1070-1072)
                # and returns the result dict with None values on failure,
                # so the inspect-the-dict path is the production failure
                # mode — the except above is defense-in-depth only.
                monitor_id = (
                    open_result.get("monitor") if open_result else None
                )
                virtual_id = (
                    open_result.get("virtual") if open_result else None
                )
                if monitor_id is None and virtual_id is None:
                    self.logger.warning(
                        "Progressive playback session open returned no "
                        f"active session (monitor={monitor_id}, "
                        f"virtual={virtual_id}); falling back to batch "
                        "playback"
                    )
                    self._progressive_playback_active = False
                    return

                self._progressive_playback_active = True
                self._progressive_playback_sample_rate = chunk.sample_rate
                self.logger.info(
                    "Progressive playback session opened: "
                    f"sample_rate={chunk.sample_rate}Hz, "
                    f"monitor_session={monitor_id}, "
                    f"virtual_session={virtual_id}"
                )

            # Write audio first when present. Covers SENTENCE_STREAM's
            # last data chunk (is_final=True with real audio_data) AND
            # TRUE_STREAM's data chunks (is_final=False). TRUE_STREAM's
            # synthetic terminal chunk has audio_data.size == 0 so this
            # branch is a no-op for it — its session close is handled
            # by the is_final block below.
            if chunk.audio_data.size > 0:
                audio_bytes = (
                    np.clip(chunk.audio_data, -1.0, 1.0) * 32767
                ).astype(np.int16).tobytes()
                try:
                    await self._audio_coordinator.play_audio_chunk(
                        audio_bytes, is_final=chunk.is_final
                    )
                except Exception:
                    self.logger.warning(
                        "Progressive playback chunk write failed (non-fatal)",
                        exc_info=True,
                    )
                # Story 18.1: per-chunk consumer-side metrics (wall-clock
                # ms — joinable by (session_id, chunk_index) against the
                # producer-side ``progressive_chunk_emit_ms``). Captured
                # AFTER play_audio_chunk returns so the arrival value
                # reflects PyAudio buffer-fill, not chunk arrival.
                # session_id passed through so the CSV stays joinable
                # when a single run captures multiple generations
                # (code-review pass M1).
                metrics.record(
                    "progressive_chunk_playback_arrival_ms",
                    time.time() * 1000.0,
                    session_id=chunk.session_id,
                    chunk_index=chunk.chunk_index,
                    is_final=chunk.is_final,
                    audio_data_size=int(chunk.audio_data.size),
                )
                if chunk.sample_rate > 0:
                    metrics.record(
                        "progressive_chunk_audio_duration_ms",
                        (chunk.audio_data.size / chunk.sample_rate)
                        * 1000.0,
                        session_id=chunk.session_id,
                        chunk_index=chunk.chunk_index,
                    )

            if chunk.is_final:
                try:
                    # wait_for_drain=True so the tail of the last chunk plays
                    # out cleanly before the PyAudio stream is closed
                    # (Story 17.3 finalization-drain follow-up; race surfaced
                    # by Story 18.3 dtype audit).
                    await self._audio_coordinator.stop_streaming_session(
                        wait_for_drain=True
                    )
                except Exception:
                    self.logger.warning(
                        "Progressive playback session close failed (non-fatal)",
                        exc_info=True,
                    )

    def _on_tts_generation_complete(self, response):
        """
        Handle TTS generation completion with dual-stream playback.

        Args:
            response: QwenTTSResponse object with audio data (numpy array)
        """
        if response.success:
            # Convert numpy array to WAV bytes for audio playback
            import io
            import soundfile as sf
            audio_bytes = None
            if response.audio_data is not None:
                buffer = io.BytesIO()
                sf.write(buffer, response.audio_data, response.sample_rate, format='WAV')
                audio_bytes = buffer.getvalue()
                self.logger.info(f"TTS generation completed successfully, {len(audio_bytes)} bytes")
            else:
                self.logger.warning("TTS generation succeeded but no audio data returned")
                if self._main_window:
                    self._main_window.set_generation_status("No audio data generated", False)
                return

            # Start dual-stream audio playback (monitor + virtual microphone)
            if self._audio_coordinator and audio_bytes:
                self.logger.debug("Starting audio playback task")
                self._run_async_task(
                    self._play_generated_audio(audio_bytes),
                    on_success=self._on_audio_playback_success,
                    on_error=self._on_audio_playback_error
                )
            else:
                self.logger.warning("Audio coordinator not available for playback")

            if self._main_window:
                self._main_window.set_generation_status("Speech generated successfully", False)
        else:
            self.logger.error(f"TTS generation failed: {response.error_message}")
            if self._main_window:
                self._main_window.set_generation_status(f"Generation failed: {response.error_message}", False)

    async def _play_generated_audio(
        self,
        audio_data: bytes,
        session_id: Optional[str] = None,
        _queue_token: Optional[str] = None,
    ):
        """
        Play generated TTS audio using AudioCoordinator dual-service architecture.

        This method routes audio through AudioCoordinator which manages both
        MonitorAudioService and VirtualMicrophoneService independently.

        Args:
            audio_data: WAV audio data from TTS generation
            session_id: Optional registry session id (Story 12.1). When None
                and a registry is available, the focal session is used as
                a load-bearing approximation per Open Question #2 approach
                (b) — the just-finalized session is the focal under tier-(b)
                of validation gap #1 at the moment dispatch begins.
            _queue_token: Internal — Story 13.2. Used by ``_dispatch_next_pending``
                to re-enter this method after the queue advances and the
                pending dispatch is being executed. The leading underscore
                signals "internal use only"; external callers (post-generation
                dispatch at ``app.py:_on_tts_generation_completed`` and the
                replay path at ``app.py:_on_replay_last_requested``) leave
                this as ``None`` and the queue_token is derived from
                ``session_id`` (or minted as a synthetic ``replay-XXXXXXXX``
                token for the replay path).
        """
        try:
            self.logger.info("Starting audio playback via AudioCoordinator")

            # Story 12.1 Task 3.6 — approach (b): if no explicit session_id
            # was threaded through (the response-side QwenTTSResponse does
            # not currently carry it; threading it would touch ~17 response-
            # construction sites in qwen_tts_service.py, exceeding the
            # story's call-site budget), snapshot focal_session_id now. The
            # session that just transitioned to READY_TO_PLAY is the focal
            # by validation gap #1 ordering.
            #
            # AI-Review H4 (2026-05-04): guard the snapshot with a state
            # check. Under back-to-back generation (a successor's
            # start_generation queued before this dispatch runs), focal
            # could be a sibling session in GENERATING — posting
            # mark_playing against it would raise InvalidSessionStateError
            # in the slot, leaving the actually-dispatching session
            # untracked. If the focal is not in READY_TO_PLAY, we leave
            # session_id None and proceed without registry coupling for
            # this dispatch; the leaked READY_TO_PLAY entry falls out of
            # focal-eligibility within the 5s decay. The proper fix is
            # to thread the id through QwenTTSResponse (approach a); this
            # is tracked as a Review Follow-up.
            if session_id is None and getattr(self, '_session_registry', None) is not None:
                from myvoice.services.sessions.generation_session import SessionState
                focal_id = self._session_registry.focal_session_id
                if focal_id is not None:
                    focal_session = self._session_registry.get(focal_id)
                    if (
                        focal_session is not None
                        and focal_session.state == SessionState.READY_TO_PLAY
                    ):
                        session_id = focal_id

            # Story 13.2 — Phase 3 (OFR-C): queue gating. The queue serializes
            # inter-session dispatch (P-8) so three rapid-fire generations
            # play in submission order rather than overlapping. Token
            # derivation and the head/re-entry decision are factored into
            # ``_derive_queue_token`` and ``_claim_queue_slot_or_defer`` so
            # the gating logic is testable in isolation (see
            # ``tests/integration/test_session_lifecycle.py::TestPlaybackQueueGatingHelpers``).
            #
            # _dispatch_started must be initialized BEFORE the gating block
            # so the ``finally`` clause can read it on the defer-path early
            # return (without this, the early ``return`` from the deferral
            # branch raises UnboundLocalError when finally reads the flag).
            # When the defer path takes the early return, the slot is
            # owned by a different token and ``_release_queue_slot_on_failure``
            # short-circuits via its token-mismatch guard.
            _dispatch_started = False
            queue_token = self._derive_queue_token(session_id, _queue_token)
            if not self._claim_queue_slot_or_defer(
                queue_token, audio_data, session_id
            ):
                return

            # Story 17.3 — Phase ⊥-Polish: if the just-completed generation
            # produced audio progressively (chunk callback opened a streaming
            # session and wrote each chunk to the audio device), the
            # assembled buffer has already played to the user's speakers.
            # Skip the batch ``play_dual_stream`` to avoid double-playback,
            # but release the queue slot so the next dispatch can advance
            # — without this release the queue would stay stuck on this
            # token because the dual-fire ``_on_playback_complete`` chain
            # never runs (no actual play_dual_stream/play_monitor_audio
            # call to drive it). The cached WAV file (written by
            # ``_save_audio_to_cache`` inside the dispatch chain) remains
            # the source of truth for Replay (Story 13.3) and
            # save-during-streaming (Story 14.3) is wired separately
            # through SessionRegistry chunk events — neither depends on
            # ``play_dual_stream`` actually firing.
            if self._progressive_playback_active:
                self.logger.info(
                    "Progressive playback already active; skipping batch "
                    f"dispatch (queue_token={queue_token})"
                )
                self._progressive_playback_active = False
                self._release_queue_slot_on_failure(queue_token)
                return

            # Get device preferences from settings
            monitor_device_id = (
                self._app_settings.monitor_device_id
                if self._app_settings and self._app_settings.monitor_device_id
                else None
            )

            virtual_device_id = (
                self._app_settings.virtual_microphone_device_id
                if self._app_settings and self._app_settings.virtual_microphone_device_id
                else None
            )

            self.logger.debug(f"Audio routing - Monitor: {monitor_device_id}, Virtual: {virtual_device_id}")

            # If virtual device is selected, ALWAYS route to BOTH monitor and virtual
            # This ensures you can hear the audio while it's also sent to the virtual mic
            if virtual_device_id:
                self.logger.info(f"[WIN11-DEBUG] Dual-stream mode activated with virtual_device_id={virtual_device_id}")

                # Find device objects using smart matching with metadata
                monitor_device = None
                virtual_device = None

                # Get windows_audio_client for smart matching
                audio_client = None
                if (hasattr(self._audio_coordinator, 'monitor_service') and
                    hasattr(self._audio_coordinator.monitor_service, 'windows_audio_client')):
                    audio_client = self._audio_coordinator.monitor_service.windows_audio_client
                    self.logger.info(f"[WIN11-DEBUG] Audio client available for smart matching: {audio_client is not None}")
                else:
                    self.logger.warning("[WIN11-DEBUG] Audio client not available - smart device matching will not work")

                # Use smart device matching for monitor device
                if monitor_device_id and audio_client and hasattr(audio_client, 'find_device_by_metadata'):
                    # Log metadata being used for smart matching
                    monitor_metadata = {
                        'device_id': monitor_device_id,
                        'device_name': self._app_settings.monitor_device_name if self._app_settings else None,
                        'host_api_name': self._app_settings.monitor_device_host_api if self._app_settings else None
                    }
                    self.logger.info(f"Attempting monitor device smart matching with metadata: {monitor_metadata}")

                    monitor_device = audio_client.find_device_by_metadata(
                        device_id=monitor_device_id,
                        device_name=self._app_settings.monitor_device_name if self._app_settings else None,
                        host_api_name=self._app_settings.monitor_device_host_api if self._app_settings else None
                    )

                    if monitor_device:
                        self.logger.info(f"[WIN11-DEBUG] SUCCESS: Found monitor device via smart matching: {monitor_device.name} (device_id={monitor_device.device_id})")
                        self.logger.info(f"[WIN11-DEBUG] DEVICE RESOLVED - Monitor: device_id={monitor_device.device_id}, name={monitor_device.name}")
                    else:
                        self.logger.warning(f"[WIN11-DEBUG] FAILED: Monitor device not found via smart matching. Metadata: {monitor_metadata}")
                        self.logger.warning("[WIN11-DEBUG] Will fall back to direct enumeration")
                elif monitor_device_id:
                    self.logger.info(f"[WIN11-DEBUG] Smart matching not available for monitor device, using direct enumeration fallback")
                    # Fallback to direct enumeration if smart matching not available
                    monitor_devices = await self._audio_coordinator.monitor_service.enumerate_monitor_devices()
                    self.logger.info(f"[WIN11-DEBUG] Enumerated {len(monitor_devices)} monitor devices for fallback matching")
                    for device in monitor_devices:
                        if device.device_id == monitor_device_id:
                            monitor_device = device
                            self.logger.info(f"[WIN11-DEBUG] Found monitor device via direct enumeration: {device.name}")
                            break
                    if not monitor_device:
                        self.logger.error(f"[WIN11-DEBUG] Monitor device not found in direct enumeration either. device_id={monitor_device_id}")
                else:
                    self.logger.warning(f"[WIN11-DEBUG] No monitor_device_id provided, monitor_device will be None")

                # Use smart device matching for virtual device
                if virtual_device_id and audio_client and hasattr(audio_client, 'find_device_by_metadata'):
                    # Log metadata being used for smart matching
                    virtual_metadata = {
                        'device_id': virtual_device_id,
                        'device_name': self._app_settings.virtual_microphone_device_name if self._app_settings else None,
                        'host_api_name': self._app_settings.virtual_microphone_device_host_api if self._app_settings else None
                    }
                    self.logger.info(f"Attempting virtual device smart matching with metadata: {virtual_metadata}")

                    virtual_device = audio_client.find_device_by_metadata(
                        device_id=virtual_device_id,
                        device_name=self._app_settings.virtual_microphone_device_name if self._app_settings else None,
                        host_api_name=self._app_settings.virtual_microphone_device_host_api if self._app_settings else None
                    )

                    if virtual_device:
                        self.logger.info(f"[WIN11-DEBUG] SUCCESS: Found virtual device via smart matching: {virtual_device.name} (device_id={virtual_device.device_id})")
                        self.logger.info(f"[WIN11-DEBUG] DEVICE RESOLVED - Virtual: device_id={virtual_device.device_id}, name={virtual_device.name}")
                    else:
                        self.logger.warning(f"[WIN11-DEBUG] FAILED: Virtual device not found via smart matching. Metadata: {virtual_metadata}")
                        self.logger.warning("[WIN11-DEBUG] Will fall back to direct enumeration")
                elif virtual_device_id:
                    self.logger.info(f"[WIN11-DEBUG] Smart matching not available for virtual device, using direct enumeration fallback")
                    # Fallback to direct enumeration if smart matching not available
                    virtual_devices = await self._audio_coordinator.virtual_service.enumerate_virtual_devices()
                    self.logger.info(f"[WIN11-DEBUG] Enumerated {len(virtual_devices)} virtual devices for fallback matching")
                    for device in virtual_devices:
                        if device.device_id == virtual_device_id:
                            virtual_device = device
                            self.logger.info(f"[WIN11-DEBUG] Found virtual device via direct enumeration: {device.name}")
                            break
                    if not virtual_device:
                        self.logger.error(f"[WIN11-DEBUG] Virtual device not found in direct enumeration either. device_id={virtual_device_id}")

                # CRITICAL FIX (Windows 11 dual audio routing):
                # If monitor_device is None after smart matching/direct enumeration,
                # fall back to system default to ensure dual-stream works properly.
                # Without this, only virtual mic plays and monitor stays silent.
                self.logger.info(f"[WIN11-DEBUG] Before fallback check: monitor_device={monitor_device}, virtual_device={virtual_device}")
                if not monitor_device:
                    self.logger.warning("[WIN11-DEBUG] ENTERING FALLBACK: Monitor device not found via smart matching, falling back to system default for dual-stream")
                    # Get default monitor device
                    monitor_devices = await self._audio_coordinator.monitor_service.enumerate_monitor_devices()
                    self.logger.info(f"[WIN11-DEBUG] Fallback enumerated {len(monitor_devices)} monitor devices")
                    if monitor_devices:
                        # Try to get Windows default output device first
                        if (hasattr(self._audio_coordinator.monitor_service, 'windows_audio_client') and
                            self._audio_coordinator.monitor_service.windows_audio_client):
                            default_device = self._audio_coordinator.monitor_service.windows_audio_client.get_default_output_device()
                            if default_device:
                                monitor_device = default_device
                                self.logger.info(f"[WIN11-DEBUG] FALLBACK SUCCESS: Using Windows default output device for dual-stream: {default_device.name}")
                            else:
                                self.logger.warning("[WIN11-DEBUG] get_default_output_device() returned None")

                        # Fallback to first available device if no default found
                        if not monitor_device:
                            monitor_device = monitor_devices[0]
                            self.logger.info(f"[WIN11-DEBUG] FALLBACK SUCCESS: Using first available monitor device for dual-stream: {monitor_device.name}")
                    else:
                        self.logger.error("[WIN11-DEBUG] FALLBACK FAILED: No monitor devices available for dual-stream playback")
                else:
                    self.logger.info(f"[WIN11-DEBUG] SKIPPING FALLBACK: monitor_device already set to {monitor_device.name if monitor_device else 'None'}")

                # CRITICAL: Windows 11 Device Collision Detection
                # Ensure monitor and virtual devices are NOT the same device.
                # If they resolve to the same device, force monitor to use Windows default.
                self.logger.info(f"[WIN11-DEBUG] Before collision check: monitor_device={monitor_device.device_id if monitor_device else 'None'}, virtual_device={virtual_device.device_id if virtual_device else 'None'}")
                if monitor_device and virtual_device:
                    if monitor_device.device_id == virtual_device.device_id:
                        self.logger.error(f"DEVICE COLLISION DETECTED: Monitor and virtual device resolved to SAME device!")
                        self.logger.error(f"Collision device: {monitor_device.name} (device_id={monitor_device.device_id})")
                        self.logger.warning("This is the root cause of Windows 11 dual-stream failure")
                        self.logger.warning("Forcing monitor to use Windows default output to fix collision")

                        # Force monitor to use Windows default output device
                        if (hasattr(self._audio_coordinator.monitor_service, 'windows_audio_client') and
                            self._audio_coordinator.monitor_service.windows_audio_client):
                            default_device = self._audio_coordinator.monitor_service.windows_audio_client.get_default_output_device()
                            if default_device and default_device.device_id != virtual_device.device_id:
                                monitor_device = default_device
                                self.logger.info(f"Monitor forced to Windows default: {monitor_device.name} (device_id={monitor_device.device_id})")
                            else:
                                self.logger.error("Windows default device is ALSO the virtual device or not found!")
                                # Last resort: try first available device that ISN'T the virtual device
                                monitor_devices = await self._audio_coordinator.monitor_service.enumerate_monitor_devices()
                                for device in monitor_devices:
                                    if device.device_id != virtual_device.device_id:
                                        monitor_device = device
                                        self.logger.info(f"Monitor forced to first non-virtual device: {monitor_device.name}")
                                        break
                    else:
                        self.logger.info(f"[WIN11-DEBUG] Device collision check PASSED: monitor_device={monitor_device.device_id}, virtual_device={virtual_device.device_id}")
                else:
                    self.logger.error(f"[WIN11-DEBUG] COLLISION CHECK SKIPPED! monitor_device={monitor_device}, virtual_device={virtual_device}")

                # Execute dual-stream routing through coordinator.
                # Story 12.1: pass session_id so the coordinator posts
                # mark_playing/mark_audible mutations to the registry.
                self.logger.info(f"[WIN11-DEBUG] Calling play_dual_stream with monitor_device={monitor_device.name if monitor_device else 'None'}, virtual_device={virtual_device.name if virtual_device else 'None'}")
                dual_result = await self._audio_coordinator.play_dual_stream(
                    audio_data=audio_data,
                    monitor_device=monitor_device,
                    virtual_device=virtual_device,
                    volume=1.0,
                    session_id=session_id,
                )

                if dual_result and dual_result.any_successful:
                    self.logger.info("Dual-stream playback started successfully (monitor + virtual mic)")
                    # Story 13.2: mark dispatch started — the dual-fire
                    # _on_playback_complete callbacks will drive the queue
                    # advance from here. If dispatch had failed before this
                    # point the finally-block would advance the queue.
                    _dispatch_started = True
                    # Story 12.1: map both task ids to the session id so
                    # _on_playback_complete can post mark_done+discard
                    # exactly once (the dual-fire dedup uses
                    # _closed_session_ids to absorb the second callback).
                    if session_id is not None:
                        if dual_result.monitor_task is not None:
                            self._task_to_session[dual_result.monitor_task.playback_id] = session_id
                        if dual_result.virtual_task is not None:
                            self._task_to_session[dual_result.virtual_task.playback_id] = session_id
                    else:
                        # Story 13.2: replay path — no registry session, but we
                        # still need to drive the queue advance exactly once
                        # despite the dual-fire callback. _task_to_replay_token
                        # gives _on_playback_complete the synthetic queue token
                        # to dedup against (_advanced_replay_tokens).
                        if dual_result.monitor_task is not None:
                            self._task_to_replay_token[dual_result.monitor_task.playback_id] = queue_token
                        if dual_result.virtual_task is not None:
                            self._task_to_replay_token[dual_result.virtual_task.playback_id] = queue_token
                    if self._main_window:
                        # Story 12.1: legacy substate path; registry-driven path now coexists
                        self._main_window.set_generation_status("Playing audio on speakers and virtual microphone", False)
                        # Story 11.4 follow-up: keep the Stop button live
                        # through playback so the user can interrupt audio.
                        # Story 12.1: legacy substate path; registry-driven path now coexists
                        self._main_window.set_playback_active(True)
                else:
                    self.logger.warning("Dual-stream playback failed")
                    if self._main_window:
                        self._main_window.set_generation_status("Audio playback failed", False)
                        # Story 12.1: legacy substate path; registry-driven path now coexists
                        self._main_window.set_playback_active(False)

            else:
                # Monitor speakers only (fallback)
                # Find the device object from device_id
                monitor_devices = await self._audio_coordinator.monitor_service.enumerate_monitor_devices()
                target_device = None

                # If monitor_device_id is None or empty, use default (first available device)
                if monitor_device_id:
                    for device in monitor_devices:
                        if device.device_id == monitor_device_id:
                            target_device = device
                            break

                    if not target_device:
                        self.logger.warning(f"Monitor device {monitor_device_id} not found, using default")

                # Fallback to first available monitor device if no specific device or device not found
                if not target_device and monitor_devices:
                    target_device = monitor_devices[0]
                    self.logger.info(f"Using default monitor device: {target_device.name}")

                # Story 12.1: post mark_playing on the monitor-only fallback
                # path mirroring play_dual_stream's behavior. This path
                # bypasses the coordinator's own posts because it calls
                # monitor_service.play_monitor_audio directly.
                #
                # AI-Review H2 (2026-05-04): defer the post until we have
                # a target device AND track whether it was posted so we
                # can clean up via set_error+discard on failure. The
                # previous unconditional post-before-dispatch leaked the
                # session in PLAYING state when no device was available
                # or play_monitor_audio returned None.
                mark_playing_posted = False
                if target_device:
                    if (
                        session_id is not None
                        and getattr(self, '_session_registry', None) is not None
                    ):
                        self._session_registry.post_mutation('mark_playing', session_id)
                        mark_playing_posted = True
                    monitor_task = await self._audio_coordinator.monitor_service.play_monitor_audio(
                        audio_data=audio_data,
                        device=target_device,
                        volume=1.0
                    )
                else:
                    self.logger.warning("No monitor devices available")
                    monitor_task = None

                if monitor_task:
                    self.logger.info("Monitor speakers playback started")
                    # Story 13.2: mark dispatch started — the monitor-only
                    # fallback fires _on_playback_complete the same way the
                    # dual-stream path does, so the queue advance is driven
                    # from there rather than the finally-block.
                    _dispatch_started = True
                    # Story 12.1: post mark_audible once the stream is
                    # running and map the task id back to session id for
                    # the playback-complete close path.
                    if mark_playing_posted:
                        self._session_registry.post_mutation('mark_audible', session_id)
                        self._task_to_session[monitor_task.playback_id] = session_id
                    elif session_id is None:
                        # Story 13.2: monitor-only fallback for the replay
                        # path. Map the task id to the synthetic token so
                        # the queue advance fires exactly once.
                        self._task_to_replay_token[monitor_task.playback_id] = queue_token
                    if self._main_window:
                        # Story 12.1: legacy substate path; registry-driven path now coexists
                        self._main_window.set_generation_status("Playing audio on speakers", False)
                        # Story 12.1: legacy substate path; registry-driven path now coexists
                        self._main_window.set_playback_active(True)
                else:
                    self.logger.warning("No audio devices available for playback")
                    # AI-Review H2: clean up the registry session if
                    # mark_playing was already queued.
                    if mark_playing_posted:
                        self._session_registry.post_mutation('set_error', session_id)
                        self._session_registry.post_mutation('discard', session_id)
                    if self._main_window:
                        self._main_window.set_generation_status("No audio devices available", False)
                        # Story 12.1: legacy substate path; registry-driven path now coexists
                        self._main_window.set_playback_active(False)

        except Exception as e:
            self.logger.error(f"Error during audio playback: {e}")
            if self._main_window:
                self._main_window.set_generation_status(f"Playback failed: {str(e)}", False)
                # Story 12.1: legacy substate path; registry-driven path now coexists
                self._main_window.set_playback_active(False)
        finally:
            # Story 13.2: failure-path cleanup. If we claimed the dispatch
            # slot but never reached play_dual_stream / play_monitor_audio
            # (no devices, exception, any_successful=False), the dual-fire
            # _on_playback_complete chain that would normally drive the
            # advance never runs, so we must release the slot ourselves.
            if not _dispatch_started:
                self._release_queue_slot_on_failure(queue_token)

    def _on_audio_playback_success(self, result):
        """Handle successful return from the playback START task.

        Note: this fires when ``_play_generated_audio`` returns — i.e.
        once ``play_dual_stream`` has dispatched the playback into the
        underlying audio services and yielded back. The audio is still
        playing at this point. The Stop-mode exit lives in
        ``_on_playback_complete``, which is wired to the audio
        coordinator's per-task completion callback.
        """
        self.logger.info("Audio playback dispatched successfully")
        if self._main_window:
            # Story 2.4: Enable replay button after successful playback (FR29)
            self._main_window.set_replay_enabled(True)

    def _on_audio_playback_error(self, error):
        """Handle audio playback error."""
        self.logger.error(f"Audio playback failed: {error}")
        if self._main_window:
            self._main_window.set_generation_status(f"Audio playback failed: {str(error)}", False)
            # Story 12.1: legacy substate path; registry-driven path now coexists
            self._main_window.set_playback_active(False)

    def _on_playback_complete(self, task_id: str):
        """Handle the audio coordinator's real playback-complete signal.

        Fires once per playback task — both the monitor-service task and
        the virtual-microphone task fire independently. We only need to
        flip the UI back to idle once, which is naturally idempotent
        because ``set_playback_active(False)`` is a no-op when already
        False. The logger output is kept terse to avoid log spam from
        the dual-task case.

        Story 11.4 follow-up: replaces the premature flip that used to
        live in ``_on_audio_playback_success``.

        Story 12.1: also posts ``mark_done`` + ``discard`` to close the
        registry session (Task 3.5). Dedup uses ``_closed_session_ids``
        because the dual-fire would otherwise post the close mutations
        twice for the same session. ``mark_done`` requires PLAYING per
        the transition graph, so a second post would raise
        ``InvalidSessionStateError`` — the dedup is load-bearing, not
        cosmetic.
        """
        self.logger.debug(f"Playback task complete: {task_id}")

        # Story 12.1: registry-side close path. Resolve session_id from the
        # mapping populated at dispatch time; if unknown, the session was
        # not registry-tracked (legacy path or replay) and we skip the posts.
        session_id = self._task_to_session.pop(task_id, None)
        # Story 13.2: parallel resolution for replay-path dispatches.
        # _task_to_replay_token only has entries for synthetic "replay-X"
        # tokens; pops to None for registry-tracked sessions and unknown
        # task ids alike.
        replay_token = self._task_to_replay_token.pop(task_id, None)

        # Story 13.2: drive the queue advance exactly once per session
        # despite the dual-fire callback. For registry-tracked sessions the
        # dedup is the existing _closed_session_ids set (load-bearing for
        # the registry close path too); for replay tokens it is the
        # _advanced_replay_tokens set. We compute should_advance here and
        # invoke the queue mutations once below the registry close block.
        should_advance_queue = False

        if (
            session_id is not None
            and session_id not in self._closed_session_ids
            and getattr(self, '_session_registry', None) is not None
        ):
            self._closed_session_ids.add(session_id)
            self._session_registry.post_mutation('mark_done', session_id)
            self._session_registry.post_mutation('discard', session_id)
            should_advance_queue = True
        elif (
            session_id is None
            and replay_token is not None
            and replay_token not in self._advanced_replay_tokens
        ):
            # Story 13.2: replay path — no registry mutations, but the queue
            # still needs to advance exactly once. The dedup set absorbs the
            # second dual-fire callback for the same replay token.
            self._advanced_replay_tokens.add(replay_token)
            should_advance_queue = True

        # Story 13.2: advance the queue cross-thread-safely. _on_playback_complete
        # fires from the audio worker thread (per
        # monitor_audio_service.py:786-790's _emit_playback_complete site
        # invoked from the playback worker), so direct mutation of the queue
        # would trigger PlaybackQueue._assert_owner_thread's RuntimeError
        # (D-2 / P-3). QMetaObject.invokeMethod with QueuedConnection
        # marshals the call onto the queue's owner thread (Qt main) via
        # the event loop; the cancel_current and _dispatch_next_pending
        # invocations execute in submission order on the next event-loop
        # drain (per Epic 11 retrospective Insight #3 — "queued before"
        # implies "fires before" once the loop drains).
        if should_advance_queue and self._playback_queue is not None:
            QMetaObject.invokeMethod(
                self._playback_queue,
                'cancel_current',
                Qt.ConnectionType.QueuedConnection,
            )
            QMetaObject.invokeMethod(
                self,
                '_dispatch_next_pending',
                Qt.ConnectionType.QueuedConnection,
            )

        if self._main_window:
            # Story 12.1: legacy substate path; registry-driven path now coexists
            self._main_window.set_playback_active(False)
            # Story 12.1: legacy substate path; registry-driven path now coexists
            self._main_window.set_generation_status("Audio playback completed", False)

    def _derive_queue_token(
        self,
        session_id: Optional[str],
        _queue_token: Optional[str],
    ) -> str:
        """Story 13.2: compute the queue token for a dispatch request.

        Three sources, in priority order:

          1. ``_queue_token`` — re-entry path from
             ``_dispatch_next_pending`` preserves the original token across
             the deferred-then-resumed dispatch (a fresh uuid would fail
             the re-entry guard in ``_claim_queue_slot_or_defer``).
          2. ``session_id`` — registry-tracked sessions use their session
             id verbatim as the queue token.
          3. Synthetic ``replay-XXXXXXXX`` — the registry-less replay
             path mints a uuid token so it participates in queue ordering
             without requiring registry state.

        Pure function (no side effects) — safe to call from any thread.
        Tested directly in
        ``TestPlaybackQueueGatingHelpers::test_derive_queue_token_*``.
        """
        if _queue_token is not None:
            return _queue_token
        if session_id is not None:
            return session_id
        return f"replay-{uuid.uuid4().hex[:8]}"

    def _claim_queue_slot_or_defer(
        self,
        queue_token: str,
        audio_data: bytes,
        session_id: Optional[str],
    ) -> bool:
        """Story 13.2: gate dispatch on the PlaybackQueue head.

        Returns ``True`` if the caller should proceed with dispatch
        (queue is absent, OR we are at the queue head and no other
        session is currently dispatching, OR this is a re-entry from
        ``_dispatch_next_pending``); ``False`` if the dispatch was
        parked in ``_pending_dispatches`` for later resumption.

        Re-entry from ``_dispatch_next_pending`` is detected via
        ``_dispatching_session_id == queue_token`` and short-circuits
        the enqueue — we already claimed the slot upstream. Without
        this guard the re-entry would loop: enqueue → check head →
        defer (non-empty queue) → park → … .

        Side effects when called: enqueues ``queue_token`` (unless
        re-entry), sets ``_dispatching_session_id = queue_token`` on
        success, parks ``_PendingDispatch`` in ``_pending_dispatches``
        on defer.

        Tested directly in
        ``TestPlaybackQueueGatingHelpers::test_claim_queue_slot_*``.
        """
        if self._playback_queue is None:
            return True
        is_reentry = (
            self._dispatching_session_id is not None
            and self._dispatching_session_id == queue_token
        )
        if is_reentry:
            return True
        self._playback_queue.enqueue(queue_token)
        # Head check + currently-dispatching check. We dispatch only if
        # (a) the queue head is this token AND (b) no other session is
        # in the middle of play_dual_stream. Depth alone cannot
        # distinguish "head is currently playing" from "head is waiting"
        # because the head occupies one slot in both cases.
        if (
            self._dispatching_session_id is not None
            or self._playback_queue.peek() != queue_token
        ):
            self._pending_dispatches[queue_token] = _PendingDispatch(
                audio_data=audio_data,
                session_id=session_id,
                queue_token=queue_token,
            )
            self.logger.debug(
                f"Story 13.2: dispatch deferred for token={queue_token} "
                f"(queue depth={self._playback_queue.depth}, "
                f"current={self._dispatching_session_id})"
            )
            return False
        self._dispatching_session_id = queue_token
        return True

    def _release_queue_slot_on_failure(self, queue_token: str) -> None:
        """Story 13.2: failure-path cleanup for ``_play_generated_audio``.

        Invoked from the method's ``finally`` block when dispatch
        claimed the slot (set ``_dispatching_session_id``) but never
        reached ``play_dual_stream`` / ``play_monitor_audio`` (no
        devices, exception, ``any_successful=False``). Without this
        cleanup the queue would stay stuck on this token because the
        dual-fire ``_on_playback_complete`` chain that normally drives
        the advance never runs.

        Safe to call unconditionally — a no-op when the queue is
        absent or this token is not the dispatching one. We are on the
        Qt main thread (qasync runs the coroutine body here), so direct
        queue calls are safe — no ``QMetaObject.invokeMethod``
        indirection required.

        Tested directly in
        ``TestPlaybackQueueGatingHelpers::test_release_queue_slot_*``.
        """
        if (
            self._playback_queue is None
            or self._dispatching_session_id != queue_token
        ):
            return
        self._dispatching_session_id = None
        self._playback_queue.cancel_current()
        self._pending_dispatches.pop(queue_token, None)
        self._dispatch_next_pending()

    @pyqtSlot()
    def _dispatch_next_pending(self) -> None:
        """Story 13.2: pull the next pending dispatch when the queue advances.

        Invoked via ``QMetaObject.invokeMethod(... QueuedConnection)`` from
        ``_on_playback_complete`` (worker-thread origin), so this slot
        always executes on the Qt main thread on the next event-loop drain
        — directly after the paired ``cancel_current`` invocation drained
        the head off the queue.

        The peek+pop sequence is safe-by-construction:
          * ``cancel_current`` ran before us (queued in order; Epic 11
            retro Insight #3) so the head is now the next session.
          * If ``peek()`` returns ``None``, the queue is empty (no pending
            dispatches either) — clear ``_dispatching_session_id`` and
            return.
          * If ``peek()`` returns a token but ``_pending_dispatches`` has
            no entry for it, the dispatch was never deferred — this is
            the "head was always the dispatching session" case and is a
            defensive no-op.
          * Otherwise we set ``_dispatching_session_id`` to the new head
            and re-enter ``_play_generated_audio`` via ``_run_async_task``.
            The re-entry guard inside ``_play_generated_audio`` (token ==
            ``_dispatching_session_id`` → ``is_reentry=True``) skips the
            enqueue and head check.
        """
        if self._playback_queue is None:
            self._dispatching_session_id = None
            return
        next_token = self._playback_queue.peek()
        if next_token is None:
            self._dispatching_session_id = None
            return
        pending = self._pending_dispatches.pop(next_token, None)
        if pending is None:
            # The new head was never deferred (its dispatch path took the
            # synchronous "head matches and no other dispatch in flight"
            # branch), so there is no audio to replay here. Defensive
            # no-op — leave _dispatching_session_id as it was set by the
            # synchronous path.
            return
        self._dispatching_session_id = next_token
        self._run_async_task(
            self._play_generated_audio(
                audio_data=pending.audio_data,
                session_id=pending.session_id,
                _queue_token=pending.queue_token,
            ),
            on_success=self._on_audio_playback_success,
            on_error=self._on_audio_playback_error,
        )

    @pyqtSlot(str)
    def _dispatch_streaming_session(self, session_id: str) -> None:
        """P-8 streaming exception (Story 13.2): dispatch a
        ``GENERATING + is_streaming=True`` session if the queue is empty.

        Architectural contract (architecture-optimization-pass.md, line 461):
        a session in the streaming substate may dispatch to the audio
        coordinator while still ``GENERATING`` *if and only if* the queue
        is empty. The streaming session counts as one queue slot, not zero.

        This entry point is hookable by Story 16.6 (TRUE_STREAM dispatch).
        It is currently unused — Epic 16 has not landed. The method exists
        to lock the integration contract: when streaming sessions become
        load-bearing, they call THIS, not ``_play_generated_audio``, to
        take the single queue slot allowed by P-8.

        TODO(Story 16.6): activate this path. Until then, calling this
        method records the queue slot but does not start streaming-chunk
        dispatch (no chunked plumbing exists pre-Epic-16). The defensive
        ``getattr(session, 'is_streaming', False)`` pattern referenced in
        the AC #6 implementation guidance is not exercised here because
        Story 13.2's ``_play_generated_audio`` path does not consult
        ``is_streaming`` at all.
        """
        if self._playback_queue is None:
            return
        if self._playback_queue.depth != 0:
            # Queue is non-empty — the streaming exception is denied.
            # The caller (Story 16.6) is expected to fall back to the
            # standard deferred path; this method does not auto-fall-back
            # because that would silently mask the policy decision.
            self.logger.debug(
                f"Story 13.2: streaming exception denied for {session_id} "
                f"(queue depth={self._playback_queue.depth}); caller must "
                f"fall back to deferred dispatch."
            )
            return
        # Queue is empty — take the single slot allowed by P-8.
        self._playback_queue.enqueue(session_id)
        self.logger.debug(
            f"[STREAMING-13.2] streaming dispatch slot reserved for "
            f"{session_id}; Epic 16 chunk-streaming plumbing is the "
            f"next integration point (Story 16.6)."
        )

    def _on_tts_generation_failed(self, error):
        """
        Handle TTS generation failure.

        Args:
            error: Exception that occurred during generation
        """
        self.logger.error(f"TTS generation failed with exception: {error}")
        if self._main_window:
            self._main_window.set_generation_status(f"Generation failed: {str(error)}", False)

    def _on_audio_coordinator_started(self, result):
        """Handle audio coordinator startup completion."""
        try:
            self.logger.info("Audio coordinator started successfully")

            # Connect audio coordinator to main window if it exists
            if self._main_window and hasattr(self, '_audio_coordinator'):
                self._main_window.set_audio_coordinator(self._audio_coordinator)
                self.logger.debug("Connected audio coordinator to main window")

            # Set up device change monitoring for runtime device updates
            if hasattr(self, '_audio_coordinator'):
                # Create task for device monitoring (fire-and-forget)
                asyncio.create_task(self._setup_device_change_monitoring_async())

            self.logger.debug("Audio coordinator startup callback completed")

        except Exception as e:
            self.logger.error(f"Error in audio coordinator started callback: {e}", exc_info=True)

    def _on_audio_coordinator_start_failed(self, error):
        """Handle audio coordinator startup failure."""
        self.logger.error(f"Audio coordinator failed to start: {error}")
        if self._main_window:
            self._main_window.set_generation_status("Audio system failed to start", False)

    async def _setup_mic_mixing_from_settings(self):
        """
        Setup microphone mixing based on current app settings.

        Called during initialization and when settings change to ensure
        mic mixing state matches the configuration.
        """
        if not hasattr(self, '_audio_coordinator') or not self._audio_coordinator:
            self.logger.debug("Audio coordinator not available, skipping mic setup")
            return

        if not hasattr(self, '_app_settings') or not self._app_settings:
            self.logger.debug("App settings not available, skipping mic setup")
            return

        try:
            mic_enabled = getattr(self._app_settings, 'mic_mixing_enabled', False)
            mic_device_id = getattr(self._app_settings, 'mic_input_device_id', None)
            # Correct setting name for virtual mic
            virtual_device_id = getattr(self._app_settings, 'virtual_microphone_device_id', None)

            self.logger.info(f"[MIC_DEBUG] ========================================")
            self.logger.info(f"[MIC_DEBUG] _setup_mic_mixing_from_settings called")
            self.logger.info(f"[MIC_DEBUG] mic_mixing_enabled: {mic_enabled}")
            self.logger.info(f"[MIC_DEBUG] mic_input_device_id: {mic_device_id}")
            self.logger.info(f"[MIC_DEBUG] virtual_microphone_device_id: {virtual_device_id}")
            self.logger.info(f"[MIC_DEBUG] mic_monitor_running: {self._audio_coordinator._mic_monitor_running}")
            self.logger.info(f"[MIC_DEBUG] continuous_passthrough_running: {self._audio_coordinator._continuous_passthrough_running}")
            if not mic_enabled:
                self.logger.info(f"[MIC_DEBUG] NOTE: Mic mixing is DISABLED - enable 'Enable Microphone Mixing' in settings")
            self.logger.info(f"[MIC_DEBUG] ========================================")

            if mic_enabled:
                # Ensure mic monitor to speakers is NOT running (that's a separate feature)
                self.logger.info("[MIC_DEBUG] Stopping any existing mic monitor...")
                await self._audio_coordinator.stop_mic_monitor_to_speakers()
                self.logger.info(f"[MIC_DEBUG] After stop: mic_monitor_running={self._audio_coordinator._mic_monitor_running}")

                # Enable mic mixing in coordinator
                self._audio_coordinator.enable_mic_mixing(True)

                # Get configured mic device
                mic_device_id = getattr(self._app_settings, 'mic_input_device_id', None)
                mic_device = None

                if mic_device_id:
                    # Find the device by ID
                    try:
                        mic_devices = await self._audio_coordinator.enumerate_mic_devices()
                        for device in mic_devices:
                            if device.device_id == mic_device_id:
                                mic_device = device
                                break
                    except Exception as e:
                        self.logger.warning(f"Error enumerating mic devices: {e}")

                # Start mic capture (uses default if no device specified)
                success = await self._audio_coordinator.start_mic_capture(mic_device)

                if success:
                    # Set mic volume from settings
                    mic_volume = getattr(self._app_settings, 'mic_volume', 1.0)
                    self._audio_coordinator.set_mic_volume(mic_volume)

                    # Start continuous passthrough for mic audio to virtual mic
                    passthrough_success = await self._audio_coordinator.start_continuous_mic_passthrough()
                    if passthrough_success:
                        self.logger.info(f"Mic mixing and passthrough started with volume {mic_volume:.2f}")
                    else:
                        self.logger.info(f"Mic mixing started (passthrough pending) with volume {mic_volume:.2f}")
                else:
                    self.logger.warning("Failed to start mic capture")
            else:
                # Disable mic mixing and stop passthrough
                await self._audio_coordinator.stop_continuous_mic_passthrough()
                await self._audio_coordinator.stop_mic_monitor_to_speakers()  # Ensure monitor is stopped
                self._audio_coordinator.enable_mic_mixing(False)
                await self._audio_coordinator.stop_mic_capture()
                self.logger.debug("Mic mixing and passthrough disabled")

        except Exception as e:
            self.logger.error(f"Error setting up mic mixing: {e}")

    def _on_settings_changed(self, new_settings):
        """
        Handle settings changes from the UI.

        Args:
            new_settings: Updated AppSettings instance
        """
        self.logger.info("Settings changed, saving and applying updates")

        try:
            # Check if model quality tier changed (before updating stored settings)
            old_tier = getattr(self._app_settings, 'model_quality_tier', 'quality') if self._app_settings else 'quality'
            new_tier = getattr(new_settings, 'model_quality_tier', 'quality')

            # Update stored settings
            self._app_settings = new_settings

            # Handle model quality tier change (no restart required)
            if old_tier != new_tier and hasattr(self, '_tts_service') and self._tts_service:
                self.logger.info(f"Model quality tier changed from '{old_tier}' to '{new_tier}'")
                self._run_async_task(
                    self._tts_service.set_quality_tier(new_tier),
                    on_success=lambda changed: self.logger.info(f"Quality tier updated: {'changed' if changed else 'no change'}"),
                    on_error=lambda error: self.logger.error(f"Failed to update quality tier: {error}")
                )

            # Update configuration manager's settings and save
            if hasattr(self, '_config_manager'):
                self._config_manager._settings = new_settings
                self._run_async_task(
                    self._config_manager.save_settings(),
                    on_success=lambda success: self.logger.debug(f"Settings saved: {success}"),
                    on_error=lambda error: self.logger.error(f"Failed to save settings: {error}")
                )

            # Update main window with new settings
            if self._main_window:
                self._main_window.update_settings(new_settings)

            # Update audio coordinator with new device settings
            if hasattr(self, '_audio_coordinator'):
                # Update app settings in coordinator
                self._audio_coordinator.app_settings = new_settings

                # Update both services with new device settings
                if new_settings.monitor_device_id:
                    self.logger.debug(f"Monitor audio device changed to: {new_settings.monitor_device_id}")

                if new_settings.virtual_microphone_device_id:
                    self.logger.debug(f"Virtual microphone device changed to: {new_settings.virtual_microphone_device_id}")

                # Apply settings to both services through coordinator
                self._run_async_task(
                    self._audio_coordinator.update_device_settings(new_settings),
                    on_success=lambda success: self.logger.info("Audio coordinator settings updated successfully"),
                    on_error=lambda error: self.logger.error(f"Failed to update audio coordinator settings: {error}")
                )

                # Handle mic mixing enable/disable based on settings
                self._run_async_task(
                    self._setup_mic_mixing_from_settings(),
                    on_success=lambda _: self.logger.debug("Mic mixing settings applied"),
                    on_error=lambda error: self.logger.error(f"Failed to apply mic mixing settings: {error}")
                )

        except Exception as e:
            self.logger.error(f"Error handling settings changes: {e}")

    def _on_device_refresh_requested(self):
        """Handle device refresh request from the UI."""
        self.logger.info("Device refresh requested")

        try:
            # Trigger device enumeration through audio coordinator
            if hasattr(self, '_audio_coordinator'):
                self._run_async_task(
                    self._audio_coordinator.enumerate_all_devices(),
                    on_success=self._on_device_refresh_complete,
                    on_error=self._on_device_refresh_failed
                )
        except Exception as e:
            self.logger.error(f"Error refreshing devices: {e}")

    def _on_device_refresh_complete(self, devices):
        """Handle completion of device refresh."""
        self.logger.info(f"Device refresh completed with {type(devices)} devices")

        # Update main window device list if settings dialog is open
        if self._main_window and hasattr(self._main_window, 'settings_dialog') and self._main_window.settings_dialog:
            # Handle dictionary structure from enumerate_all_devices()
            if isinstance(devices, dict):
                # Update monitor devices (for output)
                monitor_devices = devices.get("monitor", [])
                if monitor_devices:
                    self._main_window.settings_dialog.update_device_list(monitor_devices)
                    self.logger.info(f"Updated monitor device list with {len(monitor_devices)} devices")

                # Update virtual devices (for virtual microphone)
                virtual_devices = devices.get("virtual", [])
                self._main_window.settings_dialog.update_virtual_device_list(virtual_devices)
                self.logger.info(f"Updated virtual device list with {len(virtual_devices)} devices")

                # Update mic input devices (for microphone mixing)
                mic_devices = devices.get("mic", [])
                self._main_window.settings_dialog.update_mic_device_list(mic_devices)
                self.logger.info(f"Updated mic device list with {len(mic_devices)} devices")
            else:
                # Fallback for direct device list (backward compatibility)
                self._main_window.settings_dialog.update_device_list(devices)

    def _on_device_refresh_failed(self, error):
        """Handle device refresh failure."""
        self.logger.error(f"Device refresh failed: {error}")

    def _on_mic_device_refresh_requested(self):
        """Handle microphone input device refresh request from the UI."""
        self.logger.info("Microphone device refresh requested")

        try:
            if hasattr(self, '_audio_coordinator') and self._audio_coordinator:
                self._run_async_task(
                    self._audio_coordinator.enumerate_mic_devices(),
                    on_success=self._on_mic_device_refresh_complete,
                    on_error=self._on_mic_device_refresh_failed
                )
        except Exception as e:
            self.logger.error(f"Error refreshing mic devices: {e}")

    def _on_mic_device_refresh_complete(self, mic_devices):
        """Handle completion of microphone device refresh."""
        self.logger.info(f"Mic device refresh completed with {len(mic_devices) if mic_devices else 0} devices")

        # Update settings dialog mic device list
        if self._main_window and hasattr(self._main_window, 'settings_dialog') and self._main_window.settings_dialog:
            self._main_window.settings_dialog.update_mic_device_list(mic_devices or [])

    def _on_mic_device_refresh_failed(self, error):
        """Handle microphone device refresh failure."""
        self.logger.error(f"Mic device refresh failed: {error}")
        # Update UI with empty list on failure
        if self._main_window and hasattr(self._main_window, 'settings_dialog') and self._main_window.settings_dialog:
            self._main_window.settings_dialog.update_mic_device_list([])

    def _on_mic_monitor_toggled(self, enabled: bool, device_id: str):
        """
        Handle mic monitor toggle from settings dialog.

        Args:
            enabled: True to start monitoring, False to stop
            device_id: ID of the microphone device to monitor
        """
        self.logger.info(f"Mic monitor toggled: enabled={enabled}, device_id={device_id}")

        try:
            if not hasattr(self, '_audio_coordinator') or not self._audio_coordinator:
                self.logger.warning("Audio coordinator not available for mic monitor")
                return

            if enabled:
                # Find the device by ID
                async def start_monitor():
                    mic_device = None
                    if device_id:
                        try:
                            mic_devices = await self._audio_coordinator.enumerate_mic_devices()
                            for device in mic_devices:
                                if device.device_id == device_id:
                                    mic_device = device
                                    break
                        except Exception as e:
                            self.logger.warning(f"Error finding mic device: {e}")

                    success = await self._audio_coordinator.start_mic_monitor_to_speakers(mic_device)
                    if success:
                        self.logger.info("Mic monitor to speakers started")
                    else:
                        self.logger.warning("Failed to start mic monitor")
                        # Update UI to reflect failure
                        if self._main_window and hasattr(self._main_window, 'settings_dialog'):
                            dialog = self._main_window.settings_dialog
                            if dialog:
                                dialog.mic_monitor_checkbox.blockSignals(True)
                                dialog.mic_monitor_checkbox.setChecked(False)
                                dialog.mic_monitor_checkbox.blockSignals(False)
                                dialog._show_mic_status("Failed to start mic monitor", "error")

                self._run_async_task(start_monitor())
            else:
                # Stop monitoring
                self._run_async_task(
                    self._audio_coordinator.stop_mic_monitor_to_speakers(),
                    on_success=lambda _: self.logger.debug("Mic monitor stopped"),
                    on_error=lambda e: self.logger.error(f"Error stopping mic monitor: {e}")
                )

        except Exception as e:
            self.logger.error(f"Error handling mic monitor toggle: {e}")

    def _on_device_test_requested(self, device_id):
        """
        Handle device test request from the UI.

        Args:
            device_id: ID of the device to test
        """
        self.logger.info(f"Device test requested for device: {device_id}")

        try:
            # Run device test asynchronously to avoid blocking UI
            import asyncio

            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run test in the event loop
            if loop.is_running():
                # If loop is already running, schedule as a task
                asyncio.create_task(self._test_device_async(device_id))
            else:
                # If no loop running, run until complete
                loop.run_until_complete(self._test_device_async(device_id))

        except Exception as e:
            self.logger.error(f"Error testing device: {e}")
            if self._main_window and hasattr(self._main_window, 'settings_dialog') and self._main_window.settings_dialog:
                self._main_window.settings_dialog._show_device_status(f"Test failed: {str(e)}", "error")

    async def _test_device_async(self, device_id: str):
        """
        Test an audio device asynchronously.

        Args:
            device_id: ID of the device to test, or "default" for system default
        """
        try:
            if not self._audio_coordinator:
                if self._main_window and hasattr(self._main_window, 'settings_dialog') and self._main_window.settings_dialog:
                    self._main_window.settings_dialog._show_device_status("Audio coordinator not available", "error")
                return

            # Generate test tone (440Hz sine wave for 1 second)
            test_audio = self._generate_test_tone()

            # Enumerate monitor devices
            monitor_devices = await self._audio_coordinator.monitor_service.enumerate_monitor_devices()
            target_device = None

            # Handle "default" device selection
            if device_id == "default" or device_id is None:
                # Get Windows default output device through windows_audio_client
                default_device = None
                if hasattr(self._audio_coordinator.monitor_service, 'windows_audio_client') and \
                   self._audio_coordinator.monitor_service.windows_audio_client:
                    default_device = self._audio_coordinator.monitor_service.windows_audio_client.get_default_output_device()

                if default_device:
                    target_device = default_device
                    self.logger.info(f"Resolved default device to: {default_device.name}")
                elif monitor_devices:
                    # Fallback to first available device
                    target_device = monitor_devices[0]
                    self.logger.info(f"No default device found, using first available: {target_device.name}")
            else:
                # Find specific device by ID
                for device in monitor_devices:
                    if device.device_id == device_id:
                        target_device = device
                        break

            if not target_device:
                raise MyVoiceError(
                    severity=ErrorSeverity.ERROR,
                    code="DEVICE_NOT_FOUND",
                    user_message=f"Monitor device {device_id} not found"
                )

            # Show testing status
            if self._main_window and hasattr(self._main_window, 'settings_dialog') and self._main_window.settings_dialog:
                self._main_window.settings_dialog._show_device_status(f"Testing {target_device.name}...", "info")

            # Play test audio through the monitor service
            task = await self._audio_coordinator.monitor_service.play_monitor_audio(
                audio_data=test_audio,
                device=target_device,
                volume=0.5
            )

            if task:
                # Show success
                if self._main_window and hasattr(self._main_window, 'settings_dialog') and self._main_window.settings_dialog:
                    self._main_window.settings_dialog._show_device_status(f"Test successful on {target_device.name}", "success")
                self.logger.info(f"Device test successful for: {target_device.name}")
            else:
                # Show failure
                if self._main_window and hasattr(self._main_window, 'settings_dialog') and self._main_window.settings_dialog:
                    self._main_window.settings_dialog._show_device_status("Test failed - no audio task created", "error")

        except Exception as e:
            self.logger.error(f"Device test error: {e}")
            if self._main_window and hasattr(self._main_window, 'settings_dialog') and self._main_window.settings_dialog:
                self._main_window.settings_dialog._show_device_status(f"Test error: {str(e)}", "error")

    def _on_virtual_device_test_requested(self, device_id):
        """
        Handle virtual device test request from the UI.

        Args:
            device_id: ID of the virtual device to test
        """
        self.logger.info(f"Virtual device test requested for device: {device_id}")

        try:
            # Run virtual device test asynchronously to avoid blocking UI
            import asyncio

            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run test in the event loop
            if loop.is_running():
                # If loop is already running, schedule as a task
                asyncio.create_task(self._test_virtual_device_async(device_id))
            else:
                # If no loop running, run until complete
                loop.run_until_complete(self._test_virtual_device_async(device_id))

        except Exception as e:
            self.logger.error(f"Error testing virtual device: {e}")
            if self._main_window and hasattr(self._main_window, 'settings_dialog') and self._main_window.settings_dialog:
                self._main_window.settings_dialog._show_virtual_device_status(f"Test failed: {str(e)}", "error")

    async def _test_virtual_device_async(self, device_id: str):
        """
        Test a virtual device asynchronously.

        Args:
            device_id: ID of the virtual device to test
        """
        try:
            if not self._audio_coordinator:
                if self._main_window and hasattr(self._main_window, 'settings_dialog') and self._main_window.settings_dialog:
                    self._main_window.settings_dialog._show_virtual_device_status("Audio coordinator not available", "error")
                return

            # Find the virtual device
            all_devices_dict = await self._audio_coordinator.enumerate_all_devices()
            target_device = None

            # Search specifically in virtual devices
            for device in all_devices_dict.get("virtual", []):
                if device.device_id == device_id:
                    target_device = device
                    break

            if not target_device:
                if self._main_window and hasattr(self._main_window, 'settings_dialog') and self._main_window.settings_dialog:
                    self._main_window.settings_dialog._show_virtual_device_status("Virtual device not found", "error")
                return

            # Show testing status
            if self._main_window and hasattr(self._main_window, 'settings_dialog') and self._main_window.settings_dialog:
                self._main_window.settings_dialog._show_virtual_device_status(f"Testing {target_device.name}...", "info")

            # Generate test tone (440Hz sine wave for 1 second)
            test_audio = self._generate_test_tone()

            # Find the device object from device_id
            virtual_devices = await self._audio_coordinator.virtual_service.enumerate_virtual_devices()
            target_device = None
            for device in virtual_devices:
                if device.device_id == device_id:
                    target_device = device
                    break

            if not target_device:
                raise MyVoiceError(
                    severity=ErrorSeverity.ERROR,
                    code="DEVICE_NOT_FOUND",
                    user_message=f"Virtual device {device_id} not found"
                )

            # Play test audio through the virtual microphone service
            task = await self._audio_coordinator.virtual_service.play_virtual_microphone(
                audio_data=test_audio,
                device=target_device,
                volume=0.5
            )

            if task:
                # Show success
                if self._main_window and hasattr(self._main_window, 'settings_dialog') and self._main_window.settings_dialog:
                    self._main_window.settings_dialog._show_virtual_device_status(
                        f"Virtual device test successful: {target_device.name}", "success"
                    )
                self.logger.info(f"Virtual device test successful: {target_device.name}")
            else:
                # Show failure
                if self._main_window and hasattr(self._main_window, 'settings_dialog') and self._main_window.settings_dialog:
                    self._main_window.settings_dialog._show_virtual_device_status("Virtual device test failed", "error")
                self.logger.warning(f"Virtual device test failed for: {target_device.name}")

        except Exception as e:
            self.logger.error(f"Error testing virtual device: {e}")
            if self._main_window and hasattr(self._main_window, 'settings_dialog') and self._main_window.settings_dialog:
                self._main_window.settings_dialog._show_virtual_device_status(f"Test error: {str(e)}", "error")

    def _generate_test_tone(self, frequency: float = 440.0, duration: float = 1.0, sample_rate: int = 44100) -> bytes:
        """
        Generate a simple test tone for device testing.

        Args:
            frequency: Frequency in Hz (default 440Hz - A4 note)
            duration: Duration in seconds
            sample_rate: Sample rate in Hz

        Returns:
            bytes: WAV audio data
        """
        import math
        import struct
        import io

        # Generate sine wave
        num_samples = int(sample_rate * duration)
        samples = []

        for i in range(num_samples):
            t = i / sample_rate
            # Generate sine wave with fade in/out to avoid clicks
            amplitude = 0.3  # Keep volume moderate
            if i < sample_rate * 0.1:  # Fade in first 0.1s
                amplitude *= i / (sample_rate * 0.1)
            elif i > num_samples - sample_rate * 0.1:  # Fade out last 0.1s
                amplitude *= (num_samples - i) / (sample_rate * 0.1)

            sample = amplitude * math.sin(2 * math.pi * frequency * t)
            # Convert to 16-bit signed integer
            sample_int = int(sample * 32767)
            samples.append(sample_int)

        # Create WAV file in memory
        wav_buffer = io.BytesIO()

        # WAV header
        wav_buffer.write(b'RIFF')
        wav_buffer.write(struct.pack('<I', 36 + len(samples) * 2))  # File size
        wav_buffer.write(b'WAVE')
        wav_buffer.write(b'fmt ')
        wav_buffer.write(struct.pack('<I', 16))  # Subchunk1 size
        wav_buffer.write(struct.pack('<H', 1))   # Audio format (PCM)
        wav_buffer.write(struct.pack('<H', 1))   # Number of channels (mono)
        wav_buffer.write(struct.pack('<I', sample_rate))  # Sample rate
        wav_buffer.write(struct.pack('<I', sample_rate * 2))  # Byte rate
        wav_buffer.write(struct.pack('<H', 2))   # Block align
        wav_buffer.write(struct.pack('<H', 16))  # Bits per sample
        wav_buffer.write(b'data')
        wav_buffer.write(struct.pack('<I', len(samples) * 2))  # Data size

        # Write audio data
        for sample in samples:
            wav_buffer.write(struct.pack('<h', sample))

        return wav_buffer.getvalue()

    async def _load_app_settings_on_startup(self):
        """Load application settings during startup."""
        try:
            self.logger.info("Loading application settings during startup")

            # Load settings from configuration manager
            self._app_settings = await self._config_manager.load_settings()

            if self._app_settings:
                self.logger.info("Application settings loaded successfully")
                return self._app_settings
            else:
                # Create default settings if none exist
                from myvoice.models.app_settings import AppSettings
                self._app_settings = AppSettings()
                self.logger.info("Created default application settings")
                return self._app_settings

        except Exception as e:
            self.logger.exception(f"Error loading application settings: {e}")
            # Create default settings as fallback
            from myvoice.models.app_settings import AppSettings
            self._app_settings = AppSettings()
            return self._app_settings

    def _on_settings_loaded(self, app_settings):
        """Handle successful loading of application settings."""
        self.logger.info("Application settings loaded and applied")

        # Store settings reference
        self._app_settings = app_settings

        # Connect settings to main window if it exists
        if self._main_window:
            self._main_window.set_app_settings(app_settings)
            self.logger.debug("Connected app settings to main window")

        # Update audio coordinator with loaded settings and apply device settings
        # NOTE: Audio coordinator may not exist yet if called during configuration init
        if hasattr(self, '_audio_coordinator') and self._audio_coordinator:
            self._audio_coordinator.app_settings = app_settings

            # Apply device settings to both services through coordinator
            self._run_async_task(
                self._audio_coordinator.update_device_settings(app_settings),
                on_success=lambda success: self.logger.info("Device settings loaded successfully from settings"),
                on_error=lambda error: self.logger.warning(f"Failed to load device settings from settings: {error}")
            )

        # TTS service will use AudioCoordinator directly - no manual device syncing needed


    def _on_settings_load_failed(self, error):
        """Handle failure of application settings loading."""
        self.logger.error(f"Failed to load application settings: {error}")

        # Create default settings as fallback
        from myvoice.models.app_settings import AppSettings
        self._app_settings = AppSettings()

        # Connect default settings to main window
        if self._main_window:
            self._main_window.set_app_settings(self._app_settings)
            self.logger.debug("Connected default app settings to main window")

        # Update audio coordinator with default settings
        # NOTE: Audio coordinator may not exist yet if called during configuration init
        if hasattr(self, '_audio_coordinator') and self._audio_coordinator:
            self._audio_coordinator.app_settings = self._app_settings

    async def _setup_device_change_monitoring_async(self):
        """Set up device change monitoring for runtime device updates asynchronously."""
        try:
            self.logger.info("Setting up device change monitoring")

            # Add device notification callback through audio coordinator
            self._audio_coordinator.add_device_notification_callback(self._on_device_notification)

            # Start background device monitoring through coordinator (DIRECT AWAIT)
            await self._audio_coordinator.start_device_monitoring()
            self.logger.info("Device change monitoring started successfully")

            self.logger.debug("Device monitoring setup completed")

        except Exception as e:
            self.logger.error(f"Failed to setup device change monitoring: {e}", exc_info=True)

    def _on_device_notification(self, notification):
        """
        Handle device change notifications from the audio service.

        Args:
            notification: DeviceNotification instance with change details
        """
        try:
            self.logger.info(f"Device notification: {notification.message}")

            # Handle device disconnection of currently selected devices
            if notification.severity.name in ['WARNING', 'ERROR']:
                self._handle_device_disconnection(notification)

            # Trigger automatic device list refresh in UI
            if self._main_window and hasattr(self._main_window, 'settings_dialog') and self._main_window.settings_dialog:
                # Refresh device lists in settings dialog if it's open
                self._run_async_task(
                    self._auto_refresh_device_lists(),
                    on_success=lambda _: self.logger.debug("Device lists auto-refreshed"),
                    on_error=lambda error: self.logger.error(f"Failed to auto-refresh devices: {error}")
                )

            # Show device change notification to user
            if self._main_window:
                severity_map = {
                    'INFO': 'info',
                    'WARNING': 'warning',
                    'ERROR': 'error'
                }
                severity = severity_map.get(notification.severity.name, 'info')
                self._main_window.show_service_notification(
                    "Audio Device Change",
                    notification.message,
                    severity
                )

        except Exception as e:
            self.logger.error(f"Error handling device notification: {e}")

    def _handle_device_disconnection(self, notification):
        """
        Handle device disconnection and settings migration.

        Args:
            notification: DeviceNotification with disconnection details
        """
        try:
            if not hasattr(self, '_app_settings') or not self._app_settings:
                return

            # Check if disconnected device affects current settings
            affected_settings = []
            migration_needed = False

            # Check monitor device using notification device information
            if (notification.device and notification.device.device_id and
                self._app_settings.monitor_device_id == notification.device.device_id):
                affected_settings.append("monitor device")
                migration_needed = True

            # Check virtual microphone device
            if (notification.device and notification.device.device_id and
                self._app_settings.virtual_microphone_device_id == notification.device.device_id):
                affected_settings.append("virtual microphone device")
                migration_needed = True

            if migration_needed:
                self.logger.warning(f"Device disconnection affects: {', '.join(affected_settings)}")
                self._migrate_disconnected_device_settings(affected_settings, notification)

        except Exception as e:
            self.logger.error(f"Error handling device disconnection: {e}")

    def _migrate_disconnected_device_settings(self, affected_settings, notification):
        """
        Migrate settings when devices become unavailable.

        Args:
            affected_settings: List of affected setting types
            notification: DeviceNotification with details
        """
        try:
            self.logger.info("Migrating settings for disconnected devices")

            # Create backup of current settings
            original_settings = self._app_settings.to_dict()

            # Migrate affected settings to safe defaults
            settings_changed = False
            migration_message_parts = []

            if "monitor device" in affected_settings:
                self._app_settings.monitor_device_id = None  # Fall back to system default
                settings_changed = True
                migration_message_parts.append("Monitor output reset to system default")

            if "virtual microphone device" in affected_settings:
                self._app_settings.virtual_microphone_device_id = None  # Disable dual routing
                settings_changed = True
                migration_message_parts.append("Virtual microphone disabled")

            if settings_changed:
                # Save migrated settings
                if hasattr(self, '_config_manager'):
                    self._run_async_task(
                        self._config_manager.save_settings(),
                        on_success=lambda _: self.logger.info("Migrated settings saved"),
                        on_error=lambda error: self.logger.error(f"Failed to save migrated settings: {error}")
                    )

                # Update main window with migrated settings
                if self._main_window:
                    self._main_window.update_settings(self._app_settings)

                # Notify user about settings migration
                migration_message = "Device settings migrated:\n• " + "\n• ".join(migration_message_parts)
                device_name = notification.device.name if notification.device else "Unknown"
                migration_message += f"\n\nDisconnected device: {device_name}"

                if self._main_window:
                    self._main_window.show_service_notification(
                        "Settings Migrated",
                        migration_message,
                        "warning"
                    )

                self.logger.info(f"Settings migration completed: {migration_message_parts}")

        except Exception as e:
            self.logger.error(f"Error migrating device settings: {e}")

    async def _auto_refresh_device_lists(self):
        """Automatically refresh device lists in the UI."""
        try:
            if self._main_window and hasattr(self._main_window, 'settings_dialog') and self._main_window.settings_dialog:
                # Get updated device lists from audio coordinator
                monitor_devices = await self._audio_coordinator.monitor_service.enumerate_monitor_devices()
                virtual_devices = await self._audio_coordinator.virtual_service.enumerate_virtual_devices()
                mic_devices = await self._audio_coordinator.enumerate_mic_devices()

                # Update settings dialog device lists on the main thread
                def update_ui():
                    try:
                        self._main_window.settings_dialog.update_device_list(monitor_devices)
                        self._main_window.settings_dialog.update_virtual_device_list(virtual_devices)
                        self._main_window.settings_dialog.update_mic_device_list(mic_devices or [])
                        self.logger.debug("Auto-refreshed device lists in settings dialog")
                    except Exception as e:
                        self.logger.error(f"Error updating device lists in UI: {e}")

                # Schedule UI update on main thread
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(0, update_ui)

        except Exception as e:
            self.logger.error(f"Error in auto refresh device lists: {e}")
            raise

    def _on_voice_directory_changed(self, directory_path: str):
        """
        Handle voice directory change from the UI.

        Args:
            directory_path: New voice files directory path
        """
        self.logger.info(f"Voice directory changed to: {directory_path}")

        try:
            # Update voice manager with new directory if available
            if hasattr(self, '_voice_manager'):
                self._run_async_task(
                    self._update_voice_manager_directory(directory_path),
                    on_success=self._on_voice_directory_update_complete,
                    on_error=self._on_voice_directory_update_failed
                )

        except Exception as e:
            self.logger.error(f"Error handling voice directory change: {e}")

    async def _update_voice_manager_directory(self, directory_path: str):
        """
        Update voice manager with new directory and trigger rescan.

        Args:
            directory_path: New voice files directory path

        Returns:
            Dict with scan results
        """
        try:
            from pathlib import Path

            # Update voice manager directory
            self._voice_manager.voice_directory = Path(directory_path)
            self.logger.debug(f"Updated voice manager directory to: {directory_path}")

            # Force rescan of the new directory
            scan_results = await self._voice_manager.force_rescan()
            self.logger.info(f"Voice directory rescan completed: {scan_results.get('profiles_found', 0)} profiles found")

            return scan_results

        except Exception as e:
            self.logger.error(f"Error updating voice manager directory: {e}")
            raise

    def _on_voice_directory_update_complete(self, scan_results):
        """Handle successful voice directory update."""
        profiles_found = scan_results.get("profiles_found", 0)
        self.logger.info(f"Voice directory update completed: {profiles_found} voice profiles found")

        # Update UI with scan results
        if self._main_window:
            message = f"Voice directory updated. Found {profiles_found} voice profile(s)."
            self._main_window.show_service_notification(
                "Voice Directory Updated",
                message,
                "info"
            )

    def _on_voice_directory_update_failed(self, error):
        """Handle voice directory update failure."""
        self.logger.error(f"Voice directory update failed: {error}")

        # Show error notification
        if self._main_window:
            self._main_window.show_service_notification(
                "Voice Directory Update Failed",
                f"Failed to update voice directory: {str(error)}",
                "error"
            )

    def _on_voice_refresh_requested(self):
        """Handle voice refresh request from the UI."""
        self.logger.info("Voice refresh requested from UI")

        try:
            # Trigger voice directory rescan if voice manager available
            if hasattr(self, '_voice_manager'):
                self._run_async_task(
                    self._voice_manager.force_rescan(),
                    on_success=self._on_voice_refresh_complete,
                    on_error=self._on_voice_refresh_failed
                )
        except Exception as e:
            self.logger.error(f"Error handling voice refresh: {e}")

    def _on_voice_refresh_complete(self, scan_results):
        """Handle successful voice refresh."""
        valid_profiles = scan_results.get("valid_profiles", 0)
        self.logger.info(f"Voice refresh completed: {valid_profiles} valid profiles found")

        # Refresh voice list in settings dialog if it's open
        if self._main_window and hasattr(self._main_window, 'settings_dialog') and self._main_window.settings_dialog:
            self._main_window.settings_dialog.refresh_voice_list()

        # Show success notification
        if self._main_window:
            self._main_window.show_service_notification(
                "Voice Refresh Complete",
                f"Found {valid_profiles} valid voice profile(s)",
                "success"
            )

    def _on_voice_refresh_failed(self, error):
        """Handle voice refresh failure."""
        self.logger.error(f"Voice refresh failed: {error}")

        # Show error notification
        if self._main_window:
            self._main_window.show_service_notification(
                "Voice Refresh Failed",
                f"Failed to refresh voice profiles: {str(error)}",
                "error"
            )

    async def _auto_detect_and_configure_vb_cable(self):
        """
        Auto-detect and configure VB-Cable on first boot.

        Checks if virtual device is already configured in settings. If not,
        attempts to detect VB-Cable and automatically configure it as the
        virtual microphone device.
        """
        try:
            # Check if virtual device is already configured in settings
            if (hasattr(self, '_app_settings') and self._app_settings and
                self._app_settings.virtual_microphone_device_id):
                self.logger.info("Virtual microphone device already configured, skipping auto-detection")
                return

            self.logger.info("First boot detected - attempting VB-Cable auto-detection")

            # Initialize virtual device compatibility service for detection
            from myvoice.services.virtual_device_compatibility_service import VirtualDeviceCompatibilityService
            compat_service = VirtualDeviceCompatibilityService()
            await compat_service.start()

            try:
                # Attempt to auto-detect VB-Cable device
                vb_cable_device = await compat_service.auto_detect_vb_cable_device()

                if vb_cable_device:
                    # VB-Cable detected! Configure it automatically
                    self.logger.info(f"VB-Cable auto-detected: {vb_cable_device.name}")

                    # Update app settings with detected device
                    if hasattr(self, '_app_settings') and self._app_settings:
                        self._app_settings.virtual_microphone_device_id = vb_cable_device.device_id

                        # Save updated settings
                        if hasattr(self, '_config_manager'):
                            await self._config_manager.save_settings()
                            self.logger.info("VB-Cable device saved to settings")

                        # Update audio coordinator with new device
                        if hasattr(self, '_audio_coordinator'):
                            await self._audio_coordinator.update_device_settings(self._app_settings)
                            self.logger.info("Audio coordinator updated with VB-Cable device")

                        # Show success notification to user
                        if self._main_window:
                            self._main_window.show_service_notification(
                                "VB-Cable Detected",
                                f"VB-Cable detected and configured automatically: {vb_cable_device.name}",
                                "info"
                            )
                            self.logger.info("User notified of VB-Cable auto-configuration")
                else:
                    # VB-Cable not found - log for troubleshooting
                    self.logger.info("VB-Cable device not detected - virtual microphone not configured")

                    # Show guidance message if main window is available
                    if self._main_window:
                        self._main_window.show_service_notification(
                            "Virtual Microphone Not Configured",
                            "No VB-Cable device detected. You can configure it manually in Settings if you install VB-Cable later.",
                            "info"
                        )

            finally:
                # Clean up compatibility service
                await compat_service.stop()

        except Exception as e:
            self.logger.error(f"Error during VB-Cable auto-detection: {e}")
            # Non-critical error - continue startup

    def _on_voice_transcription_requested(self, voice_name: str):
        """
        Handle transcription request from the UI.

        Args:
            voice_name: Name of the voice profile to transcribe
        """
        self.logger.info(f"Transcription requested for voice: {voice_name}")

        # Reuse existing transcription logic
        self._on_transcription_requested(voice_name)