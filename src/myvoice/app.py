"""
MyVoice Application Controller

This module contains the main application controller that manages the overall
application lifecycle, initialization, and coordination between services and UI.
"""

import gc
import logging
import sys
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QObject, QTimer

from myvoice.models.ui_state import ServiceStatusInfo, ServiceHealthStatus
from myvoice.models.service_enums import ServiceStatus, QwenModelType


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

            # Initialize Audio Coordinator (dual-service architecture)
            from myvoice.services.audio_coordinator import AudioCoordinator
            self._audio_coordinator = AudioCoordinator(app_settings=getattr(self, '_app_settings', None))
            self.register_service("audio_coordinator", self._audio_coordinator)

            # Start audio coordinator (DIRECT AWAIT)
            await self._audio_coordinator.start()
            self.logger.info("Audio coordinator started successfully")
            self._on_audio_coordinator_started(None)

            # Auto-detect and configure VB-Cable on first boot if no virtual device configured
            await self._auto_detect_and_configure_vb_cable()

            # Initialize TTS Service (Qwen3-TTS)
            from myvoice.services.qwen_tts_service import QwenTTSService
            self._tts_service = QwenTTSService()
            self.register_service("tts", self._tts_service)

            # Set up health status callback BEFORE starting service
            self._tts_service.set_health_status_callback(self._on_tts_health_status_changed)

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
            from myvoice.ui.main_window import MainWindow
            self._main_window = MainWindow()

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
            self._main_window.whisper_init_requested.connect(self._on_whisper_init_requested)  # QA4

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

    async def _play_generated_audio(self, audio_data: bytes):
        """
        Play generated TTS audio using AudioCoordinator dual-service architecture.

        This method routes audio through AudioCoordinator which manages both
        MonitorAudioService and VirtualMicrophoneService independently.

        Args:
            audio_data: WAV audio data from TTS generation
        """
        try:
            self.logger.info("Starting audio playback via AudioCoordinator")

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

                # Execute dual-stream routing through coordinator
                self.logger.info(f"[WIN11-DEBUG] Calling play_dual_stream with monitor_device={monitor_device.name if monitor_device else 'None'}, virtual_device={virtual_device.name if virtual_device else 'None'}")
                dual_result = await self._audio_coordinator.play_dual_stream(
                    audio_data=audio_data,
                    monitor_device=monitor_device,
                    virtual_device=virtual_device,
                    volume=1.0
                )

                if dual_result and dual_result.any_successful:
                    self.logger.info("Dual-stream playback started successfully (monitor + virtual mic)")
                    if self._main_window:
                        self._main_window.set_generation_status("Playing audio on speakers and virtual microphone", False)
                else:
                    self.logger.warning("Dual-stream playback failed")
                    if self._main_window:
                        self._main_window.set_generation_status("Audio playback failed", False)

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

                if target_device:
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
                    if self._main_window:
                        self._main_window.set_generation_status("Playing audio on speakers", False)
                else:
                    self.logger.warning("No audio devices available for playback")
                    if self._main_window:
                        self._main_window.set_generation_status("No audio devices available", False)

        except Exception as e:
            self.logger.error(f"Error during audio playback: {e}")
            if self._main_window:
                self._main_window.set_generation_status(f"Playback failed: {str(e)}", False)

    def _on_audio_playback_success(self, result):
        """Handle successful audio playback completion."""
        self.logger.info("Audio playback completed successfully")
        if self._main_window:
            self._main_window.set_generation_status("Audio playback completed", False)
            # Story 2.4: Enable replay button after successful playback (FR29)
            self._main_window.set_replay_enabled(True)

    def _on_audio_playback_error(self, error):
        """Handle audio playback error."""
        self.logger.error(f"Audio playback failed: {error}")
        if self._main_window:
            self._main_window.set_generation_status(f"Audio playback failed: {str(error)}", False)

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

    def _on_settings_changed(self, new_settings):
        """
        Handle settings changes from the UI.

        Args:
            new_settings: Updated AppSettings instance
        """
        self.logger.info("Settings changed, saving and applying updates")

        try:
            # Update stored settings
            self._app_settings = new_settings

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
            else:
                # Fallback for direct device list (backward compatibility)
                self._main_window.settings_dialog.update_device_list(devices)

    def _on_device_refresh_failed(self, error):
        """Handle device refresh failure."""
        self.logger.error(f"Device refresh failed: {error}")

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

                # Update settings dialog device lists on the main thread
                def update_ui():
                    try:
                        self._main_window.settings_dialog.update_device_list(monitor_devices)
                        self._main_window.settings_dialog.update_virtual_device_list(virtual_devices)
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