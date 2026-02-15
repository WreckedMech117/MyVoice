"""
Voice Design Studio Dialog

Main dialog for creating voices using acoustic embeddings. Provides creation workflow:
- From Description: Generate voices from text descriptions with 5 variations
  - Imagine sub-tab: Voice templates and description
  - Create sub-tab: 5 variant generation and selection
  - Clone sub-tab: Voice cloning from audio samples (QA8)
- Emotions: Generate emotion variants (Neutral, Happy, Sad, Angry, Flirtatious)
- Refinement: Extract embeddings and save voice to library

Story 1.2: Voice Design Studio Dialog Shell
- FR1: User can access Voice Design Studio from the main application
- FR2: User can switch between "From Description" and "From Sample" tabs
- FR3: System displays the currently active tab with visual indication
- FR4: User can close the dialog and return to main application

Story 1.3: Voice Description Input
- FR6: User can enter a multi-line voice description
- FR11: User can enter preview text for voice generation

Acceptance Criteria (1.2):
- Menu action opens modal dialog titled "Voice Design Studio"
- Dialog displays two tabs: "From Description" and "From Sample"
- "From Description" tab is active by default with visual indicator
- Clicking tabs switches content panels
- Close button (X) closes dialog and returns to main app

Acceptance Criteria (1.3):
- "From Description" tab shows multi-line "Voice Description" text area (4-5 visible lines)
- Single-line "Preview Text" field visible
- "Generate" button initially disabled
- Generate button enables when both fields have content
- Button remains disabled if description is empty
"""

import logging
import json
import shutil
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, TYPE_CHECKING

import numpy as np
import scipy.io.wavfile as wavfile

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QPushButton, QLabel, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QKeySequence, QShortcut

from myvoice.ui.dialogs.voice_design_studio.description_path_panel import DescriptionPathPanel
from myvoice.ui.dialogs.voice_design_studio.emotions_panel import EmotionsPanel
from myvoice.ui.dialogs.voice_design_studio.refinement_panel import RefinementPanel
from myvoice.utils.session_manager import SessionManager
from myvoice.models.voice_profile import VALID_EMOTIONS

if TYPE_CHECKING:
    from myvoice.services.qwen_tts_service import QwenTTSService

# Import for torch.load deserialization of VoiceClonePromptItem
from myvoice.services.qwen_tts_service import VoiceClonePromptItem  # noqa: F401


class VoiceDesignStudioDialog(QDialog):
    """
    Voice Design Studio dialog for creating voices with acoustic embeddings.

    Provides a tabbed workflow interface:
    - From Description: Text-based voice design with 5 variations (includes Clone sub-tab for audio cloning)
    - Emotions: Generate emotion variants for the voice (Neutral, Happy, Sad, Angry, Flirtatious)
    - Refinement: Extract embeddings and save voice to library

    QA8: From Sample tab removed - clone functionality moved to From Description-Clone sub-tab.

    Signals:
        voice_saved: Emitted when a voice is successfully saved (voice_name)
        dialog_closing: Emitted when dialog is closing (has_unsaved_work)
    """

    voice_saved = pyqtSignal(str)  # voice_name
    dialog_closing = pyqtSignal(bool)  # has_unsaved_work

    def __init__(
        self,
        tts_service: Optional['QwenTTSService'] = None,
        whisper_service: Optional[object] = None,
        parent: Optional[QWidget] = None
    ):
        """
        Initialize the Voice Design Studio dialog.

        Args:
            tts_service: QwenTTSService instance for voice generation
            whisper_service: WhisperService instance for transcription
            parent: Parent widget (typically MainWindow)
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        # TTS service for voice generation
        self._tts_service = tts_service

        # Whisper service for transcription (QA3 fix)
        self._whisper_service = whisper_service

        # Session state tracking
        self._has_unsaved_work = False

        # Generation state
        self._is_generating = False
        self._current_variant = 0
        self._generation_cancelled = False

        # Create session manager for temp file lifecycle (Story 4.1)
        self._session_manager = SessionManager()
        self.logger.info(f"Session started: {self._session_manager.session_id}")

        # Setup dialog
        self._setup_dialog()
        self._create_ui()
        self._setup_accessibility()
        self._setup_shortcuts()

        # QA4: Restore existing variants from persistent session
        self._restore_existing_variants()

        # QA7: Restore session data (description, preview text) for recovery
        self._restore_session_data()

        # QA4-1: Restore emotion variants from persistent session
        self._restore_emotion_variants()

        self.logger.debug("VoiceDesignStudioDialog initialized")

    def _restore_existing_variants(self):
        """
        Restore existing variants from persistent session (QA4).

        Checks if there are variant files from a previous session and
        populates the description panel with them, allowing users to
        return and select different previously-generated variants.
        """
        if not self._session_manager.has_existing_variants():
            self.logger.debug("No existing variants to restore")
            return

        existing_variants = self._session_manager.get_existing_variants()
        self.logger.info(f"Found {len(existing_variants)} existing variants, restoring...")

        # Restore each variant to the description panel
        for wav_path in existing_variants:
            try:
                # Extract variant index from filename (variant_0.wav -> 0)
                filename = wav_path.stem  # "variant_0"
                parts = filename.split("_")
                if len(parts) >= 2:
                    variant_index = int(parts[1])

                    # Get duration
                    duration = self._get_audio_duration(wav_path)

                    # Use audio path as placeholder for embedding path
                    # (embedding extraction happens in From Sample-Clone tab)
                    self.description_panel.set_variant_complete(
                        variant_index,
                        wav_path,
                        wav_path,  # Audio as placeholder for embedding
                        duration
                    )
                    self.logger.debug(f"Restored variant {variant_index}: {wav_path.name}")

            except Exception as e:
                self.logger.warning(f"Failed to restore variant {wav_path}: {e}")

        # Switch to Create sub-tab to show restored variants
        if existing_variants:
            # QA6: Make result_group visible when restoring (normally set by set_generating())
            self.description_panel.result_group.setVisible(True)
            self.description_panel.sub_tabs.setCurrentIndex(1)  # Create tab
            self.logger.info(f"Restored {len(existing_variants)} variants from previous session")

    def _restore_session_data(self):
        """
        QA7: Restore session data (description, preview text) for recovery.

        If the user closed the dialog to test a voice, their description and
        preview text are restored so they can continue where they left off.
        """
        if not self._session_manager.has_session_data():
            self.logger.debug("No session data to restore")
            return

        session_data = self._session_manager.load_session_data()
        if not session_data:
            return

        # Restore description panel fields
        voice_description = session_data.get("voice_description", "")
        preview_text = session_data.get("preview_text", "")
        voice_name = session_data.get("voice_name", "")
        language = session_data.get("language", "Auto")

        if voice_description:
            self.description_panel.set_description(voice_description)
            self.logger.debug(f"Restored voice description: {len(voice_description)} chars")

        if preview_text:
            self.description_panel.set_preview_text(preview_text)
            self.logger.debug(f"Restored preview text: {len(preview_text)} chars")

        if voice_name:
            self.description_panel.set_voice_name(voice_name)
            # Also set in refinement panel
            self.refinement_panel.set_voice_name(voice_name)
            self.logger.debug(f"Restored voice name: {voice_name}")

        if language and language != "Auto":
            self.description_panel.set_language(language)
            self.logger.debug(f"Restored language: {language}")

        # QA Round 2 Item #6: Restore tab state
        current_tab = session_data.get("current_tab_index", 0)
        if current_tab > 0:
            self.tab_widget.setCurrentIndex(current_tab)
            self.logger.debug(f"Restored tab index: {current_tab}")

        # QA Round 2 Item #6: Check for existing emotion embeddings
        existing_embeddings = self._session_manager.get_existing_emotion_embeddings()
        if existing_embeddings:
            self.logger.info(f"Found {len(existing_embeddings)} existing emotion embeddings")
            # Mark refinement panel with existing embeddings
            for emotion, embedding_path in existing_embeddings.items():
                self.refinement_panel.set_emotion_complete(emotion, embedding_path)
            # Mark extraction complete if neutral exists and on Refinement tab
            if current_tab == 2 and "neutral" in existing_embeddings:
                self.refinement_panel.finish_extraction()

        self.logger.info("Restored session data for recovery")

    def _restore_emotion_variants(self):
        """
        QA4-1: Restore emotion variants from persistent session.

        Checks for existing emotion variant files and restores them to the
        Emotions panel, allowing users to resume work on emotion generation.
        """
        emotions = ["neutral", "happy", "sad", "angry", "flirtatious"]
        restored_count = 0

        for emotion in emotions:
            try:
                existing_variants = self._session_manager.get_existing_emotion_variants(emotion)
                if not existing_variants:
                    continue

                self.logger.debug(f"Restoring {len(existing_variants)} variants for {emotion}")

                for wav_path in existing_variants:
                    try:
                        # Extract variant index from filename (variant_0.wav -> 0)
                        filename = wav_path.stem  # "variant_0"
                        parts = filename.split("_")
                        if len(parts) >= 2:
                            variant_index = int(parts[1])
                            duration = self._get_audio_duration(wav_path)

                            # Check for corresponding embedding
                            embedding_path = wav_path.with_suffix('.pt')
                            if not embedding_path.exists():
                                embedding_path = wav_path  # Use audio as placeholder

                            self.emotions_panel.set_variant_complete(
                                emotion,
                                variant_index,
                                wav_path,
                                embedding_path,
                                duration
                            )
                            restored_count += 1

                    except Exception as e:
                        self.logger.warning(f"Failed to restore {emotion} variant {wav_path}: {e}")

                # Mark generation as finished for this emotion
                self.emotions_panel.finish_generation(emotion)

            except Exception as e:
                self.logger.warning(f"Failed to restore {emotion} variants: {e}")

        if restored_count > 0:
            self.logger.info(f"QA4-1: Restored {restored_count} emotion variants from session")

    def _get_audio_duration(self, file_path: Path) -> float:
        """
        Get audio duration for a file (QA4 helper).

        Args:
            file_path: Path to the audio file

        Returns:
            Duration in seconds, or 3.0 as fallback
        """
        try:
            import wave
            with wave.open(str(file_path), 'rb') as w:
                frames = w.getnframes()
                rate = w.getframerate()
                if rate > 0:
                    return frames / float(rate)
        except Exception:
            pass
        return 3.0  # Fallback

    def _setup_dialog(self):
        """Configure dialog properties."""
        self.setWindowTitle("Voice Design Studio")
        self.setModal(True)
        self.resize(600, 500)  # Comfortable size for the workflow

        # Set window flags for proper dialog behavior
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.WindowTitleHint |
            Qt.WindowType.WindowCloseButtonHint
        )

    def _create_ui(self):
        """Create the dialog UI with tabbed interface."""
        layout = QVBoxLayout(self)
        # QA3-2: Reduced margins and spacing for smaller screens
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        # Header
        self._create_header(layout)

        # Tab widget for voice creation workflow
        # Emotion Variants: Added Emotions and Refinement tabs between Description and Sample
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("voice_design_studio_tabs")

        # Create and add tabs in workflow order
        # Tab 0: From Description - initial voice generation (includes Clone sub-tab)
        # Tab 1: Emotions - generate emotion variants (Emotion Variants feature)
        # Tab 2: Refinement - extract embeddings and save (Emotion Variants feature)
        # QA8: From Sample tab removed - clone functionality moved to From Description-Clone sub-tab
        self._create_description_tab()
        self._create_emotions_tab()
        self._create_refinement_tab()

        # Set "From Description" as default active tab
        self.tab_widget.setCurrentIndex(0)

        # Connect tab change signal
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        layout.addWidget(self.tab_widget, 1)  # Stretch to fill space

        # Footer buttons
        self._create_footer(layout)

    def _create_header(self, layout: QVBoxLayout):
        """Create the dialog header with title and description."""
        # Title
        title_label = QLabel("Voice Design Studio")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        title_label.setObjectName("dialog_title")
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel(
            "Create custom voices with full emotion control. "
            "Choose to describe a voice or extract from an audio sample."
        )
        desc_label.setWordWrap(True)
        desc_label.setObjectName("dialog_description")
        layout.addWidget(desc_label)

    def _create_description_tab(self):
        """Create the 'From Description' tab with DescriptionPathPanel."""
        # Create the description path panel (Story 1.3)
        self.description_panel = DescriptionPathPanel()

        # Connect panel signals
        self.description_panel.generate_requested.connect(self._on_generate_requested)
        self.description_panel.content_changed.connect(self._on_description_content_changed)
        self.description_panel.save_ready_changed.connect(self._on_save_ready_changed)
        self.description_panel.regenerate_requested.connect(self._on_regenerate_requested)
        # QA3: Connect refine signal for cross-tab data transfer
        self.description_panel.refine_requested.connect(self._on_refine_requested)

        # QA8: Connect clone tab signals
        self.description_panel.clone_transcribe_requested.connect(self._on_clone_transcribe_requested)
        self.description_panel.clone_proceed_requested.connect(self._on_clone_proceed_requested)

        self.tab_widget.addTab(self.description_panel, "From Description")

    def _create_emotions_tab(self):
        """
        Create the 'Emotions' tab with EmotionsPanel.

        Emotion Variants Feature: This tab allows users to generate
        voice variants for each of the 5 emotions (Neutral, Happy, Sad, Angry, Flirtatious).
        """
        self.emotions_panel = EmotionsPanel()

        # Connect panel signals
        self.emotions_panel.generation_requested.connect(self._on_emotion_generation_requested)
        self.emotions_panel.emotion_variant_selected.connect(self._on_emotion_variant_selected)
        self.emotions_panel.proceed_to_refinement_requested.connect(self._on_proceed_to_refinement)
        self.emotions_panel.neutral_selection_changed.connect(self._on_neutral_selection_changed)

        self.tab_widget.addTab(self.emotions_panel, "Emotions")

    def _create_refinement_tab(self):
        """
        Create the 'Refinement' tab with RefinementPanel.

        Emotion Variants Feature: This tab allows users to extract embeddings
        from selected emotion samples and save the voice to the library.
        """
        self.refinement_panel = RefinementPanel()

        # Connect panel signals
        self.refinement_panel.extraction_requested.connect(self._on_batch_extraction_requested)
        self.refinement_panel.save_requested.connect(self._on_refinement_save_requested)
        self.refinement_panel.preview_requested.connect(self._on_preview_requested)
        self.refinement_panel.back_requested.connect(self._on_back_to_emotions)

        self.tab_widget.addTab(self.refinement_panel, "Refinement")

    def _on_save_ready_changed(self, is_ready: bool):
        """
        Handle save readiness changes from description panel.

        QA5: Save button removed from Description tab - this is now a no-op.
        Saving happens in From Sample-Clone tab via Extract Embedding.

        Args:
            is_ready: True if voice is ready to save (unused)
        """
        pass  # No save button in Description tab

    def _on_generate_requested(self, description: str, preview_text: str, language: str):
        """
        Handle generate request from description panel.

        Generates 5 voice variations using the TTS service with progressive reveal.

        Args:
            description: Voice description text
            preview_text: Preview text for generation
            language: Language for generation (e.g., "English", "Auto")
        """
        self.logger.info(f"Generate requested: {len(description)} char description, language={language}")

        # Check if TTS service is available
        if not self._tts_service:
            self.logger.error("TTS service not available")
            QMessageBox.warning(
                self,
                "Service Unavailable",
                "Voice generation service is not available. Please try again later."
            )
            return

        if not self._tts_service.is_running():
            self.logger.error("TTS service not running")
            QMessageBox.warning(
                self,
                "Service Not Ready",
                "Voice generation service is still initializing. Please wait a moment and try again."
            )
            return

        # Start generation
        self._is_generating = True
        self._generation_cancelled = False
        self._current_variant = 0

        # Set UI to generating state
        self.description_panel.set_generating(True, "Starting generation...")

        # Store generation parameters for potential regeneration
        self._last_description = description
        self._last_preview_text = preview_text
        self._last_language = language

        # Start generating variants
        self._generate_next_variant(description, preview_text, language)

    def _generate_next_variant(self, description: str, preview_text: str, language: str):
        """
        Generate the next voice variant.

        Args:
            description: Voice description text
            preview_text: Preview text for generation
            language: Language for generation
        """
        if self._generation_cancelled or self._current_variant >= 5:
            self._finish_generation()
            return

        variant_index = self._current_variant
        self.logger.debug(f"Generating variant {variant_index + 1}/5")

        # Update UI to show this variant is generating
        self.description_panel.set_variant_generating(variant_index)

        # Run async generation
        self._run_async_task(
            self._generate_single_variant(description, preview_text, language, variant_index),
            on_success=lambda result: self._on_variant_generated(result, description, preview_text, language),
            on_error=lambda e: self._on_variant_failed(e, variant_index, description, preview_text, language)
        )

    async def _generate_single_variant(
        self,
        description: str,
        preview_text: str,
        language: str,
        variant_index: int
    ) -> dict:
        """
        Generate a single voice variant.

        Args:
            description: Voice description text
            preview_text: Preview text for generation
            language: Language for generation
            variant_index: Index of the variant (0-4)

        Returns:
            Dictionary with audio_data, sample_rate, variant_index
        """
        self.logger.debug(f"Generating variant {variant_index + 1} with VoiceDesign model")

        # Generate voice using VoiceDesign model
        response = await self._tts_service.generate_voice_design(
            text=preview_text,
            voice_description=description,
            language=language,
            streaming=False,  # Use non-streaming for simpler handling
        )

        if not response.success:
            raise RuntimeError(response.error_message or "Voice generation failed")

        return {
            'audio_data': response.audio_data,
            'sample_rate': response.sample_rate,
            'variant_index': variant_index,
        }

    def _on_variant_generated(self, result: dict, description: str, preview_text: str, language: str):
        """
        Handle successful variant generation.

        Args:
            result: Generation result with audio_data, sample_rate, variant_index
            description: Voice description text
            preview_text: Preview text for generation
            language: Language for generation
        """
        if self._generation_cancelled:
            return

        variant_index = result['variant_index']
        audio_data = result['audio_data']
        sample_rate = result['sample_rate']

        self.logger.debug(f"Variant {variant_index + 1} generated successfully")

        # Save audio to session directory
        audio_path = self._session_manager.get_variant_path(variant_index, "wav")
        try:
            # Ensure audio data is in correct format for wavfile
            if audio_data.dtype != np.int16:
                # Normalize and convert to int16
                audio_data = np.clip(audio_data, -1.0, 1.0)
                audio_data = (audio_data * 32767).astype(np.int16)

            wavfile.write(str(audio_path), sample_rate, audio_data)
            self.logger.debug(f"Saved variant {variant_index + 1} to {audio_path}")
        except Exception as e:
            self.logger.error(f"Failed to save variant {variant_index + 1}: {e}")
            self._on_variant_failed(e, variant_index, description, preview_text, language)
            return

        # Calculate duration
        duration = len(audio_data) / sample_rate

        # Use audio path as placeholder for embedding path
        # (embedding extraction happens in From Sample-Clone tab, not here)
        embedding_path = audio_path

        # Update UI
        self.description_panel.set_variant_complete(
            variant_index,
            audio_path,
            embedding_path,
            duration
        )

        # Move to next variant
        self._current_variant = variant_index + 1
        self._generate_next_variant(description, preview_text, language)

    def _on_variant_failed(self, error: Exception, variant_index: int, description: str, preview_text: str, language: str):
        """
        Handle variant generation failure.

        Args:
            error: The exception that occurred
            variant_index: Index of the failed variant
            description: Voice description text
            preview_text: Preview text for generation
            language: Language for generation
        """
        self.logger.error(f"Variant {variant_index + 1} failed: {error}")

        # Update UI to show error for this variant
        self.description_panel.set_variant_error(variant_index, str(error))

        # Continue with next variant (don't stop on single failure)
        self._current_variant = variant_index + 1
        self._generate_next_variant(description, preview_text, language)

    def _finish_generation(self):
        """Finish the generation process."""
        self._is_generating = False
        self.description_panel.finish_generation()
        self.logger.info(f"Generation complete: {self._current_variant} variants processed")

    def _run_async_task(self, coro, on_success=None, on_error=None):
        """
        Helper to run async tasks from sync Qt signal handlers.

        Uses the shared qasync event loop.

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
        asyncio.ensure_future(_handle_task())

    def _on_regenerate_requested(self):
        """
        Handle regenerate request from description panel (Story 1.9).

        Story 4.1 REVISED: Clears previous temp audition files BEFORE
        generating new variants. This is the ONLY place temp files are deleted.
        """
        self.logger.debug("Regenerate All requested")

        # Story 1.9/4.1: Clear previous variant files before regenerating
        # This is the ONLY action that deletes temp audition files
        if self._session_manager and not self._session_manager.is_cleaned:
            deleted = self._session_manager.clear_variant_files()
            self.logger.info(f"Cleared {deleted} previous variant files before regeneration")

        # Get current description, preview text, and language
        description = self.description_panel.get_description()
        preview_text = self.description_panel.get_preview_text()
        language = self.description_panel.get_language()

        # Trigger new generation with same parameters
        # This will clear existing variants via set_generating() in the panel
        self._on_generate_requested(description, preview_text, language)

    def _on_description_content_changed(self):
        """Handle content changes in description panel."""
        # Update unsaved work flag if there's content
        has_content = self.description_panel.has_content()
        self.set_has_unsaved_work(has_content)

    # =========================================================================
    # Emotion Variants: Emotions Tab Signal Handlers
    # =========================================================================

    def _on_emotion_generation_requested(self, emotion: str, instruction: str, preview_text: str):
        """
        Handle generation request from emotions panel.

        Generates 5 voice variants for the specified emotion using the instruct parameter.

        Args:
            emotion: Emotion ID (e.g., "neutral", "happy")
            instruction: Emotion instruction text
            preview_text: Preview text for generation
        """
        self.logger.info(f"Emotion generation requested: {emotion}")

        # Check if TTS service is available
        if not self._tts_service or not self._tts_service.is_running():
            self.logger.error("TTS service not available")
            self.emotions_panel.set_generation_error(
                emotion, "Voice generation service is not available."
            )
            return

        # Get voice description from Description panel
        # QA: Support Clone workflow - use clone_transcript as description fallback
        description = self.description_panel.get_description()
        if not description:
            # Check if Clone workflow is active (has transcript and neutral embedding)
            session_data = self._session_manager.load_session_data()
            clone_transcript = session_data.get("clone_transcript") if session_data else None
            neutral_embedding = self._session_manager.get_emotion_embedding_path("neutral")

            if clone_transcript and neutral_embedding and neutral_embedding.exists():
                # Clone workflow: use transcript-based description for emotion generation
                description = f"Voice cloned from audio sample. Original transcript: {clone_transcript}"
                self.logger.info(f"Using clone transcript as description for emotion generation")
            else:
                self.emotions_panel.set_generation_error(
                    emotion, "Please generate a voice in the Description tab first."
                )
                return

        # Get preview text (use from emotions panel or fallback to description panel)
        if not preview_text:
            preview_text = self.description_panel.get_preview_text()

        # Set UI to generating state
        self.emotions_panel.set_generating(emotion, True, "Starting generation...")

        # Generate 5 variants for this emotion
        self._emotion_generation_state = {
            'emotion': emotion,
            'instruction': instruction,
            'description': description,
            'preview_text': preview_text,
            'current_variant': 0,
            'cancelled': False
        }

        self._generate_next_emotion_variant()

    def _generate_next_emotion_variant(self):
        """Generate the next emotion variant."""
        state = getattr(self, '_emotion_generation_state', None)
        if not state or state['cancelled'] or state['current_variant'] >= 5:
            self._finish_emotion_generation()
            return

        emotion = state['emotion']
        variant_index = state['current_variant']

        self.logger.debug(f"Generating {emotion} variant {variant_index + 1}/5")
        self.emotions_panel.set_variant_generating(emotion, variant_index)

        # Run async generation
        self._run_async_task(
            self._generate_single_emotion_variant(state, variant_index),
            on_success=self._on_emotion_variant_generated,
            on_error=lambda e: self._on_emotion_variant_failed(e, variant_index)
        )

    async def _generate_single_emotion_variant(self, state: dict, variant_index: int) -> dict:
        """
        Generate a single emotion variant using VoiceDesign model with instruct.

        Args:
            state: Generation state dictionary
            variant_index: Index of the variant (0-4)

        Returns:
            Dictionary with audio_data, sample_rate, variant_index, emotion
        """
        emotion = state['emotion']
        instruction = state['instruction']
        description = state['description']
        preview_text = state['preview_text']

        self.logger.debug(f"Generating {emotion} variant {variant_index + 1}")

        # Combine voice description with emotion instruction
        full_description = f"{description}\n\nEmotion instruction: {instruction}"

        # Get language from description panel
        language = self.description_panel.get_language()

        # Generate voice using VoiceDesign model
        response = await self._tts_service.generate_voice_design(
            text=preview_text,
            voice_description=full_description,
            language=language,
            streaming=False,
        )

        if not response.success:
            raise RuntimeError(response.error_message or "Voice generation failed")

        return {
            'audio_data': response.audio_data,
            'sample_rate': response.sample_rate,
            'variant_index': variant_index,
            'emotion': emotion,
        }

    def _on_emotion_variant_generated(self, result: dict):
        """Handle successful emotion variant generation."""
        state = getattr(self, '_emotion_generation_state', None)
        if not state or state['cancelled']:
            return

        emotion = result['emotion']
        variant_index = result['variant_index']
        audio_data = result['audio_data']
        sample_rate = result['sample_rate']

        self.logger.debug(f"{emotion} variant {variant_index + 1} generated")

        # Save audio to session directory (emotion-specific path)
        audio_path = self._session_manager.get_emotion_variant_path(emotion, variant_index, "wav")
        try:
            if audio_data.dtype != np.int16:
                audio_data = np.clip(audio_data, -1.0, 1.0)
                audio_data = (audio_data * 32767).astype(np.int16)

            wavfile.write(str(audio_path), sample_rate, audio_data)
            self.logger.debug(f"Saved {emotion} variant {variant_index + 1} to {audio_path}")
        except Exception as e:
            self.logger.error(f"Failed to save {emotion} variant {variant_index + 1}: {e}")
            self._on_emotion_variant_failed(e, variant_index)
            return

        # Calculate duration
        duration = len(audio_data) / sample_rate

        # Use audio path as placeholder for embedding (extraction happens in Refinement)
        embedding_path = audio_path

        # Update UI
        self.emotions_panel.set_variant_complete(emotion, variant_index, audio_path, embedding_path, duration)

        # Move to next variant
        state['current_variant'] = variant_index + 1
        self._generate_next_emotion_variant()

    def _on_emotion_variant_failed(self, error: Exception, variant_index: int):
        """Handle emotion variant generation failure."""
        state = getattr(self, '_emotion_generation_state', None)
        if not state:
            return

        emotion = state['emotion']
        self.logger.error(f"{emotion} variant {variant_index + 1} failed: {error}")

        self.emotions_panel.set_variant_error(emotion, variant_index, str(error))

        # Continue with next variant
        state['current_variant'] = variant_index + 1
        self._generate_next_emotion_variant()

    def _finish_emotion_generation(self):
        """Finish emotion variant generation."""
        state = getattr(self, '_emotion_generation_state', None)
        if state:
            emotion = state['emotion']
            self.emotions_panel.finish_generation(emotion)
            self.logger.info(f"{emotion} generation complete")
        self._emotion_generation_state = None

    def _on_emotion_variant_selected(self, emotion: str, variant_index: int):
        """Handle emotion variant selection."""
        self.logger.debug(f"Emotion variant selected: {emotion} variant {variant_index}")
        self.set_has_unsaved_work(True)

    def _on_neutral_selection_changed(self, has_selection: bool):
        """Handle neutral selection state change."""
        self.logger.debug(f"Neutral selection changed: {has_selection}")
        # Neutral selection is required to proceed to Refinement

    def _on_proceed_to_refinement(self):
        """
        Handle proceed to refinement request from emotions panel.

        Transfers all selected emotion samples to the Refinement panel.
        """
        self.logger.info("Proceeding to Refinement tab")

        # Get all selected emotion samples
        selected_emotions = self.emotions_panel.get_selected_emotions()
        if not selected_emotions:
            self.logger.warning("No emotions selected")
            return

        if "neutral" not in selected_emotions:
            self.logger.warning("Neutral not selected - required for proceeding")
            return

        # Build samples dict for refinement panel: emotion -> (audio_path, embedding_path)
        samples = {}
        for emotion in selected_emotions:
            paths = self.emotions_panel.get_selected_variant_paths(emotion)
            if paths:
                audio_path, embedding_path = paths
                samples[emotion] = (audio_path, embedding_path)

        # Get voice description for saving
        description = self.description_panel.get_description()

        # QA Round 2 Item #2: Preserve voice name across tab navigation
        # Get voice name BEFORE clear() to restore it after
        existing_voice_name = self.refinement_panel.get_voice_name()
        if not existing_voice_name:
            # Fallback to description panel voice name
            existing_voice_name = self.description_panel.get_voice_name()

        # Set up refinement panel
        self.refinement_panel.clear()
        self.refinement_panel.set_samples(samples)
        self.refinement_panel.set_voice_description(description)

        # QA Round 2 Item #2: Restore voice name after clear()
        if existing_voice_name:
            self.refinement_panel.set_voice_name(existing_voice_name)

        # Switch to Refinement tab (index 2)
        self.tab_widget.setCurrentIndex(2)

        self.logger.info(f"Transferred {len(samples)} emotion samples to Refinement")

    # =========================================================================
    # Emotion Variants: Refinement Tab Signal Handlers
    # =========================================================================

    def _on_batch_extraction_requested(self):
        """
        Handle batch extraction request from refinement panel.

        Extracts embeddings for all selected emotion samples sequentially.
        """
        self.logger.info("Batch extraction requested")

        emotions = self.refinement_panel.get_sample_emotions()
        if not emotions:
            self.logger.warning("No samples to extract")
            return

        # Check TTS service
        if not self._tts_service or not self._tts_service.is_running():
            self.refinement_panel.set_extraction_error("Voice service not available")
            return

        # Start extraction
        self.refinement_panel.set_extracting(True, "Starting extraction...")

        self._batch_extraction_state = {
            'emotions': emotions,
            'current_index': 0,
            'total': len(emotions),
            'cancelled': False
        }

        self._extract_next_emotion_embedding()

    def _extract_next_emotion_embedding(self):
        """Extract embedding for the next emotion in the batch."""
        state = getattr(self, '_batch_extraction_state', None)
        if not state or state['cancelled'] or state['current_index'] >= state['total']:
            self._finish_batch_extraction()
            return

        emotions = state['emotions']
        index = state['current_index']
        emotion = emotions[index]

        self.logger.debug(f"Extracting embedding for {emotion} ({index + 1}/{state['total']})")

        # Update UI
        self.refinement_panel.set_extraction_progress(index + 1, state['total'], emotion)
        self.refinement_panel.set_emotion_extracting(emotion)

        # Get audio path for this emotion
        audio_paths = self.refinement_panel.get_audio_paths()
        audio_path = audio_paths.get(emotion)

        if not audio_path or not audio_path.exists():
            self.logger.error(f"Audio file not found for {emotion}")
            self._on_emotion_extraction_failed(Exception(f"Audio not found"), emotion)
            return

        # Run async extraction
        self._run_async_task(
            self._extract_emotion_embedding(emotion, audio_path),
            on_success=self._on_emotion_extraction_complete,
            on_error=lambda e: self._on_emotion_extraction_failed(e, emotion)
        )

    async def _extract_emotion_embedding(self, emotion: str, audio_path: Path) -> dict:
        """
        Extract embedding for a single emotion.

        Args:
            emotion: Emotion ID
            audio_path: Path to the audio sample

        Returns:
            Dictionary with emotion and embedding_path
        """
        import torch

        self.logger.debug(f"Extracting embedding from {audio_path}")

        # QA8: Get ref_text from session data - required for embedding extraction
        # Priority: clone_transcript (Clone workflow) > preview_text (From Scratch workflow)
        ref_text = ""
        session_data = self._session_manager.load_session_data()
        if session_data:
            if session_data.get("clone_transcript"):
                ref_text = session_data["clone_transcript"]
                self.logger.info(f"Using clone transcript for {emotion} extraction: {ref_text[:50]}...")
            elif session_data.get("preview_text"):
                ref_text = session_data["preview_text"]
                self.logger.info(f"Using preview text for {emotion} extraction: {ref_text[:50]}...")

        if not ref_text:
            raise ValueError(f"No ref_text available for {emotion} extraction. Please ensure preview text is provided.")

        voice_clone_prompt = await self._tts_service.create_voice_clone_prompt(
            ref_audio=audio_path,
            ref_text=ref_text
        )

        # Move tensors to CPU before saving for cross-device compatibility
        if voice_clone_prompt.ref_code is not None:
            voice_clone_prompt.ref_code = voice_clone_prompt.ref_code.cpu()
        if voice_clone_prompt.ref_spk_embedding is not None:
            voice_clone_prompt.ref_spk_embedding = voice_clone_prompt.ref_spk_embedding.cpu()

        # Save embedding to session directory (emotion-specific path)
        embedding_path = self._session_manager.get_emotion_embedding_path(emotion)
        torch.save(voice_clone_prompt, str(embedding_path))

        # Verify save was successful by reloading
        try:
            verify_prompt = torch.load(str(embedding_path), map_location='cpu', weights_only=False)
            if verify_prompt.ref_spk_embedding is None:
                raise ValueError("Embedding verification failed: ref_spk_embedding is None")
            self.logger.info(f"Saved and verified {emotion} embedding to {embedding_path}")
        except Exception as verify_error:
            self.logger.error(f"Embedding verification failed: {verify_error}")
            raise RuntimeError(f"Failed to save {emotion} embedding correctly: {verify_error}")

        return {
            'emotion': emotion,
            'embedding_path': embedding_path
        }

    def _on_emotion_extraction_complete(self, result: dict):
        """Handle successful emotion embedding extraction."""
        state = getattr(self, '_batch_extraction_state', None)
        if not state or state['cancelled']:
            return

        emotion = result['emotion']
        embedding_path = result['embedding_path']

        self.logger.debug(f"{emotion} embedding extraction complete")

        # Update UI
        self.refinement_panel.set_emotion_complete(emotion, embedding_path)

        # Move to next emotion
        state['current_index'] += 1
        self._extract_next_emotion_embedding()

    def _on_emotion_extraction_failed(self, error: Exception, emotion: str):
        """Handle emotion embedding extraction failure."""
        self.logger.error(f"{emotion} extraction failed: {error}")

        self.refinement_panel.set_emotion_error(emotion, str(error)[:30])

        # Continue with next emotion
        state = getattr(self, '_batch_extraction_state', None)
        if state:
            state['current_index'] += 1
            self._extract_next_emotion_embedding()

    def _finish_batch_extraction(self):
        """Finish batch extraction and generate preview."""
        self.logger.info("Batch extraction complete")

        self.refinement_panel.finish_extraction()
        self._batch_extraction_state = None

        # Generate preview using neutral embedding
        self._generate_refinement_preview()

    def _generate_refinement_preview(self):
        """Generate a preview audio using the neutral embedding."""
        embedding_paths = self.refinement_panel.get_embedding_paths()
        neutral_embedding = embedding_paths.get("neutral")

        if not neutral_embedding or not neutral_embedding.exists():
            self.logger.warning("No neutral embedding for preview")
            return

        preview_text = self.description_panel.get_preview_text()
        language = self.description_panel.get_language()

        # QA Round 2 Item #3: Show loading indicator before async generation
        self.refinement_panel.set_preview_generating(True)

        self._run_async_task(
            self._generate_preview_audio(neutral_embedding, preview_text, language),
            on_success=self._on_refinement_preview_complete,
            on_error=self._on_refinement_preview_error
        )

    def _on_refinement_preview_error(self, error):
        """QA Round 2 Item #3: Handle preview generation error."""
        self.logger.error(f"Preview generation failed: {error}")
        self.refinement_panel.set_preview_generating(False)
        self.refinement_panel.preview_status.setText("Preview generation failed")
        self.refinement_panel.preview_status.setStyleSheet("color: red;")

    async def _generate_preview_audio(
        self,
        embedding_path: Path,
        preview_text: str,
        language: str
    ) -> Path:
        """Generate preview audio using an embedding."""
        response = await self._tts_service.generate_with_embedding(
            text=preview_text,
            embedding_path=embedding_path,
            language=language if language != "Auto" else "Auto",
            streaming=False,
        )

        if not response.success:
            raise RuntimeError(response.error_message or "Preview generation failed")

        # Save preview audio
        audio_data = response.audio_data
        sample_rate = response.sample_rate

        preview_path = self._session_manager.session_dir / "refinement_preview.wav"

        if audio_data.dtype != np.int16:
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)

        wavfile.write(str(preview_path), sample_rate, audio_data)

        return preview_path

    def _on_refinement_preview_complete(self, preview_path: Path):
        """Handle successful preview generation."""
        self.logger.info(f"Refinement preview generated: {preview_path}")
        self.refinement_panel.set_preview_audio(preview_path)

    def _on_preview_requested(self):
        """Handle preview request from refinement panel."""
        self._generate_refinement_preview()

    def _on_back_to_emotions(self):
        """Handle back to emotions request."""
        self.logger.debug("Returning to Emotions tab")
        self.tab_widget.setCurrentIndex(1)  # Emotions tab

    def _on_refinement_save_requested(self, voice_name: str, selected_emotions: list):
        """
        Handle save request from refinement panel.

        Saves the voice with emotion subfolders to the library.

        Args:
            voice_name: Name for the saved voice
            selected_emotions: List of emotion IDs with complete extraction
        """
        self.logger.info(f"Saving voice '{voice_name}' with emotions: {selected_emotions}")

        if not voice_name:
            self.refinement_panel.set_save_error("Voice name is required")
            return

        if not selected_emotions or "neutral" not in selected_emotions:
            self.refinement_panel.set_save_error("At least Neutral emotion is required")
            return

        # Validate voice name
        invalid_chars = '<>:"/\\|?*'
        if any(char in voice_name for char in invalid_chars):
            self.refinement_panel.set_save_error(f"Name cannot contain: {invalid_chars}")
            return

        # Determine save location
        save_dir = self._get_embeddings_save_dir() / voice_name

        # Check for existing voice
        if save_dir.exists():
            reply = QMessageBox.question(
                self, "Voice Exists",
                f"A voice named '{voice_name}' already exists.\n\nDo you want to replace it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Start save
        self.refinement_panel.set_saving(True, "Saving voice...")

        try:
            # Create voice directory
            save_dir.mkdir(parents=True, exist_ok=True)

            # Get paths from refinement panel
            embedding_paths = self.refinement_panel.get_embedding_paths()
            audio_paths = self.refinement_panel.get_audio_paths()

            # Copy embeddings and audio for each emotion
            for emotion in selected_emotions:
                if emotion in embedding_paths:
                    # Create emotion subfolder
                    emotion_dir = save_dir / emotion
                    emotion_dir.mkdir(exist_ok=True)

                    # Copy embedding
                    src_embedding = embedding_paths[emotion]
                    if src_embedding and src_embedding.exists():
                        dest_embedding = emotion_dir / "embedding.pt"
                        shutil.copy2(src_embedding, dest_embedding)
                        self.logger.debug(f"Copied {emotion} embedding")

                    # Copy source audio
                    src_audio = audio_paths.get(emotion)
                    if src_audio and src_audio.exists():
                        dest_audio = emotion_dir / "source_audio.wav"
                        shutil.copy2(src_audio, dest_audio)
                        self.logger.debug(f"Copied {emotion} source audio")

            # Copy preview audio if exists
            preview_path = self._session_manager.session_dir / "refinement_preview.wav"
            if preview_path.exists():
                dest_preview = save_dir / "preview.wav"
                shutil.copy2(preview_path, dest_preview)

            # Create metadata.json with Emotion Variants schema (v2.0)
            description = self.refinement_panel.get_voice_description()

            # Get transcription (ref_text) for TTS generation
            # Priority: clone_transcript (Clone workflow) > preview_text (VoiceDesign workflow)
            session_data = self._session_manager.load_session_data()
            transcription = None
            if session_data:
                transcription = session_data.get("clone_transcript") or session_data.get("preview_text")

            metadata = {
                "version": "2.0",
                "name": voice_name,
                "description": description,
                "transcription": transcription,  # ref_text for TTS generation
                "voice_type": "embedding",
                "available_emotions": selected_emotions,
                "emotion_capable": True,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            metadata_path = save_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # Success
            self.refinement_panel.set_save_complete(voice_name)
            self.set_has_unsaved_work(False)
            self.voice_saved.emit(voice_name)

            # QA Round 2 Item #4: Change Cancel to Close after successful save
            self.cancel_button.setText("Close")

            self.logger.info(f"Voice '{voice_name}' saved to {save_dir} with {len(selected_emotions)} emotions")

        except Exception as e:
            self.logger.error(f"Failed to save voice: {e}")
            self.refinement_panel.set_save_error(str(e))

    # =========================================================================
    # QA8: From Sample tab removed - clone functionality moved to Clone sub-tab
    # Core transcription/extraction methods retained for Clone sub-tab use
    # =========================================================================

    async def _transcribe_audio(self, file_path: str) -> str:
        """
        Transcribe audio file using whisper service.

        Args:
            file_path: Path to the audio file

        Returns:
            Transcribed text
        """
        from pathlib import Path

        self.logger.debug(f"Starting transcription of {file_path}")

        result = await self._whisper_service.transcribe_file(
            file_path=Path(file_path),
            language=None  # Auto-detect
        )

        return result.text

    async def _extract_voice_embedding(
        self,
        audio_path: str,
        transcript: str,
        preview_text: str,
        language: str
    ) -> Path:
        """
        Extract voice embedding from audio sample (QA6 fix).

        This method:
        1. Extracts the voice clone prompt (embedding) from reference audio
        2. Saves the embedding to session directory as embedding.pt
        3. Generates a preview using that embedding to confirm voice capture
        4. Returns the preview audio path

        Args:
            audio_path: Path to the audio file
            transcript: Transcript of the audio
            preview_text: Text to use for preview generation
            language: Language for TTS generation

        Returns:
            Path to the generated preview audio
        """
        import torch
        from pathlib import Path as PathLib

        ref_audio_path = PathLib(audio_path)
        self.logger.debug(f"Extracting embedding from {audio_path}")

        # Step 1: Extract the voice clone prompt (acoustic embedding)
        self.logger.info("Step 1: Extracting voice clone prompt...")
        try:
            voice_clone_prompt = await self._tts_service.create_voice_clone_prompt(
                ref_audio=ref_audio_path,
                ref_text=transcript
            )
        except Exception as e:
            self.logger.error(f"Voice clone prompt extraction failed: {e}")
            raise RuntimeError(f"Embedding extraction failed: {e}")

        # Move tensors to CPU before saving for cross-device compatibility
        if voice_clone_prompt.ref_code is not None:
            voice_clone_prompt.ref_code = voice_clone_prompt.ref_code.cpu()
        if voice_clone_prompt.ref_spk_embedding is not None:
            voice_clone_prompt.ref_spk_embedding = voice_clone_prompt.ref_spk_embedding.cpu()

        # Step 2: Save embedding to session directory
        embedding_path = self._session_manager.session_dir / "embedding.pt"
        torch.save(voice_clone_prompt, str(embedding_path))

        # Verify save was successful by reloading
        try:
            verify_prompt = torch.load(str(embedding_path), map_location='cpu', weights_only=False)
            if verify_prompt.ref_spk_embedding is None:
                raise ValueError("Embedding verification failed: ref_spk_embedding is None")
            self.logger.info(f"Step 2: Saved and verified embedding to {embedding_path}")
        except Exception as verify_error:
            self.logger.error(f"Embedding verification failed: {verify_error}")
            raise RuntimeError(f"Failed to save embedding correctly: {verify_error}")

        # Step 3: Generate preview using the extracted embedding
        self.logger.info("Step 3: Generating preview with embedding...")
        response = await self._tts_service.generate_with_embedding(
            text=preview_text,
            embedding_path=embedding_path,
            language=language if language != "Auto" else "Auto",
            streaming=False,
        )

        if not response.success:
            raise RuntimeError(response.error_message or "Preview generation failed")

        # Step 4: Save preview audio to session directory
        audio_data = response.audio_data
        sample_rate = response.sample_rate

        preview_path = self._session_manager.session_dir / "extraction_preview.wav"

        # Convert and save audio
        if audio_data.dtype != np.int16:
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)

        wavfile.write(str(preview_path), sample_rate, audio_data)
        self.logger.info(f"Step 4: Saved extraction preview to {preview_path}")

        return preview_path

    # =========================================================================
    # QA8: Clone Tab Handlers
    # =========================================================================

    def _on_clone_transcribe_requested(self, file_path: str):
        """
        Handle transcription request from Clone tab (QA8).

        Args:
            file_path: Path to the audio file to transcribe
        """
        self.logger.info(f"Clone transcription requested for: {file_path}")

        # Try to get whisper_service from parent if not set
        if not self._whisper_service:
            self.logger.debug("Whisper service not set, trying to get from parent")
            parent = self.parent()
            if parent and hasattr(parent, 'whisper_service') and parent.whisper_service:
                self._whisper_service = parent.whisper_service
                self.logger.info("Got whisper_service from parent")

        # Check if whisper service is available
        if not self._whisper_service:
            self.logger.error("Whisper service not available")
            self.description_panel.set_clone_transcription_error(
                "Transcription service not available. Please close this dialog and try again in a moment."
            )
            return

        # Check if service is running
        try:
            if hasattr(self._whisper_service, 'is_running') and not self._whisper_service.is_running():
                self.logger.error("Whisper service not running")
                self.description_panel.set_clone_transcription_error(
                    "Transcription service is still initializing. Please wait a moment and try again."
                )
                return
        except Exception as e:
            self.logger.warning(f"Could not check whisper service status: {e}")

        # Start transcription
        self.description_panel.set_clone_transcribing(True, "Transcribing...")

        # Run async transcription
        self._run_async_task(
            self._transcribe_audio(file_path),
            on_success=self._on_clone_transcription_complete,
            on_error=self._on_clone_transcription_failed
        )

    def _on_clone_transcription_complete(self, text: str):
        """
        Handle successful clone transcription (QA8).

        Args:
            text: Transcribed text
        """
        self.logger.info(f"Clone transcription complete: {len(text)} characters")
        self.description_panel.set_clone_transcription_complete(text)

    def _on_clone_transcription_failed(self, error: Exception):
        """
        Handle clone transcription failure (QA8).

        Args:
            error: The exception that occurred
        """
        self.logger.error(f"Clone transcription failed: {error}")
        self.description_panel.set_clone_transcription_error(str(error))

    def _on_clone_proceed_requested(self, audio_path: str, transcript: str, voice_name: str):
        """
        Handle Proceed request from Clone tab (QA8).

        Transfers sample data to Emotions tab for building voice with emotion variants.

        Args:
            audio_path: Path to the audio file
            transcript: Transcript of the audio
            voice_name: Name for the voice
        """
        self.logger.info(f"Clone proceed requested: audio={audio_path}, name={voice_name}")

        from pathlib import Path as PathLib
        audio_path_obj = PathLib(audio_path)

        # Set voice name in Refinement panel
        if voice_name:
            self.refinement_panel.set_voice_name(voice_name)

        # Set voice description from transcript (if available)
        if transcript:
            self.emotions_panel.set_voice_description(
                f"Voice cloned from sample. Original transcript: {transcript}"
            )

        # Set preview text in emotions panel from transcript
        if transcript:
            preview_text = transcript[:100] + "..." if len(transcript) > 100 else transcript
            self.emotions_panel.set_preview_text(preview_text)

        # Transfer sample audio to Emotions-Neutral tab
        if audio_path_obj.exists():
            self.emotions_panel.load_neutral_sample(
                audio_path=audio_path_obj,
                embedding_path=None,  # No embedding yet
                duration=None  # Will be auto-detected
            )
            self.logger.info(f"Transferred sample to Neutral tab: {audio_path}")

        # Save session data for recovery (QA8: include clone_transcript for extraction)
        self._session_manager.save_session_data(
            voice_description=transcript or "",
            preview_text=transcript[:100] if transcript else "",
            voice_name=voice_name or "",
            language="Auto",
            clone_transcript=transcript or ""  # QA8: Store for ref_text during extraction
        )

        # Switch to Emotions tab (index 1)
        self.tab_widget.setCurrentIndex(1)
        self.logger.info("Switched to Emotions tab from Clone tab")

    # =========================================================================
    # Footer
    # =========================================================================

    def _create_footer(self, layout: QVBoxLayout):
        """Create the footer with New Voice and Cancel buttons.

        QA8: Saving happens in Refinement tab or From Description-Clone sub-tab.
        """
        footer_layout = QHBoxLayout()
        footer_layout.setSpacing(8)

        # QA Round 2 Item #6: Left side - New Voice button to start fresh
        self.new_voice_button = QPushButton("New Voice")
        self.new_voice_button.setObjectName("new_voice_button")
        self.new_voice_button.setToolTip("Clear session and start with a new voice")
        self.new_voice_button.clicked.connect(self._on_new_voice_clicked)
        footer_layout.addWidget(self.new_voice_button)

        footer_layout.addStretch()

        # Right side - Cancel button only
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("cancel_button")
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        footer_layout.addWidget(self.cancel_button)

        layout.addLayout(footer_layout)

    def _setup_accessibility(self):
        """Configure accessibility properties for the dialog."""
        # Dialog
        self.setAccessibleName("Voice Design Studio")
        self.setAccessibleDescription(
            "Dialog for creating custom voices from descriptions, cloning from audio, and adding emotion variants"
        )

        # Tab widget
        self.tab_widget.setAccessibleName("Voice creation workflow")
        self.tab_widget.setAccessibleDescription(
            "Workflow tabs: From Description, Emotions, and Refinement"
        )

        # Buttons
        self.new_voice_button.setAccessibleName("New Voice")
        self.new_voice_button.setAccessibleDescription("Clear session and start a new voice")
        self.cancel_button.setAccessibleName("Cancel")
        self.cancel_button.setAccessibleDescription("Close dialog without saving")

    def _setup_shortcuts(self):
        """Configure keyboard shortcuts."""
        # Escape to close
        escape_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        escape_shortcut.activated.connect(self._on_cancel_clicked)

        # Tab switching with Ctrl+1 through Ctrl+3
        # QA8: Removed Ctrl+4 (From Sample tab removed)
        tab1_shortcut = QShortcut(QKeySequence("Ctrl+1"), self)
        tab1_shortcut.activated.connect(lambda: self.tab_widget.setCurrentIndex(0))

        tab2_shortcut = QShortcut(QKeySequence("Ctrl+2"), self)
        tab2_shortcut.activated.connect(lambda: self.tab_widget.setCurrentIndex(1))

        tab3_shortcut = QShortcut(QKeySequence("Ctrl+3"), self)
        tab3_shortcut.activated.connect(lambda: self.tab_widget.setCurrentIndex(2))

    def _on_tab_changed(self, index: int):
        """
        Handle tab change events.

        QA8: Updated for 3 tabs (Description, Emotions, Refinement)

        Args:
            index: Index of the newly selected tab
        """
        tab_names = ["From Description", "Emotions", "Refinement"]
        tab_name = tab_names[index] if index < len(tab_names) else "Unknown"
        self.logger.debug(f"Tab changed to: {tab_name} (index {index})")

        # Sync voice description to Emotions panel when switching to it
        if index == 1:  # Emotions tab
            description = self.description_panel.get_description()
            preview_text = self.description_panel.get_preview_text()
            self.emotions_panel.set_voice_description(description)
            self.emotions_panel.set_preview_text(preview_text)

    def _on_cancel_clicked(self):
        """Handle cancel button click or Escape key."""
        self.logger.debug("Cancel clicked")

        # Story 4.2: Show confirmation if user has unsaved selection
        if self._has_unsaved_selection():
            reply = QMessageBox.question(
                self, "Unsaved Selection",
                "You have a selected variant that hasn't been saved.\n\n"
                "Your generated variants will be preserved and you can save them later.\n\n"
                "Close anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return  # User cancelled, stay in dialog

        self.dialog_closing.emit(self._has_unsaved_work)

        # Story 4.1 REVISED: Do NOT cleanup session on close
        # Temp files persist so user can reopen and save additional variants
        # Cleanup happens only on Regenerate (Story 1.9) or orphan cleanup (>24h)

        self.reject()

    def _on_new_voice_clicked(self):
        """
        QA Round 2 Item #6: Handle New Voice button click.

        Clears the session and resets all panels for a fresh start.
        """
        self.logger.debug("New Voice clicked")

        # Confirm if there's unsaved work
        if self._has_unsaved_work or self._has_unsaved_selection():
            reply = QMessageBox.question(
                self, "Start New Voice",
                "This will clear your current session.\n\n"
                "Any unsaved variants will be lost.\n\n"
                "Start a new voice?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Clear session files
        if self._session_manager:
            self._session_manager.cleanup()
            # Reinitialize session manager for new session
            self._session_manager = SessionManager()

        # Clear all panels
        self.description_panel.clear()
        self.emotions_panel.clear()
        self.refinement_panel.clear()

        # Reset to first tab
        self.tab_widget.setCurrentIndex(0)

        # Reset unsaved work flag
        self._has_unsaved_work = False

        # Reset cancel button text
        self.cancel_button.setText("Cancel")

        self.logger.info("Started new voice session")

    def _on_save_clicked(self):
        """Handle save button click."""
        self.logger.debug("Save clicked")

        # Get current tab
        if self.tab_widget.currentIndex() == 0:  # Description tab
            self._save_from_description()
        else:
            # Sample tab save will be implemented in Story 2.x
            self.logger.warning("Save from sample tab not yet implemented")

    def _save_from_description(self):
        """Save voice from description panel (Story 1.5)."""
        voice_name = self.description_panel.get_voice_name()
        embedding_path = self.description_panel.get_generated_embedding_path()
        description = self.description_panel.get_description()

        if not voice_name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a voice name.")
            return

        if not embedding_path or not embedding_path.exists():
            QMessageBox.warning(self, "No Embedding", "No generated embedding found. Please generate a voice first.")
            return

        # Validate voice name (no special characters that would break paths)
        invalid_chars = '<>:"/\\|?*'
        if any(char in voice_name for char in invalid_chars):
            QMessageBox.warning(
                self, "Invalid Name",
                f"Voice name cannot contain: {invalid_chars}"
            )
            return

        # Determine save location
        save_dir = self._get_embeddings_save_dir() / voice_name

        # Check for existing voice (overwrite warning)
        if save_dir.exists():
            reply = QMessageBox.question(
                self, "Voice Exists",
                f"A voice named '{voice_name}' already exists.\n\nDo you want to replace it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Start save operation
        self.description_panel.set_saving(True, "Saving...")

        try:
            # Create directory
            save_dir.mkdir(parents=True, exist_ok=True)

            # Copy embedding file
            dest_embedding = save_dir / "embedding.pt"
            shutil.copy2(embedding_path, dest_embedding)

            # Copy audio preview if exists
            audio_path = self.description_panel.get_generated_audio_path()
            if audio_path and audio_path.exists():
                dest_audio = save_dir / "preview.wav"
                shutil.copy2(audio_path, dest_audio)

            # Get transcription (ref_text) for TTS generation - use preview text
            transcription = self.description_panel.get_preview_text()

            # Create metadata.json
            metadata = {
                "name": voice_name,
                "description": description,
                "transcription": transcription,  # ref_text for TTS generation
                "voice_type": "designed",
                "emotion_capable": True,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            metadata_path = save_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # Success
            self.description_panel.set_save_complete(voice_name)
            self.set_has_unsaved_work(False)
            self.voice_saved.emit(voice_name)

            # QA Round 2 Item #4: Change Cancel to Close after successful save
            self.cancel_button.setText("Close")

            self.logger.info(f"Voice saved successfully: {voice_name} at {save_dir}")

        except Exception as e:
            self.logger.error(f"Failed to save voice: {e}")
            self.description_panel.set_save_error(str(e))

    def _get_embeddings_save_dir(self) -> Path:
        """
        Get the directory for saving voice embeddings.

        Returns:
            Path to voice_files/embeddings directory
        """
        # QA8: Use portable paths for consistency with VoiceProfileManager loading
        # This ensures saved voices are found when the app rescans
        from myvoice.utils.portable_paths import get_voice_files_path
        base_dir = get_voice_files_path() / "embeddings"
        return base_dir

    def _on_refine_requested(self):
        """
        QA7: Handle refine request from description panel.

        Transfers all 5 generated variants to the Emotions-Neutral tab
        for emotion variant creation workflow.

        Flow:
        1. Get all 5 variant paths from description panel
        2. Copy variant files to emotion_variants/neutral/ in session
        3. Load variants into Emotions panel Neutral tab
        4. Pre-select the variant that was selected in Description
        5. Copy Voice Name to Refinement tab
        6. Set voice description and preview text in Emotions panel
        7. Switch to Emotions tab
        """
        self.logger.info("QA7: Refine voice requested - transferring to Emotions tab")

        # Get selected variant index first
        selected_index = self.description_panel.get_selected_variant_index()
        if selected_index is None:
            self.logger.warning("No variant selected for refinement")
            QMessageBox.warning(
                self, "No Selection",
                "Please select a voice variant to refine."
            )
            return

        # Get all variant paths (may include None for incomplete)
        all_variant_paths = self.description_panel.get_all_variant_paths()
        if not all_variant_paths or not any(v for v in all_variant_paths if v):
            self.logger.warning("No complete variants found")
            QMessageBox.warning(
                self, "No Variants",
                "No generated variants found. Please generate voices first."
            )
            return

        # Copy variant files to emotion_variants/neutral/
        copied_variants = []
        for idx, variant_data in enumerate(all_variant_paths):
            if variant_data is None:
                copied_variants.append(None)
                continue

            audio_path, embedding_path, duration = variant_data
            if audio_path and audio_path.exists():
                # Get destination path in emotion_variants/neutral/
                dest_path = self._session_manager.get_emotion_variant_path("neutral", idx, "wav")
                try:
                    shutil.copy2(audio_path, dest_path)
                    # Use dest_path as both audio and embedding (placeholder)
                    copied_variants.append((dest_path, dest_path, duration or 3.0))
                    self.logger.debug(f"Copied variant {idx} to {dest_path}")
                except Exception as e:
                    self.logger.error(f"Failed to copy variant {idx}: {e}")
                    copied_variants.append(None)
            else:
                copied_variants.append(None)

        # Filter out None entries for loading (but keep indices correct)
        valid_variants = [(audio, emb, dur) for v in copied_variants if v for audio, emb, dur in [v]]
        if not valid_variants:
            self.logger.error("No variants were copied successfully")
            return

        # Get voice description and preview text
        description = self.description_panel.get_description()
        preview_text = self.description_panel.get_preview_text()
        voice_name = self.description_panel.get_voice_name()
        language = self.description_panel.get_language()

        # QA7: Save session data for recovery
        self._session_manager.save_session_data(
            voice_description=description,
            preview_text=preview_text,
            voice_name=voice_name,
            language=language
        )

        # Set up Emotions panel
        self.emotions_panel.set_voice_description(description)
        self.emotions_panel.set_preview_text(preview_text)

        # Load variants into Neutral tab with selection
        # Build list preserving indices
        variants_for_loading = []
        for v in copied_variants:
            if v:
                variants_for_loading.append(v)
        self.emotions_panel.load_neutral_variants(variants_for_loading, selected_index)

        # Copy voice name to Refinement tab
        if voice_name:
            self.refinement_panel.set_voice_name(voice_name)

        # Switch to Emotions tab (index 1)
        self.tab_widget.setCurrentIndex(1)

        # QA Round 2 Item #1: Switch to Happy tab instead of Neutral
        # This prevents users from accidentally regenerating Neutral when they
        # think "Generate Variants" does all emotions
        self.emotions_panel.select_emotion_tab("happy")

        self.logger.info(
            f"QA7: Transferred {len(valid_variants)} variants to Emotions-Happy tab, "
            f"selected={selected_index}, voice_name='{voice_name}'"
        )

    def closeEvent(self, event):
        """
        Handle dialog close event.

        Emits dialog_closing signal. Story 4.1 REVISED: Does NOT cleanup
        session files - they persist so user can save additional variants later.

        Args:
            event: Close event
        """
        self.logger.debug("Dialog close event")

        # QA Round 2 Item #6: Save current tab state for restoration
        current_tab = self.tab_widget.currentIndex()
        self._session_manager.save_session_data(current_tab_index=current_tab)
        self.logger.debug(f"Saved tab state: {current_tab}")

        # Story 4.2: Show confirmation if user has unsaved selection
        if self._has_unsaved_selection():
            reply = QMessageBox.question(
                self, "Unsaved Selection",
                "You have a selected variant that hasn't been saved.\n\n"
                "Your generated variants will be preserved and you can save them later.\n\n"
                "Close anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()  # Cancel the close
                return

        self.dialog_closing.emit(self._has_unsaved_work)

        # Story 4.1 REVISED: Do NOT cleanup session on close
        # Temp files persist so user can reopen and save additional variants
        # Cleanup happens only on Regenerate (Story 1.9) or orphan cleanup (>24h)

        super().closeEvent(event)

    def _cleanup_session(self):
        """
        Clean up the session directory and all temp files (Story 4.2).

        NOTE: This is NOT called on dialog close per Story 4.1 revision.
        Reserved for orphan cleanup (sessions >24h old) on app restart.
        For variant cleanup on Regenerate, use _session_manager.clear_variant_files().
        """
        if self._session_manager and not self._session_manager.is_cleaned:
            self.logger.info(f"Cleaning up session: {self._session_manager.session_id}")
            self._session_manager.cleanup()

    @property
    def session_dir(self) -> Path:
        """
        Get the session directory path for temp file storage.

        Returns:
            Path to the session directory
        """
        return self._session_manager.session_dir

    def get_variant_output_path(self, variant_index: int, extension: str) -> Path:
        """
        Get output path for a variant file within the session directory.

        Use this when generating variants to ensure files are written to
        the session directory for proper cleanup (Story 4.1).

        Args:
            variant_index: Index of the variant (1-5)
            extension: File extension without dot ("pt" or "wav")

        Returns:
            Path to write the variant file

        Example:
            embedding_path = dialog.get_variant_output_path(1, "pt")
            audio_path = dialog.get_variant_output_path(1, "wav")
        """
        return self._session_manager.get_variant_path(variant_index, extension)

    def set_has_unsaved_work(self, has_work: bool):
        """
        Update the unsaved work flag.

        Args:
            has_work: True if there is unsaved work
        """
        self._has_unsaved_work = has_work

    def _has_unsaved_selection(self) -> bool:
        """
        Check if user has a variant selected but not saved (Story 4.2).

        Used to show confirmation dialog on close.

        Returns:
            True if there's a selected variant that hasn't been saved
        """
        # Check description tab
        if self.tab_widget.currentIndex() == 0:
            return (self.description_panel.has_variant_selection() and
                    not self.description_panel.get_voice_name())

        # Check sample tab (future: add similar logic when implemented)
        return False

    def get_active_path(self) -> str:
        """
        Get the currently active creation path.

        Returns:
            "description" or "sample" based on active tab
        """
        return "description" if self.tab_widget.currentIndex() == 0 else "sample"
