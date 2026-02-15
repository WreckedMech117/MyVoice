"""
Voice Variant List Widget

Container for 5 VoiceVariantRow widgets with progressive reveal logic.
Manages generation queue, selection state, and audio playback coordination.

Story 1.7: Generate 5 Variations with Progressive Reveal
Story 1.8: Audition and Select Variant
Story 1.9: Regenerate All Variations
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QButtonGroup, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from myvoice.ui.dialogs.voice_design_studio.voice_variant_row import VoiceVariantRow, VariantState


class VoiceVariantList(QWidget):
    """
    Container widget for 5 voice variant rows with progressive reveal.

    Manages:
    - 5 VoiceVariantRow widgets in a vertical layout
    - Single selection via radio button group
    - Audio playback coordination (only one playing at a time)
    - Progressive reveal during generation
    - Status display ("5 variations ready", "Generating 2/5...")

    Signals:
        variant_selected: Emitted when a variant is selected (variant_index)
        variant_deselected: Emitted when selection is cleared
        all_complete: Emitted when all 5 variants are generated
        generation_progress: Emitted during generation (completed_count, total_count)
    """

    variant_selected = pyqtSignal(int)  # variant_index
    variant_deselected = pyqtSignal()
    all_complete = pyqtSignal()
    generation_progress = pyqtSignal(int, int)  # completed, total

    NUM_VARIANTS = 5

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the Voice Variant List.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        self._selected_index: Optional[int] = None
        self._currently_playing: Optional[int] = None
        self._is_generating = False

        # Audio player for playback
        self._media_player = QMediaPlayer()
        self._audio_output = QAudioOutput()
        self._media_player.setAudioOutput(self._audio_output)

        self._create_ui()
        self._setup_connections()
        self._setup_accessibility()

        self.logger.debug("VoiceVariantList initialized")

    def _create_ui(self):
        """Create the list UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Header with status
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        self.status_label = QLabel("")
        self.status_label.setObjectName("variant_list_status")
        header_layout.addWidget(self.status_label)

        header_layout.addStretch()

        # Regenerate All button (hidden until variants exist)
        self.regenerate_button = QPushButton("Regenerate All")
        self.regenerate_button.setObjectName("regenerate_all_button")
        self.regenerate_button.setVisible(False)
        self.regenerate_button.setMinimumWidth(120)
        header_layout.addWidget(self.regenerate_button)

        layout.addLayout(header_layout)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        # Create 5 variant rows
        self._variant_rows: List[VoiceVariantRow] = []
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)

        for i in range(self.NUM_VARIANTS):
            row = VoiceVariantRow(i)
            self._variant_rows.append(row)
            self._button_group.addButton(row.radio_button, i)
            layout.addWidget(row)

        # Connect button group for exclusive selection
        self._button_group.idToggled.connect(self._on_selection_changed)

    def _setup_connections(self):
        """Setup signal connections."""
        # Connect each row's signals
        for row in self._variant_rows:
            row.play_requested.connect(self._on_play_requested)
            row.stop_requested.connect(self._on_stop_requested)

        # Media player signals
        self._media_player.playbackStateChanged.connect(self._on_playback_state_changed)
        self._media_player.errorOccurred.connect(self._on_media_error)

    def _setup_accessibility(self):
        """Configure accessibility properties."""
        self.setAccessibleName("Voice Variations")
        self.setAccessibleDescription(
            "List of 5 generated voice variations. Select one to save."
        )
        self.regenerate_button.setAccessibleName("Regenerate All")
        self.regenerate_button.setAccessibleDescription(
            "Generate a new set of 5 voice variations"
        )

    def _on_selection_changed(self, button_id: int, checked: bool):
        """Handle selection changes from button group."""
        if checked:
            self._selected_index = button_id
            self.variant_selected.emit(button_id)
            self.logger.debug(f"Variant {button_id} selected")
        elif button_id == self._selected_index:
            self._selected_index = None
            self.variant_deselected.emit()
            self.logger.debug("Selection cleared")

    def _on_play_requested(self, variant_index: int):
        """Handle play request from a variant row."""
        # Stop any currently playing audio
        if self._currently_playing is not None:
            self._stop_playback()

        # Start playback for requested variant
        row = self._variant_rows[variant_index]
        audio_path = row.get_audio_path()

        if audio_path and audio_path.exists():
            self._currently_playing = variant_index
            row.set_playing(True)
            self._media_player.setSource(QUrl.fromLocalFile(str(audio_path)))
            self._media_player.play()
            self.logger.debug(f"Playing variant {variant_index}")

    def _on_stop_requested(self, variant_index: int):
        """Handle stop request from a variant row."""
        if self._currently_playing == variant_index:
            self._stop_playback()

    def _stop_playback(self):
        """Stop current playback."""
        if self._currently_playing is not None:
            self._variant_rows[self._currently_playing].set_playing(False)
            self._media_player.stop()
            self._currently_playing = None
            self.logger.debug("Playback stopped")

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState):
        """Handle media player state changes."""
        if state == QMediaPlayer.PlaybackState.StoppedState:
            if self._currently_playing is not None:
                self._variant_rows[self._currently_playing].set_playing(False)
                self._currently_playing = None

    def _on_media_error(self, error: QMediaPlayer.Error, error_string: str):
        """Handle media player errors."""
        self.logger.error(f"Media player error: {error_string}")
        if self._currently_playing is not None:
            self._variant_rows[self._currently_playing].set_playing(False)
            self._currently_playing = None

    def _update_status(self):
        """Update the status label based on current state."""
        completed = sum(1 for row in self._variant_rows if row.is_complete())
        errors = sum(1 for row in self._variant_rows if row.is_error())
        generating = any(row.get_state() == VariantState.GENERATING for row in self._variant_rows)

        if generating:
            # Find which variant is generating
            for i, row in enumerate(self._variant_rows):
                if row.get_state() == VariantState.GENERATING:
                    self.status_label.setText(f"Generating variant {i + 1}/{self.NUM_VARIANTS}...")
                    break
        elif completed == self.NUM_VARIANTS:
            self.status_label.setText(f"{self.NUM_VARIANTS} variations ready")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        elif completed > 0:
            if errors > 0:
                self.status_label.setText(f"{completed} ready, {errors} failed")
                self.status_label.setStyleSheet("color: orange;")
            else:
                self.status_label.setText(f"{completed}/{self.NUM_VARIANTS} variations ready")
                self.status_label.setStyleSheet("")
        else:
            self.status_label.setText("")
            self.status_label.setStyleSheet("")

        # Show regenerate button if any variants exist
        has_variants = completed > 0 or errors > 0
        self.regenerate_button.setVisible(has_variants and not self._is_generating)

    # Public API

    def start_generation(self):
        """
        Start the generation process - shows all rows in waiting state.

        Call this before generating variants to show the placeholder rows.
        """
        self._is_generating = True
        self._selected_index = None
        self._stop_playback()

        # Set all rows to waiting state
        for row in self._variant_rows:
            row.clear()

        self._update_status()
        self.logger.debug("Generation started - all rows in waiting state")

    def set_variant_generating(self, variant_index: int):
        """
        Set a specific variant to generating state.

        Args:
            variant_index: Index of variant (0-4)
        """
        if 0 <= variant_index < self.NUM_VARIANTS:
            self._variant_rows[variant_index].set_generating()
            self._update_status()
            self.logger.debug(f"Variant {variant_index} set to generating")

    def set_variant_complete(
        self,
        variant_index: int,
        audio_path: Path,
        embedding_path: Optional[Path],
        duration: float = None
    ):
        """
        Set a specific variant to complete state.

        Args:
            variant_index: Index of variant (0-4)
            audio_path: Path to generated audio file
            embedding_path: Path to generated embedding file (QA7: Optional for uploaded samples)
            duration: Audio duration in seconds
        """
        if 0 <= variant_index < self.NUM_VARIANTS:
            self._variant_rows[variant_index].set_complete(audio_path, embedding_path, duration)
            self._update_status()

            # Check if all complete
            completed = sum(1 for row in self._variant_rows if row.is_complete())
            self.generation_progress.emit(completed, self.NUM_VARIANTS)

            if completed == self.NUM_VARIANTS:
                self._is_generating = False
                self.all_complete.emit()
                self._update_status()
                self.logger.info("All variants complete")

    def set_variant_error(self, variant_index: int, error_message: str = "Failed"):
        """
        Set a specific variant to error state.

        Args:
            variant_index: Index of variant (0-4)
            error_message: Error message to display
        """
        if 0 <= variant_index < self.NUM_VARIANTS:
            self._variant_rows[variant_index].set_error(error_message)
            self._update_status()
            self.logger.debug(f"Variant {variant_index} set to error: {error_message}")

    def finish_generation(self):
        """
        Mark generation as finished (even if not all complete).

        Call this when generation is cancelled or complete.
        """
        self._is_generating = False
        self._update_status()
        self.logger.debug("Generation finished")

    def get_selected_index(self) -> Optional[int]:
        """
        Get the currently selected variant index.

        Returns:
            Selected variant index (0-4), or None if no selection
        """
        return self._selected_index

    def get_selected_variant(self) -> Optional[Tuple[Path, Path]]:
        """
        Get the selected variant's paths.

        Returns:
            Tuple of (audio_path, embedding_path), or None if no selection
        """
        if self._selected_index is not None:
            row = self._variant_rows[self._selected_index]
            if row.is_complete():
                return (row.get_audio_path(), row.get_embedding_path())
        return None

    def get_variant_row(self, variant_index: int) -> Optional[VoiceVariantRow]:
        """
        Get a specific variant row.

        Args:
            variant_index: Index of variant (0-4)

        Returns:
            VoiceVariantRow widget, or None if invalid index
        """
        if 0 <= variant_index < self.NUM_VARIANTS:
            return self._variant_rows[variant_index]
        return None

    def get_complete_count(self) -> int:
        """Get the number of successfully completed variants."""
        return sum(1 for row in self._variant_rows if row.is_complete())

    def get_all_variant_paths(self) -> List[Tuple[Path, Path, float]]:
        """
        QA7: Get paths and duration for all complete variants.

        Returns:
            List of (audio_path, embedding_path, duration) tuples for complete variants.
            Indexed by variant position (0-4), None entries omitted.
        """
        result = []
        for row in self._variant_rows:
            if row.is_complete():
                result.append((
                    row.get_audio_path(),
                    row.get_embedding_path(),
                    row.get_duration()
                ))
            else:
                # Include None for incomplete variants to preserve index
                result.append(None)
        return result

    def is_generating(self) -> bool:
        """Check if generation is in progress."""
        return self._is_generating

    def has_selection(self) -> bool:
        """Check if a variant is selected."""
        return self._selected_index is not None

    def has_complete_variants(self) -> bool:
        """Check if any variants are complete."""
        return any(row.is_complete() for row in self._variant_rows)

    def select_variant(self, variant_index: int) -> bool:
        """
        QA7: Programmatically select a variant.

        Args:
            variant_index: Index of variant to select (0-4)

        Returns:
            True if selection was successful, False otherwise
        """
        if not 0 <= variant_index < self.NUM_VARIANTS:
            self.logger.warning(f"Invalid variant index: {variant_index}")
            return False

        row = self._variant_rows[variant_index]
        if not row.is_complete():
            self.logger.warning(f"Cannot select incomplete variant {variant_index}")
            return False

        # Set the radio button to trigger selection
        row.radio_button.setChecked(True)
        self.logger.debug(f"Programmatically selected variant {variant_index}")
        return True

    def clear_selection(self):
        """Clear the current selection."""
        if self._selected_index is not None:
            self._variant_rows[self._selected_index].set_selected(False)
            self._selected_index = None

    def clear(self):
        """Clear all variants and reset to initial state."""
        self._stop_playback()
        self._selected_index = None
        self._is_generating = False

        for row in self._variant_rows:
            row.clear()

        self._update_status()
        self.logger.debug("Variant list cleared")

    def stop_playback(self):
        """Stop any currently playing audio."""
        self._stop_playback()
