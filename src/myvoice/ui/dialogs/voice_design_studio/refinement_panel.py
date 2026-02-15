"""
Refinement Panel for Voice Design Studio

Panel for batch embedding extraction and voice saving. Displays selected emotion
samples, extracts embeddings, and saves the voice profile with emotion subfolders.

Emotion Variants Feature:
- Shows selected samples from Emotions tab
- Extracts embeddings for all selected emotions sequentially
- Creates folder structure: embeddings/{voice_name}/{emotion}/embedding.pt
- Saves metadata.json with available_emotions list

Story: UI: Create RefinementPanel component
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from enum import Enum, auto

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QLineEdit, QProgressBar, QScrollArea,
    QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QUrl
from PyQt6.QtGui import QFont
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from myvoice.models.voice_profile import VALID_EMOTIONS


# Emotion display configuration
EMOTION_CONFIG = {
    "neutral": {"emoji": "😐", "name": "Neutral", "index": 0},
    "happy": {"emoji": "😄", "name": "Happy", "index": 1},
    "sad": {"emoji": "😢", "name": "Sad", "index": 2},
    "angry": {"emoji": "😠", "name": "Angry", "index": 3},
    "flirtatious": {"emoji": "😏", "name": "Flirtatious", "index": 4},
}


class ExtractionState(Enum):
    """State of embedding extraction for a sample."""
    PENDING = auto()
    EXTRACTING = auto()
    COMPLETE = auto()
    ERROR = auto()


class EmotionSampleRow(QWidget):
    """
    A single row displaying an emotion sample with playback and status.

    Shows: Emotion icon | Emotion name | [Play] button | Status indicator
    """

    play_requested = pyqtSignal(str)  # emotion
    stop_requested = pyqtSignal(str)  # emotion

    def __init__(
        self,
        emotion: str,
        audio_path: Optional[Path] = None,
        parent: Optional[QWidget] = None
    ):
        """
        Initialize the emotion sample row.

        Args:
            emotion: Emotion ID (e.g., "neutral", "happy")
            audio_path: Path to the audio sample file
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{emotion}")

        self._emotion = emotion
        self._config = EMOTION_CONFIG[emotion]
        self._audio_path = audio_path
        self._state = ExtractionState.PENDING
        self._is_playing = False
        self._embedding_path: Optional[Path] = None
        self._spinner_frame = 0

        # Spinner animation timer
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(100)
        self._spinner_timer.timeout.connect(self._animate_spinner)

        self._create_ui()
        self._setup_accessibility()
        self._update_ui_for_state()

    def _create_ui(self):
        """Create the row UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(12)

        # Emotion icon and name
        emoji = self._config["emoji"]
        name = self._config["name"]

        self.emotion_label = QLabel(f"{emoji} {name}")
        self.emotion_label.setObjectName(f"emotion_label_{self._emotion}")
        font = QFont()
        font.setBold(True)
        self.emotion_label.setFont(font)
        self.emotion_label.setMinimumWidth(120)
        layout.addWidget(self.emotion_label)

        # Play button
        self.play_button = QPushButton("Play")
        self.play_button.setObjectName(f"play_button_{self._emotion}")
        self.play_button.setMinimumWidth(60)
        self.play_button.setEnabled(self._audio_path is not None)
        self.play_button.clicked.connect(self._on_play_clicked)
        layout.addWidget(self.play_button)

        # Stretch
        layout.addStretch()

        # Status label
        self.status_label = QLabel("Pending")
        self.status_label.setObjectName(f"status_label_{self._emotion}")
        self.status_label.setMinimumWidth(100)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.status_label)

        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(40)
        self.setAutoFillBackground(True)

    def _setup_accessibility(self):
        """Configure accessibility properties."""
        name = self._config["name"]
        self.play_button.setAccessibleName(f"Play {name} sample")
        self.play_button.setAccessibleDescription(f"Play the audio sample for {name} emotion")
        self.status_label.setAccessibleName(f"{name} extraction status")

    def _on_play_clicked(self):
        """Handle play button click."""
        if self._is_playing:
            self.stop_requested.emit(self._emotion)
        else:
            self.play_requested.emit(self._emotion)

    def _update_ui_for_state(self):
        """Update UI based on current state."""
        if self._state == ExtractionState.PENDING:
            # QA5-1: Changed from "Pending" to "Ready to Extract" for clarity
            self.status_label.setText("Ready to Extract")
            self.status_label.setStyleSheet("color: #888;")
            self._spinner_timer.stop()

        elif self._state == ExtractionState.EXTRACTING:
            self._spinner_frame = 0
            self.status_label.setText("Extracting...")
            self.status_label.setStyleSheet("color: #0066cc;")
            self._spinner_timer.start()

        elif self._state == ExtractionState.COMPLETE:
            self.status_label.setText("✓ Complete")
            self.status_label.setStyleSheet("color: #008800;")
            self._spinner_timer.stop()

        elif self._state == ExtractionState.ERROR:
            self.status_label.setText("✗ Error")
            self.status_label.setStyleSheet("color: #cc0000;")
            self._spinner_timer.stop()

    def _animate_spinner(self):
        """Animate the spinner in status label."""
        spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"]
        self._spinner_frame = (self._spinner_frame + 1) % len(spinner_chars)
        self.status_label.setText(f"{spinner_chars[self._spinner_frame]} Extracting...")

    # Public API

    def get_emotion(self) -> str:
        """Get the emotion ID."""
        return self._emotion

    def get_audio_path(self) -> Optional[Path]:
        """Get the audio sample path."""
        return self._audio_path

    def set_audio_path(self, path: Path):
        """Set the audio sample path."""
        self._audio_path = path
        self.play_button.setEnabled(path is not None and path.exists())

    def get_state(self) -> ExtractionState:
        """Get the current extraction state."""
        return self._state

    def set_state(self, state: ExtractionState):
        """Set the extraction state."""
        self._state = state
        self._update_ui_for_state()

    def set_extracting(self):
        """Set to extracting state."""
        self._state = ExtractionState.EXTRACTING
        self._update_ui_for_state()

    def set_complete(self, embedding_path: Path):
        """Set to complete state with embedding path."""
        self._state = ExtractionState.COMPLETE
        self._embedding_path = embedding_path
        self._update_ui_for_state()

    def set_error(self, error_message: str = "Extraction failed"):
        """Set to error state."""
        self._state = ExtractionState.ERROR
        self.status_label.setText(f"✗ {error_message[:15]}..." if len(error_message) > 15 else f"✗ {error_message}")
        self._spinner_timer.stop()

    def get_embedding_path(self) -> Optional[Path]:
        """Get the extracted embedding path."""
        return self._embedding_path

    def set_playing(self, is_playing: bool):
        """Set the playing state."""
        self._is_playing = is_playing
        self.play_button.setText("Stop" if is_playing else "Play")

    def is_complete(self) -> bool:
        """Check if extraction is complete."""
        return self._state == ExtractionState.COMPLETE


class RefinementPanel(QWidget):
    """
    Panel for batch embedding extraction and voice saving.

    Displays selected emotion samples, extracts embeddings sequentially,
    and saves the voice profile with the correct folder structure.

    Signals:
        extraction_requested: Emitted when user clicks Extract All
        save_requested: Emitted when user clicks Save (voice_name, selected_emotions)
        preview_requested: Emitted when preview should be generated
        back_requested: Emitted when user clicks Back to Emotions
    """

    extraction_requested = pyqtSignal()
    save_requested = pyqtSignal(str, list)  # voice_name, List[str] emotions
    preview_requested = pyqtSignal()
    back_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the Refinement Panel.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        self._sample_rows: Dict[str, EmotionSampleRow] = {}
        self._is_extracting = False
        self._is_saving = False
        self._preview_path: Optional[Path] = None
        self._voice_description: str = ""
        self._is_generating_preview = False  # QA Round 2 Item #3: Preview loading state

        # Audio player for preview
        self._media_player = QMediaPlayer()
        self._audio_output = QAudioOutput()
        self._media_player.setAudioOutput(self._audio_output)
        self._currently_playing: Optional[str] = None

        self._create_ui()
        self._setup_connections()
        self._setup_accessibility()

        self.logger.debug("RefinementPanel initialized")

    def _create_ui(self):
        """Create the panel UI."""
        layout = QVBoxLayout(self)
        # QA3-2: Reduced margins and spacing for smaller screens
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Header
        header_label = QLabel(
            "Review selected samples and extract embeddings to create your voice."
        )
        header_label.setWordWrap(True)
        header_label.setObjectName("refinement_header")
        layout.addWidget(header_label)

        # Samples list section
        self._create_samples_section(layout)

        # Extract button section
        self._create_extract_section(layout)

        # Preview section
        self._create_preview_section(layout)

        # Save section
        self._create_save_section(layout)

        # Stretch at bottom
        layout.addStretch()

    def _create_samples_section(self, parent_layout: QVBoxLayout):
        """Create the samples list section."""
        samples_group = QGroupBox("Selected Emotion Samples")
        samples_group.setObjectName("samples_group")
        samples_layout = QVBoxLayout(samples_group)
        samples_layout.setSpacing(4)

        # Scroll area for samples
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(150)
        scroll_area.setMaximumHeight(250)

        # Container for sample rows
        self.samples_container = QWidget()
        self.samples_container.setObjectName("samples_container")
        self.samples_layout = QVBoxLayout(self.samples_container)
        self.samples_layout.setContentsMargins(0, 0, 0, 0)
        self.samples_layout.setSpacing(2)

        # Placeholder label (shown when no samples)
        self.no_samples_label = QLabel("No samples selected. Go back to Emotions tab.")
        self.no_samples_label.setObjectName("no_samples_label")
        self.no_samples_label.setStyleSheet("color: #888; font-style: italic;")
        self.no_samples_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.samples_layout.addWidget(self.no_samples_label)

        scroll_area.setWidget(self.samples_container)
        samples_layout.addWidget(scroll_area)

        parent_layout.addWidget(samples_group)

    def _create_extract_section(self, parent_layout: QVBoxLayout):
        """Create the extract button section."""
        extract_layout = QHBoxLayout()
        extract_layout.setSpacing(12)

        # Progress bar
        self.extract_progress = QProgressBar()
        self.extract_progress.setObjectName("extract_progress")
        self.extract_progress.setMinimum(0)
        self.extract_progress.setMaximum(100)
        self.extract_progress.setValue(0)
        self.extract_progress.setVisible(False)
        self.extract_progress.setMinimumWidth(200)
        extract_layout.addWidget(self.extract_progress)

        # Status label
        self.extract_status = QLabel("")
        self.extract_status.setObjectName("extract_status")
        extract_layout.addWidget(self.extract_status, 1)

        # Extract button
        self.extract_button = QPushButton("Extract All Embeddings")
        self.extract_button.setObjectName("extract_button")
        self.extract_button.setMinimumWidth(180)
        self.extract_button.setEnabled(False)
        extract_layout.addWidget(self.extract_button)

        parent_layout.addLayout(extract_layout)

    def _create_preview_section(self, parent_layout: QVBoxLayout):
        """Create the preview section."""
        self.preview_group = QGroupBox("Voice Preview")
        self.preview_group.setObjectName("preview_group")
        self.preview_group.setVisible(False)
        preview_layout = QVBoxLayout(self.preview_group)
        preview_layout.setSpacing(8)

        # Preview info
        preview_info = QLabel(
            "Preview generated using Neutral embedding. "
            "Listen to verify the voice quality."
        )
        preview_info.setWordWrap(True)
        preview_info.setStyleSheet("color: #666;")
        preview_layout.addWidget(preview_info)

        # Preview controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(12)

        # QA Round 2 Item #3: Progress indicator for preview generation
        self.preview_progress = QProgressBar()
        self.preview_progress.setObjectName("preview_progress")
        self.preview_progress.setRange(0, 0)  # Indeterminate
        self.preview_progress.setVisible(False)
        self.preview_progress.setMaximumWidth(150)
        self.preview_progress.setMaximumHeight(16)
        controls_layout.addWidget(self.preview_progress)

        self.preview_play_button = QPushButton("▶ Play Preview")
        self.preview_play_button.setObjectName("preview_play_button")
        self.preview_play_button.setMinimumWidth(120)
        self.preview_play_button.setEnabled(False)  # QA Round 2 Item #3: Disabled until ready
        controls_layout.addWidget(self.preview_play_button)

        self.preview_status = QLabel("")
        self.preview_status.setObjectName("preview_status")
        controls_layout.addWidget(self.preview_status, 1)

        preview_layout.addLayout(controls_layout)

        parent_layout.addWidget(self.preview_group)

    def _create_save_section(self, parent_layout: QVBoxLayout):
        """Create the save section."""
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        parent_layout.addWidget(separator)

        # Save group
        save_group = QGroupBox("Save Voice")
        save_group.setObjectName("save_group")
        save_layout = QVBoxLayout(save_group)
        save_layout.setSpacing(8)

        # Voice name input
        name_layout = QHBoxLayout()
        name_layout.setSpacing(8)

        name_label = QLabel("Voice Name:")
        name_label.setObjectName("voice_name_label")
        name_layout.addWidget(name_label)

        self.voice_name_edit = QLineEdit()
        self.voice_name_edit.setObjectName("voice_name_edit")
        self.voice_name_edit.setPlaceholderText("Enter a name for your voice")
        self.voice_name_edit.setMinimumWidth(200)
        name_layout.addWidget(self.voice_name_edit, 1)

        save_layout.addLayout(name_layout)

        # Button row
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        # Back button
        self.back_button = QPushButton("← Back to Emotions")
        self.back_button.setObjectName("back_button")
        self.back_button.setMinimumWidth(150)
        button_layout.addWidget(self.back_button)

        button_layout.addStretch()

        # Save status
        self.save_status = QLabel("")
        self.save_status.setObjectName("save_status")
        button_layout.addWidget(self.save_status)

        # Save button
        self.save_button = QPushButton("Save to Library")
        self.save_button.setObjectName("save_button")
        self.save_button.setMinimumWidth(140)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)

        save_layout.addLayout(button_layout)

        parent_layout.addWidget(save_group)

    def _setup_connections(self):
        """Setup signal connections."""
        self.extract_button.clicked.connect(self._on_extract_clicked)
        self.preview_play_button.clicked.connect(self._on_preview_play_clicked)
        self.save_button.clicked.connect(self._on_save_clicked)
        self.back_button.clicked.connect(self._on_back_clicked)
        self.voice_name_edit.textChanged.connect(self._update_save_button_state)

        # Media player signals
        self._media_player.playbackStateChanged.connect(self._on_playback_state_changed)
        self._media_player.errorOccurred.connect(self._on_media_error)

    def _setup_accessibility(self):
        """Configure accessibility properties."""
        self.extract_button.setAccessibleName("Extract All Embeddings")
        self.extract_button.setAccessibleDescription(
            "Extract voice embeddings for all selected emotion samples"
        )

        self.preview_play_button.setAccessibleName("Play Preview")
        self.preview_play_button.setAccessibleDescription(
            "Play a preview of the voice using the neutral embedding"
        )

        self.voice_name_edit.setAccessibleName("Voice Name")
        self.voice_name_edit.setAccessibleDescription(
            "Enter a name to save your voice to the library"
        )

        self.save_button.setAccessibleName("Save to Library")
        self.save_button.setAccessibleDescription(
            "Save the voice with all extracted embeddings to your library"
        )

        self.back_button.setAccessibleName("Back to Emotions")
        self.back_button.setAccessibleDescription(
            "Return to the Emotions tab to modify selections"
        )

    def _on_extract_clicked(self):
        """Handle Extract All button click."""
        self.logger.debug("Extract All clicked")
        self.extraction_requested.emit()

    def _on_preview_play_clicked(self):
        """Handle preview play button click."""
        if self._media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._media_player.stop()
            self.preview_play_button.setText("▶ Play Preview")
        elif self._preview_path and self._preview_path.exists():
            self._media_player.setSource(QUrl.fromLocalFile(str(self._preview_path)))
            self._media_player.play()
            self.preview_play_button.setText("■ Stop")

    def _on_save_clicked(self):
        """Handle Save button click."""
        voice_name = self.get_voice_name()
        selected_emotions = self.get_complete_emotions()
        self.logger.debug(f"Save clicked: name={voice_name}, emotions={selected_emotions}")
        self.save_requested.emit(voice_name, selected_emotions)

    def _on_back_clicked(self):
        """Handle Back button click."""
        self.logger.debug("Back to Emotions clicked")
        self.back_requested.emit()

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState):
        """Handle media player state changes."""
        if state == QMediaPlayer.PlaybackState.StoppedState:
            self.preview_play_button.setText("▶ Play Preview")
            if self._currently_playing:
                row = self._sample_rows.get(self._currently_playing)
                if row:
                    row.set_playing(False)
                self._currently_playing = None

    def _on_media_error(self, error: QMediaPlayer.Error, error_string: str):
        """Handle media player errors."""
        self.logger.error(f"Media player error: {error_string}")
        self.preview_status.setText(f"Playback error: {error_string}")
        self.preview_status.setStyleSheet("color: red;")

    def _on_sample_play_requested(self, emotion: str):
        """Handle play request from a sample row."""
        # Stop current playback
        if self._currently_playing:
            row = self._sample_rows.get(self._currently_playing)
            if row:
                row.set_playing(False)
            self._media_player.stop()

        # Start new playback
        row = self._sample_rows.get(emotion)
        if row and row.get_audio_path():
            audio_path = row.get_audio_path()
            if audio_path.exists():
                self._currently_playing = emotion
                row.set_playing(True)
                self._media_player.setSource(QUrl.fromLocalFile(str(audio_path)))
                self._media_player.play()

    def _on_sample_stop_requested(self, emotion: str):
        """Handle stop request from a sample row."""
        if self._currently_playing == emotion:
            self._media_player.stop()

    def _update_save_button_state(self):
        """Update save button enabled state."""
        has_name = bool(self.get_voice_name())
        has_neutral = self._is_emotion_complete("neutral")
        not_saving = not self._is_saving
        not_extracting = not self._is_extracting

        self.save_button.setEnabled(
            has_name and has_neutral and not_saving and not_extracting
        )

    def _is_emotion_complete(self, emotion: str) -> bool:
        """Check if an emotion's extraction is complete."""
        row = self._sample_rows.get(emotion)
        return row is not None and row.is_complete()

    def _add_sample_row(self, emotion: str, audio_path: Path) -> EmotionSampleRow:
        """Add a sample row for an emotion."""
        row = EmotionSampleRow(emotion, audio_path)
        row.play_requested.connect(self._on_sample_play_requested)
        row.stop_requested.connect(self._on_sample_stop_requested)

        self._sample_rows[emotion] = row
        self.samples_layout.addWidget(row)

        return row

    def _clear_sample_rows(self):
        """Clear all sample rows."""
        # Stop playback
        self._media_player.stop()
        self._currently_playing = None

        # Remove all rows
        for row in self._sample_rows.values():
            row.deleteLater()
        self._sample_rows.clear()

        # Show placeholder
        self.no_samples_label.setVisible(True)

    # Public API

    def set_samples(self, samples: Dict[str, Tuple[Path, Path]]):
        """
        Set the emotion samples to display.

        Args:
            samples: Dict mapping emotion -> (audio_path, embedding_path_or_none)
        """
        self._clear_sample_rows()
        self.no_samples_label.setVisible(False)

        # Add rows in VALID_EMOTIONS order, with neutral first
        # QA5-1: Always start as PENDING ("Ready to Extract"), never auto-mark as Complete
        # The user must click "Extract All Embeddings" to mark samples as Complete
        for emotion in VALID_EMOTIONS:
            if emotion in samples:
                audio_path, embedding_path = samples[emotion]
                self._add_sample_row(emotion, audio_path)
                # Store the embedding path for later use, but don't mark as complete yet

        self._update_extract_button_state()
        self._update_save_button_state()
        self.logger.debug(f"Set {len(samples)} samples: {list(samples.keys())}")

    def add_sample(self, emotion: str, audio_path: Path):
        """
        Add a single emotion sample.

        Args:
            emotion: Emotion ID
            audio_path: Path to the audio sample
        """
        if emotion in self._sample_rows:
            self._sample_rows[emotion].set_audio_path(audio_path)
        else:
            self.no_samples_label.setVisible(False)
            self._add_sample_row(emotion, audio_path)

        self._update_extract_button_state()

    def _update_extract_button_state(self):
        """Update extract button enabled state."""
        has_samples = len(self._sample_rows) > 0
        has_neutral = "neutral" in self._sample_rows
        not_extracting = not self._is_extracting

        self.extract_button.setEnabled(
            has_samples and has_neutral and not_extracting
        )

    def set_extracting(self, is_extracting: bool, message: str = ""):
        """
        Set the extraction state.

        Args:
            is_extracting: Whether extraction is in progress
            message: Status message to display
        """
        self._is_extracting = is_extracting
        self.extract_progress.setVisible(is_extracting)
        self.extract_button.setEnabled(not is_extracting and len(self._sample_rows) > 0)
        self.save_button.setEnabled(False)

        if is_extracting:
            self.extract_status.setText(message or "Extracting embeddings...")
            self.extract_status.setStyleSheet("color: blue;")
        else:
            self._update_extract_button_state()
            self._update_save_button_state()

    def set_extraction_progress(self, current: int, total: int, emotion: str = ""):
        """
        Update extraction progress.

        Args:
            current: Current item being processed (1-based)
            total: Total items to process
            emotion: Current emotion being processed
        """
        progress = int((current / total) * 100) if total > 0 else 0
        self.extract_progress.setValue(progress)

        if emotion:
            config = EMOTION_CONFIG.get(emotion, {})
            name = config.get("name", emotion)
            self.extract_status.setText(f"Extracting {name} ({current}/{total})...")

    def set_emotion_extracting(self, emotion: str):
        """Set a specific emotion to extracting state."""
        row = self._sample_rows.get(emotion)
        if row:
            row.set_extracting()

    def set_emotion_complete(self, emotion: str, embedding_path: Path):
        """Set a specific emotion to complete state."""
        row = self._sample_rows.get(emotion)
        if row:
            row.set_complete(embedding_path)
        self._update_save_button_state()

    def set_emotion_error(self, emotion: str, error_message: str):
        """Set a specific emotion to error state."""
        row = self._sample_rows.get(emotion)
        if row:
            row.set_error(error_message)

    def finish_extraction(self):
        """Mark extraction as finished."""
        self._is_extracting = False
        self.extract_progress.setVisible(False)
        self.extract_status.setText("All embeddings extracted successfully.")
        self.extract_status.setStyleSheet("color: green;")
        self._update_extract_button_state()
        self._update_save_button_state()

        # Show preview section
        self.preview_group.setVisible(True)

    def set_extraction_error(self, error_message: str):
        """Set extraction error state."""
        self._is_extracting = False
        self.extract_progress.setVisible(False)
        self.extract_status.setText(f"Error: {error_message}")
        self.extract_status.setStyleSheet("color: red;")
        self._update_extract_button_state()

    def set_preview_generating(self, is_generating: bool = True):
        """
        QA Round 2 Item #3: Set preview generation loading state.

        Args:
            is_generating: True if preview is being generated
        """
        self._is_generating_preview = is_generating

        if is_generating:
            self.preview_group.setVisible(True)
            self.preview_progress.setVisible(True)
            self.preview_play_button.setEnabled(False)
            self.preview_status.setText("Generating preview...")
            self.preview_status.setStyleSheet("color: blue;")
            self.logger.debug("Preview generation started")
        else:
            self.preview_progress.setVisible(False)
            self.logger.debug("Preview generation stopped")

    def set_preview_audio(self, preview_path: Path):
        """
        Set the preview audio path.

        Args:
            preview_path: Path to the preview audio file
        """
        self._preview_path = preview_path
        self._is_generating_preview = False

        # QA Round 2 Item #3: Update UI for ready state
        self.preview_group.setVisible(True)
        self.preview_progress.setVisible(False)
        self.preview_play_button.setEnabled(True)
        self.preview_status.setText("Preview ready.")
        self.preview_status.setStyleSheet("color: green;")

    def get_voice_name(self) -> str:
        """Get the entered voice name."""
        return self.voice_name_edit.text().strip()

    def set_voice_name(self, name: str):
        """Set the voice name."""
        self.voice_name_edit.setText(name)

    def set_voice_description(self, description: str):
        """Store the voice description for saving."""
        self._voice_description = description

    def get_voice_description(self) -> str:
        """Get the stored voice description."""
        return self._voice_description

    def get_sample_emotions(self) -> List[str]:
        """Get list of emotions that have samples."""
        return list(self._sample_rows.keys())

    def get_complete_emotions(self) -> List[str]:
        """Get list of emotions with complete extraction."""
        return [
            emotion for emotion, row in self._sample_rows.items()
            if row.is_complete()
        ]

    def get_embedding_paths(self) -> Dict[str, Path]:
        """
        Get mapping of emotion -> embedding path for complete extractions.

        Returns:
            Dict mapping emotion ID to embedding Path
        """
        return {
            emotion: row.get_embedding_path()
            for emotion, row in self._sample_rows.items()
            if row.is_complete() and row.get_embedding_path()
        }

    def get_audio_paths(self) -> Dict[str, Path]:
        """
        Get mapping of emotion -> audio path for all samples.

        Returns:
            Dict mapping emotion ID to audio Path
        """
        return {
            emotion: row.get_audio_path()
            for emotion, row in self._sample_rows.items()
            if row.get_audio_path()
        }

    def set_saving(self, is_saving: bool, message: str = ""):
        """
        Set the saving state.

        Args:
            is_saving: Whether save is in progress
            message: Status message to display
        """
        self._is_saving = is_saving
        self.save_button.setEnabled(not is_saving)
        self.voice_name_edit.setEnabled(not is_saving)
        self.back_button.setEnabled(not is_saving)

        if is_saving:
            self.save_status.setText(message or "Saving...")
            self.save_status.setStyleSheet("color: blue;")
        else:
            self._update_save_button_state()

    def set_save_complete(self, voice_name: str):
        """Set save complete state."""
        self._is_saving = False
        self.save_status.setText(f"Voice '{voice_name}' saved successfully!")
        self.save_status.setStyleSheet("color: green;")
        self.voice_name_edit.setEnabled(True)
        self.back_button.setEnabled(True)

    def set_save_error(self, error_message: str):
        """Set save error state."""
        self._is_saving = False
        self.save_status.setText(f"Save failed: {error_message}")
        self.save_status.setStyleSheet("color: red;")
        self.voice_name_edit.setEnabled(True)
        self.back_button.setEnabled(True)
        self._update_save_button_state()

    def clear(self):
        """Clear all samples and reset state."""
        self._clear_sample_rows()
        self._is_extracting = False
        self._is_saving = False
        self._preview_path = None
        self._voice_description = ""

        self.extract_progress.setValue(0)
        self.extract_progress.setVisible(False)
        self.extract_status.setText("")
        self.preview_group.setVisible(False)
        self.preview_status.setText("")
        self.voice_name_edit.clear()
        self.save_status.setText("")

        self._update_extract_button_state()
        self._update_save_button_state()

        self.logger.debug("RefinementPanel cleared")
