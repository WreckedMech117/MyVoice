"""
Sample Path Panel

Panel for the "From Sample" tab in Voice Design Studio.
Allows users to upload an audio file for voice extraction.

Story 2.1: Upload and Validate Audio File
- Browse button for file selection
- File dialog filters for WAV, MP3, M4A formats
- Valid file shows filename, duration, format info
- Warning for audio < 3 seconds or > 30 seconds (not blocking)
- Play/stop button for uploaded audio preview
- Error message for unsupported formats

Story 2.2: Auto-Transcribe with Whisper
- Multi-line transcript field appears after file upload
- "Auto-Transcribe" button triggers Whisper transcription
- Progress indicator during transcription (indeterminate)
- Completed transcription populates text area (editable)
- Error handling with retry option

Story 2.3: Extract Embedding and Preview
- Preview Text field and Extract Embedding button visible
- Button disabled if preview text empty
- Clicking shows progress "Extracting...", loads Base model if needed
- Preview audio auto-plays on extraction completion
- Play/stop button for replay
- Voice Name field and Save Voice button available on success
- Errors show clear message with suggestions

Story 2.4: Re-Extract After Edit
- Visual indicator when transcript changed since last extraction
- Extract Embedding button remains available after first extraction
- Re-extraction clears previous preview, shows new result
- Only most recent extraction retained
- Changing Preview Text allows testing same embedding with different text

Acceptance Criteria (2.1):
- "From Sample" tab shows Browse button for file selection
- File dialog filters for WAV, MP3, M4A formats
- Valid file shows filename, duration, format info
- Warning for audio < 3 seconds or > 30 seconds (not blocking)
- Play/stop button for uploaded audio preview
- Error message for unsupported formats

Acceptance Criteria (2.2):
- Multi-line "Transcript" text area appears after file upload
- User can manually enter/edit transcript
- "Auto-Transcribe" button visible for automatic transcription
- Clicking Auto-Transcribe shows progress indicator (indeterminate)
- Transcribed text appears in editable field
- Transcription errors show message with "Retry" button

Acceptance Criteria (2.3):
- Preview Text field and Extract Embedding button visible after file upload
- Extract Embedding button disabled if preview text empty
- Clicking shows progress "Extracting..." with indeterminate progress
- Preview audio auto-plays on extraction completion
- Play/stop button available for replay
- Voice Name field and Save Voice button available on success
- Errors show clear message with suggestions (cleaner sample, check transcript)

Acceptance Criteria (2.4):
- Visual indicator shows when transcript changed since last extraction
- Extract Embedding button remains available after first extraction
- Re-extraction clears previous preview, shows new result
- Only most recent extraction is retained
- Changing Preview Text works with same embedding
"""

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QFileDialog, QLineEdit,
    QTextEdit, QProgressBar, QComboBox, QTabWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from myvoice.ui.dialogs.voice_design_studio.audio_player_widget import AudioPlayerWidget


class SamplePathPanel(QWidget):
    """
    Panel for creating voices from audio samples.

    Provides file upload, validation, transcription, extraction, and preview
    functionality for extracting voice embeddings from audio recordings.

    Signals:
        file_loaded: Emitted when a valid audio file is loaded (file_path)
        content_changed: Emitted when file selection changes
        save_ready_changed: Emitted when save readiness changes (has_embedding and has_name)
        transcribe_requested: Emitted when Auto-Transcribe button is clicked (file_path)
        transcript_changed: Emitted when transcript text changes
        extract_requested: Emitted when Extract Embedding is clicked (audio_path, transcript, preview_text)
        preview_text_changed: Emitted when preview text field changes
    """

    file_loaded = pyqtSignal(str)  # file_path
    content_changed = pyqtSignal()  # Emitted when file changes
    save_ready_changed = pyqtSignal(bool)  # Emitted when save readiness changes
    transcribe_requested = pyqtSignal(str)  # file_path - Emitted when Auto-Transcribe clicked
    transcript_changed = pyqtSignal()  # Emitted when transcript text changes
    extract_requested = pyqtSignal(str, str, str, str)  # audio_path, transcript, preview_text, language - Story 2.3
    preview_text_changed = pyqtSignal()  # Emitted when preview text changes - Story 2.3
    save_voice_requested = pyqtSignal(str)  # voice_name - QA4: Emitted when Save Voice clicked in Clone tab
    refinement_requested = pyqtSignal()  # QA7: Emitted when Refinement button clicked in Clone tab

    # Supported audio formats
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a'}
    FILE_FILTER = "Audio Files (*.wav *.mp3 *.m4a);;WAV Files (*.wav);;MP3 Files (*.mp3);;M4A Files (*.m4a);;All Files (*)"

    # Duration thresholds (in seconds)
    MIN_DURATION_WARNING = 3.0  # Warn if < 3 seconds
    MAX_DURATION_WARNING = 30.0  # Warn if > 30 seconds

    # Supported languages for Qwen3-TTS (full names required)
    SUPPORTED_LANGUAGES = [
        "Auto",       # Auto-detect
        "English",
        "Chinese",
        "Japanese",
        "Korean",
        "German",
        "French",
        "Russian",
        "Portuguese",
        "Spanish",
        "Italian",
    ]

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the Sample Path Panel.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        # File state
        self._audio_path: Optional[Path] = None
        self._audio_duration: float = 0.0
        self._audio_format: str = ""
        self._has_error = False

        # Transcription state (Story 2.2)
        self._is_transcribing = False
        self._transcription_error = False

        # Extraction state (Story 2.3)
        self._is_extracting = False
        self._extraction_error = False
        self._generated_audio_path: Optional[Path] = None

        # Re-extraction tracking (Story 2.4)
        self._last_extracted_transcript: Optional[str] = None
        self._transcript_modified_since_extraction = False

        # Save state
        self._is_saving = False
        self._generated_embedding_path: Optional[Path] = None

        self._create_ui()
        self._setup_connections()
        self._setup_accessibility()

        self.logger.debug("SamplePathPanel initialized")

    def _create_ui(self):
        """Create the panel UI with Sample/Clone sub-tabs (QA3 fix)."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Create sub-tab widget for Sample/Clone workflow
        self.sub_tabs = QTabWidget()
        self.sub_tabs.setObjectName("sample_sub_tabs")
        self.sub_tabs.setDocumentMode(True)  # Cleaner look for nested tabs

        # Create Sample tab (input phase)
        self.sample_tab = QWidget()
        self.sample_tab.setObjectName("sample_tab")
        sample_layout = QVBoxLayout(self.sample_tab)
        sample_layout.setContentsMargins(8, 8, 8, 8)
        sample_layout.setSpacing(10)

        # File upload section
        self._create_upload_section(sample_layout)

        # File info section (hidden until file loaded)
        self._create_file_info_section(sample_layout)

        sample_layout.addStretch()

        # Create Clone tab (output phase)
        self.clone_tab = QWidget()
        self.clone_tab.setObjectName("clone_tab")
        clone_layout = QVBoxLayout(self.clone_tab)
        clone_layout.setContentsMargins(8, 8, 8, 8)
        clone_layout.setSpacing(10)

        # Transcript section (Story 2.2 - hidden until file loaded)
        self._create_transcript_section(clone_layout)

        # Extract section (Story 2.3 - hidden until file loaded)
        self._create_extract_section(clone_layout)

        clone_layout.addStretch()

        # Add tabs to sub-tab widget
        self.sub_tabs.addTab(self.sample_tab, "Sample")
        self.sub_tabs.addTab(self.clone_tab, "Clone")

        # Connect tab change to handle state
        self.sub_tabs.currentChanged.connect(self._on_sub_tab_changed)

        layout.addWidget(self.sub_tabs)

    def _create_upload_section(self, layout: QVBoxLayout):
        """Create the file upload section."""
        upload_group = QGroupBox("Audio Sample")
        upload_group.setObjectName("upload_group")
        upload_layout = QVBoxLayout(upload_group)
        upload_layout.setSpacing(8)

        # Instruction label
        instruction_label = QLabel(
            "Upload an audio recording to extract a voice embedding. "
            "Supported formats: WAV, MP3, M4A. Best results with 3-30 seconds of clear speech."
        )
        instruction_label.setWordWrap(True)
        instruction_label.setObjectName("upload_instruction")
        upload_layout.addWidget(instruction_label)

        # Browse button row
        browse_layout = QHBoxLayout()
        browse_layout.setSpacing(8)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.setObjectName("browse_button")
        self.browse_button.setMinimumWidth(100)
        browse_layout.addWidget(self.browse_button)

        # Selected file label
        self.file_label = QLabel("No file selected")
        self.file_label.setObjectName("selected_file_label")
        self.file_label.setStyleSheet("color: #666; font-style: italic;")
        browse_layout.addWidget(self.file_label, 1)

        upload_layout.addLayout(browse_layout)

        layout.addWidget(upload_group)

    def _create_file_info_section(self, layout: QVBoxLayout):
        """Create the file information section."""
        self.info_group = QGroupBox("File Information")
        self.info_group.setObjectName("file_info_group")
        self.info_group.setVisible(False)  # Hidden until file loaded
        info_layout = QVBoxLayout(self.info_group)
        info_layout.setSpacing(8)

        # File details row
        details_layout = QHBoxLayout()
        details_layout.setSpacing(16)

        # Duration
        duration_layout = QVBoxLayout()
        duration_title = QLabel("Duration:")
        duration_title.setStyleSheet("font-weight: bold;")
        duration_layout.addWidget(duration_title)
        self.duration_value = QLabel("0:00")
        self.duration_value.setObjectName("duration_value")
        duration_font = QFont()
        duration_font.setFamily("monospace")
        self.duration_value.setFont(duration_font)
        duration_layout.addWidget(self.duration_value)
        details_layout.addLayout(duration_layout)

        # Format
        format_layout = QVBoxLayout()
        format_title = QLabel("Format:")
        format_title.setStyleSheet("font-weight: bold;")
        format_layout.addWidget(format_title)
        self.format_value = QLabel("Unknown")
        self.format_value.setObjectName("format_value")
        format_layout.addWidget(self.format_value)
        details_layout.addLayout(format_layout)

        details_layout.addStretch()
        info_layout.addLayout(details_layout)

        # Warning/status label
        self.warning_label = QLabel("")
        self.warning_label.setObjectName("warning_label")
        self.warning_label.setWordWrap(True)
        self.warning_label.setVisible(False)
        info_layout.addWidget(self.warning_label)

        # Audio player for preview
        player_layout = QHBoxLayout()
        player_label = QLabel("Preview:")
        player_label.setStyleSheet("font-weight: bold;")
        player_layout.addWidget(player_label)

        self.audio_player = AudioPlayerWidget()
        self.audio_player.setObjectName("sample_audio_player")
        player_layout.addWidget(self.audio_player, 1)

        info_layout.addLayout(player_layout)

        # Voice name input (for saving)
        self._create_save_section(info_layout)

        layout.addWidget(self.info_group)

    def _create_save_section(self, layout: QVBoxLayout):
        """Create the save section with voice name input and Next button."""
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

        layout.addLayout(name_layout)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setObjectName("status_label")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # QA4: Next button to guide user to Clone tab
        next_layout = QHBoxLayout()
        next_layout.addStretch()

        self.next_button = QPushButton("Next →")
        self.next_button.setObjectName("next_button")
        self.next_button.setMinimumWidth(100)
        self.next_button.setToolTip("Proceed to Clone tab for transcription and extraction")
        self.next_button.clicked.connect(self._on_next_clicked)
        next_layout.addWidget(self.next_button)

        layout.addLayout(next_layout)

    def _create_transcript_section(self, layout: QVBoxLayout):
        """Create the transcript section (Story 2.2)."""
        self.transcript_group = QGroupBox("Transcript")
        self.transcript_group.setObjectName("transcript_group")
        self.transcript_group.setVisible(False)  # Hidden until file loaded
        transcript_layout = QVBoxLayout(self.transcript_group)
        transcript_layout.setSpacing(8)

        # Instruction label
        transcript_instruction = QLabel(
            "Enter the text spoken in the audio sample, or use Auto-Transcribe "
            "to automatically detect it."
        )
        transcript_instruction.setWordWrap(True)
        transcript_instruction.setObjectName("transcript_instruction")
        transcript_layout.addWidget(transcript_instruction)

        # Transcript text area
        self.transcript_edit = QTextEdit()
        self.transcript_edit.setObjectName("transcript_edit")
        self.transcript_edit.setPlaceholderText(
            "Enter or paste the transcript of what is spoken in the audio..."
        )
        self.transcript_edit.setMinimumHeight(80)
        self.transcript_edit.setMaximumHeight(120)
        transcript_layout.addWidget(self.transcript_edit)

        # Transcript modified indicator (Story 2.4 - hidden until transcript changes after extraction)
        self.transcript_modified_label = QLabel(
            "Transcript modified since last extraction. Re-extract to update the voice."
        )
        self.transcript_modified_label.setObjectName("transcript_modified_label")
        self.transcript_modified_label.setWordWrap(True)
        self.transcript_modified_label.setStyleSheet(
            "color: #856404; background-color: #fff3cd; padding: 6px; "
            "border-radius: 4px; border: 1px solid #ffeeba;"
        )
        self.transcript_modified_label.setVisible(False)
        transcript_layout.addWidget(self.transcript_modified_label)

        # Auto-Transcribe button and progress row
        transcribe_layout = QHBoxLayout()
        transcribe_layout.setSpacing(8)

        # Progress bar (hidden by default)
        self.transcribe_progress = QProgressBar()
        self.transcribe_progress.setObjectName("transcribe_progress")
        self.transcribe_progress.setRange(0, 0)  # Indeterminate
        self.transcribe_progress.setVisible(False)
        self.transcribe_progress.setMaximumWidth(150)
        self.transcribe_progress.setMaximumHeight(16)
        transcribe_layout.addWidget(self.transcribe_progress)

        # Progress/status label
        self.transcribe_status = QLabel("")
        self.transcribe_status.setObjectName("transcribe_status")
        transcribe_layout.addWidget(self.transcribe_status, 1)

        # Retry button (hidden by default) - QA4: increased width to prevent text cutoff
        self.transcribe_retry_button = QPushButton("Retry")
        self.transcribe_retry_button.setObjectName("transcribe_retry_button")
        self.transcribe_retry_button.setVisible(False)
        self.transcribe_retry_button.setMinimumWidth(80)
        transcribe_layout.addWidget(self.transcribe_retry_button)

        # Auto-Transcribe button - QA4: increased width to prevent text cutoff
        self.transcribe_button = QPushButton("Auto-Transcribe")
        self.transcribe_button.setObjectName("transcribe_button")
        self.transcribe_button.setMinimumWidth(140)
        transcribe_layout.addWidget(self.transcribe_button)

        transcript_layout.addLayout(transcribe_layout)

        layout.addWidget(self.transcript_group)

    def _create_extract_section(self, layout: QVBoxLayout):
        """Create the extract embedding section (Story 2.3)."""
        self.extract_group = QGroupBox("Extract Voice")
        self.extract_group.setObjectName("extract_group")
        self.extract_group.setVisible(False)  # Hidden until file loaded
        extract_layout = QVBoxLayout(self.extract_group)
        extract_layout.setSpacing(8)

        # Instruction label
        extract_instruction = QLabel(
            "Enter preview text to hear how the extracted voice sounds. "
            "This helps verify the voice was captured correctly."
        )
        extract_instruction.setWordWrap(True)
        extract_instruction.setObjectName("extract_instruction")
        extract_layout.addWidget(extract_instruction)

        # Preview text input row
        preview_text_layout = QHBoxLayout()
        preview_text_layout.setSpacing(8)

        preview_text_label = QLabel("Preview Text:")
        preview_text_label.setObjectName("preview_text_label")
        preview_text_layout.addWidget(preview_text_label)

        self.preview_text_edit = QLineEdit()
        self.preview_text_edit.setObjectName("preview_text_edit")
        self.preview_text_edit.setPlaceholderText("Enter text to hear in the extracted voice...")
        preview_text_layout.addWidget(self.preview_text_edit, 1)

        extract_layout.addLayout(preview_text_layout)

        # Language selection row
        lang_layout = QHBoxLayout()
        lang_layout.setSpacing(8)

        lang_label = QLabel("Language:")
        lang_label.setObjectName("language_label")
        lang_layout.addWidget(lang_label)

        self.language_combo = QComboBox()
        self.language_combo.setObjectName("language_combo")
        for lang in self.SUPPORTED_LANGUAGES:
            self.language_combo.addItem(lang)
        self.language_combo.setCurrentText("Auto")  # Default to auto-detect
        self.language_combo.setMinimumWidth(120)
        lang_layout.addWidget(self.language_combo)

        lang_layout.addStretch()
        extract_layout.addLayout(lang_layout)

        # Extract button and progress row
        extract_button_layout = QHBoxLayout()
        extract_button_layout.setSpacing(8)

        # Progress bar (hidden by default)
        self.extract_progress = QProgressBar()
        self.extract_progress.setObjectName("extract_progress")
        self.extract_progress.setRange(0, 0)  # Indeterminate
        self.extract_progress.setVisible(False)
        self.extract_progress.setMaximumWidth(150)
        self.extract_progress.setMaximumHeight(16)
        extract_button_layout.addWidget(self.extract_progress)

        # Status label
        self.extract_status = QLabel("")
        self.extract_status.setObjectName("extract_status")
        self.extract_status.setWordWrap(True)
        extract_button_layout.addWidget(self.extract_status, 1)

        # Retry button (hidden by default)
        self.extract_retry_button = QPushButton("Retry")
        self.extract_retry_button.setObjectName("extract_retry_button")
        self.extract_retry_button.setVisible(False)
        self.extract_retry_button.setMinimumWidth(60)
        extract_button_layout.addWidget(self.extract_retry_button)

        # Extract Embedding button
        self.extract_button = QPushButton("Extract Embedding")
        self.extract_button.setObjectName("extract_button")
        self.extract_button.setMinimumWidth(140)
        self.extract_button.setEnabled(False)  # Disabled until preview text entered
        extract_button_layout.addWidget(self.extract_button)

        extract_layout.addLayout(extract_button_layout)

        # Preview audio section (hidden until extraction complete)
        self._create_preview_audio_section(extract_layout)

        layout.addWidget(self.extract_group)

    def _create_preview_audio_section(self, layout: QVBoxLayout):
        """Create the preview audio section (Story 2.3, QA4)."""
        # Preview audio group (hidden until extraction complete)
        self.preview_audio_group = QGroupBox("Voice Preview")
        self.preview_audio_group.setObjectName("preview_audio_group")
        self.preview_audio_group.setVisible(False)
        preview_audio_layout = QVBoxLayout(self.preview_audio_group)
        preview_audio_layout.setSpacing(8)

        # Preview audio player
        self.preview_audio_player = AudioPlayerWidget()
        self.preview_audio_player.setObjectName("preview_audio_player")
        preview_audio_layout.addWidget(self.preview_audio_player)

        # Success message
        self.preview_success_label = QLabel(
            "Voice extracted successfully! Listen to the preview above, "
            "then save your voice."
        )
        self.preview_success_label.setWordWrap(True)
        self.preview_success_label.setStyleSheet("color: green;")
        self.preview_success_label.setObjectName("preview_success_label")
        preview_audio_layout.addWidget(self.preview_success_label)

        # QA4: Save Voice button in Clone tab (visible only after extraction)
        # QA7: Added Refinement button for building full voice with all emotions
        save_layout = QHBoxLayout()
        save_layout.addStretch()

        # QA7: Refinement button - navigate to Emotions tab for full voice building
        self.clone_refinement_button = QPushButton("Refinement")
        self.clone_refinement_button.setObjectName("clone_refinement_button")
        self.clone_refinement_button.setMinimumWidth(120)
        self.clone_refinement_button.setToolTip(
            "Build a full voice with multiple emotion variants in the Emotions tab"
        )
        self.clone_refinement_button.clicked.connect(self._on_refinement_clicked)
        save_layout.addWidget(self.clone_refinement_button)

        self.clone_save_button = QPushButton("Save Voice")
        self.clone_save_button.setObjectName("clone_save_button")
        self.clone_save_button.setMinimumWidth(120)
        self.clone_save_button.setEnabled(False)  # Enabled when voice name entered
        self.clone_save_button.clicked.connect(self._on_save_clicked)
        save_layout.addWidget(self.clone_save_button)

        preview_audio_layout.addLayout(save_layout)

        # Save feedback label (QA4)
        self.save_feedback_label = QLabel("")
        self.save_feedback_label.setObjectName("save_feedback_label")
        self.save_feedback_label.setWordWrap(True)
        preview_audio_layout.addWidget(self.save_feedback_label)

        layout.addWidget(self.preview_audio_group)

    def _setup_connections(self):
        """Setup signal connections."""
        self.browse_button.clicked.connect(self._on_browse_clicked)
        self.voice_name_edit.textChanged.connect(self._on_voice_name_changed)

        # Transcript connections (Story 2.2)
        self.transcribe_button.clicked.connect(self._on_transcribe_clicked)
        self.transcribe_retry_button.clicked.connect(self._on_transcribe_clicked)
        self.transcript_edit.textChanged.connect(self._on_transcript_changed)

        # Extract connections (Story 2.3)
        self.extract_button.clicked.connect(self._on_extract_clicked)
        self.extract_retry_button.clicked.connect(self._on_extract_clicked)
        self.preview_text_edit.textChanged.connect(self._on_preview_text_changed)

    def _setup_accessibility(self):
        """Configure accessibility properties."""
        self.browse_button.setAccessibleName("Browse for audio file")
        self.browse_button.setAccessibleDescription(
            "Open file dialog to select an audio file for voice extraction"
        )

        self.voice_name_edit.setAccessibleName("Voice Name")
        self.voice_name_edit.setAccessibleDescription(
            "Enter a name to save your extracted voice"
        )

        # Transcript accessibility (Story 2.2)
        self.transcript_edit.setAccessibleName("Transcript")
        self.transcript_edit.setAccessibleDescription(
            "Enter the text spoken in the audio sample"
        )

        self.transcribe_button.setAccessibleName("Auto-Transcribe")
        self.transcribe_button.setAccessibleDescription(
            "Automatically transcribe the audio using speech recognition"
        )

        self.transcribe_retry_button.setAccessibleName("Retry transcription")
        self.transcribe_retry_button.setAccessibleDescription(
            "Retry the failed transcription"
        )

        # Transcript modified indicator accessibility (Story 2.4)
        self.transcript_modified_label.setAccessibleName("Transcript modified warning")
        self.transcript_modified_label.setAccessibleDescription(
            "Indicates the transcript has been changed since the last extraction"
        )

        # Extract accessibility (Story 2.3)
        self.preview_text_edit.setAccessibleName("Preview Text")
        self.preview_text_edit.setAccessibleDescription(
            "Enter text to hear how the extracted voice sounds"
        )

        self.language_combo.setAccessibleName("Language")
        self.language_combo.setAccessibleDescription(
            "Select the language for voice extraction preview"
        )

        self.extract_button.setAccessibleName("Extract Embedding")
        self.extract_button.setAccessibleDescription(
            "Extract voice embedding from the audio sample and generate preview"
        )

        self.extract_retry_button.setAccessibleName("Retry extraction")
        self.extract_retry_button.setAccessibleDescription(
            "Retry the failed voice extraction"
        )

        # QA4: Next button accessibility
        self.next_button.setAccessibleName("Next")
        self.next_button.setAccessibleDescription(
            "Proceed to the Clone tab for transcription and voice extraction"
        )

    def _on_sub_tab_changed(self, index: int):
        """Handle sub-tab change (QA3)."""
        tab_names = ["Sample", "Clone"]
        self.logger.debug(f"Sub-tab changed to: {tab_names[index] if index < len(tab_names) else index}")

    def _on_next_clicked(self):
        """Handle Next button click - advance to Clone tab (QA4)."""
        self.sub_tabs.setCurrentIndex(1)  # Clone tab
        self.logger.debug("User clicked Next - switched to Clone tab")

    def _on_browse_clicked(self):
        """Handle Browse button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            self.FILE_FILTER
        )

        if file_path:
            self._load_audio_file(Path(file_path))

    def _load_audio_file(self, file_path: Path):
        """
        Load and validate an audio file.

        Args:
            file_path: Path to the audio file
        """
        self.logger.debug(f"Loading audio file: {file_path}")

        # Check if file exists
        if not file_path.exists():
            self._show_error("File not found")
            return

        # Check format
        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            self._show_error(f"Unsupported format. Please use WAV, MP3, or M4A")
            return

        # Clear previous error
        self._has_error = False

        # Store file path
        self._audio_path = file_path

        # Get audio info
        self._audio_duration = self._get_audio_duration(file_path)
        self._audio_format = self._get_audio_format(file_path)

        # Update UI
        self.file_label.setText(file_path.name)
        self.file_label.setStyleSheet("")  # Remove italic style

        self.duration_value.setText(self._format_duration(self._audio_duration))
        self.format_value.setText(self._audio_format)

        # Set audio player
        self.audio_player.set_audio_file(file_path)

        # Show file info, transcript, and extract sections
        self.info_group.setVisible(True)
        self.transcript_group.setVisible(True)
        self.extract_group.setVisible(True)

        # QA4: Do NOT auto-switch to Clone tab - let user enter voice name first
        # User will click "Next" button when ready to proceed to Clone tab

        # Check duration warnings
        self._check_duration_warnings()

        # Emit signals
        self.file_loaded.emit(str(file_path))
        self.content_changed.emit()
        self.save_ready_changed.emit(self._is_save_ready_internal())

        self.logger.info(f"Audio file loaded: {file_path.name}, {self._audio_duration:.1f}s, {self._audio_format}")

    def _get_audio_duration(self, file_path: Path) -> float:
        """
        Get the duration of an audio file.

        Uses Python's built-in wave module for WAV files (most reliable),
        falls back to mutagen for other formats.

        Args:
            file_path: Path to the audio file

        Returns:
            Duration in seconds
        """
        suffix = file_path.suffix.lower()

        # For WAV files, use Python's built-in wave module (most reliable)
        if suffix == '.wav':
            try:
                import wave
                with wave.open(str(file_path), 'rb') as w:
                    frames = w.getnframes()
                    rate = w.getframerate()
                    if rate > 0:
                        duration = frames / float(rate)
                        self.logger.debug(f"WAV duration via wave module: {duration:.2f}s")
                        return duration
            except Exception as e:
                self.logger.debug(f"Could not get WAV duration with wave module: {e}")

        # For other formats (MP3, M4A), try mutagen
        try:
            from mutagen import File as MutagenFile
            audio = MutagenFile(str(file_path))
            if audio and audio.info:
                return audio.info.length
        except ImportError:
            self.logger.warning("mutagen not available for duration detection")
        except Exception as e:
            self.logger.debug(f"Could not get duration with mutagen: {e}")

        # Fallback estimate
        self.logger.warning(f"Could not determine duration for {file_path.name}, using estimate")
        return 5.0

    def _get_audio_format(self, file_path: Path) -> str:
        """
        Get format information for an audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            Format string (e.g., "MP3, 44.1kHz")
        """
        suffix = file_path.suffix.lower()
        format_name = suffix[1:].upper()  # Remove dot and uppercase

        try:
            from mutagen import File as MutagenFile
            audio = MutagenFile(str(file_path))
            if audio and audio.info:
                sample_rate = getattr(audio.info, 'sample_rate', None)
                if sample_rate:
                    return f"{format_name}, {sample_rate / 1000:.1f}kHz"
        except Exception as e:
            self.logger.debug(f"Could not get format info: {e}")

        return format_name

    def _format_duration(self, seconds: float) -> str:
        """
        Format duration as M:SS.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted string like "0:08"
        """
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes}:{secs:02d}"

    def _check_duration_warnings(self):
        """Check duration and show appropriate warnings."""
        if self._audio_duration < self.MIN_DURATION_WARNING:
            self._show_warning(
                f"Audio should be at least {int(self.MIN_DURATION_WARNING)} seconds for best results"
            )
        elif self._audio_duration > self.MAX_DURATION_WARNING:
            self._show_warning(
                f"Only the first {int(self.MAX_DURATION_WARNING)} seconds will be used"
            )
        else:
            self._hide_warning()

    def _show_warning(self, message: str):
        """Show a warning message."""
        self.warning_label.setText(f"Warning: {message}")
        self.warning_label.setStyleSheet("color: #b8860b; background-color: #fff8dc; padding: 8px; border-radius: 4px;")
        self.warning_label.setVisible(True)

    def _hide_warning(self):
        """Hide the warning message."""
        self.warning_label.setVisible(False)
        self.warning_label.setText("")

    def _show_error(self, message: str):
        """Show an error message."""
        self._has_error = True
        self._audio_path = None
        self.file_label.setText(f"Error: {message}")
        self.file_label.setStyleSheet("color: red;")
        self.info_group.setVisible(False)
        self.logger.error(f"Audio file error: {message}")

    def _on_voice_name_changed(self):
        """Handle voice name field changes."""
        self.save_ready_changed.emit(self._is_save_ready_internal())

        # QA4: Enable/disable Clone tab save button based on name + extraction status
        has_name = bool(self.get_voice_name())
        has_extraction = self._generated_audio_path is not None
        self.clone_save_button.setEnabled(has_name and has_extraction and not self._is_saving)

    def _on_save_clicked(self):
        """Handle Save Voice button click in Clone tab (QA4)."""
        voice_name = self.get_voice_name()
        if not voice_name:
            self.save_feedback_label.setText("Please enter a voice name first.")
            self.save_feedback_label.setStyleSheet("color: #b8860b;")
            return

        if not self._generated_audio_path:
            self.save_feedback_label.setText("Please extract the voice first.")
            self.save_feedback_label.setStyleSheet("color: #b8860b;")
            return

        # Emit signal for parent to handle save
        self.save_voice_requested.emit(voice_name)

    def _on_refinement_clicked(self):
        """Handle Refinement button click in Clone tab (QA7).

        Emits refinement_requested signal to navigate to Emotions tab
        for building a full voice with multiple emotion variants.
        """
        self.logger.debug("Refinement button clicked - requesting transition to Emotions tab")
        self.refinement_requested.emit()

    def _is_save_ready_internal(self) -> bool:
        """Check if save is ready (internal helper)."""
        has_file = self._audio_path is not None
        has_name = bool(self.get_voice_name())
        has_embedding = self._generated_embedding_path is not None or self._generated_audio_path is not None
        not_saving = not self._is_saving
        return has_file and has_name and has_embedding and not_saving

    # Public API

    def get_audio_path(self) -> Optional[Path]:
        """
        Get the loaded audio file path.

        Returns:
            Path to audio file, or None if not loaded
        """
        return self._audio_path

    def get_audio_duration(self) -> float:
        """
        Get the loaded audio duration.

        Returns:
            Duration in seconds
        """
        return self._audio_duration

    def get_voice_name(self) -> str:
        """
        Get the entered voice name.

        Returns:
            Voice name (stripped of whitespace)
        """
        return self.voice_name_edit.text().strip()

    def set_voice_name(self, name: str):
        """
        Set the voice name field.

        Args:
            name: Voice name to set
        """
        self.voice_name_edit.setText(name)

    def has_audio_file(self) -> bool:
        """
        Check if an audio file is loaded.

        Returns:
            True if a valid audio file is loaded
        """
        return self._audio_path is not None and not self._has_error

    def has_content(self) -> bool:
        """
        Check if the panel has any content.

        Returns:
            True if an audio file is loaded
        """
        return self.has_audio_file()

    def has_error(self) -> bool:
        """
        Check if there was an error loading the file.

        Returns:
            True if there was an error
        """
        return self._has_error

    def is_save_ready(self) -> bool:
        """
        Check if the voice is ready to be saved.

        Returns:
            True if audio file loaded and voice name entered
        """
        return self._is_save_ready_internal()

    def clear(self):
        """Clear all state and reset the panel."""
        self._audio_path = None
        self._audio_duration = 0.0
        self._audio_format = ""
        self._has_error = False
        self._is_saving = False
        self._generated_embedding_path = None

        # Reset transcription state (Story 2.2)
        self._is_transcribing = False
        self._transcription_error = False

        self.file_label.setText("No file selected")
        self.file_label.setStyleSheet("color: #666; font-style: italic;")
        self.info_group.setVisible(False)
        self.voice_name_edit.clear()
        self.audio_player.clear()
        self._hide_warning()
        self.status_label.setText("")

        # Reset transcript UI (Story 2.2)
        self.transcript_group.setVisible(False)
        self.transcript_edit.clear()
        self.transcribe_progress.setVisible(False)
        self.transcribe_status.setText("")
        self.transcribe_retry_button.setVisible(False)
        self.transcribe_button.setEnabled(True)
        self.transcript_edit.setEnabled(True)

        # Reset extraction state (Story 2.3)
        self._is_extracting = False
        self._extraction_error = False
        self._generated_audio_path = None

        # Reset re-extraction tracking (Story 2.4)
        self._last_extracted_transcript = None
        self._transcript_modified_since_extraction = False
        self.transcript_modified_label.setVisible(False)

        # Reset extract UI (Story 2.3)
        self.extract_group.setVisible(False)
        self.preview_text_edit.clear()
        self.extract_progress.setVisible(False)
        self.extract_status.setText("")
        self.extract_retry_button.setVisible(False)
        self.extract_button.setEnabled(False)
        self.preview_text_edit.setEnabled(True)
        self.preview_audio_group.setVisible(False)
        self.preview_audio_player.clear()

        # QA4: Reset Clone tab save button
        self.clone_save_button.setEnabled(False)
        self.clone_save_button.setText("Save Voice")
        self.save_feedback_label.setText("")

        self.save_ready_changed.emit(False)

    # Save state management (for future stories)

    def get_generated_embedding_path(self) -> Optional[Path]:
        """
        Get the path to the generated embedding file.

        Returns:
            Path to embedding .pt file, or None if not generated
        """
        return self._generated_embedding_path

    def set_saving(self, is_saving: bool, message: str = "Saving..."):
        """
        Set the saving state.

        Args:
            is_saving: True if save is in progress
            message: Progress message to display
        """
        self._is_saving = is_saving

        if is_saving:
            self.status_label.setText(message)
            self.status_label.setStyleSheet("color: blue;")
            self.voice_name_edit.setEnabled(False)
        else:
            self.voice_name_edit.setEnabled(True)

        self.save_ready_changed.emit(self._is_save_ready_internal())

    def is_saving(self) -> bool:
        """
        Check if save is in progress.

        Returns:
            True if saving
        """
        return self._is_saving

    # QA4: Clone tab save feedback methods

    def set_clone_save_in_progress(self, message: str = "Saving..."):
        """Set Clone tab save button to in-progress state (QA4)."""
        self._is_saving = True
        self.clone_save_button.setEnabled(False)
        self.clone_save_button.setText(message)
        self.save_feedback_label.setText("")
        self.voice_name_edit.setEnabled(False)

    def set_clone_save_complete(self, voice_name: str):
        """Set Clone tab save as complete with success feedback (QA4)."""
        self._is_saving = False
        self.clone_save_button.setText("Save Voice")
        self.clone_save_button.setEnabled(True)
        self.voice_name_edit.setEnabled(True)

        # Show success feedback
        self.save_feedback_label.setText(f"✓ Voice '{voice_name}' saved successfully!")
        self.save_feedback_label.setStyleSheet("color: green; font-weight: bold;")

        self.logger.info(f"Clone tab save complete: {voice_name}")

    def set_clone_save_error(self, error_message: str):
        """Set Clone tab save error with feedback (QA4)."""
        self._is_saving = False
        self.clone_save_button.setText("Save Voice")
        self.clone_save_button.setEnabled(True)
        self.voice_name_edit.setEnabled(True)

        # Show error feedback
        self.save_feedback_label.setText(f"✗ Save failed: {error_message}")
        self.save_feedback_label.setStyleSheet("color: red;")

        self.logger.error(f"Clone tab save error: {error_message}")

    # Transcription methods (Story 2.2)

    def _on_transcribe_clicked(self):
        """Handle Auto-Transcribe button click."""
        if not self._audio_path:
            return

        # Reset error state
        self._transcription_error = False
        self.transcribe_retry_button.setVisible(False)

        # Emit signal for parent to handle transcription
        self.transcribe_requested.emit(str(self._audio_path))

    def _on_transcript_changed(self):
        """Handle transcript text changes."""
        # Story 2.4: Check if transcript modified since last extraction
        self._update_transcript_modified_state()

        self.transcript_changed.emit()

    def _update_transcript_modified_state(self):
        """Update the transcript modified indicator (Story 2.4)."""
        if self._last_extracted_transcript is not None:
            current_transcript = self.get_transcript()
            is_modified = current_transcript != self._last_extracted_transcript
            self._transcript_modified_since_extraction = is_modified
            self.transcript_modified_label.setVisible(is_modified)
        else:
            self._transcript_modified_since_extraction = False
            self.transcript_modified_label.setVisible(False)

    def set_transcribing(self, is_transcribing: bool, message: str = "Transcribing..."):
        """
        Set the transcription state.

        Args:
            is_transcribing: True if transcription is in progress
            message: Status message to display
        """
        self._is_transcribing = is_transcribing

        if is_transcribing:
            # Show progress indicators
            self.transcribe_progress.setVisible(True)
            self.transcribe_status.setText(message)
            self.transcribe_status.setStyleSheet("color: blue;")

            # Disable controls during transcription
            self.transcribe_button.setEnabled(False)
            self.transcript_edit.setEnabled(False)
            self.transcribe_retry_button.setVisible(False)
        else:
            # Hide progress
            self.transcribe_progress.setVisible(False)
            self.transcribe_button.setEnabled(True)
            self.transcript_edit.setEnabled(True)

    def set_transcription_complete(self, transcript: str):
        """
        Set transcription as complete with the result.

        Args:
            transcript: The transcribed text
        """
        self._is_transcribing = False
        self._transcription_error = False

        # Update UI
        self.transcribe_progress.setVisible(False)
        self.transcribe_button.setEnabled(True)
        self.transcript_edit.setEnabled(True)
        self.transcribe_retry_button.setVisible(False)

        # Set transcript text
        self.transcript_edit.setText(transcript)

        # Show success status briefly
        self.transcribe_status.setText("Transcription complete")
        self.transcribe_status.setStyleSheet("color: green;")

        self.logger.info("Transcription completed successfully")

    def set_transcription_error(self, error_message: str):
        """
        Set transcription error state.

        Args:
            error_message: Error message to display
        """
        self._is_transcribing = False
        self._transcription_error = True

        # Update UI
        self.transcribe_progress.setVisible(False)
        self.transcribe_button.setEnabled(True)
        self.transcript_edit.setEnabled(True)

        # Show error and retry button
        self.transcribe_status.setText(f"Error: {error_message}")
        self.transcribe_status.setStyleSheet("color: red;")
        self.transcribe_retry_button.setVisible(True)

        self.logger.error(f"Transcription error: {error_message}")

    def get_transcript(self) -> str:
        """
        Get the transcript text.

        Returns:
            Transcript text (stripped of whitespace)
        """
        return self.transcript_edit.toPlainText().strip()

    def set_transcript(self, text: str):
        """
        Set the transcript text.

        Args:
            text: Transcript text to set
        """
        self.transcript_edit.setText(text)

    def has_transcript(self) -> bool:
        """
        Check if a transcript exists.

        Returns:
            True if transcript text is not empty
        """
        return bool(self.get_transcript())

    def is_transcribing(self) -> bool:
        """
        Check if transcription is in progress.

        Returns:
            True if transcription is running
        """
        return self._is_transcribing

    def has_transcription_error(self) -> bool:
        """
        Check if there was a transcription error.

        Returns:
            True if transcription failed
        """
        return self._transcription_error

    # Extraction methods (Story 2.3)

    def _on_extract_clicked(self):
        """Handle Extract Embedding button click."""
        if not self._audio_path:
            return

        preview_text = self.get_preview_text()
        if not preview_text:
            return

        # Reset error state
        self._extraction_error = False
        self.extract_retry_button.setVisible(False)

        # Emit signal for parent to handle extraction
        self.extract_requested.emit(
            str(self._audio_path),
            self.get_transcript(),
            preview_text,
            self.get_language()
        )

    def _on_preview_text_changed(self):
        """Handle preview text field changes."""
        # Enable/disable extract button based on preview text
        has_text = bool(self.get_preview_text())
        self.extract_button.setEnabled(has_text and not self._is_extracting)

        # Emit signal
        self.preview_text_changed.emit()

    def set_extracting(self, is_extracting: bool, message: str = "Extracting..."):
        """
        Set the extraction state.

        Args:
            is_extracting: True if extraction is in progress
            message: Status message to display
        """
        self._is_extracting = is_extracting

        if is_extracting:
            # Show progress indicators
            self.extract_progress.setVisible(True)
            self.extract_status.setText(message)
            self.extract_status.setStyleSheet("color: blue;")

            # Disable controls during extraction
            self.extract_button.setEnabled(False)
            self.preview_text_edit.setEnabled(False)
            self.extract_retry_button.setVisible(False)

            # Hide previous preview if any
            self.preview_audio_group.setVisible(False)
        else:
            # Hide progress
            self.extract_progress.setVisible(False)
            self.preview_text_edit.setEnabled(True)
            # extract_button re-enabled based on preview_text

            # Re-enable button if there's preview text
            has_text = bool(self.get_preview_text())
            self.extract_button.setEnabled(has_text)

    def set_extraction_complete(self, audio_path: Path, auto_play: bool = True):
        """
        Set extraction as complete with the generated audio.

        Args:
            audio_path: Path to the generated preview audio file
            auto_play: Whether to auto-play the preview (default True)
        """
        self._is_extracting = False
        self._extraction_error = False
        self._generated_audio_path = audio_path

        # Story 2.4: Record the transcript used for this extraction
        self._last_extracted_transcript = self.get_transcript()
        self._transcript_modified_since_extraction = False
        self.transcript_modified_label.setVisible(False)

        # Update UI
        self.extract_progress.setVisible(False)
        self.preview_text_edit.setEnabled(True)
        self.extract_retry_button.setVisible(False)

        # Re-enable button
        has_text = bool(self.get_preview_text())
        self.extract_button.setEnabled(has_text)

        # Clear extraction status
        self.extract_status.setText("")

        # Show preview section and load audio
        self.preview_audio_group.setVisible(True)
        self.preview_audio_player.set_audio_file(audio_path)

        # Auto-play if requested
        if auto_play:
            self.preview_audio_player.play()

        # QA4: Enable Clone tab save button if voice name entered
        has_name = bool(self.get_voice_name())
        self.clone_save_button.setEnabled(has_name)
        self.save_feedback_label.setText("")  # Clear any previous feedback

        # Update save readiness
        self.save_ready_changed.emit(self._is_save_ready_internal())

        self.logger.info(f"Extraction completed: {audio_path}")

    def set_extraction_error(self, error_message: str, suggestions: Optional[str] = None):
        """
        Set extraction error state.

        Args:
            error_message: Error message to display
            suggestions: Optional suggestions for fixing the issue
        """
        self._is_extracting = False
        self._extraction_error = True

        # Update UI
        self.extract_progress.setVisible(False)
        self.preview_text_edit.setEnabled(True)

        # Re-enable button
        has_text = bool(self.get_preview_text())
        self.extract_button.setEnabled(has_text)

        # Show error and retry button
        error_text = f"Error: {error_message}"
        if suggestions:
            error_text += f"\n{suggestions}"
        self.extract_status.setText(error_text)
        self.extract_status.setStyleSheet("color: red;")
        self.extract_retry_button.setVisible(True)

        # Hide preview section
        self.preview_audio_group.setVisible(False)

        self.logger.error(f"Extraction error: {error_message}")

    def get_preview_text(self) -> str:
        """
        Get the preview text.

        Returns:
            Preview text (stripped of whitespace)
        """
        return self.preview_text_edit.text().strip()

    def get_language(self) -> str:
        """
        Get the selected language for voice extraction.

        Returns:
            Selected language name (e.g., "English", "Chinese", "Auto")
        """
        return self.language_combo.currentText()

    def set_language(self, language: str):
        """
        Set the selected language.

        Args:
            language: Language name to select (must be in SUPPORTED_LANGUAGES)
        """
        index = self.language_combo.findText(language)
        if index >= 0:
            self.language_combo.setCurrentIndex(index)

    def set_preview_text(self, text: str):
        """
        Set the preview text.

        Args:
            text: Preview text to set
        """
        self.preview_text_edit.setText(text)

    def has_preview_text(self) -> bool:
        """
        Check if preview text exists.

        Returns:
            True if preview text is not empty
        """
        return bool(self.get_preview_text())

    def is_extracting(self) -> bool:
        """
        Check if extraction is in progress.

        Returns:
            True if extraction is running
        """
        return self._is_extracting

    def has_extraction_error(self) -> bool:
        """
        Check if there was an extraction error.

        Returns:
            True if extraction failed
        """
        return self._extraction_error

    def get_generated_audio_path(self) -> Optional[Path]:
        """
        Get the path to the generated preview audio.

        Returns:
            Path to preview audio file, or None if not generated
        """
        return self._generated_audio_path

    def has_generated_preview(self) -> bool:
        """
        Check if a preview has been successfully generated.

        Returns:
            True if preview audio exists
        """
        return self._generated_audio_path is not None and not self._extraction_error

    # Re-extraction methods (Story 2.4)

    def is_transcript_modified_since_extraction(self) -> bool:
        """
        Check if transcript has been modified since last extraction.

        Returns:
            True if transcript was modified after an extraction
        """
        return self._transcript_modified_since_extraction

    def get_last_extracted_transcript(self) -> Optional[str]:
        """
        Get the transcript used for the last extraction.

        Returns:
            Transcript text used for last extraction, or None if no extraction done
        """
        return self._last_extracted_transcript

    def has_had_extraction(self) -> bool:
        """
        Check if any extraction has been performed.

        Returns:
            True if at least one extraction has been done
        """
        return self._last_extracted_transcript is not None

    # QA3: Cross-tab data transfer

    def load_from_description(
        self,
        audio_path: Path,
        transcript: str = "",
        voice_name: str = ""
    ):
        """
        Load data transferred from the Description tab (QA3).

        This enables the "Save Voice" → "Refine in Sample Tab" flow where
        a user generates a voice from description, then wants to refine
        it using the voice cloning workflow.

        Args:
            audio_path: Path to the audio sample from description generation
            transcript: Preview text to use as transcript
            voice_name: Voice name to pre-fill
        """
        self.logger.info(
            f"Loading from description tab: {audio_path}, "
            f"transcript={len(transcript)} chars, name='{voice_name}'"
        )

        # Load the audio file
        self._load_audio_file(audio_path)

        # Set transcript if provided
        if transcript:
            self.set_transcript(transcript)

        # Set voice name if provided
        if voice_name:
            self.set_voice_name(voice_name)

        # Emit signals to indicate data was transferred
        self.content_changed.emit()
        self.save_ready_changed.emit(self._is_save_ready_internal())
