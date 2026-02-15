"""
Emotions Panel for Voice Design Studio

Panel for the "Emotions" tab in Voice Design Studio. Allows users to generate
5 emotion variants (Neutral, Happy, Sad, Angry, Flirtatious) for embedding voices.
Each emotion has its own sub-tab with variant generation and selection.

Emotion Variants Feature:
- EMBEDDING voices can have multiple embeddings, one per emotion
- Each emotion tab contains 5 variant rows for generation/selection
- Neutral is required; other emotions are optional
- "Proceed to Refinement" button enables when Neutral is selected

Story: UI: Create EmotionsPanel component
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QTextEdit, QComboBox, QTabWidget, QSplitter,
    QFrame, QSizePolicy, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from myvoice.ui.dialogs.voice_design_studio.voice_variant_list import VoiceVariantList
from myvoice.models.voice_profile import VALID_EMOTIONS


# Emotion display configuration: emoji, display name, tab order
EMOTION_CONFIG = {
    "neutral": {"emoji": "😐", "name": "Neutral", "index": 0, "required": True},
    "happy": {"emoji": "😄", "name": "Happy", "index": 1, "required": False},
    "sad": {"emoji": "😢", "name": "Sad", "index": 2, "required": False},
    "angry": {"emoji": "😠", "name": "Angry", "index": 3, "required": False},
    "flirtatious": {"emoji": "😏", "name": "Flirtatious", "index": 4, "required": False},
}

# Default emotion instruction prompts (same as EmotionPreset)
EMOTION_PROMPTS = {
    "neutral": "Speak in a calm, balanced, and matter-of-fact tone with steady pacing and even intonation, conveying composed professionalism without emotional inflection",
    "happy": "Speak with bright, joyful enthusiasm and warmth, voice lifted with genuine delight, occasional light laughter in the breath, upbeat tempo with melodic rises expressing pure happiness",
    "sad": "Speak with deep sorrow and heaviness, voice trembling with grief, shaky breaths between phrases, slow wavering delivery as if holding back tears, pitch dropping with melancholy and despair",
    "angry": "Speak with fierce intensity and sharp edges, voice tight with barely contained rage, forceful emphasis on words, clipped aggressive pacing, seething undertone building to explosive bursts",
    "flirtatious": "Speak with playful, teasing allure and a coy smile in the voice, breathy whispered hints, drawn-out vowels with suggestive pauses, warm intimate tone dripping with charming mischief",
}


class EmotionTabWidget(QWidget):
    """
    Widget for a single emotion tab containing:
    - Voice description (QA7: editable for per-emotion modification)
    - Emotion instruction input
    - Variant list (5 variants)
    - Generate button
    """

    generate_requested = pyqtSignal(str, str, str)  # emotion, instruction, preview_text
    upload_completed = pyqtSignal(str, str)  # QA7: emotion, audio_path - emitted when upload finishes
    variant_selected = pyqtSignal(str, int)  # emotion, variant_index
    variant_deselected = pyqtSignal(str)  # emotion

    def __init__(
        self,
        emotion: str,
        parent: Optional[QWidget] = None
    ):
        """
        Initialize the emotion tab widget.

        Args:
            emotion: Emotion ID (e.g., "neutral", "happy")
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{emotion}")
        self._emotion = emotion
        self._config = EMOTION_CONFIG[emotion]
        self._voice_description: str = ""
        self._preview_text: str = ""
        self._is_generating = False

        self._create_ui()
        self._setup_connections()
        self._setup_accessibility()

        self.logger.debug(f"EmotionTabWidget initialized for {emotion}")

    def _create_ui(self):
        """Create the tab UI with 3-column layout."""
        layout = QVBoxLayout(self)
        # QA3-2: Reduced margins and spacing for smaller screens
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Main content area using splitter for resizable columns
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Column 1: Voice Description (QA7: editable)
        col1_widget = self._create_description_column()
        splitter.addWidget(col1_widget)

        # Column 2: Emotion Instruction
        col2_widget = self._create_instruction_column()
        splitter.addWidget(col2_widget)

        # Column 3: Variant List
        col3_widget = self._create_variant_column()
        splitter.addWidget(col3_widget)

        # Set initial column sizes (1:1:1.5 ratio)
        splitter.setSizes([200, 200, 300])

        layout.addWidget(splitter, 1)  # Stretch factor 1

        # Bottom row: Generate button and status
        self._create_generate_section(layout)

    def _create_description_column(self) -> QWidget:
        """Create the voice description column."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(8)

        # Header
        header = QLabel("Voice Description")
        header.setObjectName("description_header")
        font = QFont()
        font.setBold(True)
        header.setFont(font)
        layout.addWidget(header)

        # Description display (QA7: editable so users can modify per-emotion)
        self.description_display = QTextEdit()
        self.description_display.setObjectName("voice_description_display")
        self.description_display.setReadOnly(False)  # QA7: Made editable
        self.description_display.setPlaceholderText("No voice description available")
        self.description_display.setMinimumHeight(100)
        self.description_display.setMaximumHeight(200)
        layout.addWidget(self.description_display)

        layout.addStretch()
        return widget

    def _create_instruction_column(self) -> QWidget:
        """Create the emotion instruction column."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(8)

        # Header with emotion emoji
        emoji = self._config["emoji"]
        name = self._config["name"]
        header = QLabel(f"{emoji} {name} Emotion")
        header.setObjectName("emotion_header")
        font = QFont()
        font.setBold(True)
        header.setFont(font)
        layout.addWidget(header)

        # Instruction template dropdown
        template_layout = QHBoxLayout()
        template_layout.setSpacing(8)

        template_label = QLabel("Template:")
        template_layout.addWidget(template_label)

        self.template_combo = QComboBox()
        self.template_combo.setObjectName("emotion_template_combo")
        self.template_combo.addItem("Default", EMOTION_PROMPTS[self._emotion])
        self.template_combo.addItem("Custom", "")
        self.template_combo.setMinimumWidth(100)
        template_layout.addWidget(self.template_combo, 1)

        layout.addLayout(template_layout)

        # Instruction text area
        instruction_label = QLabel("Emotion Instruction:")
        layout.addWidget(instruction_label)

        self.instruction_edit = QTextEdit()
        self.instruction_edit.setObjectName("emotion_instruction_edit")
        self.instruction_edit.setPlainText(EMOTION_PROMPTS[self._emotion])
        self.instruction_edit.setPlaceholderText(
            "Describe how the voice should express this emotion..."
        )
        self.instruction_edit.setMinimumHeight(80)
        self.instruction_edit.setMaximumHeight(150)
        layout.addWidget(self.instruction_edit)

        # Required indicator for neutral
        if self._config["required"]:
            required_label = QLabel("* Required for saving")
            required_label.setObjectName("required_indicator")
            required_label.setStyleSheet("color: #cc6600; font-style: italic;")
            layout.addWidget(required_label)

        layout.addStretch()
        return widget

    def _create_variant_column(self) -> QWidget:
        """Create the variant list column."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 0, 0, 0)
        layout.setSpacing(8)

        # Header
        header = QLabel("Variants")
        header.setObjectName("variants_header")
        font = QFont()
        font.setBold(True)
        header.setFont(font)
        layout.addWidget(header)

        # Variant list widget
        self.variant_list = VoiceVariantList()
        self.variant_list.setObjectName(f"variant_list_{self._emotion}")
        layout.addWidget(self.variant_list, 1)

        return widget

    def _create_generate_section(self, parent_layout: QVBoxLayout):
        """Create the generate button and status section."""
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        parent_layout.addWidget(separator)

        # Generate row
        generate_layout = QHBoxLayout()
        generate_layout.setSpacing(12)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setObjectName("emotion_status_label")
        generate_layout.addWidget(self.status_label, 1)

        # QA7: Upload Sample button - alternative to generation
        self.upload_button = QPushButton("Upload Sample")
        self.upload_button.setObjectName(f"upload_{self._emotion}_button")
        self.upload_button.setMinimumWidth(120)
        self.upload_button.setToolTip(
            "Upload an existing audio sample for this emotion variant"
        )
        generate_layout.addWidget(self.upload_button)

        # Generate button
        emoji = self._config["emoji"]
        self.generate_button = QPushButton(f"Generate {emoji} Variants")
        self.generate_button.setObjectName(f"generate_{self._emotion}_button")
        self.generate_button.setMinimumWidth(180)
        self.generate_button.setEnabled(False)  # Disabled until description set
        generate_layout.addWidget(self.generate_button)

        parent_layout.addLayout(generate_layout)

    def _setup_connections(self):
        """Setup signal connections."""
        # Template selection
        self.template_combo.currentIndexChanged.connect(self._on_template_changed)

        # QA7: Description text changes (now editable)
        self.description_display.textChanged.connect(self._update_generate_button_state)

        # Instruction text changes
        self.instruction_edit.textChanged.connect(self._on_instruction_changed)

        # Variant list signals
        self.variant_list.variant_selected.connect(self._on_variant_selected)
        self.variant_list.variant_deselected.connect(self._on_variant_deselected)
        self.variant_list.all_complete.connect(self._on_all_complete)

        # Generate button
        self.generate_button.clicked.connect(self._on_generate_clicked)

        # QA7: Upload button
        self.upload_button.clicked.connect(self._on_upload_clicked)

        # Regenerate button from variant list
        self.variant_list.regenerate_button.clicked.connect(self._on_generate_clicked)

    def _setup_accessibility(self):
        """Configure accessibility properties."""
        name = self._config["name"]

        self.description_display.setAccessibleName(f"{name} Voice Description")
        self.description_display.setAccessibleDescription(
            f"Editable voice description for {name} emotion - modify if needed"
        )

        self.template_combo.setAccessibleName(f"{name} Instruction Template")
        self.template_combo.setAccessibleDescription(
            f"Select a template for {name} emotion instruction"
        )

        self.instruction_edit.setAccessibleName(f"{name} Emotion Instruction")
        self.instruction_edit.setAccessibleDescription(
            f"Enter the instruction for how the voice should express {name} emotion"
        )

        self.generate_button.setAccessibleName(f"Generate {name} Variants")
        self.generate_button.setAccessibleDescription(
            f"Generate 5 voice variants with {name} emotion"
        )

        # QA7: Upload button accessibility
        self.upload_button.setAccessibleName(f"Upload {name} Sample")
        self.upload_button.setAccessibleDescription(
            f"Upload an existing audio sample for {name} emotion"
        )

    def _on_template_changed(self, index: int):
        """Handle template selection change."""
        template_text = self.template_combo.currentData()
        if template_text:  # Default template
            self.instruction_edit.setPlainText(template_text)
        # Custom template leaves current text as-is

    def _on_instruction_changed(self):
        """Handle instruction text changes."""
        self._update_generate_button_state()

    def _on_variant_selected(self, variant_index: int):
        """Handle variant selection from the list."""
        self.logger.debug(f"Variant {variant_index} selected for {self._emotion}")
        self.variant_selected.emit(self._emotion, variant_index)

    def _on_variant_deselected(self):
        """Handle variant deselection."""
        self.logger.debug(f"Variant deselected for {self._emotion}")
        self.variant_deselected.emit(self._emotion)

    def _on_all_complete(self):
        """Handle all variants complete."""
        self._is_generating = False
        self._update_status("5 variants ready. Select one.")
        self._update_generate_button_state()

    def _on_generate_clicked(self):
        """Handle generate button click."""
        instruction = self.get_instruction()
        self.logger.debug(f"Generate clicked for {self._emotion}")
        self.generate_requested.emit(self._emotion, instruction, self._preview_text)

    def _on_upload_clicked(self):
        """QA7: Handle upload sample button click."""
        self.logger.debug(f"Upload clicked for {self._emotion}")

        # Supported audio formats
        file_filter = "Audio Files (*.wav *.mp3 *.m4a);;WAV Files (*.wav);;MP3 Files (*.mp3);;M4A Files (*.m4a);;All Files (*)"

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {self._config['name']} Voice Sample",
            "",
            file_filter
        )

        if not file_path:
            return  # User cancelled

        audio_path = Path(file_path)
        if not audio_path.exists():
            QMessageBox.warning(
                self,
                "File Not Found",
                f"The selected file could not be found:\n{file_path}"
            )
            return

        # Validate file extension
        valid_extensions = {'.wav', '.mp3', '.m4a'}
        if audio_path.suffix.lower() not in valid_extensions:
            QMessageBox.warning(
                self,
                "Unsupported Format",
                f"Please select a WAV, MP3, or M4A audio file.\n"
                f"Selected: {audio_path.suffix}"
            )
            return

        # Load the uploaded sample as a variant
        self._load_uploaded_sample(audio_path)

    def _load_uploaded_sample(self, audio_path: Path):
        """QA7: Load an uploaded sample into the first variant slot."""
        try:
            # Get audio duration using mutagen or other method
            duration = self._get_audio_duration(audio_path)

            # Clear any existing variants and show the uploaded one in slot 0
            self.variant_list.clear()

            # We don't have an embedding yet - just the audio file
            # The embedding will be extracted later in the Refinement step
            # For now, show the audio in the variant list with a placeholder embedding path
            self.variant_list.set_variant_complete(
                variant_index=0,
                audio_path=audio_path,
                embedding_path=None,  # Will be set when embedding is extracted
                duration=duration
            )

            self._update_status(f"Sample uploaded: {audio_path.name}")
            self.logger.info(f"Uploaded sample for {self._emotion}: {audio_path}")

            # Emit signal so parent can handle the upload
            self.upload_completed.emit(self._emotion, str(audio_path))

            # Auto-select the uploaded variant
            self.variant_list.select_variant(0)

        except Exception as e:
            self.logger.error(f"Failed to load uploaded sample: {e}")
            QMessageBox.warning(
                self,
                "Upload Failed",
                f"Failed to load the audio sample:\n{str(e)}"
            )

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds."""
        try:
            # Try using mutagen for accurate duration
            import mutagen
            audio = mutagen.File(audio_path)
            if audio is not None and hasattr(audio, 'info') and hasattr(audio.info, 'length'):
                return audio.info.length
        except Exception:
            pass

        try:
            # Fallback: try scipy for WAV files
            if audio_path.suffix.lower() == '.wav':
                import scipy.io.wavfile as wavfile
                sample_rate, data = wavfile.read(audio_path)
                return len(data) / sample_rate
        except Exception:
            pass

        # Default estimate if we can't determine duration
        return 5.0

    def _update_generate_button_state(self):
        """Update generate button enabled state."""
        # QA7: Use actual text from display (may be modified by user)
        has_description = bool(self.description_display.toPlainText().strip())
        has_instruction = bool(self.get_instruction())
        not_generating = not self._is_generating

        self.generate_button.setEnabled(
            has_description and has_instruction and not_generating
        )

    def _update_status(self, message: str, is_error: bool = False):
        """Update status label."""
        self.status_label.setText(message)
        if is_error:
            self.status_label.setStyleSheet("color: red;")
        else:
            self.status_label.setStyleSheet("color: green;")

    # Public API

    def get_emotion(self) -> str:
        """Get the emotion ID."""
        return self._emotion

    def get_instruction(self) -> str:
        """Get the current emotion instruction text."""
        return self.instruction_edit.toPlainText().strip()

    def set_instruction(self, instruction: str):
        """Set the emotion instruction text."""
        self.instruction_edit.setPlainText(instruction)

    def get_voice_description(self) -> str:
        """Get the current voice description (QA7: may be modified by user)."""
        return self.description_display.toPlainText().strip()

    def set_voice_description(self, description: str):
        """Set the voice description (QA7: editable by user)."""
        self._voice_description = description
        self.description_display.setPlainText(description)
        self._update_generate_button_state()

    def set_preview_text(self, text: str):
        """Set the preview text for generation."""
        self._preview_text = text
        self._update_generate_button_state()

    def set_generating(self, is_generating: bool, message: str = "Generating..."):
        """Set the generation state."""
        self._is_generating = is_generating
        self.generate_button.setEnabled(not is_generating)
        if is_generating:
            self.status_label.setText(message)
            self.status_label.setStyleSheet("color: blue;")
            self.variant_list.start_generation()

    def set_variant_generating(self, variant_index: int):
        """Set a specific variant to generating state."""
        self.variant_list.set_variant_generating(variant_index)
        self.status_label.setText(f"Generating variant {variant_index + 1}/5...")

    def set_variant_complete(
        self,
        variant_index: int,
        audio_path: Path,
        embedding_path: Path,
        duration: float = None
    ):
        """Set a specific variant to complete state."""
        self.variant_list.set_variant_complete(
            variant_index, audio_path, embedding_path, duration
        )

    def set_variant_error(self, variant_index: int, error_message: str):
        """Set a specific variant to error state."""
        self.variant_list.set_variant_error(variant_index, error_message)

    def finish_generation(self):
        """Mark generation as finished."""
        self._is_generating = False
        self.variant_list.finish_generation()
        self._update_generate_button_state()

    def set_generation_error(self, error_message: str):
        """Set generation error state."""
        self._is_generating = False
        self._update_status(f"Error: {error_message}", is_error=True)
        self._update_generate_button_state()

    def get_selected_variant_index(self) -> Optional[int]:
        """Get the selected variant index, or None if none selected."""
        return self.variant_list.get_selected_index()

    def get_selected_variant_paths(self) -> Optional[tuple]:
        """Get the selected variant's paths (audio_path, embedding_path)."""
        return self.variant_list.get_selected_variant()

    def has_selection(self) -> bool:
        """Check if a variant is selected."""
        return self.variant_list.has_selection()

    def has_complete_variants(self) -> bool:
        """Check if any variants are complete."""
        return self.variant_list.has_complete_variants()

    def clear(self):
        """Clear all variants and reset state."""
        self.variant_list.clear()
        self._is_generating = False
        self.status_label.setText("")
        self._update_generate_button_state()


class EmotionsPanel(QWidget):
    """
    Panel for generating emotion variants for embedding voices.

    Contains 5 emotion sub-tabs (Neutral, Happy, Sad, Angry, Flirtatious),
    each with variant generation and selection capability.

    Signals:
        emotion_variant_selected: Emitted when any emotion's variant is selected
        generation_requested: Emitted when generation is requested for an emotion
        proceed_to_refinement_requested: Emitted when Proceed button is clicked
        neutral_selection_changed: Emitted when neutral's selection state changes
    """

    emotion_variant_selected = pyqtSignal(str, int)  # emotion, variant_index
    generation_requested = pyqtSignal(str, str, str)  # emotion, instruction, preview_text
    proceed_to_refinement_requested = pyqtSignal()
    neutral_selection_changed = pyqtSignal(bool)  # has_selection

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the Emotions Panel.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        self._voice_description: str = ""
        self._preview_text: str = ""
        self._emotion_tabs: Dict[str, EmotionTabWidget] = {}

        self._create_ui()
        self._setup_connections()
        self._setup_accessibility()

        self.logger.debug("EmotionsPanel initialized")

    def _create_ui(self):
        """Create the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # QA3-2: Reduced spacing for smaller screens
        layout.setSpacing(4)

        # Header with instructions
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)

        instruction_label = QLabel(
            "Generate variants for each emotion. Neutral is required; others are optional."
        )
        instruction_label.setWordWrap(True)
        instruction_label.setObjectName("emotions_instruction")
        header_layout.addWidget(instruction_label, 1)

        layout.addLayout(header_layout)

        # Emotion tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("emotion_tabs")
        self.tab_widget.setDocumentMode(True)

        # Create tabs for each emotion in order
        for emotion_id in ["neutral", "happy", "sad", "angry", "flirtatious"]:
            config = EMOTION_CONFIG[emotion_id]
            tab = EmotionTabWidget(emotion_id)
            self._emotion_tabs[emotion_id] = tab

            # Tab title with emoji
            tab_title = f"{config['emoji']} {config['name']}"
            self.tab_widget.addTab(tab, tab_title)

        layout.addWidget(self.tab_widget, 1)

        # Bottom section: Proceed button
        self._create_proceed_section(layout)

    def _create_proceed_section(self, parent_layout: QVBoxLayout):
        """Create the proceed to refinement section."""
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        parent_layout.addWidget(separator)

        # Proceed row
        proceed_layout = QHBoxLayout()
        proceed_layout.setSpacing(12)

        # Status/info label
        self.proceed_status = QLabel("Select a Neutral variant to proceed.")
        self.proceed_status.setObjectName("proceed_status")
        proceed_layout.addWidget(self.proceed_status, 1)

        # Proceed button
        self.proceed_button = QPushButton("Proceed to Refinement")
        self.proceed_button.setObjectName("proceed_to_refinement_button")
        self.proceed_button.setMinimumWidth(180)
        self.proceed_button.setEnabled(False)  # Disabled until neutral selected
        self.proceed_button.setToolTip(
            "Continue to refinement tab (requires Neutral variant selection)"
        )
        proceed_layout.addWidget(self.proceed_button)

        parent_layout.addLayout(proceed_layout)

    def _setup_connections(self):
        """Setup signal connections."""
        # Connect each emotion tab's signals
        for emotion_id, tab in self._emotion_tabs.items():
            # Connect generation request
            tab.generate_requested.connect(
                lambda e, i, p, em=emotion_id: self._on_generate_requested(e, i, p)
            )

            # Connect variant selection
            tab.variant_selected.connect(self._on_emotion_variant_selected)
            tab.variant_deselected.connect(self._on_emotion_variant_deselected)

        # Proceed button
        self.proceed_button.clicked.connect(self._on_proceed_clicked)

    def _setup_accessibility(self):
        """Configure accessibility properties."""
        self.tab_widget.setAccessibleName("Emotion Tabs")
        self.tab_widget.setAccessibleDescription(
            "Tabs for generating voice variants with different emotions"
        )

        self.proceed_button.setAccessibleName("Proceed to Refinement")
        self.proceed_button.setAccessibleDescription(
            "Continue to the refinement tab after selecting emotion variants"
        )

    def _on_generate_requested(self, emotion: str, instruction: str, preview_text: str):
        """Handle generation request from an emotion tab."""
        self.logger.debug(f"Generation requested for {emotion}")
        self.generation_requested.emit(emotion, instruction, preview_text)

    def _on_emotion_variant_selected(self, emotion: str, variant_index: int):
        """Handle variant selection from any emotion tab."""
        self.logger.debug(f"Variant {variant_index} selected for {emotion}")
        self.emotion_variant_selected.emit(emotion, variant_index)

        # Update proceed button if this is neutral
        if emotion == "neutral":
            self._update_proceed_button_state()
            self.neutral_selection_changed.emit(True)

    def _on_emotion_variant_deselected(self, emotion: str):
        """Handle variant deselection from any emotion tab."""
        self.logger.debug(f"Variant deselected for {emotion}")

        # Update proceed button if this is neutral
        if emotion == "neutral":
            self._update_proceed_button_state()
            self.neutral_selection_changed.emit(False)

    def _on_proceed_clicked(self):
        """Handle proceed button click."""
        self.logger.debug("Proceed to Refinement clicked")
        self.proceed_to_refinement_requested.emit()

    def _update_proceed_button_state(self):
        """Update proceed button based on neutral selection."""
        neutral_tab = self._emotion_tabs.get("neutral")
        if neutral_tab and neutral_tab.has_selection():
            self.proceed_button.setEnabled(True)
            self.proceed_status.setText("Neutral variant selected. Ready to proceed.")
            self.proceed_status.setStyleSheet("color: green;")
        else:
            self.proceed_button.setEnabled(False)
            self.proceed_status.setText("Select a Neutral variant to proceed.")
            self.proceed_status.setStyleSheet("")

    def _update_proceed_status(self):
        """Update the proceed status label with selection summary."""
        selected_emotions = self.get_selected_emotions()
        count = len(selected_emotions)

        if count == 0:
            self.proceed_status.setText("Select a Neutral variant to proceed.")
            self.proceed_status.setStyleSheet("")
        elif "neutral" not in selected_emotions:
            self.proceed_status.setText(
                f"{count} emotion(s) selected, but Neutral is required."
            )
            self.proceed_status.setStyleSheet("color: orange;")
        else:
            emotion_list = ", ".join(
                EMOTION_CONFIG[e]["name"] for e in selected_emotions
            )
            self.proceed_status.setText(f"Selected: {emotion_list}")
            self.proceed_status.setStyleSheet("color: green;")

    # Public API

    def set_voice_description(self, description: str):
        """
        Set the voice description for all emotion tabs.

        Args:
            description: Voice description text
        """
        self._voice_description = description
        for tab in self._emotion_tabs.values():
            tab.set_voice_description(description)

    def set_preview_text(self, text: str):
        """
        Set the preview text for all emotion tabs.

        Args:
            text: Preview text for TTS generation
        """
        self._preview_text = text
        for tab in self._emotion_tabs.values():
            tab.set_preview_text(text)

    def get_emotion_tab(self, emotion: str) -> Optional[EmotionTabWidget]:
        """
        Get the tab widget for a specific emotion.

        Args:
            emotion: Emotion ID (e.g., "neutral", "happy")

        Returns:
            EmotionTabWidget for the emotion, or None if not found
        """
        return self._emotion_tabs.get(emotion)

    def set_generating(self, emotion: str, is_generating: bool, message: str = ""):
        """
        Set generation state for an emotion tab.

        Args:
            emotion: Emotion ID
            is_generating: Whether generation is in progress
            message: Optional status message
        """
        tab = self._emotion_tabs.get(emotion)
        if tab:
            tab.set_generating(is_generating, message)

    def set_variant_generating(self, emotion: str, variant_index: int):
        """Set a specific variant to generating state."""
        tab = self._emotion_tabs.get(emotion)
        if tab:
            tab.set_variant_generating(variant_index)

    def set_variant_complete(
        self,
        emotion: str,
        variant_index: int,
        audio_path: Path,
        embedding_path: Path,
        duration: float = None
    ):
        """Set a specific variant to complete state."""
        tab = self._emotion_tabs.get(emotion)
        if tab:
            tab.set_variant_complete(variant_index, audio_path, embedding_path, duration)

    def set_variant_error(self, emotion: str, variant_index: int, error_message: str):
        """Set a specific variant to error state."""
        tab = self._emotion_tabs.get(emotion)
        if tab:
            tab.set_variant_error(variant_index, error_message)

    def finish_generation(self, emotion: str):
        """Mark generation as finished for an emotion."""
        tab = self._emotion_tabs.get(emotion)
        if tab:
            tab.finish_generation()

    def set_generation_error(self, emotion: str, error_message: str):
        """Set generation error for an emotion."""
        tab = self._emotion_tabs.get(emotion)
        if tab:
            tab.set_generation_error(error_message)

    def get_selected_emotions(self) -> List[str]:
        """
        Get list of emotions that have a selected variant.

        Returns:
            List of emotion IDs with selections
        """
        selected = []
        for emotion_id, tab in self._emotion_tabs.items():
            if tab.has_selection():
                selected.append(emotion_id)
        return selected

    def get_selected_variant_paths(self, emotion: str) -> Optional[tuple]:
        """
        Get the selected variant's paths for an emotion.

        Args:
            emotion: Emotion ID

        Returns:
            Tuple of (audio_path, embedding_path), or None
        """
        tab = self._emotion_tabs.get(emotion)
        if tab:
            return tab.get_selected_variant_paths()
        return None

    def has_neutral_selection(self) -> bool:
        """Check if neutral emotion has a variant selected."""
        neutral_tab = self._emotion_tabs.get("neutral")
        return neutral_tab.has_selection() if neutral_tab else False

    def has_any_selection(self) -> bool:
        """Check if any emotion has a variant selected."""
        return len(self.get_selected_emotions()) > 0

    def get_emotion_instruction(self, emotion: str) -> str:
        """
        Get the emotion instruction text for an emotion.

        Args:
            emotion: Emotion ID

        Returns:
            Instruction text, or empty string if not found
        """
        tab = self._emotion_tabs.get(emotion)
        return tab.get_instruction() if tab else ""

    def set_emotion_instruction(self, emotion: str, instruction: str):
        """
        Set the emotion instruction text for an emotion.

        Args:
            emotion: Emotion ID
            instruction: Instruction text
        """
        tab = self._emotion_tabs.get(emotion)
        if tab:
            tab.set_instruction(instruction)

    def select_emotion_tab(self, emotion: str):
        """
        Switch to a specific emotion tab.

        Args:
            emotion: Emotion ID to select
        """
        if emotion in EMOTION_CONFIG:
            self.tab_widget.setCurrentIndex(EMOTION_CONFIG[emotion]["index"])

    def clear(self):
        """Clear all emotion tabs and reset state."""
        for tab in self._emotion_tabs.values():
            tab.clear()
        self._update_proceed_button_state()
        self.logger.debug("EmotionsPanel cleared")

    def clear_emotion(self, emotion: str):
        """
        Clear a specific emotion's variants.

        Args:
            emotion: Emotion ID to clear
        """
        tab = self._emotion_tabs.get(emotion)
        if tab:
            tab.clear()
            if emotion == "neutral":
                self._update_proceed_button_state()

    # QA7: Methods for receiving transferred data from Description panel

    def load_neutral_variants(
        self,
        variants: List[tuple],
        selected_index: Optional[int] = None
    ):
        """
        QA7: Load pre-generated variants into the Neutral tab.

        Called when user clicks "Refine Voice" in Description panel to carry
        over all 5 generations to Emotions-Neutral tab.

        Args:
            variants: List of (audio_path, embedding_path, duration) tuples
            selected_index: Index of the variant that was selected (0-4), or None
        """
        neutral_tab = self._emotion_tabs.get("neutral")
        if not neutral_tab:
            self.logger.error("Neutral tab not found")
            return

        # Clear any existing variants
        neutral_tab.clear()

        # Load each variant into the neutral tab
        for idx, (audio_path, embedding_path, duration) in enumerate(variants):
            if idx >= 5:
                break  # Max 5 variants
            neutral_tab.set_variant_complete(idx, audio_path, embedding_path, duration)

        # Mark generation as finished
        neutral_tab.finish_generation()

        # Pre-select the variant that was selected in Description panel
        if selected_index is not None and 0 <= selected_index < len(variants):
            neutral_tab.variant_list.select_variant(selected_index)

        self._update_proceed_button_state()
        self.logger.info(f"Loaded {len(variants)} variants into Neutral tab, selected={selected_index}")

    def load_neutral_sample(
        self,
        audio_path: Path,
        embedding_path: Optional[Path] = None,
        duration: float = None
    ):
        """
        QA7: Load a single sample into the Neutral tab (from Sample-Clone workflow).

        Called when user clicks "Refinement" in Sample-Clone tab to transfer
        the uploaded sample to Emotions-Neutral tab.

        Args:
            audio_path: Path to the audio sample file
            embedding_path: Path to embedding file if already extracted (optional)
            duration: Audio duration in seconds (optional, will be detected if not provided)
        """
        neutral_tab = self._emotion_tabs.get("neutral")
        if not neutral_tab:
            self.logger.error("Neutral tab not found")
            return

        # Clear any existing variants
        neutral_tab.clear()

        # Get duration if not provided
        if duration is None:
            duration = neutral_tab._get_audio_duration(audio_path) if hasattr(neutral_tab, '_get_audio_duration') else 5.0

        # Load the sample as variant 0
        neutral_tab.set_variant_complete(
            variant_index=0,
            audio_path=audio_path,
            embedding_path=embedding_path,
            duration=duration
        )

        # Mark generation as finished
        neutral_tab.finish_generation()

        # Auto-select the sample
        neutral_tab.variant_list.select_variant(0)

        self._update_proceed_button_state()
        self.logger.info(f"Loaded sample into Neutral tab: {audio_path}")
