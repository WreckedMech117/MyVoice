"""
Description Path Panel

Panel for the "From Description" tab in Voice Design Studio.
Allows users to enter a voice description and preview text to generate voice variations.

Story 1.3: Voice Description Input
- FR6: User can enter a multi-line voice description
- FR11: User can enter preview text for voice generation

Story 1.4: Generate Single Voice Variation
- Clicking Generate shows progress indicator "Generating..."
- Generate button disables during generation, UI stays responsive
- Audio player widget appears with play button and duration on completion
- Play/stop functionality works for preview audio
- Errors display clear message with retry option

Story 1.7: Generate 5 Variations with Progressive Reveal
- Clicking "Generate 5 Variations" shows 5 rows immediately in "Waiting..." state
- Row 1 shows spinner/"Generating..." while rows 2-5 remain waiting
- Completed rows show play button and duration, next row starts generating
- Can play completed variants while others still generating
- All 5 complete shows "5 variations ready" status

Acceptance Criteria (1.3):
- "From Description" tab shows multi-line "Voice Description" text area (4-5 visible lines)
- Single-line "Preview Text" field visible
- "Generate" button initially disabled
- Generate button enables when both fields have content
- Button remains disabled if description is empty

Acceptance Criteria (1.4):
- Clicking Generate shows progress indicator "Generating..."
- Generate button disables during generation, UI stays responsive
- Audio player widget appears with play button and duration on completion
- Play/stop functionality works for preview audio
- Errors display clear message with retry option

Acceptance Criteria (1.7):
- Clicking "Generate 5 Variations" shows 5 rows in "Waiting..." state
- Progressive reveal as each variant completes
- Can play completed variants while others generate
- Status shows "5 variations ready" when all complete
"""

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTextEdit, QLineEdit, QPushButton, QGroupBox, QProgressBar,
    QComboBox, QTabWidget, QToolButton, QStyle, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QIcon

from myvoice.ui.dialogs.voice_design_studio.audio_player_widget import AudioPlayerWidget
from myvoice.ui.dialogs.voice_design_studio.voice_variant_list import VoiceVariantList
from myvoice.ui.dialogs.voice_design_studio.add_category_dialog import AddCategoryDialog
from myvoice.ui.dialogs.voice_design_studio.add_template_dialog import AddTemplateDialog
from myvoice.ui.dialogs.voice_design_studio.edit_template_dialog import EditTemplateDialog
from myvoice.services.category_service import CategoryService
from myvoice.services.template_service import TemplateService


class DescriptionPathPanel(QWidget):
    """
    Panel for creating voices from text descriptions.

    Provides input fields for voice description and preview text,
    with a Generate button that enables when both fields have content.
    Shows progress during generation and audio player on completion.
    After generation, shows voice name input for saving.

    Signals:
        generate_requested: Emitted when user clicks Generate (description, preview_text)
        content_changed: Emitted when any input field content changes
        retry_requested: Emitted when user clicks Retry after an error
        save_ready_changed: Emitted when save readiness changes (has_embedding and has_name)
    """

    generate_requested = pyqtSignal(str, str, str)  # description, preview_text, language
    content_changed = pyqtSignal()  # Emitted when inputs change
    retry_requested = pyqtSignal()  # Emitted when retry button clicked
    save_ready_changed = pyqtSignal(bool)  # Emitted when save readiness changes
    regenerate_requested = pyqtSignal()  # Emitted when regenerate button clicked (Story 1.9)
    variant_selected = pyqtSignal(int)  # Emitted when a variant is selected (Story 1.8)
    refine_requested = pyqtSignal()  # QA3: Emitted when user wants to refine voice in Sample tab

    # QA8: Clone sub-tab signals
    clone_file_loaded = pyqtSignal(str)  # audio_path - Emitted when audio file loaded in Clone tab
    clone_transcribe_requested = pyqtSignal(str)  # file_path - Emitted when Auto-Transcribe clicked
    clone_proceed_requested = pyqtSignal(str, str, str)  # audio_path, transcript, voice_name - To transfer to Emotions tab

    # Default preview text suggestion
    DEFAULT_PREVIEW_TEXT = "Hello, this is a preview of my voice."

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

    # Vocal Recipe formula guidance (Story 1.10)
    VOCAL_RECIPE_FORMULA = "[Age/Gender] + [Texture] + [Personality] + [Accent]"

    # Add Category option text (Story 3.1)
    ADD_CATEGORY_OPTION = "+ Add Category"

    # Add Template option text (Story 3.2)
    ADD_TEMPLATE_OPTION = "+ Add Template"

    # QA8: Clone tab audio format constants
    CLONE_SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a'}
    CLONE_FILE_FILTER = "Audio Files (*.wav *.mp3 *.m4a);;WAV Files (*.wav);;MP3 Files (*.mp3);;M4A Files (*.m4a);;All Files (*)"
    CLONE_MIN_DURATION_WARNING = 3.0  # Warn if < 3 seconds
    CLONE_MAX_DURATION_WARNING = 30.0  # Warn if > 30 seconds

    # Built-in voice templates organized by category (Story 1.10)
    VOICE_TEMPLATES = {
        "Narrators": {
            "Audiobook Narrator": "A mature male voice in his 50s with a rich, resonant baritone. "
                "Warm and inviting with excellent diction. Speaks with a measured pace, "
                "perfect for long-form storytelling. Neutral American accent.",
            "Documentary Narrator": "A distinguished female voice in her 40s with an authoritative "
                "yet approachable tone. Clear and articulate with subtle gravitas. "
                "British RP accent with modern inflections.",
            "Children's Storyteller": "A bright, expressive female voice in her 30s with playful "
                "energy and warmth. Animated and engaging with a gentle quality. "
                "Friendly American accent, perfect for young audiences.",
            "Meditation Guide": "A calm, soothing female voice in her 40s with a gentle, "
                "peaceful quality. Slow, deliberate pacing with a warm, nurturing tone. "
                "Soft-spoken with neutral accent.",
        },
        "Characters": {
            "Grumpy Craftsman": "A weathered male voice in his 60s with a gravelly, rough texture. "
                "Speaks in short, direct sentences with underlying warmth beneath gruff exterior. "
                "Working-class British accent with regional dialect hints.",
            "High-Energy Anime": "A youthful, energetic voice with exaggerated expressiveness. "
                "Bright and enthusiastic with rapid-fire delivery. Sharp articulation "
                "with dramatic emotional range. Japanese-influenced English.",
            "Wise Mentor": "A deep, thoughtful male voice in his 70s with quiet authority. "
                "Measured, contemplative speech with occasional humor. Slight rasp "
                "that conveys experience. Warm, grandfatherly quality.",
            "Villain Mastermind": "A smooth, sophisticated male voice in his 40s with menacing "
                "undertones. Calculated and precise with cold intelligence. "
                "Aristocratic British accent with silky, dangerous charm.",
        },
        "Regional": {
            "Southern Comfort": "A warm, friendly female voice in her 30s with authentic Southern "
                "American accent. Honey-sweet tone with natural hospitality. "
                "Relaxed pacing with charming colloquialisms.",
            "New York Professional": "A confident, sharp female voice in her 30s with distinct "
                "New York accent. Fast-paced and assertive with street-smart edge. "
                "Direct and no-nonsense delivery.",
            "Irish Storyteller": "A melodic male voice in his 50s with warm Irish brogue. "
                "Musical quality with natural rhythm. Friendly and engaging with "
                "a twinkle of mischief in the tone.",
            "Australian Casual": "An easygoing male voice in his 30s with relaxed Australian accent. "
                "Friendly and approachable with laid-back delivery. Natural, "
                "conversational style with occasional humor.",
        },
    }

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the Description Path Panel.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Category service for custom categories (Story 3.1)
        self._category_service = CategoryService()

        # Template service for custom templates (Story 3.2)
        self._template_service = TemplateService()

        # Generation state
        self._is_generating = False
        self._has_error = False
        self._generated_audio_path: Optional[Path] = None
        self._generated_embedding_path: Optional[Path] = None

        # Save state
        self._is_saving = False

        # QA8: Clone tab state
        self._clone_audio_path: Optional[Path] = None
        self._clone_audio_duration: float = 0.0
        self._clone_audio_format: str = ""
        self._clone_has_error = False
        self._clone_is_transcribing = False
        self._clone_transcription_error = False

        self._create_ui()
        self._setup_connections()
        self._setup_accessibility()

        self.logger.debug("DescriptionPathPanel initialized")

    def _create_ui(self):
        """Create the panel UI with Imagine/Create sub-tabs (QA3 fix)."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # QA3-2: Reduced spacing for smaller screens
        layout.setSpacing(4)

        # Create sub-tab widget for Imagine/Create workflow
        self.sub_tabs = QTabWidget()
        self.sub_tabs.setObjectName("description_sub_tabs")
        self.sub_tabs.setDocumentMode(True)  # Cleaner look for nested tabs

        # Create Imagine tab (input phase)
        self.imagine_tab = QWidget()
        self.imagine_tab.setObjectName("imagine_tab")
        imagine_layout = QVBoxLayout(self.imagine_tab)
        # QA3-2: Reduced margins and spacing for smaller screens
        imagine_layout.setContentsMargins(4, 4, 4, 4)
        imagine_layout.setSpacing(6)

        # Template selection section (Story 1.10)
        self._create_template_section(imagine_layout)

        # Voice Description section (reduced height)
        self._create_description_section(imagine_layout)

        # Preview Text section (increased height)
        self._create_preview_text_section(imagine_layout)

        # Generate button and progress section
        self._create_generate_section(imagine_layout)

        imagine_layout.addStretch()

        # Create "Create" tab (output phase)
        self.create_tab = QWidget()
        self.create_tab.setObjectName("create_tab")
        create_layout = QVBoxLayout(self.create_tab)
        # QA3-2: Reduced margins and spacing for smaller screens
        create_layout.setContentsMargins(4, 4, 4, 4)
        create_layout.setSpacing(6)

        # Result section (variant list and save options)
        self._create_result_section(create_layout)

        create_layout.addStretch()

        # QA8: Create Clone tab (voice cloning from audio sample)
        self.clone_tab = QWidget()
        self.clone_tab.setObjectName("clone_tab")
        clone_layout = QVBoxLayout(self.clone_tab)
        # QA3-2: Reduced margins and spacing for smaller screens
        clone_layout.setContentsMargins(4, 4, 4, 4)
        clone_layout.setSpacing(6)

        # Clone tab sections
        self._create_clone_upload_section(clone_layout)
        self._create_clone_info_section(clone_layout)
        self._create_clone_transcript_section(clone_layout)
        self._create_clone_proceed_section(clone_layout)

        clone_layout.addStretch()

        # Add tabs to sub-tab widget
        self.sub_tabs.addTab(self.imagine_tab, "Imagine")
        self.sub_tabs.addTab(self.create_tab, "Create")
        self.sub_tabs.addTab(self.clone_tab, "Clone")  # QA8: New Clone tab

        # Connect tab change to handle state
        self.sub_tabs.currentChanged.connect(self._on_sub_tab_changed)

        layout.addWidget(self.sub_tabs)

    def _create_template_section(self, layout: QVBoxLayout):
        """Create the template selection section (Story 1.10)."""
        # Group box for visual grouping
        template_group = QGroupBox("Voice Templates")
        template_group.setObjectName("template_group")
        template_layout = QVBoxLayout(template_group)
        template_layout.setSpacing(8)

        # Vocal Recipe formula guidance
        formula_label = QLabel(
            f"<b>Vocal Recipe:</b> {self.VOCAL_RECIPE_FORMULA}"
        )
        formula_label.setObjectName("vocal_recipe_formula")
        formula_label.setWordWrap(True)
        formula_label.setStyleSheet("color: #666; font-style: italic;")
        template_layout.addWidget(formula_label)

        # Dropdowns row
        dropdowns_layout = QHBoxLayout()
        dropdowns_layout.setSpacing(12)

        # Category dropdown
        category_layout = QVBoxLayout()
        category_label = QLabel("Category:")
        category_label.setObjectName("category_label")
        category_layout.addWidget(category_label)

        self.category_combo = QComboBox()
        self.category_combo.setObjectName("category_combo")
        self._populate_category_dropdown()
        self.category_combo.setMinimumWidth(150)
        category_layout.addWidget(self.category_combo)
        dropdowns_layout.addLayout(category_layout)

        # Template dropdown
        template_dd_layout = QVBoxLayout()
        template_label = QLabel("Template:")
        template_label.setObjectName("template_label")
        template_dd_layout.addWidget(template_label)

        # Template combo with edit/delete buttons (Story 3.3)
        template_row_layout = QHBoxLayout()
        template_row_layout.setSpacing(4)

        self.template_combo = QComboBox()
        self.template_combo.setObjectName("template_combo")
        self.template_combo.addItem("-- Select Template --")
        self.template_combo.setEnabled(False)  # Disabled until category selected
        self.template_combo.setMinimumWidth(200)
        template_row_layout.addWidget(self.template_combo)

        # Edit button (Story 3.3, QA3: Icon button)
        self.edit_template_button = QToolButton()
        self.edit_template_button.setObjectName("edit_template_button")
        edit_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView)
        self.edit_template_button.setIcon(edit_icon)
        self.edit_template_button.setFixedSize(28, 28)
        self.edit_template_button.setEnabled(False)
        self.edit_template_button.setToolTip("Edit Template")
        self.edit_template_button.setAccessibleName("Edit Template")
        self.edit_template_button.setAccessibleDescription("Edit the selected custom template")
        template_row_layout.addWidget(self.edit_template_button)

        # Delete button (Story 3.3, QA3: Icon button)
        self.delete_template_button = QToolButton()
        self.delete_template_button.setObjectName("delete_template_button")
        delete_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon)
        self.delete_template_button.setIcon(delete_icon)
        self.delete_template_button.setFixedSize(28, 28)
        self.delete_template_button.setEnabled(False)
        self.delete_template_button.setToolTip("Delete Template")
        self.delete_template_button.setAccessibleName("Delete Template")
        self.delete_template_button.setAccessibleDescription("Delete the selected custom template")
        template_row_layout.addWidget(self.delete_template_button)

        template_dd_layout.addLayout(template_row_layout)
        dropdowns_layout.addLayout(template_dd_layout)

        dropdowns_layout.addStretch()
        template_layout.addLayout(dropdowns_layout)

        layout.addWidget(template_group)

    def _create_description_section(self, layout: QVBoxLayout):
        """Create the voice description input section."""
        # Group box for visual grouping
        desc_group = QGroupBox("Voice Description")
        desc_layout = QVBoxLayout(desc_group)
        desc_layout.setSpacing(8)

        # Instruction label
        instruction_label = QLabel(
            "Describe the voice you want to create. Include details like "
            "age, gender, accent, and vocal qualities."
        )
        instruction_label.setWordWrap(True)
        instruction_label.setObjectName("description_instruction")
        desc_layout.addWidget(instruction_label)

        # Multi-line text area (QA3: reduced to ~2-3 lines ≈ 60px height)
        self.description_edit = QTextEdit()
        self.description_edit.setObjectName("voice_description_edit")
        self.description_edit.setPlaceholderText(
            "Example: A warm, friendly female voice in her 30s with a slight "
            "British accent. Speaks clearly with a gentle, reassuring tone."
        )
        self.description_edit.setMinimumHeight(60)  # QA3: Reduced from 100px
        self.description_edit.setMaximumHeight(80)  # QA3: Reduced from 150px
        desc_layout.addWidget(self.description_edit)

        layout.addWidget(desc_group)

    def _create_preview_text_section(self, layout: QVBoxLayout):
        """Create the preview text input section with language selection."""
        # Group box for visual grouping
        preview_group = QGroupBox("Preview Text")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setSpacing(8)

        # Instruction label
        preview_instruction = QLabel(
            "Enter the text the generated voice will speak for preview."
        )
        preview_instruction.setWordWrap(True)
        preview_instruction.setObjectName("preview_instruction")
        preview_layout.addWidget(preview_instruction)

        # Multi-line text input (QA3: doubled height for better visibility)
        self.preview_text_edit = QTextEdit()
        self.preview_text_edit.setObjectName("preview_text_edit")
        self.preview_text_edit.setPlaceholderText(self.DEFAULT_PREVIEW_TEXT)
        self.preview_text_edit.setMinimumHeight(50)  # QA3: Doubled from single line
        self.preview_text_edit.setMaximumHeight(70)  # QA3: Allow slight growth
        preview_layout.addWidget(self.preview_text_edit)

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
        preview_layout.addLayout(lang_layout)

        layout.addWidget(preview_group)

    def _create_generate_section(self, layout: QVBoxLayout):
        """Create the generate button and progress section."""
        generate_layout = QHBoxLayout()
        generate_layout.setSpacing(12)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("generate_progress_bar")
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumWidth(150)
        self.progress_bar.setMaximumHeight(16)
        generate_layout.addWidget(self.progress_bar)

        # Progress label (hidden by default)
        self.progress_label = QLabel("Generating...")
        self.progress_label.setObjectName("generate_progress_label")
        self.progress_label.setVisible(False)
        generate_layout.addWidget(self.progress_label)

        generate_layout.addStretch()

        # Generate button (Story 1.7: renamed to "Generate 5 Variations")
        self.generate_button = QPushButton("Generate 5 Variations")
        self.generate_button.setObjectName("generate_button")
        self.generate_button.setEnabled(False)  # Disabled until both fields have content
        self.generate_button.setMinimumWidth(150)
        generate_layout.addWidget(self.generate_button)

        layout.addLayout(generate_layout)

    def _create_result_section(self, layout: QVBoxLayout):
        """Create the result section with variant list and save options."""
        # Result group (hidden until generation starts - Story 1.7)
        self.result_group = QGroupBox("Voice Variations")
        self.result_group.setObjectName("result_group")
        self.result_group.setVisible(False)
        result_layout = QVBoxLayout(self.result_group)
        result_layout.setSpacing(8)

        # Voice Variant List widget (Story 1.7: Progressive Reveal)
        self.variant_list = VoiceVariantList()
        result_layout.addWidget(self.variant_list)

        # Legacy audio player (hidden, kept for backward compatibility)
        self.audio_player = AudioPlayerWidget()
        self.audio_player.setVisible(False)
        result_layout.addWidget(self.audio_player)

        # Voice Name input (for saving)
        self._create_save_section(result_layout)

        layout.addWidget(self.result_group)

    def _create_save_section(self, layout: QVBoxLayout):
        """Create the save section with voice name input."""
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

        # QA3: Refine Voice button - transfers to Sample tab for voice cloning
        self.refine_button = QPushButton("Refine Voice")
        self.refine_button.setObjectName("refine_button")
        self.refine_button.setToolTip(
            "Transfer this voice to the Sample tab for refinement using voice cloning"
        )
        self.refine_button.setEnabled(False)  # Enabled when variant is selected
        self.refine_button.setMinimumWidth(100)
        name_layout.addWidget(self.refine_button)

        layout.addLayout(name_layout)

        # Status/error section
        status_layout = QHBoxLayout()
        status_layout.setSpacing(8)

        self.status_label = QLabel("")
        self.status_label.setObjectName("status_label")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label, 1)

        # Retry button (hidden by default)
        self.retry_button = QPushButton("Retry")
        self.retry_button.setObjectName("retry_button")
        self.retry_button.setVisible(False)
        self.retry_button.setMinimumWidth(80)
        status_layout.addWidget(self.retry_button)

        layout.addLayout(status_layout)

    # =========================================================================
    # QA8: Clone Tab UI Sections
    # =========================================================================

    def _create_clone_upload_section(self, layout: QVBoxLayout):
        """Create the audio upload section for Clone tab (QA8)."""
        upload_group = QGroupBox("Audio Sample")
        upload_group.setObjectName("clone_upload_group")
        upload_layout = QVBoxLayout(upload_group)
        upload_layout.setSpacing(8)

        # Instruction label (compact)
        instruction_label = QLabel(
            "Upload audio (WAV/MP3/M4A, 3-30 sec) to clone a voice."
        )
        instruction_label.setObjectName("clone_upload_instruction")
        upload_layout.addWidget(instruction_label)

        # Browse button row
        browse_layout = QHBoxLayout()
        browse_layout.setSpacing(8)

        self.clone_browse_button = QPushButton("Browse...")
        self.clone_browse_button.setObjectName("clone_browse_button")
        self.clone_browse_button.setMinimumWidth(100)
        browse_layout.addWidget(self.clone_browse_button)

        # Selected file label
        self.clone_file_label = QLabel("No file selected")
        self.clone_file_label.setObjectName("clone_selected_file_label")
        self.clone_file_label.setStyleSheet("color: #666; font-style: italic;")
        browse_layout.addWidget(self.clone_file_label, 1)

        upload_layout.addLayout(browse_layout)

        layout.addWidget(upload_group)

    def _create_clone_info_section(self, layout: QVBoxLayout):
        """Create file info section for Clone tab (QA8) - compact layout."""
        self.clone_info_group = QGroupBox("Sample Info")
        self.clone_info_group.setObjectName("clone_file_info_group")
        self.clone_info_group.setVisible(False)  # Hidden until file loaded
        info_layout = QVBoxLayout(self.clone_info_group)
        info_layout.setSpacing(4)
        info_layout.setContentsMargins(8, 8, 8, 8)

        # Combined row: Duration | Format | Player
        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        # Duration (inline)
        self.clone_duration_value = QLabel("0:00")
        self.clone_duration_value.setObjectName("clone_duration_value")
        top_row.addWidget(QLabel("Duration:"))
        top_row.addWidget(self.clone_duration_value)

        # Format (inline)
        self.clone_format_value = QLabel("WAV")
        self.clone_format_value.setObjectName("clone_format_value")
        top_row.addWidget(QLabel("Format:"))
        top_row.addWidget(self.clone_format_value)

        top_row.addStretch()

        # Audio player (compact)
        self.clone_audio_player = AudioPlayerWidget()
        self.clone_audio_player.setObjectName("clone_audio_player")
        top_row.addWidget(self.clone_audio_player)

        info_layout.addLayout(top_row)

        # Warning label (hidden by default)
        self.clone_warning_label = QLabel("")
        self.clone_warning_label.setObjectName("clone_warning_label")
        self.clone_warning_label.setWordWrap(True)
        self.clone_warning_label.setVisible(False)
        info_layout.addWidget(self.clone_warning_label)

        # Voice name input (same row)
        name_layout = QHBoxLayout()
        name_layout.setSpacing(8)
        name_label = QLabel("Voice Name:")
        name_label.setObjectName("clone_voice_name_label")
        name_layout.addWidget(name_label)

        self.clone_voice_name_edit = QLineEdit()
        self.clone_voice_name_edit.setObjectName("clone_voice_name_edit")
        self.clone_voice_name_edit.setPlaceholderText("Enter a name for your voice")
        name_layout.addWidget(self.clone_voice_name_edit, 1)

        info_layout.addLayout(name_layout)

        layout.addWidget(self.clone_info_group)

    def _create_clone_transcript_section(self, layout: QVBoxLayout):
        """Create transcript section for Clone tab (QA8) - compact layout."""
        self.clone_transcript_group = QGroupBox("Transcript")
        self.clone_transcript_group.setObjectName("clone_transcript_group")
        self.clone_transcript_group.setVisible(False)  # Hidden until file loaded
        transcript_layout = QVBoxLayout(self.clone_transcript_group)
        transcript_layout.setSpacing(4)
        transcript_layout.setContentsMargins(8, 8, 8, 8)

        # Transcript text area (reduced height)
        self.clone_transcript_edit = QTextEdit()
        self.clone_transcript_edit.setObjectName("clone_transcript_edit")
        self.clone_transcript_edit.setPlaceholderText(
            "Enter transcript or click Auto-Transcribe..."
        )
        self.clone_transcript_edit.setMinimumHeight(50)
        self.clone_transcript_edit.setMaximumHeight(80)
        transcript_layout.addWidget(self.clone_transcript_edit)

        # Transcribe button row
        transcribe_layout = QHBoxLayout()
        transcribe_layout.setSpacing(8)

        # Progress bar (hidden by default)
        self.clone_transcribe_progress = QProgressBar()
        self.clone_transcribe_progress.setObjectName("clone_transcribe_progress")
        self.clone_transcribe_progress.setRange(0, 0)  # Indeterminate
        self.clone_transcribe_progress.setVisible(False)
        self.clone_transcribe_progress.setMaximumWidth(150)
        self.clone_transcribe_progress.setMaximumHeight(16)
        transcribe_layout.addWidget(self.clone_transcribe_progress)

        # Status label
        self.clone_transcribe_status = QLabel("")
        self.clone_transcribe_status.setObjectName("clone_transcribe_status")
        transcribe_layout.addWidget(self.clone_transcribe_status, 1)

        # Retry button (hidden by default)
        self.clone_transcribe_retry_button = QPushButton("Retry")
        self.clone_transcribe_retry_button.setObjectName("clone_transcribe_retry_button")
        self.clone_transcribe_retry_button.setVisible(False)
        self.clone_transcribe_retry_button.setMinimumWidth(80)
        transcribe_layout.addWidget(self.clone_transcribe_retry_button)

        # Auto-Transcribe button
        self.clone_transcribe_button = QPushButton("Auto-Transcribe")
        self.clone_transcribe_button.setObjectName("clone_transcribe_button")
        self.clone_transcribe_button.setMinimumWidth(140)
        transcribe_layout.addWidget(self.clone_transcribe_button)

        transcript_layout.addLayout(transcribe_layout)

        layout.addWidget(self.clone_transcript_group)

    def _create_clone_proceed_section(self, layout: QVBoxLayout):
        """Create proceed button section for Clone tab (QA8) - compact layout."""
        # No group box - just a button row for compactness
        self.clone_proceed_group = QWidget()
        self.clone_proceed_group.setObjectName("clone_proceed_group")
        self.clone_proceed_group.setVisible(False)  # Hidden until file loaded
        proceed_layout = QHBoxLayout(self.clone_proceed_group)
        proceed_layout.setContentsMargins(0, 8, 0, 0)

        proceed_layout.addStretch()

        self.clone_proceed_button = QPushButton("Proceed to Emotions →")
        self.clone_proceed_button.setObjectName("clone_proceed_button")
        self.clone_proceed_button.setMinimumWidth(160)
        self.clone_proceed_button.setToolTip(
            "Transfer sample to Emotions tab for building voice with emotion variants"
        )
        proceed_layout.addWidget(self.clone_proceed_button)

        layout.addWidget(self.clone_proceed_group)

    def _setup_connections(self):
        """Setup signal connections for input validation."""
        # Connect text changes to validation
        self.description_edit.textChanged.connect(self._on_content_changed)
        self.preview_text_edit.textChanged.connect(self._on_content_changed)

        # Connect generate button
        self.generate_button.clicked.connect(self._on_generate_clicked)

        # Connect retry button
        self.retry_button.clicked.connect(self._on_retry_clicked)

        # Connect voice name changes to save readiness
        self.voice_name_edit.textChanged.connect(self._on_voice_name_changed)

        # Connect variant list signals (Story 1.7, 1.8)
        self.variant_list.variant_selected.connect(self._on_variant_selected)
        self.variant_list.variant_deselected.connect(self._on_variant_deselected)
        self.variant_list.all_complete.connect(self._on_all_variants_complete)
        self.variant_list.regenerate_button.clicked.connect(self._on_regenerate_clicked)

        # Connect template selection (Story 1.10)
        self.category_combo.currentIndexChanged.connect(self._on_category_changed)
        self.template_combo.currentIndexChanged.connect(self._on_template_changed)

        # Connect edit/delete buttons (Story 3.3)
        self.edit_template_button.clicked.connect(self._on_edit_template_clicked)
        self.delete_template_button.clicked.connect(self._on_delete_template_clicked)

        # Connect refine button (QA3)
        self.refine_button.clicked.connect(self._on_refine_clicked)

        # QA8: Clone tab connections
        self.clone_browse_button.clicked.connect(self._on_clone_browse_clicked)
        self.clone_transcribe_button.clicked.connect(self._on_clone_transcribe_clicked)
        self.clone_transcribe_retry_button.clicked.connect(self._on_clone_transcribe_clicked)
        self.clone_proceed_button.clicked.connect(self._on_clone_proceed_clicked)

    def _setup_accessibility(self):
        """Configure accessibility properties."""
        # Category combo (Story 1.10)
        self.category_combo.setAccessibleName("Voice Category")
        self.category_combo.setAccessibleDescription(
            "Select a voice template category"
        )

        # Template combo (Story 1.10)
        self.template_combo.setAccessibleName("Voice Template")
        self.template_combo.setAccessibleDescription(
            "Select a voice template to populate the description"
        )

        # Description edit
        self.description_edit.setAccessibleName("Voice Description")
        self.description_edit.setAccessibleDescription(
            "Enter a description of the voice you want to create"
        )

        # Preview text edit
        self.preview_text_edit.setAccessibleName("Preview Text")
        self.preview_text_edit.setAccessibleDescription(
            "Enter the text that will be spoken in the voice preview"
        )

        # Language combo
        self.language_combo.setAccessibleName("Language")
        self.language_combo.setAccessibleDescription(
            "Select the language for voice generation"
        )

        # Generate button
        self.generate_button.setAccessibleName("Generate")
        self.generate_button.setAccessibleDescription(
            "Generate voice preview from description"
        )

        # Retry button
        self.retry_button.setAccessibleName("Retry")
        self.retry_button.setAccessibleDescription(
            "Retry the failed generation"
        )

        # Voice name edit
        self.voice_name_edit.setAccessibleName("Voice Name")
        self.voice_name_edit.setAccessibleDescription(
            "Enter a name to save your generated voice"
        )

        # Refine button (QA3)
        self.refine_button.setAccessibleName("Refine Voice")
        self.refine_button.setAccessibleDescription(
            "Transfer this voice to the Emotions tab for building with emotion variants"
        )

        # QA8: Clone tab accessibility
        self.clone_browse_button.setAccessibleName("Browse for audio file")
        self.clone_browse_button.setAccessibleDescription(
            "Open file dialog to select an audio file for voice cloning"
        )
        self.clone_voice_name_edit.setAccessibleName("Voice Name")
        self.clone_voice_name_edit.setAccessibleDescription(
            "Enter a name for your cloned voice"
        )
        self.clone_transcript_edit.setAccessibleName("Transcript")
        self.clone_transcript_edit.setAccessibleDescription(
            "Enter the text spoken in the audio sample"
        )
        self.clone_transcribe_button.setAccessibleName("Auto-Transcribe")
        self.clone_transcribe_button.setAccessibleDescription(
            "Automatically transcribe the audio using speech recognition"
        )
        self.clone_proceed_button.setAccessibleName("Proceed to Emotions")
        self.clone_proceed_button.setAccessibleDescription(
            "Transfer sample to Emotions tab for building voice with emotion variants"
        )

    def _on_content_changed(self):
        """Handle content changes in input fields."""
        # Update generate button state (only if not generating)
        if not self._is_generating:
            self._update_generate_button_state()

        # Clear previous results when content changes
        if self._generated_audio_path:
            self._clear_result()

        # Emit content changed signal
        self.content_changed.emit()

    def _on_category_changed(self, index: int):
        """Handle category dropdown selection change (Story 1.10, 3.1, 3.2)."""
        if index <= 0:
            # "-- Select Category --" selected, disable template dropdown
            self.template_combo.clear()
            self.template_combo.addItem("-- Select Template --")
            self.template_combo.setEnabled(False)
            self.logger.debug("Category cleared")
            return

        # Get selected category name
        category = self.category_combo.currentText()

        # Story 3.1: Handle "+ Add Category" selection
        if category == self.ADD_CATEGORY_OPTION:
            self._on_add_category_selected()
            return

        # Populate template dropdown with templates for this category
        self._populate_template_dropdown(category)

        self.logger.debug(f"Category changed to: {category}")

    def _populate_template_dropdown(self, category: str):
        """
        Populate the template dropdown for a given category.

        Story 3.2: Includes built-in templates, custom templates, and "+ Add Template" option.

        Args:
            category: Category name to load templates for
        """
        self.template_combo.clear()
        self.template_combo.addItem("-- Select Template --")

        has_templates = False

        # Add built-in templates for this category
        if category in self.VOICE_TEMPLATES:
            for template_name in self.VOICE_TEMPLATES[category].keys():
                self.template_combo.addItem(template_name)
            has_templates = True

        # Add custom templates for this category (Story 3.2)
        custom_templates = self._template_service.get_templates_for_category(category)
        for template_name in custom_templates.keys():
            self.template_combo.addItem(template_name)
            has_templates = True

        # Add "+ Add Template" option at the end (Story 3.2)
        self.template_combo.addItem(self.ADD_TEMPLATE_OPTION)

        self.template_combo.setEnabled(True)

    def _on_template_changed(self, index: int):
        """Handle template dropdown selection change (Story 1.10, 3.2, 3.3)."""
        # Story 3.3: Disable edit/delete buttons by default
        self.edit_template_button.setEnabled(False)
        self.delete_template_button.setEnabled(False)

        if index <= 0:
            # "-- Select Template --" selected, don't change description
            return

        # Get selected category and template
        category = self.category_combo.currentText()
        template_name = self.template_combo.currentText()

        # Story 3.2: Handle "+ Add Template" selection
        if template_name == self.ADD_TEMPLATE_OPTION:
            self._on_add_template_selected()
            return

        # Get template description and populate the field
        template_description = None
        is_custom = False

        # Check built-in templates first
        if category in self.VOICE_TEMPLATES:
            templates = self.VOICE_TEMPLATES[category]
            if template_name in templates:
                template_description = templates[template_name]

        # Check custom templates (Story 3.2)
        if template_description is None:
            custom_templates = self._template_service.get_templates_for_category(category)
            if template_name in custom_templates:
                template_description = custom_templates[template_name]
                is_custom = True

        if template_description:
            self.description_edit.setPlainText(template_description)
            self.logger.debug(f"Applied template: {category}/{template_name}")

        # Story 3.3: Enable edit/delete buttons for custom templates only
        if is_custom:
            self.edit_template_button.setEnabled(True)
            self.delete_template_button.setEnabled(True)

    def _populate_category_dropdown(self):
        """
        Populate the category dropdown with built-in and custom categories.

        Story 3.1: Includes built-in categories, custom categories, and "+ Add Category" option.
        """
        self.category_combo.clear()
        self.category_combo.addItem("-- Select Category --")

        # Add built-in categories
        for category in self.VOICE_TEMPLATES.keys():
            self.category_combo.addItem(category)

        # Add custom categories (Story 3.1)
        custom_categories = self._category_service.get_custom_categories()
        for category in custom_categories:
            self.category_combo.addItem(category)

        # Add "+ Add Category" option at the end (Story 3.1)
        self.category_combo.addItem(self.ADD_CATEGORY_OPTION)

        self.logger.debug(
            f"Populated categories: {len(self.VOICE_TEMPLATES)} built-in, "
            f"{len(custom_categories)} custom"
        )

    def _on_add_category_selected(self):
        """
        Handle "+ Add Category" selection from dropdown.

        Story 3.1: Shows dialog to create new category.
        """
        # Reset dropdown to placeholder while dialog is open
        self.category_combo.blockSignals(True)
        self.category_combo.setCurrentIndex(0)
        self.category_combo.blockSignals(False)

        # Disable template dropdown
        self.template_combo.clear()
        self.template_combo.addItem("-- Select Template --")
        self.template_combo.setEnabled(False)

        # Get all existing category names for validation
        all_categories = self._get_all_category_names()

        # Show add category dialog
        new_name = AddCategoryDialog.get_new_category(self, all_categories)

        if new_name:
            # Add category via service
            builtin_categories = list(self.VOICE_TEMPLATES.keys())
            success, error = self._category_service.add_category(new_name, builtin_categories)

            if success:
                self.logger.info(f"Created custom category: {new_name}")
                # Refresh dropdown and select new category
                self._populate_category_dropdown()
                # Find and select the new category
                index = self.category_combo.findText(new_name)
                if index >= 0:
                    self.category_combo.setCurrentIndex(index)
            else:
                self.logger.error(f"Failed to create category: {error}")
                # Show error to user
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Cannot Create Category",
                    error
                )

    def _get_all_category_names(self) -> list[str]:
        """
        Get all category names (built-in + custom).

        Returns:
            List of all category names
        """
        names = list(self.VOICE_TEMPLATES.keys())
        names.extend(self._category_service.get_custom_categories())
        return names

    def _on_add_template_selected(self):
        """
        Handle "+ Add Template" selection from dropdown.

        Story 3.2: Shows dialog to create new template with current description pre-populated.
        """
        # Get current category
        category = self.category_combo.currentText()

        # Reset template dropdown to placeholder while dialog is open
        self.template_combo.blockSignals(True)
        self.template_combo.setCurrentIndex(0)
        self.template_combo.blockSignals(False)

        # Get current description to pre-populate
        current_description = self.get_description()

        # Get existing template names in this category
        existing_templates = self._get_template_names_for_category(category)

        # Show add template dialog
        name, description = AddTemplateDialog.get_new_template(
            self,
            category=category,
            current_description=current_description,
            existing_templates=existing_templates
        )

        if name and description:
            # Add template via service
            success, error = self._template_service.add_template(
                category, name, description, self.VOICE_TEMPLATES
            )

            if success:
                self.logger.info(f"Created custom template: {category}/{name}")
                # Refresh template dropdown and select the new template
                self._populate_template_dropdown(category)
                # Find and select the new template
                index = self.template_combo.findText(name)
                if index >= 0:
                    self.template_combo.setCurrentIndex(index)
            else:
                self.logger.error(f"Failed to create template: {error}")
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Cannot Create Template",
                    error
                )

    def _get_template_names_for_category(self, category: str) -> list[str]:
        """
        Get all template names for a category (built-in + custom).

        Args:
            category: Category name

        Returns:
            List of template names
        """
        names = []

        # Built-in templates
        if category in self.VOICE_TEMPLATES:
            names.extend(self.VOICE_TEMPLATES[category].keys())

        # Custom templates
        custom_templates = self._template_service.get_templates_for_category(category)
        names.extend(custom_templates.keys())

        return names

    def _on_edit_template_clicked(self):
        """
        Handle Edit button click for custom template.

        Story 3.3: Shows edit dialog for the selected custom template.
        """
        category = self.category_combo.currentText()
        template_name = self.template_combo.currentText()

        # Get current description from service
        custom_templates = self._template_service.get_templates_for_category(category)
        if template_name not in custom_templates:
            self.logger.warning(f"Template not found for edit: {category}/{template_name}")
            return

        current_description = custom_templates[template_name]

        # Get existing template names (excluding current)
        existing_templates = self._get_template_names_for_category(category)

        # Show edit dialog
        new_name, new_description = EditTemplateDialog.edit_template(
            self,
            category=category,
            template_name=template_name,
            template_description=current_description,
            existing_templates=existing_templates
        )

        if new_name and new_description:
            # Update template via service
            success, error = self._template_service.update_template(
                category, template_name, new_name, new_description, self.VOICE_TEMPLATES
            )

            if success:
                self.logger.info(f"Updated template: {category}/{template_name} -> {new_name}")
                # Refresh dropdown and select updated template
                self._populate_template_dropdown(category)
                index = self.template_combo.findText(new_name)
                if index >= 0:
                    self.template_combo.setCurrentIndex(index)
                # Update description field if name changed
                self.description_edit.setPlainText(new_description)
            else:
                self.logger.error(f"Failed to update template: {error}")
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Cannot Update Template",
                    error
                )

    def _on_delete_template_clicked(self):
        """
        Handle Delete button click for custom template.

        Story 3.3: Shows confirmation dialog and deletes the template.
        """
        from PyQt6.QtWidgets import QMessageBox

        category = self.category_combo.currentText()
        template_name = self.template_combo.currentText()

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Delete Template",
            f"Are you sure you want to delete the template '{template_name}'?\n\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Delete template via service
        success, error = self._template_service.remove_template(category, template_name)

        if success:
            self.logger.info(f"Deleted template: {category}/{template_name}")
            # Refresh dropdown and reset to placeholder
            self._populate_template_dropdown(category)
            self.template_combo.setCurrentIndex(0)
            # Disable edit/delete buttons
            self.edit_template_button.setEnabled(False)
            self.delete_template_button.setEnabled(False)
        else:
            self.logger.error(f"Failed to delete template: {error}")
            QMessageBox.warning(
                self,
                "Cannot Delete Template",
                error
            )

    def _update_generate_button_state(self):
        """Update the Generate button enabled state based on input content."""
        description = self.get_description()
        preview_text = self.get_preview_text()

        # Enable only if both fields have content and not currently generating
        can_generate = bool(description) and bool(preview_text) and not self._is_generating
        self.generate_button.setEnabled(can_generate)

    def _on_generate_clicked(self):
        """Handle Generate button click."""
        description = self.get_description()
        preview_text = self.get_preview_text()
        language = self.get_language()

        self.logger.debug(f"Generate clicked: description={len(description)} chars, preview={len(preview_text)} chars, language={language}")
        self.generate_requested.emit(description, preview_text, language)

    def _on_retry_clicked(self):
        """Handle Retry button click."""
        self.logger.debug("Retry clicked")
        self._clear_error()
        self.retry_requested.emit()

    def _on_voice_name_changed(self):
        """Handle voice name field changes."""
        was_ready = self._is_save_ready_internal()
        # Emit signal if save readiness changed
        is_ready = self._is_save_ready_internal()
        if was_ready != is_ready:
            self.save_ready_changed.emit(is_ready)
        # Always emit on change to allow external validation
        self.save_ready_changed.emit(is_ready)

    def _on_variant_selected(self, variant_index: int):
        """Handle variant selection from the variant list (Story 1.8)."""
        self.logger.debug(f"Variant {variant_index} selected")
        # Get the selected variant's embedding path
        row = self.variant_list.get_variant_row(variant_index)
        if row and row.is_complete():
            self._generated_embedding_path = row.get_embedding_path()
            self._generated_audio_path = row.get_audio_path()
            self.save_ready_changed.emit(self._is_save_ready_internal())
            # QA3: Enable refine button when variant is selected
            self.refine_button.setEnabled(True)
        self.variant_selected.emit(variant_index)

    def _on_variant_deselected(self):
        """Handle variant deselection from the variant list."""
        self.logger.debug("Variant deselected")
        # Clear the selected paths when no variant is selected
        # Don't clear completely - keep the paths for the previously selected variant
        self.save_ready_changed.emit(self._is_save_ready_internal())
        # QA3: Disable refine button when no variant selected
        self.refine_button.setEnabled(False)

    def _on_refine_clicked(self):
        """Handle Refine Voice button click (QA3)."""
        self.logger.debug("Refine Voice clicked")
        self.refine_requested.emit()

    # =========================================================================
    # QA8: Clone Tab Handlers
    # =========================================================================

    def _on_clone_browse_clicked(self):
        """Handle Browse button click in Clone tab (QA8)."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            self.CLONE_FILE_FILTER
        )

        if file_path:
            self._load_clone_audio_file(Path(file_path))

    def _load_clone_audio_file(self, file_path: Path):
        """Load and validate an audio file for cloning (QA8)."""
        self.logger.debug(f"Loading clone audio file: {file_path}")

        # Check if file exists
        if not file_path.exists():
            self._show_clone_error("File not found")
            return

        # Check format
        suffix = file_path.suffix.lower()
        if suffix not in self.CLONE_SUPPORTED_FORMATS:
            self._show_clone_error("Unsupported format. Please use WAV, MP3, or M4A")
            return

        # Clear previous error
        self._clone_has_error = False

        # Store file path
        self._clone_audio_path = file_path

        # Get audio info
        self._clone_audio_duration = self._get_clone_audio_duration(file_path)
        self._clone_audio_format = self._get_clone_audio_format(file_path)

        # Update UI
        self.clone_file_label.setText(file_path.name)
        self.clone_file_label.setStyleSheet("")  # Remove italic style

        self.clone_duration_value.setText(self._format_clone_duration(self._clone_audio_duration))
        self.clone_format_value.setText(self._clone_audio_format)

        # Set audio player
        self.clone_audio_player.set_audio_file(file_path)

        # Show file info, transcript, and proceed sections
        self.clone_info_group.setVisible(True)
        self.clone_transcript_group.setVisible(True)
        self.clone_proceed_group.setVisible(True)

        # Check duration warnings
        self._check_clone_duration_warnings()

        # Emit signal
        self.clone_file_loaded.emit(str(file_path))

        self.logger.info(f"Clone audio file loaded: {file_path.name}, {self._clone_audio_duration:.1f}s")

    def _get_clone_audio_duration(self, file_path: Path) -> float:
        """Get the duration of an audio file for cloning (QA8)."""
        suffix = file_path.suffix.lower()

        # For WAV files, use Python's built-in wave module
        if suffix == '.wav':
            try:
                import wave
                with wave.open(str(file_path), 'rb') as w:
                    frames = w.getnframes()
                    rate = w.getframerate()
                    if rate > 0:
                        return frames / float(rate)
            except Exception as e:
                self.logger.debug(f"Could not get WAV duration: {e}")

        # For other formats, try mutagen
        try:
            from mutagen import File as MutagenFile
            audio = MutagenFile(str(file_path))
            if audio and audio.info:
                return audio.info.length
        except Exception as e:
            self.logger.debug(f"Could not get duration with mutagen: {e}")

        # Fallback estimate
        return 5.0

    def _get_clone_audio_format(self, file_path: Path) -> str:
        """Get format information for an audio file (QA8)."""
        suffix = file_path.suffix.lower()
        format_name = suffix[1:].upper()

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

    def _format_clone_duration(self, seconds: float) -> str:
        """Format duration as M:SS (QA8)."""
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes}:{secs:02d}"

    def _check_clone_duration_warnings(self):
        """Check duration and show appropriate warnings (QA8)."""
        if self._clone_audio_duration < self.CLONE_MIN_DURATION_WARNING:
            self._show_clone_warning(
                f"Audio should be at least {int(self.CLONE_MIN_DURATION_WARNING)} seconds for best results"
            )
        elif self._clone_audio_duration > self.CLONE_MAX_DURATION_WARNING:
            self._show_clone_warning(
                f"Only the first {int(self.CLONE_MAX_DURATION_WARNING)} seconds will be used"
            )
        else:
            self._hide_clone_warning()

    def _show_clone_warning(self, message: str):
        """Show a warning message in Clone tab (QA8)."""
        self.clone_warning_label.setText(f"Warning: {message}")
        self.clone_warning_label.setStyleSheet(
            "color: #b8860b; background-color: #fff8dc; padding: 8px; border-radius: 4px;"
        )
        self.clone_warning_label.setVisible(True)

    def _hide_clone_warning(self):
        """Hide the warning message in Clone tab (QA8)."""
        self.clone_warning_label.setVisible(False)
        self.clone_warning_label.setText("")

    def _show_clone_error(self, message: str):
        """Show an error message in Clone tab (QA8)."""
        self._clone_has_error = True
        self._clone_audio_path = None
        self.clone_file_label.setText(f"Error: {message}")
        self.clone_file_label.setStyleSheet("color: red;")
        self.clone_info_group.setVisible(False)
        self.clone_transcript_group.setVisible(False)
        self.clone_proceed_group.setVisible(False)
        self.logger.error(f"Clone audio file error: {message}")

    def _on_clone_transcribe_clicked(self):
        """Handle Auto-Transcribe button click in Clone tab (QA8)."""
        if not self._clone_audio_path:
            return

        # Reset error state
        self._clone_transcription_error = False
        self.clone_transcribe_retry_button.setVisible(False)

        # Emit signal for parent to handle transcription
        self.clone_transcribe_requested.emit(str(self._clone_audio_path))

    def _on_clone_proceed_clicked(self):
        """Handle Proceed button click in Clone tab (QA8)."""
        if not self._clone_audio_path:
            return

        audio_path = str(self._clone_audio_path)
        transcript = self.get_clone_transcript()
        voice_name = self.get_clone_voice_name()

        self.logger.info(f"Clone proceed: audio={audio_path}, transcript_len={len(transcript)}, name={voice_name}")

        # Emit signal for parent to handle transfer to Emotions tab
        self.clone_proceed_requested.emit(audio_path, transcript, voice_name)

    # QA8: Clone tab public methods for state management

    def set_clone_transcribing(self, is_transcribing: bool, message: str = "Transcribing..."):
        """Set the clone transcription state (QA8)."""
        self._clone_is_transcribing = is_transcribing

        if is_transcribing:
            self.clone_transcribe_progress.setVisible(True)
            self.clone_transcribe_status.setText(message)
            self.clone_transcribe_status.setStyleSheet("color: blue;")
            self.clone_transcribe_button.setEnabled(False)
            self.clone_transcript_edit.setEnabled(False)
            self.clone_transcribe_retry_button.setVisible(False)
        else:
            self.clone_transcribe_progress.setVisible(False)
            self.clone_transcribe_button.setEnabled(True)
            self.clone_transcript_edit.setEnabled(True)

    def set_clone_transcription_complete(self, transcript: str):
        """Set clone transcription as complete (QA8)."""
        self._clone_is_transcribing = False
        self._clone_transcription_error = False

        self.clone_transcribe_progress.setVisible(False)
        self.clone_transcribe_button.setEnabled(True)
        self.clone_transcript_edit.setEnabled(True)
        self.clone_transcribe_retry_button.setVisible(False)

        self.clone_transcript_edit.setText(transcript)
        self.clone_transcribe_status.setText("Transcription complete")
        self.clone_transcribe_status.setStyleSheet("color: green;")

        self.logger.info("Clone transcription completed successfully")

    def set_clone_transcription_error(self, error_message: str):
        """Set clone transcription error state (QA8)."""
        self._clone_is_transcribing = False
        self._clone_transcription_error = True

        self.clone_transcribe_progress.setVisible(False)
        self.clone_transcribe_button.setEnabled(True)
        self.clone_transcript_edit.setEnabled(True)

        self.clone_transcribe_status.setText(f"Error: {error_message}")
        self.clone_transcribe_status.setStyleSheet("color: red;")
        self.clone_transcribe_retry_button.setVisible(True)

        self.logger.error(f"Clone transcription error: {error_message}")

    def get_clone_transcript(self) -> str:
        """Get the clone transcript text (QA8)."""
        return self.clone_transcript_edit.toPlainText().strip()

    def set_clone_transcript(self, text: str):
        """Set the clone transcript text (QA8)."""
        self.clone_transcript_edit.setText(text)

    def get_clone_voice_name(self) -> str:
        """Get the clone voice name (QA8)."""
        return self.clone_voice_name_edit.text().strip()

    def set_clone_voice_name(self, name: str):
        """Set the clone voice name (QA8)."""
        self.clone_voice_name_edit.setText(name)

    def get_clone_audio_path(self) -> Optional[Path]:
        """Get the clone audio file path (QA8)."""
        return self._clone_audio_path

    def has_clone_audio_file(self) -> bool:
        """Check if a clone audio file is loaded (QA8)."""
        return self._clone_audio_path is not None and not self._clone_has_error

    def clear_clone_tab(self):
        """Clear all clone tab state (QA8)."""
        self._clone_audio_path = None
        self._clone_audio_duration = 0.0
        self._clone_audio_format = ""
        self._clone_has_error = False
        self._clone_is_transcribing = False
        self._clone_transcription_error = False

        self.clone_file_label.setText("No file selected")
        self.clone_file_label.setStyleSheet("color: #666; font-style: italic;")
        self.clone_info_group.setVisible(False)
        self.clone_transcript_group.setVisible(False)
        self.clone_proceed_group.setVisible(False)
        self.clone_voice_name_edit.clear()
        self.clone_transcript_edit.clear()
        self.clone_audio_player.clear()
        self._hide_clone_warning()
        self.clone_transcribe_status.setText("")
        self.clone_transcribe_retry_button.setVisible(False)

    def _on_all_variants_complete(self):
        """Handle all variants complete from the variant list (Story 1.7)."""
        self.logger.debug("All 5 variants complete")
        self._is_generating = False
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self._update_generate_button_state()
        self.status_label.setText("Select a variant to save.")
        self.status_label.setStyleSheet("color: green;")

    def _on_regenerate_clicked(self):
        """Handle regenerate button click from variant list (Story 1.9)."""
        self.logger.debug("Regenerate All clicked")
        self.regenerate_requested.emit()

    def _on_sub_tab_changed(self, index: int):
        """Handle sub-tab change (QA3, QA8)."""
        tab_names = ["Imagine", "Create", "Clone"]  # QA8: Added Clone tab
        self.logger.debug(f"Sub-tab changed to: {tab_names[index] if index < len(tab_names) else index}")

    def _is_save_ready_internal(self) -> bool:
        """Check if save is ready (internal helper)."""
        # Story 1.8: Check if a variant is selected OR we have a single embedding
        has_selection = self.variant_list.has_selection()
        has_embedding = self._generated_embedding_path is not None
        has_name = bool(self.get_voice_name())
        not_saving = not self._is_saving
        return (has_embedding or has_selection) and has_name and not_saving

    def _clear_result(self):
        """Clear the generated result."""
        self._generated_audio_path = None
        self._generated_embedding_path = None
        self.result_group.setVisible(False)
        self.audio_player.clear()
        self.variant_list.clear()  # Story 1.7: Clear variant list
        self.voice_name_edit.clear()
        self.save_ready_changed.emit(False)

    def _clear_error(self):
        """Clear error state."""
        self._has_error = False
        self.status_label.setText("")
        self.status_label.setStyleSheet("")
        self.retry_button.setVisible(False)

    # Public API for generation state management

    def set_generating(self, is_generating: bool, message: str = "Generating..."):
        """
        Set the generation state.

        Args:
            is_generating: True if generation is in progress
            message: Progress message to display
        """
        self._is_generating = is_generating
        self._has_error = False

        # Update UI elements
        self.progress_bar.setVisible(is_generating)
        self.progress_label.setVisible(is_generating)
        self.progress_label.setText(message)

        # Disable generate button during generation
        self.generate_button.setEnabled(not is_generating and bool(self.get_description()))

        # Clear any previous error
        if is_generating:
            self._clear_error()
            self._clear_result()
            # Story 1.7: Show variant list and start generation
            self.result_group.setVisible(True)
            self.variant_list.start_generation()
            # QA3: Switch to Create tab when generation starts
            self.sub_tabs.setCurrentIndex(1)  # Create tab

        self.logger.debug(f"Generation state: {is_generating}, message: {message}")

    def set_generation_complete(self, audio_path: str | Path, embedding_path: str | Path = None):
        """
        Set generation complete with audio file and optional embedding.

        NOTE: For Story 1.7, use set_variant_complete() instead to update individual variants.
        This method is kept for backward compatibility with single-variant generation.

        Args:
            audio_path: Path to the generated audio file
            embedding_path: Path to the generated embedding .pt file (for saving)
        """
        self._is_generating = False
        self._has_error = False
        self._generated_audio_path = Path(audio_path) if audio_path else None
        self._generated_embedding_path = Path(embedding_path) if embedding_path else None

        # Hide progress
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

        # Show result
        if self._generated_audio_path and self._generated_audio_path.exists():
            self.result_group.setVisible(True)
            self.audio_player.set_audio_file(self._generated_audio_path)
            self.status_label.setText("Preview generated successfully. Enter a name to save.")
            self.status_label.setStyleSheet("color: green;")
            # Emit save ready signal (will be false until name is entered)
            self.save_ready_changed.emit(self._is_save_ready_internal())
        else:
            self.set_generation_error("Generated audio file not found.")

        # Re-enable generate button
        self._update_generate_button_state()

        self.logger.debug(f"Generation complete: audio={audio_path}, embedding={embedding_path}")

    # Story 1.7: Variant-specific generation methods

    def set_variant_generating(self, variant_index: int):
        """
        Set a specific variant to generating state (Story 1.7).

        Args:
            variant_index: Index of the variant (0-4)
        """
        self.variant_list.set_variant_generating(variant_index)
        self.progress_label.setText(f"Generating variant {variant_index + 1}/5...")
        self.logger.debug(f"Variant {variant_index} generating")

    def set_variant_complete(
        self,
        variant_index: int,
        audio_path: Path,
        embedding_path: Path,
        duration: float = None
    ):
        """
        Set a specific variant to complete state (Story 1.7).

        Args:
            variant_index: Index of the variant (0-4)
            audio_path: Path to generated audio file
            embedding_path: Path to generated embedding file
            duration: Audio duration in seconds
        """
        self.variant_list.set_variant_complete(variant_index, audio_path, embedding_path, duration)

        # Update status to prompt selection
        completed = self.variant_list.get_complete_count()
        if completed < 5:
            self.status_label.setText(f"{completed}/5 variants ready. Select one to save.")
        else:
            self.status_label.setText("5 variations ready. Select one to save.")
        self.status_label.setStyleSheet("color: green;")

        self.logger.debug(f"Variant {variant_index} complete: {audio_path}")

    def set_variant_error(self, variant_index: int, error_message: str = "Generation failed"):
        """
        Set a specific variant to error state (Story 1.7).

        Args:
            variant_index: Index of the variant (0-4)
            error_message: Error message to display
        """
        self.variant_list.set_variant_error(variant_index, error_message)
        self.logger.debug(f"Variant {variant_index} error: {error_message}")

    def finish_generation(self):
        """
        Mark generation as finished (Story 1.7).

        Called when all variants are complete or generation is cancelled.
        """
        self._is_generating = False
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.variant_list.finish_generation()
        self._update_generate_button_state()
        self.logger.debug("Generation finished")

    def set_generation_error(self, error_message: str):
        """
        Set generation error state.

        Args:
            error_message: Error message to display
        """
        self._is_generating = False
        self._has_error = True

        # Hide progress
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

        # Show error
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet("color: red;")
        self.retry_button.setVisible(True)

        # Re-enable generate button
        self._update_generate_button_state()

        self.logger.error(f"Generation error: {error_message}")

    def is_generating(self) -> bool:
        """
        Check if generation is in progress.

        Returns:
            True if generating
        """
        return self._is_generating

    def has_error(self) -> bool:
        """
        Check if there was a generation error.

        Returns:
            True if there was an error
        """
        return self._has_error

    def get_generated_audio_path(self) -> Optional[Path]:
        """
        Get the path to the generated audio file.

        Returns:
            Path to audio file, or None if not generated
        """
        return self._generated_audio_path

    # Existing public API

    def get_description(self) -> str:
        """
        Get the current voice description text.

        Returns:
            Voice description text (stripped of whitespace)
        """
        return self.description_edit.toPlainText().strip()

    def get_preview_text(self) -> str:
        """
        Get the current preview text.

        Returns:
            Preview text (stripped of whitespace), or default if empty
        """
        text = self.preview_text_edit.toPlainText().strip()
        # Use placeholder as default if field is empty
        return text if text else self.DEFAULT_PREVIEW_TEXT

    def get_language(self) -> str:
        """
        Get the selected language for voice generation.

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

    def set_description(self, description: str):
        """
        Set the voice description text.

        Args:
            description: Voice description text
        """
        self.description_edit.setPlainText(description)

    def set_preview_text(self, text: str):
        """
        Set the preview text.

        Args:
            text: Preview text
        """
        self.preview_text_edit.setPlainText(text)

    def clear(self):
        """Clear all input fields and results."""
        self.description_edit.clear()
        self.preview_text_edit.clear()
        self._clear_result()
        self._clear_error()
        self._is_generating = False
        self._is_saving = False
        self.variant_list.clear()  # Story 1.7: Clear variant list

    def set_generate_enabled(self, enabled: bool):
        """
        Override the automatic Generate button state.

        Useful for disabling during generation.

        Args:
            enabled: Whether the button should be enabled
        """
        self.generate_button.setEnabled(enabled)

    def is_generate_enabled(self) -> bool:
        """
        Check if the Generate button is enabled.

        Returns:
            True if Generate button is enabled
        """
        return self.generate_button.isEnabled()

    def has_content(self) -> bool:
        """
        Check if either input field has content.

        Returns:
            True if description or preview text has content
        """
        return bool(self.get_description()) or bool(self.preview_text_edit.toPlainText().strip())

    def has_generated_audio(self) -> bool:
        """
        Check if there is generated audio available.

        Returns:
            True if audio has been generated
        """
        return self._generated_audio_path is not None and self._generated_audio_path.exists()

    # Story 1.5: Save Voice to Library API

    def get_voice_name(self) -> str:
        """
        Get the entered voice name for saving.

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

    def get_generated_embedding_path(self) -> Optional[Path]:
        """
        Get the path to the generated embedding file.

        Returns:
            Path to embedding .pt file, or None if not generated
        """
        return self._generated_embedding_path

    def is_save_ready(self) -> bool:
        """
        Check if the voice is ready to be saved.

        Returns:
            True if embedding exists and voice name is entered
        """
        return self._is_save_ready_internal()

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

        # Emit save readiness change
        self.save_ready_changed.emit(self._is_save_ready_internal())

        self.logger.debug(f"Saving state: {is_saving}, message: {message}")

    def is_saving(self) -> bool:
        """
        Check if save is in progress.

        Returns:
            True if saving
        """
        return self._is_saving

    def set_save_complete(self, voice_name: str):
        """
        Set save complete state.

        Args:
            voice_name: Name of the saved voice
        """
        self._is_saving = False
        self.status_label.setText(f"Voice '{voice_name}' saved successfully!")
        self.status_label.setStyleSheet("color: green;")
        self.voice_name_edit.setEnabled(True)

        self.logger.debug(f"Save complete: {voice_name}")

    def set_save_error(self, error_message: str):
        """
        Set save error state.

        Args:
            error_message: Error message to display
        """
        self._is_saving = False
        self.status_label.setText(f"Save failed: {error_message}")
        self.status_label.setStyleSheet("color: red;")
        self.voice_name_edit.setEnabled(True)

        # Emit save readiness (should be ready to retry)
        self.save_ready_changed.emit(self._is_save_ready_internal())

        self.logger.error(f"Save error: {error_message}")

    # Story 1.7/1.8: Variant selection API

    def get_selected_variant_index(self) -> int | None:
        """
        Get the currently selected variant index.

        Returns:
            Selected variant index (0-4), or None if no selection
        """
        return self.variant_list.get_selected_index()

    def get_selected_variant_paths(self) -> tuple[Path, Path] | None:
        """
        Get the selected variant's paths.

        Returns:
            Tuple of (audio_path, embedding_path), or None if no selection
        """
        return self.variant_list.get_selected_variant()

    def has_variant_selection(self) -> bool:
        """
        Check if a variant is selected.

        Returns:
            True if a variant is selected
        """
        return self.variant_list.has_selection()

    def has_complete_variants(self) -> bool:
        """
        Check if any variants are complete.

        Returns:
            True if at least one variant is complete
        """
        return self.variant_list.has_complete_variants()

    def get_complete_variant_count(self) -> int:
        """
        Get the number of successfully completed variants.

        Returns:
            Count of complete variants (0-5)
        """
        return self.variant_list.get_complete_count()

    def get_all_variant_paths(self) -> list:
        """
        QA7: Get all variant paths and durations.

        Returns:
            List of (audio_path, embedding_path, duration) tuples for all complete variants.
            Entries may be None for incomplete variants.
        """
        return self.variant_list.get_all_variant_paths()
