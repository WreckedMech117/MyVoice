"""
Voice Variant Row Widget

A single row displaying one voice variation with states:
- Waiting: Dimmed, shows "Waiting..."
- Generating: Shows spinner and "Generating..."
- Complete: Shows play button and duration
- Playing: Shows stop button and playing indicator
- Selected: Radio button checked, row highlighted

Story 1.7: Generate 5 Variations with Progressive Reveal
Story 1.8: Audition and Select Variant
"""

import logging
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QRadioButton,
    QSizePolicy, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QEvent
from PyQt6.QtGui import QFont, QCursor

from myvoice.ui.dialogs.voice_design_studio.audio_player_widget import AudioPlayerWidget


class VariantState(Enum):
    """State of a voice variant row."""
    WAITING = auto()      # Waiting to be generated (dimmed)
    GENERATING = auto()   # Currently generating (spinner)
    COMPLETE = auto()     # Generation complete (ready to play)
    ERROR = auto()        # Generation failed


class VoiceVariantRow(QWidget):
    """
    A single row displaying one voice variation.

    Supports different states: waiting, generating, complete, playing.
    Includes selection via radio button and audio playback.

    Signals:
        selection_changed: Emitted when radio button selection changes (variant_index, is_selected)
        play_requested: Emitted when play button clicked (variant_index)
        stop_requested: Emitted when stop button clicked (variant_index)
    """

    selection_changed = pyqtSignal(int, bool)  # variant_index, is_selected
    play_requested = pyqtSignal(int)  # variant_index
    stop_requested = pyqtSignal(int)  # variant_index

    def __init__(self, variant_index: int, parent: Optional[QWidget] = None):
        """
        Initialize the Voice Variant Row.

        Args:
            variant_index: Zero-based index of this variant (0-4)
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{variant_index}")

        self._variant_index = variant_index
        self._state = VariantState.WAITING
        self._audio_path: Optional[Path] = None
        self._embedding_path: Optional[Path] = None
        self._duration: Optional[float] = None
        self._is_playing = False
        self._spinner_frame = 0
        self._is_hovered = False  # QA5: Track hover state for manual hover styling

        # Spinner animation timer (created before UI update)
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(100)  # 100ms per frame
        self._spinner_timer.timeout.connect(self._animate_spinner)

        self._create_ui()
        self._setup_connections()
        self._setup_accessibility()
        self._update_ui_for_state()

        self.logger.debug(f"VoiceVariantRow {variant_index} initialized")

    def _create_ui(self):
        """Create the row UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Radio button for selection
        self.radio_button = QRadioButton()
        self.radio_button.setObjectName(f"variant_{self._variant_index}_radio")
        self.radio_button.setEnabled(False)  # Disabled until complete
        layout.addWidget(self.radio_button)

        # Variant number label
        self.variant_label = QLabel(f"Variant {self._variant_index + 1}")
        self.variant_label.setObjectName(f"variant_{self._variant_index}_label")
        self.variant_label.setMinimumWidth(70)
        font = QFont()
        font.setBold(True)
        self.variant_label.setFont(font)
        layout.addWidget(self.variant_label)

        # Status label (shows state text or duration)
        self.status_label = QLabel("Waiting...")
        self.status_label.setObjectName(f"variant_{self._variant_index}_status")
        self.status_label.setMinimumWidth(100)
        layout.addWidget(self.status_label)

        # Stretch to push buttons to the right
        layout.addStretch()

        # Play/Stop button
        self.play_button = QPushButton("Play")
        self.play_button.setObjectName(f"variant_{self._variant_index}_play")
        self.play_button.setMinimumWidth(60)
        self.play_button.setVisible(False)
        layout.addWidget(self.play_button)

        # Duration label
        self.duration_label = QLabel("")
        self.duration_label.setObjectName(f"variant_{self._variant_index}_duration")
        self.duration_label.setMinimumWidth(50)
        self.duration_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.duration_label.setVisible(False)
        layout.addWidget(self.duration_label)

        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(40)

        # QA6: Enable auto-fill background for stylesheet background-color to work
        self.setAutoFillBackground(True)

    def _setup_connections(self):
        """Setup signal connections."""
        self.radio_button.toggled.connect(self._on_radio_toggled)
        self.play_button.clicked.connect(self._on_play_clicked)

    def mousePressEvent(self, event):
        """
        Handle mouse clicks on the entire row (QA3: Improve selection UX).

        Clicking anywhere on the row selects that variant when enabled.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            # Only select if the radio button is enabled (variant is complete)
            if self.radio_button.isEnabled() and not self.radio_button.isChecked():
                self.radio_button.setChecked(True)
        super().mousePressEvent(event)

    def enterEvent(self, event):
        """Handle mouse enter - show clickable cursor and hover highlight when selectable (QA3/QA5)."""
        if self.radio_button.isEnabled():
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            self._is_hovered = True
            self._update_highlight()  # QA5: Apply hover styling
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave - restore default cursor and remove hover highlight (QA3/QA5)."""
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self._is_hovered = False
        self._update_highlight()  # QA5: Remove hover styling
        super().leaveEvent(event)

    def _setup_accessibility(self):
        """Configure accessibility properties."""
        self.radio_button.setAccessibleName(f"Select Variant {self._variant_index + 1}")
        self.radio_button.setAccessibleDescription(
            f"Select variant {self._variant_index + 1} as the voice to save"
        )
        self.play_button.setAccessibleName(f"Play Variant {self._variant_index + 1}")
        self.play_button.setAccessibleDescription(
            f"Play the audio preview for variant {self._variant_index + 1}"
        )

    def _on_radio_toggled(self, checked: bool):
        """Handle radio button toggle."""
        self.selection_changed.emit(self._variant_index, checked)
        self._update_highlight()

    def _on_play_clicked(self):
        """Handle play/stop button click."""
        if self._is_playing:
            self.stop_requested.emit(self._variant_index)
        else:
            self.play_requested.emit(self._variant_index)

    def _update_ui_for_state(self):
        """Update UI elements based on current state."""
        if self._state == VariantState.WAITING:
            self.status_label.setText("Waiting...")
            self.status_label.setStyleSheet("color: #888;")
            self.radio_button.setEnabled(False)
            self.play_button.setVisible(False)
            self.duration_label.setVisible(False)
            self._spinner_timer.stop()
            self.setEnabled(True)
            self._set_dimmed(True)

        elif self._state == VariantState.GENERATING:
            self._spinner_frame = 0
            self.status_label.setText("Generating...")
            self.status_label.setStyleSheet("color: #0066cc;")
            self.radio_button.setEnabled(False)
            self.play_button.setVisible(False)
            self.duration_label.setVisible(False)
            self._spinner_timer.start()
            self.setEnabled(True)
            self._set_dimmed(False)

        elif self._state == VariantState.COMPLETE:
            duration_str = self._format_duration(self._duration) if self._duration else "0:00"
            self.status_label.setText("Ready")
            self.status_label.setStyleSheet("color: #008800;")
            self.radio_button.setEnabled(True)
            self.play_button.setVisible(True)
            self.play_button.setText("Play")
            self.duration_label.setVisible(True)
            self.duration_label.setText(duration_str)
            self._spinner_timer.stop()
            self.setEnabled(True)
            self._set_dimmed(False)

        elif self._state == VariantState.ERROR:
            self.status_label.setText("Failed")
            self.status_label.setStyleSheet("color: #cc0000;")
            self.radio_button.setEnabled(False)
            self.play_button.setVisible(False)
            self.duration_label.setVisible(False)
            self._spinner_timer.stop()
            self.setEnabled(True)
            self._set_dimmed(True)

        self._update_highlight()

    def _set_dimmed(self, dimmed: bool):
        """Set the dimmed appearance for the row."""
        opacity = "0.5" if dimmed else "1.0"
        self.variant_label.setStyleSheet(f"opacity: {opacity};")

    def _update_highlight(self):
        """Update row highlight based on selection and hover state (QA3/QA5/QA6: Enhanced visual feedback)."""
        # QA6: Use direct style without class selector for reliable PyQt6 styling
        if self.radio_button.isChecked():
            # Strong visual highlight with border when selected - MORE VISIBLE
            self.setStyleSheet("""
                background-color: #cce5ff;
                border: 3px solid #0066cc;
                border-radius: 8px;
            """)
            # Also highlight the variant label for extra visibility
            self.variant_label.setStyleSheet("color: #0066cc; font-weight: bold;")
        elif self._state == VariantState.COMPLETE:
            # QA5: Manual hover state handling (PyQt6 :hover doesn't work on custom QWidget)
            if self._is_hovered:
                # Hover state - highlighted background
                self.setStyleSheet("""
                    background-color: #e6f2ff;
                    border: 2px solid #99c2ff;
                    border-radius: 6px;
                """)
                self.variant_label.setStyleSheet("font-weight: bold;")
            else:
                # Normal state - subtle border
                self.setStyleSheet("""
                    background-color: transparent;
                    border: 1px solid #ddd;
                    border-radius: 6px;
                """)
                self.variant_label.setStyleSheet("font-weight: bold;")
        else:
            self.setStyleSheet("")
            self.variant_label.setStyleSheet("")

    def _animate_spinner(self):
        """Animate the spinner in the status label."""
        spinner_chars = ["", "", "", "", "", "", "", ""]
        self._spinner_frame = (self._spinner_frame + 1) % len(spinner_chars)
        self.status_label.setText(f"{spinner_chars[self._spinner_frame]} Generating...")

    def _format_duration(self, duration: Optional[float]) -> str:
        """Format duration in seconds to M:SS format."""
        if duration is None:
            return "0:00"
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        return f"{minutes}:{seconds:02d}"

    # Public API

    def get_variant_index(self) -> int:
        """Get the variant index."""
        return self._variant_index

    def get_state(self) -> VariantState:
        """Get the current state."""
        return self._state

    def set_waiting(self):
        """Set to waiting state."""
        self._state = VariantState.WAITING
        self._audio_path = None
        self._embedding_path = None
        self._duration = None
        self._is_playing = False
        self._update_ui_for_state()
        self.logger.debug(f"Variant {self._variant_index} set to WAITING")

    def set_generating(self):
        """Set to generating state."""
        self._state = VariantState.GENERATING
        self._update_ui_for_state()
        self.logger.debug(f"Variant {self._variant_index} set to GENERATING")

    def set_complete(self, audio_path: Path, embedding_path: Optional[Path], duration: float = None):
        """
        Set to complete state with audio and embedding paths.

        Args:
            audio_path: Path to the generated audio file
            embedding_path: Path to the generated embedding file (QA7: Optional for uploaded samples)
            duration: Audio duration in seconds (optional)
        """
        self._state = VariantState.COMPLETE
        self._audio_path = audio_path
        self._embedding_path = embedding_path
        self._duration = duration
        self._update_ui_for_state()
        self.logger.debug(f"Variant {self._variant_index} set to COMPLETE: {audio_path}")

    def set_error(self, error_message: str = "Generation failed"):
        """
        Set to error state.

        Args:
            error_message: Error message to display
        """
        self._state = VariantState.ERROR
        self.status_label.setText(error_message[:20] + "..." if len(error_message) > 20 else error_message)
        self._update_ui_for_state()
        self.logger.debug(f"Variant {self._variant_index} set to ERROR: {error_message}")

    def set_playing(self, is_playing: bool):
        """
        Set the playing state.

        Args:
            is_playing: True if audio is playing
        """
        self._is_playing = is_playing
        if self._state == VariantState.COMPLETE:
            self.play_button.setText("Stop" if is_playing else "Play")

    def is_playing(self) -> bool:
        """Check if audio is playing."""
        return self._is_playing

    def is_selected(self) -> bool:
        """Check if this variant is selected."""
        return self.radio_button.isChecked()

    def set_selected(self, selected: bool):
        """
        Set the selection state.

        Args:
            selected: True to select this variant
        """
        self.radio_button.setChecked(selected)

    def get_audio_path(self) -> Optional[Path]:
        """Get the audio file path."""
        return self._audio_path

    def get_embedding_path(self) -> Optional[Path]:
        """Get the embedding file path."""
        return self._embedding_path

    def get_duration(self) -> Optional[float]:
        """Get the audio duration in seconds."""
        return self._duration

    def is_complete(self) -> bool:
        """Check if variant is in complete state."""
        return self._state == VariantState.COMPLETE

    def is_error(self) -> bool:
        """Check if variant is in error state."""
        return self._state == VariantState.ERROR

    def clear(self):
        """Clear the variant and reset to waiting state."""
        self.set_waiting()
        self.radio_button.setChecked(False)
