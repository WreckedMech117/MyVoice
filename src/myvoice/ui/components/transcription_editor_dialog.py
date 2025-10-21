"""
Transcription Editor Dialog Component

This module implements a dialog for viewing, editing, and managing voice transcriptions
with preview functionality, save/cancel options, and progress feedback for generation.
"""

import logging
import asyncio
from typing import Optional, Callable
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel,
    QProgressBar, QGroupBox, QFormLayout, QSpinBox, QCheckBox, QMessageBox,
    QSplitter, QWidget, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread, pyqtSlot
from PyQt6.QtGui import QFont, QTextCursor, QTextCharFormat, QColor

from myvoice.models.voice_profile import VoiceProfile, TranscriptionStatus
from myvoice.models.transcription_result import TranscriptionResult
from myvoice.models.error import MyVoiceError
from myvoice.services.background_transcription_manager import (
    BackgroundTranscriptionManager, TranscriptionNotification, NotificationType
)


class TranscriptionGenerationWorker(QThread):
    """Worker thread for background transcription generation."""

    progress_updated = pyqtSignal(str, float)  # stage, progress_percent
    transcription_completed = pyqtSignal(TranscriptionResult)  # result
    transcription_failed = pyqtSignal(str)  # error_message

    def __init__(self, voice_profile: VoiceProfile, transcription_service, parent=None):
        """
        Initialize transcription worker.

        Args:
            voice_profile: Voice profile to transcribe
            transcription_service: Service for performing transcription
            parent: Parent object
        """
        super().__init__(parent)
        self.voice_profile = voice_profile
        self.transcription_service = transcription_service
        self.is_cancelled = False

    def run(self):
        """Run transcription generation in background thread."""
        try:
            if self.is_cancelled:
                return

            self.progress_updated.emit("Starting transcription...", 10.0)

            # Perform transcription
            result = self.transcription_service.transcribe_file(
                file_path=self.voice_profile.file_path,
                word_timestamps=True
            )

            if self.is_cancelled:
                return

            self.progress_updated.emit("Transcription completed", 100.0)
            self.transcription_completed.emit(result)

        except Exception as e:
            if not self.is_cancelled:
                self.transcription_failed.emit(str(e))

    def cancel(self):
        """Cancel the transcription operation."""
        self.is_cancelled = True


class TranscriptionEditorDialog(QDialog):
    """
    Dialog for editing and managing voice transcriptions.

    Features:
    - View existing transcription or generate new one
    - Edit transcription text with validation
    - Save/cancel functionality
    - Progress feedback during generation
    - Transcription metadata display

    Signals:
        transcription_saved: Emitted when transcription is saved (voice_profile, text)
        generation_requested: Emitted when user requests new transcription generation
    """

    transcription_saved = pyqtSignal(VoiceProfile, str)  # profile, transcription_text
    generation_requested = pyqtSignal(VoiceProfile)  # profile

    def __init__(self, voice_profile: VoiceProfile, transcription_service=None,
                 voice_manager=None, background_transcription_manager: Optional[BackgroundTranscriptionManager] = None,
                 parent: Optional[QWidget] = None):
        """
        Initialize the transcription editor dialog.

        Args:
            voice_profile: Voice profile to edit transcription for
            transcription_service: Service for generating transcriptions
            voice_manager: Voice profile manager for saving transcriptions
            background_transcription_manager: Manager for background transcription processing
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Store references
        self.voice_profile = voice_profile
        self.transcription_service = transcription_service
        self.voice_manager = voice_manager
        self.background_transcription_manager = background_transcription_manager

        # UI state
        self.original_text = ""
        self.has_unsaved_changes = False
        self.generation_worker: Optional[TranscriptionGenerationWorker] = None

        # UI components
        self.transcription_edit: Optional[QTextEdit] = None
        self.progress_bar: Optional[QProgressBar] = None
        self.progress_label: Optional[QLabel] = None
        self.generate_button: Optional[QPushButton] = None
        self.save_button: Optional[QPushButton] = None
        self.cancel_button: Optional[QPushButton] = None
        self.metadata_labels: dict = {}

        # Setup dialog
        self._setup_dialog()
        self._create_ui()
        self._load_transcription()
        self._setup_connections()

        self.logger.debug(f"TranscriptionEditorDialog initialized for {voice_profile.name}")

    def _setup_dialog(self):
        """Configure dialog properties."""
        self.setWindowTitle(f"Edit Transcription - {self.voice_profile.name}")
        self.setModal(True)
        self.resize(700, 500)

        # Set window flags
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.WindowTitleHint |
            Qt.WindowType.WindowCloseButtonHint
        )

    def _create_ui(self):
        """Create the dialog UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # Create splitter for main content
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left panel: Transcription editing
        left_panel = self._create_transcription_panel()
        splitter.addWidget(left_panel)

        # Right panel: Metadata and controls
        right_panel = self._create_metadata_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions (70% transcription, 30% metadata)
        splitter.setSizes([500, 200])

        # Progress section
        progress_group = self._create_progress_section()
        layout.addWidget(progress_group)

        # Button section
        button_layout = self._create_button_section()
        layout.addLayout(button_layout)

    def _create_transcription_panel(self) -> QWidget:
        """Create the transcription editing panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Header
        header_label = QLabel("Transcription Text")
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(11)
        header_label.setFont(header_font)
        layout.addWidget(header_label)

        # Text editor
        self.transcription_edit = QTextEdit()
        self.transcription_edit.setPlaceholderText(
            "Enter or edit the transcription text for this voice sample...\n\n"
            "Tips:\n"
            "• Keep the text clear and accurate\n"
            "• Include proper punctuation\n"
            "• Remove background noise descriptions"
        )

        # Set font for better readability
        edit_font = QFont("Segoe UI", 10)
        self.transcription_edit.setFont(edit_font)

        # Set minimum height
        self.transcription_edit.setMinimumHeight(300)

        layout.addWidget(self.transcription_edit)

        # Word count label
        self.word_count_label = QLabel("Words: 0")
        self.word_count_label.setProperty("class", "caption")
        layout.addWidget(self.word_count_label)

        return panel

    def _create_metadata_panel(self) -> QWidget:
        """Create the metadata display panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Voice profile info
        profile_group = QGroupBox("Voice Profile")
        profile_layout = QFormLayout(profile_group)

        # Profile details
        self.metadata_labels['name'] = QLabel(self.voice_profile.name)
        self.metadata_labels['duration'] = QLabel(
            f"{self.voice_profile.duration:.1f}s" if self.voice_profile.duration else "Unknown"
        )
        self.metadata_labels['status'] = QLabel(self.voice_profile.transcription_status.value.title())

        profile_layout.addRow("Name:", self.metadata_labels['name'])
        profile_layout.addRow("Duration:", self.metadata_labels['duration'])
        profile_layout.addRow("Status:", self.metadata_labels['status'])

        layout.addWidget(profile_group)

        # Transcription info
        trans_group = QGroupBox("Transcription Info")
        trans_layout = QFormLayout(trans_group)

        self.metadata_labels['confidence'] = QLabel("N/A")
        self.metadata_labels['model'] = QLabel("N/A")
        self.metadata_labels['language'] = QLabel("N/A")

        trans_layout.addRow("Confidence:", self.metadata_labels['confidence'])
        trans_layout.addRow("Model:", self.metadata_labels['model'])
        trans_layout.addRow("Language:", self.metadata_labels['language'])

        layout.addWidget(trans_group)

        # Generation controls
        controls_group = QGroupBox("Actions")
        controls_layout = QVBoxLayout(controls_group)

        self.generate_button = QPushButton("🎤 Generate Transcription")
        self.generate_button.setToolTip("Generate new transcription using Whisper")
        controls_layout.addWidget(self.generate_button)

        # Auto-generation options
        self.auto_save_checkbox = QCheckBox("Auto-save when generated")
        self.auto_save_checkbox.setChecked(True)
        controls_layout.addWidget(self.auto_save_checkbox)

        layout.addWidget(controls_group)

        # Add stretch to push everything to top
        layout.addStretch()

        return panel

    def _create_progress_section(self) -> QWidget:
        """Create the progress feedback section."""
        group = QGroupBox("Progress")
        layout = QVBoxLayout(group)

        # Progress label
        self.progress_label = QLabel("Ready")
        layout.addWidget(self.progress_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Initially hide the progress group
        group.setVisible(False)

        return group

    def _create_button_section(self) -> QHBoxLayout:
        """Create the dialog button section."""
        layout = QHBoxLayout()
        layout.addStretch()

        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setMinimumWidth(80)
        layout.addWidget(self.cancel_button)

        # Save button
        self.save_button = QPushButton("Save")
        self.save_button.setMinimumWidth(80)
        self.save_button.setDefault(True)
        layout.addWidget(self.save_button)

        return layout

    def _setup_connections(self):
        """Setup signal/slot connections."""
        # Text editing
        self.transcription_edit.textChanged.connect(self._on_text_changed)

        # Buttons
        self.generate_button.clicked.connect(self._on_generate_clicked)
        self.save_button.clicked.connect(self._on_save_clicked)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)

    def _load_transcription(self):
        """Load existing transcription into the editor."""
        try:
            # Load current transcription text
            transcription_text = ""

            if self.voice_profile.transcription:
                transcription_text = self.voice_profile.transcription
            elif self.voice_manager:
                # Try to load from file
                loaded_result = self.voice_manager.load_transcription(self.voice_profile.file_path)
                if loaded_result:
                    transcription_text = loaded_result.text

            # Set text in editor
            self.transcription_edit.setPlainText(transcription_text)
            self.original_text = transcription_text
            self.has_unsaved_changes = False

            # Update metadata
            self._update_metadata()

            # Update word count
            self._update_word_count()

            self.logger.debug("Transcription loaded successfully")

        except Exception as e:
            self.logger.exception(f"Error loading transcription: {e}")
            QMessageBox.warning(
                self, "Load Error",
                f"Failed to load transcription: {str(e)}"
            )

    def _update_metadata(self):
        """Update the metadata display."""
        try:
            # Update transcription status
            self.metadata_labels['status'].setText(
                self.voice_profile.transcription_status.value.title()
            )

            # Update transcription info if available
            if self.voice_profile.transcription_confidence is not None:
                confidence_pct = self.voice_profile.transcription_confidence * 100
                self.metadata_labels['confidence'].setText(f"{confidence_pct:.1f}%")

            if self.voice_profile.transcription_model:
                self.metadata_labels['model'].setText(self.voice_profile.transcription_model)

            # For now, assume English - could be enhanced to detect language
            self.metadata_labels['language'].setText("English")

        except Exception as e:
            self.logger.error(f"Error updating metadata: {e}")

    def _update_word_count(self):
        """Update the word count display."""
        try:
            text = self.transcription_edit.toPlainText().strip()
            if text:
                word_count = len(text.split())
                self.word_count_label.setText(f"Words: {word_count}")
            else:
                self.word_count_label.setText("Words: 0")
        except Exception as e:
            self.logger.error(f"Error updating word count: {e}")

    @pyqtSlot()
    def _on_text_changed(self):
        """Handle text changes in the editor."""
        try:
            current_text = self.transcription_edit.toPlainText()
            self.has_unsaved_changes = (current_text != self.original_text)

            # Update save button state
            self.save_button.setEnabled(self.has_unsaved_changes)

            # Update word count
            self._update_word_count()

        except Exception as e:
            self.logger.error(f"Error handling text change: {e}")

    @pyqtSlot()
    def _on_generate_clicked(self):
        """Handle generate transcription button click."""
        try:
            if not self.transcription_service:
                QMessageBox.warning(
                    self, "Service Unavailable",
                    "Transcription service is not available. Please check your settings."
                )
                return

            # Check if there are unsaved changes
            if self.has_unsaved_changes:
                reply = QMessageBox.question(
                    self, "Unsaved Changes",
                    "You have unsaved changes. Generate new transcription anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return

            # Start generation
            self._start_transcription_generation()

        except Exception as e:
            self.logger.exception(f"Error starting transcription generation: {e}")
            QMessageBox.critical(
                self, "Generation Error",
                f"Failed to start transcription generation: {str(e)}"
            )

    def _start_transcription_generation(self):
        """Start background transcription generation."""
        try:
            # Use background transcription manager if available
            if self.background_transcription_manager:
                self._start_background_transcription()
            else:
                self._start_legacy_transcription()

        except Exception as e:
            self.logger.exception(f"Error starting generation: {e}")
            self._hide_progress()
            self._reset_generate_button()
            raise

    def _start_background_transcription(self):
        """Start transcription using background transcription manager."""
        try:
            # Show progress
            self._show_progress("Queuing transcription...", 0.0)

            # Disable generation button
            self.generate_button.setEnabled(False)
            self.generate_button.setText("Queued...")

            # Set up notification callback
            self.background_transcription_manager.add_notification_callback(self._on_background_notification)

            # Request transcription with high priority (since user is waiting)
            from myvoice.services.transcription_queue_service import QueuePriority

            # Use QTimer to schedule the async call
            QTimer.singleShot(0, lambda: self._request_background_transcription(QueuePriority.HIGH))

        except Exception as e:
            self.logger.exception(f"Error starting background transcription: {e}")
            self._on_generation_failed(str(e))

    def _start_legacy_transcription(self):
        """Start transcription using legacy worker thread."""
        # Show progress
        self._show_progress("Preparing transcription...", 0.0)

        # Disable generation button
        self.generate_button.setEnabled(False)
        self.generate_button.setText("Generating...")

        # Create and start worker thread
        self.generation_worker = TranscriptionGenerationWorker(
            self.voice_profile,
            self.transcription_service,
            self
        )

        # Connect worker signals
        self.generation_worker.progress_updated.connect(self._on_generation_progress)
        self.generation_worker.transcription_completed.connect(self._on_generation_completed)
        self.generation_worker.transcription_failed.connect(self._on_generation_failed)
        self.generation_worker.finished.connect(self._on_generation_finished)

        # Start generation
        self.generation_worker.start()

        self.logger.info("Started legacy transcription generation")

    def _show_progress(self, message: str, progress: float):
        """Show progress feedback."""
        self.progress_label.setText(message)
        self.progress_bar.setValue(int(progress))
        self.progress_bar.parentWidget().setVisible(True)
        self.progress_bar.setVisible(True)

    def _hide_progress(self):
        """Hide progress feedback."""
        self.progress_bar.parentWidget().setVisible(False)

    def _reset_generate_button(self):
        """Reset the generate button to normal state."""
        self.generate_button.setEnabled(True)
        self.generate_button.setText("🎤 Generate Transcription")

    @pyqtSlot(str, float)
    def _on_generation_progress(self, message: str, progress: float):
        """Handle generation progress updates."""
        self._show_progress(message, progress)

    @pyqtSlot(TranscriptionResult)
    def _on_generation_completed(self, result: TranscriptionResult):
        """Handle successful transcription generation."""
        try:
            # Update the text editor
            self.transcription_edit.setPlainText(result.text)

            # Update voice profile
            self.voice_profile.set_transcription_result(
                transcription_text=result.text,
                confidence=result.confidence,
                model_name=result.model_name if hasattr(result, 'model_name') else None
            )

            # Update metadata display
            self._update_metadata()

            # Auto-save if enabled
            if self.auto_save_checkbox.isChecked():
                self._save_transcription()
            else:
                # Mark as having changes
                self.original_text = ""  # Force unsaved state
                self._on_text_changed()

            self._show_progress("Generation completed successfully!", 100.0)

            # Hide progress after a delay
            QTimer.singleShot(2000, self._hide_progress)

            self.logger.info("Transcription generation completed successfully")

        except Exception as e:
            self.logger.exception(f"Error handling generation completion: {e}")
            QMessageBox.critical(
                self, "Generation Error",
                f"Error processing generated transcription: {str(e)}"
            )

    @pyqtSlot(str)
    def _on_generation_failed(self, error_message: str):
        """Handle transcription generation failure."""
        self.logger.error(f"Transcription generation failed: {error_message}")

        self._hide_progress()

        QMessageBox.critical(
            self, "Generation Failed",
            f"Failed to generate transcription:\n\n{error_message}"
        )

    @pyqtSlot()
    def _on_generation_finished(self):
        """Handle generation thread completion."""
        self._reset_generate_button()

        if self.generation_worker:
            self.generation_worker.deleteLater()
            self.generation_worker = None

    @pyqtSlot()
    def _on_save_clicked(self):
        """Handle save button click."""
        try:
            self._save_transcription()
        except Exception as e:
            self.logger.exception(f"Error saving transcription: {e}")
            QMessageBox.critical(
                self, "Save Error",
                f"Failed to save transcription: {str(e)}"
            )

    def _save_transcription(self):
        """Save the current transcription."""
        try:
            transcription_text = self.transcription_edit.toPlainText().strip()

            if not transcription_text:
                reply = QMessageBox.question(
                    self, "Empty Transcription",
                    "The transcription is empty. Save anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return

            # Update voice profile
            self.voice_profile.transcription = transcription_text
            self.voice_profile.update_transcription_status(TranscriptionStatus.COMPLETED)

            # Save through voice manager if available
            if self.voice_manager:
                self.voice_manager.save_transcription(
                    self.voice_profile.file_path,
                    transcription_text
                )

            # Update state
            self.original_text = transcription_text
            self.has_unsaved_changes = False
            self.save_button.setEnabled(False)

            # Update metadata
            self._update_metadata()

            # Emit signal
            self.transcription_saved.emit(self.voice_profile, transcription_text)

            # Show success message briefly
            self.progress_label.setText("Transcription saved successfully!")
            self.progress_label.parentWidget().setVisible(True)
            QTimer.singleShot(2000, lambda: self.progress_label.parentWidget().setVisible(False))

            self.logger.info("Transcription saved successfully")

        except Exception as e:
            self.logger.exception(f"Error saving transcription: {e}")
            raise

    @pyqtSlot()
    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        self._close_dialog()

    def _close_dialog(self):
        """Close the dialog with unsaved changes check."""
        try:
            if self.has_unsaved_changes:
                reply = QMessageBox.question(
                    self, "Unsaved Changes",
                    "You have unsaved changes. Close without saving?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return

            # Cancel any ongoing generation
            if self.generation_worker and self.generation_worker.isRunning():
                self.generation_worker.cancel()
                self.generation_worker.wait(3000)  # Wait up to 3 seconds

            self.accept()

        except Exception as e:
            self.logger.exception(f"Error closing dialog: {e}")
            self.reject()

    async def _request_background_transcription(self, priority):
        """Request transcription using background manager."""
        try:
            success = await self.background_transcription_manager.request_transcription(
                self.voice_profile,
                priority=priority,
                force_regenerate=True
            )

            if success:
                QTimer.singleShot(0, lambda: self._show_progress("Transcription queued for processing...", 10.0))
                self.logger.info("Transcription queued successfully")
            else:
                QTimer.singleShot(0, lambda: self._on_generation_failed("Failed to queue transcription"))

        except Exception as e:
            self.logger.exception(f"Error requesting background transcription: {e}")
            QTimer.singleShot(0, lambda: self._on_generation_failed(str(e)))

    def _on_background_notification(self, notification: TranscriptionNotification):
        """Handle background transcription notifications."""
        try:
            # Only handle notifications for our voice profile
            if notification.voice_profile.name != self.voice_profile.name:
                return

            if notification.type == NotificationType.PROGRESS:
                QTimer.singleShot(0, lambda: self._show_progress(
                    notification.message,
                    notification.progress_percent or 50.0
                ))
            elif notification.type == NotificationType.COMPLETED:
                QTimer.singleShot(0, lambda: self._on_background_completed(notification))
            elif notification.type == NotificationType.FAILED:
                QTimer.singleShot(0, lambda: self._on_generation_failed(notification.message))

        except Exception as e:
            self.logger.exception(f"Error handling background notification: {e}")

    def _on_background_completed(self, notification: TranscriptionNotification):
        """Handle background transcription completion."""
        try:
            # Reload the transcription
            self._load_transcription()

            # Hide progress and reset button
            self._hide_progress()
            self._reset_generate_button()

            # Remove the notification callback
            if self.background_transcription_manager:
                self.background_transcription_manager.remove_notification_callback(self._on_background_notification)

            # Show success message
            self._show_progress("Transcription completed successfully!", 100.0)
            QTimer.singleShot(2000, self._hide_progress)  # Hide after 2 seconds

            self.logger.info("Background transcription completed successfully")

        except Exception as e:
            self.logger.exception(f"Error handling background completion: {e}")

    def closeEvent(self, event):
        """Handle dialog close event."""
        # Clean up background transcription callback
        if self.background_transcription_manager:
            try:
                self.background_transcription_manager.remove_notification_callback(self._on_background_notification)
            except Exception as e:
                self.logger.error(f"Error removing notification callback: {e}")

        self._close_dialog()
        if self.result() == QDialog.DialogCode.Rejected:
            event.ignore()
        else:
            event.accept()