"""
Audio Player Widget

Reusable audio player component with play/stop functionality and duration display.
Used in Voice Design Studio for previewing generated voice variations.

QA4 Update: Uses QMediaPlayer for internal playback (no external player popup).

Story 1.4: Generate Single Voice Variation
- Audio player widget appears with play button and duration on completion
- Play/stop functionality works for preview audio
"""

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QLabel
)
from PyQt6.QtCore import Qt, pyqtSignal, QUrl
from PyQt6.QtGui import QFont

# Try to import QMediaPlayer for internal playback
try:
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    MULTIMEDIA_AVAILABLE = True
except ImportError:
    MULTIMEDIA_AVAILABLE = False


class AudioPlayerWidget(QWidget):
    """
    Compact audio player widget with play/stop and duration display.

    QA4: Uses QMediaPlayer for internal playback (no external player popup).
    Falls back to disabled state if Qt Multimedia is not available.

    Signals:
        playback_started: Emitted when playback starts
        playback_stopped: Emitted when playback stops
    """

    playback_started = pyqtSignal()
    playback_stopped = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the audio player widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        self._audio_path: Optional[Path] = None
        self._duration_seconds: float = 0.0
        self._is_playing = False

        # QA4: Initialize QMediaPlayer for internal playback
        self._media_player: Optional[QMediaPlayer] = None
        self._audio_output: Optional[QAudioOutput] = None

        if MULTIMEDIA_AVAILABLE:
            self._setup_media_player()
        else:
            self.logger.warning("Qt Multimedia not available - audio playback disabled")

        self._create_ui()
        self._setup_accessibility()

        self.logger.debug(f"AudioPlayerWidget initialized (multimedia={'available' if MULTIMEDIA_AVAILABLE else 'unavailable'})")

    def _setup_media_player(self):
        """Setup QMediaPlayer for internal audio playback (QA4)."""
        try:
            self._media_player = QMediaPlayer(self)
            self._audio_output = QAudioOutput(self)
            self._media_player.setAudioOutput(self._audio_output)

            # Connect signals for playback state tracking
            self._media_player.playbackStateChanged.connect(self._on_playback_state_changed)
            self._media_player.errorOccurred.connect(self._on_media_error)
            self._media_player.durationChanged.connect(self._on_duration_changed)

            self.logger.debug("QMediaPlayer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize QMediaPlayer: {e}")
            self._media_player = None
            self._audio_output = None

    def _create_ui(self):
        """Create the widget UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Play/Stop button
        self.play_button = QPushButton("Play")
        self.play_button.setObjectName("audio_play_button")
        self.play_button.setMinimumWidth(70)
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._on_play_clicked)
        layout.addWidget(self.play_button)

        # Duration label
        self.duration_label = QLabel("0:00")
        self.duration_label.setObjectName("audio_duration_label")
        self.duration_label.setMinimumWidth(40)
        duration_font = QFont()
        duration_font.setFamily("monospace")
        self.duration_label.setFont(duration_font)
        layout.addWidget(self.duration_label)

        # Status indicator (optional visual feedback)
        self.status_label = QLabel("")
        self.status_label.setObjectName("audio_status_label")
        layout.addWidget(self.status_label, 1)  # Stretch to fill

    def _setup_accessibility(self):
        """Configure accessibility properties."""
        self.play_button.setAccessibleName("Play audio")
        self.play_button.setAccessibleDescription("Play or stop the audio preview")

        self.duration_label.setAccessibleName("Duration")
        self.duration_label.setAccessibleDescription("Audio duration in minutes and seconds")

    def set_audio_file(self, file_path: str | Path):
        """
        Set the audio file to play.

        Args:
            file_path: Path to the audio file
        """
        self._audio_path = Path(file_path) if file_path else None

        if self._audio_path and self._audio_path.exists():
            # Set source for media player
            if self._media_player:
                url = QUrl.fromLocalFile(str(self._audio_path))
                self._media_player.setSource(url)
                self.play_button.setEnabled(True)
            else:
                # No media player available
                self.play_button.setEnabled(False)
                self.status_label.setText("Playback unavailable")

            # Get duration
            self._duration_seconds = self._get_audio_duration(self._audio_path)
            self.duration_label.setText(self._format_duration(self._duration_seconds))
            self.status_label.setText("")
            self.logger.debug(f"Audio file set: {self._audio_path} ({self._duration_seconds:.1f}s)")
        else:
            self.play_button.setEnabled(False)
            self.duration_label.setText("0:00")
            self.status_label.setText("")
            if file_path:
                self.logger.warning(f"Audio file not found: {file_path}")

    def _get_audio_duration(self, file_path: Path) -> float:
        """
        Get the duration of an audio file.

        Uses Python's built-in wave module for WAV files (most reliable),
        falls back to mutagen for other formats.

        Args:
            file_path: Path to the audio file

        Returns:
            Duration in seconds, or 0 if unable to determine
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
            pass
        except Exception as e:
            self.logger.debug(f"Could not get duration with mutagen: {e}")

        # Fallback: estimate based on file size (rough approximation)
        try:
            file_size = file_path.stat().st_size
            # Assume 16-bit stereo 44.1kHz WAV (176400 bytes/second)
            # This is a rough estimate
            estimated_duration = file_size / 176400
            return max(estimated_duration, 1.0)  # At least 1 second
        except Exception:
            return 3.0  # Default fallback

    def _format_duration(self, seconds: float) -> str:
        """
        Format duration as M:SS.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted string like "1:23"
        """
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes}:{secs:02d}"

    def _on_play_clicked(self):
        """Handle play/stop button click."""
        if self._is_playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        """Start audio playback using QMediaPlayer (QA4: internal playback)."""
        if not self._audio_path or not self._audio_path.exists():
            self.status_label.setText("No audio file")
            return

        if not self._media_player:
            self.status_label.setText("Playback unavailable")
            return

        try:
            # QA4: Use QMediaPlayer for internal playback
            self._media_player.play()
            self.logger.debug(f"Started internal playback: {self._audio_path}")

        except Exception as e:
            self.logger.error(f"Error starting playback: {e}")
            self.status_label.setText("Playback failed")

    def _stop_playback(self):
        """Stop audio playback (QA4: actually stops the player)."""
        if self._media_player:
            self._media_player.stop()
        self._on_playback_finished()

    def _on_playback_state_changed(self, state):
        """
        Handle QMediaPlayer playback state changes (QA4).

        Args:
            state: QMediaPlayer.PlaybackState enum
        """
        if not MULTIMEDIA_AVAILABLE:
            return

        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._is_playing = True
            self.play_button.setText("Stop")
            self.status_label.setText("")
            self.playback_started.emit()
            self.logger.debug("Playback started")

        elif state == QMediaPlayer.PlaybackState.StoppedState:
            self._is_playing = False
            self.play_button.setText("Play")
            self.playback_stopped.emit()
            self.logger.debug("Playback stopped")

        elif state == QMediaPlayer.PlaybackState.PausedState:
            self._is_playing = False
            self.play_button.setText("Play")
            self.logger.debug("Playback paused")

    def _on_media_error(self, error, error_string=""):
        """
        Handle QMediaPlayer errors (QA4).

        Args:
            error: QMediaPlayer.Error enum
            error_string: Error message string
        """
        self.logger.error(f"Media player error: {error} - {error_string}")
        self.status_label.setText("Playback error")
        self._is_playing = False
        self.play_button.setText("Play")

    def _on_duration_changed(self, duration_ms: int):
        """
        Handle duration change from QMediaPlayer (QA4).

        Args:
            duration_ms: Duration in milliseconds
        """
        if duration_ms > 0:
            duration_sec = duration_ms / 1000.0
            # Only update if significantly different (media player may be more accurate)
            if abs(duration_sec - self._duration_seconds) > 0.5:
                self._duration_seconds = duration_sec
                self.duration_label.setText(self._format_duration(self._duration_seconds))
                self.logger.debug(f"Duration updated from media player: {duration_sec:.2f}s")

    def _on_playback_finished(self):
        """Handle playback completion."""
        self._is_playing = False
        self.play_button.setText("Play")
        self.playback_stopped.emit()

    def clear(self):
        """Clear the audio file and reset state."""
        # Stop any current playback
        if self._media_player:
            self._media_player.stop()
            self._media_player.setSource(QUrl())

        self._audio_path = None
        self._duration_seconds = 0.0
        self._is_playing = False
        self.play_button.setEnabled(False)
        self.play_button.setText("Play")
        self.duration_label.setText("0:00")
        self.status_label.setText("")

    def get_audio_path(self) -> Optional[Path]:
        """
        Get the current audio file path.

        Returns:
            Path to the audio file, or None if not set
        """
        return self._audio_path

    def get_duration(self) -> float:
        """
        Get the audio duration.

        Returns:
            Duration in seconds
        """
        return self._duration_seconds

    def is_playing(self) -> bool:
        """
        Check if audio is currently playing.

        Returns:
            True if playing
        """
        return self._is_playing

    def play(self):
        """
        Start audio playback.

        Does nothing if no audio is loaded or already playing.
        """
        if self._audio_path and not self._is_playing:
            self._start_playback()

    def stop(self):
        """
        Stop audio playback.

        Does nothing if not currently playing.
        """
        if self._is_playing:
            self._stop_playback()
