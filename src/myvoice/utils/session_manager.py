"""
Session Manager for Voice Design Studio

Manages temporary session directories for voice generation workflows.
QA4: Uses a single persistent session directory instead of unique UUIDs,
allowing generated samples to persist across dialog close/reopen.

Session Directory Structure:
    voice_files/
        design_sessions/
            current/                  # QA4: Single persistent session
                variant_0.wav         # Description tab variants (legacy)
                variant_1.wav
                ...
                extraction_preview.wav
                emotion_variants/     # Emotion Variants feature
                    neutral/
                        variant_0.wav ... variant_4.wav
                        selected.wav
                    happy/
                        variant_0.wav ... variant_4.wav
                        selected.wav
                    ... (other emotions)
                embeddings/           # Extracted embeddings during session
                    neutral/embedding.pt
                    happy/embedding.pt
                    ...

Cleanup Behavior (Story 4.1 REVISED + QA4):
    - Variant files are ONLY cleared when user clicks Regenerate (Story 1.9)
    - Dialog close does NOT delete temp files (user may save more variants later)
    - Orphan sessions (>24h old) are cleaned on app restart (Story 4.2)
    - QA4: Single session persists until Regenerate clears it

Usage:
    from myvoice.utils.session_manager import SessionManager

    # Create/get persistent session (directory reused)
    manager = SessionManager()

    # Get path for variant files (description tab)
    embedding_path = manager.get_variant_path(0, "pt")
    audio_path = manager.get_variant_path(0, "wav")

    # Get path for emotion variant files (Emotion Variants feature)
    emotion_audio = manager.get_emotion_variant_path("happy", 0, "wav")
    selected_audio = manager.get_emotion_selected_path("happy")
    embedding = manager.get_emotion_embedding_path("happy")

    # Clear variant files on Regenerate (Story 1.9)
    manager.clear_variant_files()
    manager.clear_emotion_variants("happy")  # Clear specific emotion

    # Full cleanup for orphan sessions only (Story 4.2)
    SessionManager.cleanup_orphan_sessions()
"""

import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

from myvoice.utils.portable_paths import get_voice_files_path

logger = logging.getLogger(__name__)

# Orphan session threshold in seconds (24 hours)
ORPHAN_SESSION_THRESHOLD_SECONDS = 24 * 60 * 60

# QA4: Single persistent session directory name
PERSISTENT_SESSION_NAME = "current"


class SessionManager:
    """
    Manages temporary session directory lifecycle for Voice Design Studio.

    QA4: Uses a single persistent session directory instead of unique UUIDs.
    This allows generated samples to persist across dialog close/reopen,
    so users can return and select different previously-generated variants.

    Story 4.1 REVISED Cleanup Behavior:
    - clear_variant_files(): Deletes variant .pt/.wav files (called on Regenerate)
    - cleanup(): Deletes entire session directory (reserved for orphan cleanup)
    - Dialog close does NOT trigger cleanup - files persist for additional saves

    Attributes:
        session_id: Session identifier (always "current" for persistent mode)
        session_dir: Path to the session directory
        is_cleaned: Whether full cleanup has been called

    Example:
        manager = SessionManager()
        variant_path = manager.get_variant_path(0, "wav")
        # ... generate audio to variant_path ...
        manager.clear_variant_files()  # On Regenerate only
    """

    def __init__(self, base_dir: Optional[Path] = None, use_persistent: bool = True):
        """
        Initialize SessionManager and create/access session directory.

        Args:
            base_dir: Base directory for design_sessions folder.
                      Defaults to voice_files path from portable_paths.
            use_persistent: QA4: If True, use single persistent session directory.
                           If False, create a unique session (for legacy/testing).
        """
        self._is_cleaned = False

        # Determine base directory
        if base_dir is None:
            base_dir = get_voice_files_path()

        # QA4: Use persistent session directory by default
        if use_persistent:
            self._session_id = PERSISTENT_SESSION_NAME
        else:
            import uuid
            self._session_id = str(uuid.uuid4())

        # Create session directory: base_dir/design_sessions/session_id
        self._design_sessions_dir = base_dir / "design_sessions"
        self._session_dir = self._design_sessions_dir / self._session_id

        # Create directory if it doesn't exist
        self._session_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Session directory: {self._session_dir} (persistent={use_persistent})")

    @property
    def session_id(self) -> str:
        """Get the session ID."""
        return self._session_id

    @property
    def session_dir(self) -> Path:
        """Get the path to the session directory."""
        return self._session_dir

    @property
    def is_cleaned(self) -> bool:
        """Check if cleanup has been called."""
        return self._is_cleaned

    def get_variant_path(self, variant_index: int, extension: str) -> Path:
        """
        Get the file path for a variant file.

        Args:
            variant_index: Index of the variant (0-4 for description path)
            extension: File extension without dot (e.g., "pt", "wav")

        Returns:
            Path to the variant file within the session directory

        Example:
            path = manager.get_variant_path(0, "wav")
            # Returns: session_dir/variant_0.wav
        """
        # Ensure extension doesn't have leading dot
        ext = extension.lstrip(".")
        return self._session_dir / f"variant_{variant_index}.{ext}"

    def get_existing_variants(self) -> list[Path]:
        """
        Get list of existing variant files in the session (QA4).

        Returns:
            List of paths to existing variant .wav files, sorted by index
        """
        variants = []
        if self._session_dir.exists():
            for wav_file in sorted(self._session_dir.glob("variant_*.wav")):
                variants.append(wav_file)
        return variants

    def has_existing_variants(self) -> bool:
        """
        Check if there are existing variant files from a previous session (QA4).

        Returns:
            True if variant files exist
        """
        return len(self.get_existing_variants()) > 0

    def clear_variant_files(self) -> int:
        """
        Delete all variant files in the session directory (Story 4.1/1.9).

        Called when user clicks Regenerate - clears previous audition files
        while preserving the session directory for new variants.

        Returns:
            Number of files deleted
        """
        if self._is_cleaned:
            logger.debug(f"Session already cleaned: {self._session_id}")
            return 0

        deleted_count = 0
        try:
            if self._session_dir.exists():
                # Delete variant_*.pt, variant_*.wav, and extraction_preview.wav
                patterns = ["variant_*.pt", "variant_*.wav", "extraction_preview.wav"]
                for pattern in patterns:
                    for file_path in self._session_dir.glob(pattern):
                        try:
                            file_path.unlink()
                            deleted_count += 1
                            logger.debug(f"Deleted variant file: {file_path.name}")
                        except Exception as e:
                            logger.warning(f"Failed to delete {file_path}: {e}")

                logger.info(f"Cleared {deleted_count} variant files from session {self._session_id}")
        except Exception as e:
            logger.error(f"Error clearing variant files in session {self._session_id}: {e}")

        return deleted_count

    # =========================================================================
    # Emotion Variants: Emotion-specific path and cleanup methods
    # =========================================================================

    def get_emotion_variants_dir(self) -> Path:
        """
        Get the emotion_variants subdirectory path.

        Returns:
            Path to session_dir/emotion_variants/
        """
        return self._session_dir / "emotion_variants"

    def get_emotion_variant_path(self, emotion: str, variant_index: int, extension: str) -> Path:
        """
        Get the file path for an emotion-specific variant file.

        Emotion Variants: Stores variant files in emotion subfolders.

        Args:
            emotion: Emotion name (e.g., "neutral", "happy")
            variant_index: Index of the variant (0-4)
            extension: File extension without dot (e.g., "wav")

        Returns:
            Path to the variant file: session_dir/emotion_variants/{emotion}/variant_{index}.{ext}

        Example:
            path = manager.get_emotion_variant_path("happy", 0, "wav")
            # Returns: session_dir/emotion_variants/happy/variant_0.wav
        """
        ext = extension.lstrip(".")
        emotion_dir = self.get_emotion_variants_dir() / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)
        return emotion_dir / f"variant_{variant_index}.{ext}"

    def get_emotion_selected_path(self, emotion: str) -> Path:
        """
        Get the path for the selected sample of an emotion.

        Emotion Variants: Each emotion can have one selected sample.

        Args:
            emotion: Emotion name (e.g., "neutral", "happy")

        Returns:
            Path to the selected file: session_dir/emotion_variants/{emotion}/selected.wav
        """
        emotion_dir = self.get_emotion_variants_dir() / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)
        return emotion_dir / "selected.wav"

    def get_emotion_embedding_path(self, emotion: str) -> Path:
        """
        Get the path for an emotion's extracted embedding.

        Emotion Variants: Embeddings are stored during the session for preview.

        Args:
            emotion: Emotion name (e.g., "neutral", "happy")

        Returns:
            Path to the embedding: session_dir/embeddings/{emotion}/embedding.pt
        """
        embedding_dir = self._session_dir / "embeddings" / emotion
        embedding_dir.mkdir(parents=True, exist_ok=True)
        return embedding_dir / "embedding.pt"

    def get_existing_emotion_variants(self, emotion: str) -> List[Path]:
        """
        Get list of existing variant files for a specific emotion.

        Args:
            emotion: Emotion name (e.g., "neutral", "happy")

        Returns:
            List of paths to existing variant .wav files, sorted by index
        """
        emotion_dir = self.get_emotion_variants_dir() / emotion
        if not emotion_dir.exists():
            return []

        variants = []
        for wav_file in sorted(emotion_dir.glob("variant_*.wav")):
            variants.append(wav_file)
        return variants

    def has_emotion_selected(self, emotion: str) -> bool:
        """
        Check if an emotion has a selected sample.

        Args:
            emotion: Emotion name

        Returns:
            True if selected.wav exists for this emotion
        """
        selected_path = self.get_emotion_variants_dir() / emotion / "selected.wav"
        return selected_path.exists()

    def get_existing_emotion_embeddings(self) -> dict:
        """
        QA Round 2 Item #6: Get existing emotion embeddings from session.

        Returns:
            Dict mapping emotion name to embedding Path if exists
        """
        embeddings_dir = self._session_dir / "embeddings"
        if not embeddings_dir.exists():
            return {}

        result = {}
        for emotion_dir in embeddings_dir.iterdir():
            if emotion_dir.is_dir():
                embedding_path = emotion_dir / "embedding.pt"
                if embedding_path.exists():
                    result[emotion_dir.name] = embedding_path
        return result

    def get_selected_emotions(self) -> List[str]:
        """
        Get list of emotions that have selected samples.

        Returns:
            List of emotion names with selected.wav files
        """
        emotions_dir = self.get_emotion_variants_dir()
        if not emotions_dir.exists():
            return []

        selected = []
        for emotion_dir in emotions_dir.iterdir():
            if emotion_dir.is_dir() and (emotion_dir / "selected.wav").exists():
                selected.append(emotion_dir.name)
        return selected

    def clear_emotion_variants(self, emotion: str) -> int:
        """
        Clear all variant files for a specific emotion.

        Called when user regenerates variants for one emotion.

        Args:
            emotion: Emotion name to clear

        Returns:
            Number of files deleted
        """
        emotion_dir = self.get_emotion_variants_dir() / emotion
        if not emotion_dir.exists():
            return 0

        deleted_count = 0
        try:
            # Delete variant_*.wav files but preserve selected.wav
            for wav_file in emotion_dir.glob("variant_*.wav"):
                try:
                    wav_file.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted emotion variant: {wav_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete {wav_file}: {e}")

            logger.info(f"Cleared {deleted_count} variants for emotion '{emotion}'")
        except Exception as e:
            logger.error(f"Error clearing emotion variants for '{emotion}': {e}")

        return deleted_count

    def clear_all_emotion_variants(self) -> int:
        """
        Clear all emotion variant files and embeddings.

        Called when starting a new voice design workflow.

        Returns:
            Number of files deleted
        """
        deleted_count = 0

        # Clear emotion_variants directory
        emotion_variants_dir = self.get_emotion_variants_dir()
        if emotion_variants_dir.exists():
            try:
                shutil.rmtree(emotion_variants_dir)
                deleted_count += 1
                logger.info(f"Cleared emotion_variants directory")
            except Exception as e:
                logger.error(f"Error clearing emotion_variants: {e}")

        # Clear embeddings directory
        embeddings_dir = self._session_dir / "embeddings"
        if embeddings_dir.exists():
            try:
                shutil.rmtree(embeddings_dir)
                deleted_count += 1
                logger.info(f"Cleared embeddings directory")
            except Exception as e:
                logger.error(f"Error clearing embeddings: {e}")

        return deleted_count

    # QA7: Session data persistence for recovery

    def save_session_data(
        self,
        voice_description: str = "",
        preview_text: str = "",
        voice_name: str = "",
        language: str = "Auto",
        clone_transcript: str = "",
        current_tab_index: int = 0
    ) -> bool:
        """
        QA7: Save session data for recovery.

        Saves voice description and preview text to session_data.json so the
        user can recover their work if they close the dialog to test a voice.

        Args:
            voice_description: The voice description text
            preview_text: The preview/sample text
            voice_name: The voice name (if entered)
            language: The selected language
            clone_transcript: QA8: Transcript from Clone tab for embedding extraction
            current_tab_index: QA Round 2 Item #6: Current main tab index

        Returns:
            True if save was successful, False otherwise
        """
        import json

        # QA Round 2 Item #6: Merge with existing session data instead of overwriting
        existing_data = self.load_session_data() or {}

        # Update only non-empty values (allows partial updates)
        session_data = existing_data.copy()
        if voice_description:
            session_data["voice_description"] = voice_description
        if preview_text:
            session_data["preview_text"] = preview_text
        if voice_name:
            session_data["voice_name"] = voice_name
        if language and language != "Auto":
            session_data["language"] = language
        if clone_transcript:
            session_data["clone_transcript"] = clone_transcript
        if current_tab_index > 0:
            session_data["current_tab_index"] = current_tab_index
        session_data["saved_at"] = datetime.now().isoformat()

        session_data_path = self._session_dir / "session_data.json"
        try:
            with open(session_data_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved session data to {session_data_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")
            return False

    def load_session_data(self) -> Optional[dict]:
        """
        QA7: Load saved session data for recovery.

        Returns:
            Dictionary with session data, or None if no saved data exists

        Example return:
            {
                "voice_description": "...",
                "preview_text": "...",
                "voice_name": "...",
                "language": "Auto",
                "saved_at": "2026-02-05T..."
            }
        """
        import json

        session_data_path = self._session_dir / "session_data.json"
        if not session_data_path.exists():
            logger.debug("No saved session data found")
            return None

        try:
            with open(session_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded session data from {session_data_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load session data: {e}")
            return None

    def has_session_data(self) -> bool:
        """QA7: Check if session data exists for recovery."""
        return (self._session_dir / "session_data.json").exists()

    def cleanup(self) -> None:
        """
        Remove the session directory and all its contents.

        This method is idempotent - calling it multiple times is safe.
        Errors during cleanup are logged but not raised.

        Note: For persistent sessions, this should only be called for
        orphan cleanup, not on dialog close.
        """
        if self._is_cleaned:
            logger.debug(f"Session already cleaned: {self._session_id}")
            return

        try:
            if self._session_dir.exists():
                shutil.rmtree(self._session_dir)
                logger.info(f"Cleaned up session directory: {self._session_dir}")
            else:
                logger.debug(f"Session directory already removed: {self._session_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up session {self._session_id}: {e}")
        finally:
            self._is_cleaned = True

    def __enter__(self) -> "SessionManager":
        """Context manager entry - returns self."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - cleanup regardless of exception."""
        self.cleanup()
        return False  # Don't suppress exceptions

    def __repr__(self) -> str:
        """Return string representation."""
        return f"SessionManager(session_id={self._session_id}, cleaned={self._is_cleaned})"

    @classmethod
    def cleanup_orphan_sessions(
        cls,
        base_dir: Optional[Path] = None,
        max_age_seconds: int = ORPHAN_SESSION_THRESHOLD_SECONDS
    ) -> Tuple[int, int]:
        """
        Clean up orphaned session directories older than threshold (Story 4.2).

        Called on application startup to remove abandoned session directories
        that are older than the specified threshold (default 24 hours).

        QA4: The persistent "current" session is preserved unless older than threshold.

        Args:
            base_dir: Base directory containing design_sessions folder.
                      Defaults to voice_files path from portable_paths.
            max_age_seconds: Maximum age in seconds before a session is considered
                            orphaned. Default is 24 hours (86400 seconds).

        Returns:
            Tuple of (sessions_cleaned, sessions_preserved)

        Example:
            # On app startup
            cleaned, preserved = SessionManager.cleanup_orphan_sessions()
            logger.info(f"Cleaned {cleaned} orphan sessions, preserved {preserved} recent")
        """
        if base_dir is None:
            base_dir = get_voice_files_path()

        design_sessions_dir = base_dir / "design_sessions"

        if not design_sessions_dir.exists():
            logger.debug("No design_sessions directory found - nothing to clean")
            return (0, 0)

        current_time = time.time()
        sessions_cleaned = 0
        sessions_preserved = 0

        try:
            # Find all session directories (including legacy session_* and current)
            for session_dir in design_sessions_dir.iterdir():
                if not session_dir.is_dir():
                    continue

                # QA4: Include both legacy session_* and new persistent sessions
                if not (session_dir.name.startswith("session_") or
                        session_dir.name == PERSISTENT_SESSION_NAME):
                    continue

                try:
                    # Get directory modification time
                    dir_mtime = session_dir.stat().st_mtime
                    age_seconds = current_time - dir_mtime

                    if age_seconds > max_age_seconds:
                        # Session is orphaned - delete it
                        shutil.rmtree(session_dir)
                        sessions_cleaned += 1
                        age_hours = age_seconds / 3600
                        logger.info(f"Cleaned orphan session: {session_dir.name} (age: {age_hours:.1f}h)")
                    else:
                        # Session is recent - preserve it
                        sessions_preserved += 1
                        age_hours = age_seconds / 3600
                        logger.debug(f"Preserved recent session: {session_dir.name} (age: {age_hours:.1f}h)")

                except Exception as e:
                    logger.warning(f"Error processing session {session_dir.name}: {e}")

        except Exception as e:
            logger.error(f"Error scanning design_sessions directory: {e}")

        if sessions_cleaned > 0:
            logger.info(f"Orphan session cleanup: {sessions_cleaned} cleaned, {sessions_preserved} preserved")

        return (sessions_cleaned, sessions_preserved)

    @classmethod
    def get_design_sessions_dir(cls, base_dir: Optional[Path] = None) -> Path:
        """
        Get the path to the design_sessions directory.

        Args:
            base_dir: Base directory. Defaults to voice_files path.

        Returns:
            Path to design_sessions directory
        """
        if base_dir is None:
            base_dir = get_voice_files_path()
        return base_dir / "design_sessions"
