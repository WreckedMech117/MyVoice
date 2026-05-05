"""
Tests for Session Manager

Tests the session_manager module which handles temporary session directory
lifecycle for Voice Design Studio (Story 4.1).
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import shutil
import uuid

# Skip this module if PyQt6 is not available (required for myvoice imports)
pytest.importorskip("PyQt6", reason="PyQt6 required for myvoice imports")


class TestSessionManagerCreation:
    """Tests for SessionManager initialization and directory creation."""

    def test_creates_session_directory_on_init(self, tmp_path):
        """Test that SessionManager creates session directory on initialization."""
        from myvoice.utils.session_manager import SessionManager

        manager = SessionManager(base_dir=tmp_path)

        assert manager.session_dir.exists(), "Session directory should be created"
        assert manager.session_dir.is_dir(), "Session directory should be a directory"

    def test_session_directory_has_uuid_in_name(self, tmp_path):
        """Test that session directory name contains a UUID."""
        from myvoice.utils.session_manager import SessionManager

        manager = SessionManager(base_dir=tmp_path)

        # Directory name should match pattern: session_{uuid}
        dir_name = manager.session_dir.name
        assert dir_name.startswith("session_"), f"Should start with 'session_': {dir_name}"

        # Extract UUID part and verify it's valid
        uuid_part = dir_name[8:]  # Remove "session_" prefix
        try:
            uuid.UUID(uuid_part)
        except ValueError:
            pytest.fail(f"Directory name should contain valid UUID: {dir_name}")

    def test_session_directory_under_design_sessions(self, tmp_path):
        """Test that session directory is created under design_sessions folder."""
        from myvoice.utils.session_manager import SessionManager

        manager = SessionManager(base_dir=tmp_path)

        # Session should be under base_dir/design_sessions/session_{uuid}
        assert manager.session_dir.parent.name == "design_sessions", \
            f"Should be under design_sessions: {manager.session_dir}"

    def test_session_id_property(self, tmp_path):
        """Test that session_id property returns the UUID."""
        from myvoice.utils.session_manager import SessionManager

        manager = SessionManager(base_dir=tmp_path)

        # session_id should be a valid UUID string
        session_id = manager.session_id
        assert isinstance(session_id, str), "session_id should be a string"

        try:
            uuid.UUID(session_id)
        except ValueError:
            pytest.fail(f"session_id should be valid UUID: {session_id}")

    def test_uses_voice_files_by_default(self):
        """Test that SessionManager uses voice_files as default base_dir."""
        from myvoice.utils.session_manager import SessionManager
        from myvoice.utils.portable_paths import get_voice_files_path

        manager = SessionManager()

        expected_base = get_voice_files_path()
        assert manager.session_dir.parent.parent == expected_base, \
            f"Should use voice_files as base: {manager.session_dir}"

        # Cleanup
        manager.cleanup()


class TestSessionManagerCleanup:
    """Tests for SessionManager cleanup functionality."""

    def test_cleanup_removes_session_directory(self, tmp_path):
        """Test that cleanup() removes the entire session directory."""
        from myvoice.utils.session_manager import SessionManager

        manager = SessionManager(base_dir=tmp_path)
        session_dir = manager.session_dir

        # Create some files in session directory
        (session_dir / "test.pt").touch()
        (session_dir / "preview.wav").touch()

        assert session_dir.exists(), "Session directory should exist before cleanup"

        manager.cleanup()

        assert not session_dir.exists(), "Session directory should be removed after cleanup"

    def test_cleanup_removes_all_files(self, tmp_path):
        """Test that cleanup removes all .pt and .wav files."""
        from myvoice.utils.session_manager import SessionManager

        manager = SessionManager(base_dir=tmp_path)
        session_dir = manager.session_dir

        # Create multiple files
        files = [
            "embedding_1.pt",
            "embedding_2.pt",
            "preview_1.wav",
            "preview_2.wav",
            "metadata.json",
        ]
        for filename in files:
            (session_dir / filename).touch()

        manager.cleanup()

        # Verify no files remain
        assert not session_dir.exists(), "Session directory should be completely removed"

    def test_cleanup_handles_nonexistent_directory(self, tmp_path):
        """Test that cleanup handles already-deleted directories gracefully."""
        from myvoice.utils.session_manager import SessionManager

        manager = SessionManager(base_dir=tmp_path)
        session_dir = manager.session_dir

        # Manually delete the directory first
        shutil.rmtree(session_dir)

        # Cleanup should not raise an error
        manager.cleanup()  # Should not raise

    def test_cleanup_handles_subdirectories(self, tmp_path):
        """Test that cleanup removes nested subdirectories."""
        from myvoice.utils.session_manager import SessionManager

        manager = SessionManager(base_dir=tmp_path)
        session_dir = manager.session_dir

        # Create nested structure
        subdir = session_dir / "variants"
        subdir.mkdir()
        (subdir / "variant_1.pt").touch()
        (subdir / "variant_2.pt").touch()

        manager.cleanup()

        assert not session_dir.exists(), "Session directory and subdirs should be removed"

    def test_cleanup_leaves_parent_directory(self, tmp_path):
        """Test that cleanup leaves design_sessions directory intact."""
        from myvoice.utils.session_manager import SessionManager

        manager = SessionManager(base_dir=tmp_path)
        design_sessions_dir = manager.session_dir.parent

        manager.cleanup()

        assert design_sessions_dir.exists(), "design_sessions directory should remain"

    def test_is_cleaned_property(self, tmp_path):
        """Test that is_cleaned property reflects cleanup state."""
        from myvoice.utils.session_manager import SessionManager

        manager = SessionManager(base_dir=tmp_path)

        assert not manager.is_cleaned, "Should not be cleaned initially"

        manager.cleanup()

        assert manager.is_cleaned, "Should be cleaned after cleanup()"


class TestSessionManagerMultipleSessions:
    """Tests for multiple SessionManager instances."""

    def test_multiple_sessions_have_unique_directories(self, tmp_path):
        """Test that each SessionManager instance has a unique directory."""
        from myvoice.utils.session_manager import SessionManager

        manager1 = SessionManager(base_dir=tmp_path)
        manager2 = SessionManager(base_dir=tmp_path)

        assert manager1.session_dir != manager2.session_dir, \
            "Each session should have unique directory"
        assert manager1.session_id != manager2.session_id, \
            "Each session should have unique ID"

        # Cleanup
        manager1.cleanup()
        manager2.cleanup()

    def test_cleanup_one_session_preserves_others(self, tmp_path):
        """Test that cleaning up one session doesn't affect others."""
        from myvoice.utils.session_manager import SessionManager

        manager1 = SessionManager(base_dir=tmp_path)
        manager2 = SessionManager(base_dir=tmp_path)

        # Create files in both
        (manager1.session_dir / "file1.pt").touch()
        (manager2.session_dir / "file2.pt").touch()

        manager1.cleanup()

        assert not manager1.session_dir.exists(), "Session 1 should be cleaned"
        assert manager2.session_dir.exists(), "Session 2 should be preserved"
        assert (manager2.session_dir / "file2.pt").exists(), "Session 2 files preserved"

        # Cleanup
        manager2.cleanup()


class TestSessionManagerContextManager:
    """Tests for SessionManager context manager protocol."""

    def test_works_as_context_manager(self, tmp_path):
        """Test that SessionManager can be used with 'with' statement."""
        from myvoice.utils.session_manager import SessionManager

        with SessionManager(base_dir=tmp_path) as manager:
            session_dir = manager.session_dir
            assert session_dir.exists(), "Session directory should exist in context"

        # After context exits, directory should be cleaned up
        assert not session_dir.exists(), "Session should be cleaned after context exit"

    def test_context_manager_cleans_up_on_exception(self, tmp_path):
        """Test that context manager cleans up even when exception occurs."""
        from myvoice.utils.session_manager import SessionManager

        session_dir = None
        try:
            with SessionManager(base_dir=tmp_path) as manager:
                session_dir = manager.session_dir
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert session_dir is not None, "Session directory should have been created"
        assert not session_dir.exists(), "Session should be cleaned after exception"


class TestSessionManagerFilePaths:
    """Tests for SessionManager file path helpers."""

    def test_get_variant_path(self, tmp_path):
        """Test that get_variant_path returns correct path structure."""
        from myvoice.utils.session_manager import SessionManager

        manager = SessionManager(base_dir=tmp_path)

        variant_path = manager.get_variant_path(1, "pt")

        assert variant_path.parent == manager.session_dir, "Should be under session dir"
        assert "variant_1" in variant_path.name, "Should include variant number"
        assert variant_path.suffix == ".pt", "Should have correct extension"

        # Cleanup
        manager.cleanup()

    def test_get_variant_path_for_wav(self, tmp_path):
        """Test get_variant_path for wav files."""
        from myvoice.utils.session_manager import SessionManager

        manager = SessionManager(base_dir=tmp_path)

        variant_path = manager.get_variant_path(3, "wav")

        assert "variant_3" in variant_path.name, "Should include variant number"
        assert variant_path.suffix == ".wav", "Should have .wav extension"

        # Cleanup
        manager.cleanup()


class TestSessionManagerOrphanCleanup:
    """Tests for cleanup_orphan_sessions() class method (Story 4.2)."""

    def test_cleanup_orphan_sessions_removes_old_sessions(self, tmp_path):
        """Test that orphan sessions older than threshold are removed."""
        from myvoice.utils.session_manager import SessionManager
        import time

        # Create design_sessions directory
        design_sessions = tmp_path / "design_sessions"
        design_sessions.mkdir()

        # Create an "old" session (simulate by setting mtime to past)
        old_session = design_sessions / "session_old-uuid-1234"
        old_session.mkdir()
        (old_session / "variant_1.pt").touch()

        # Set modification time to 25 hours ago
        old_time = time.time() - (25 * 60 * 60)
        import os
        os.utime(old_session, (old_time, old_time))

        # Run cleanup with 24-hour threshold
        cleaned, preserved = SessionManager.cleanup_orphan_sessions(
            base_dir=tmp_path,
            max_age_seconds=24 * 60 * 60
        )

        assert cleaned == 1, "Should clean 1 orphan session"
        assert preserved == 0, "Should preserve 0 sessions"
        assert not old_session.exists(), "Old session should be removed"

    def test_cleanup_orphan_sessions_preserves_recent_sessions(self, tmp_path):
        """Test that recent sessions are preserved."""
        from myvoice.utils.session_manager import SessionManager

        # Create design_sessions directory
        design_sessions = tmp_path / "design_sessions"
        design_sessions.mkdir()

        # Create a recent session (default mtime is now)
        recent_session = design_sessions / "session_recent-uuid-5678"
        recent_session.mkdir()
        (recent_session / "variant_1.wav").touch()

        # Run cleanup
        cleaned, preserved = SessionManager.cleanup_orphan_sessions(
            base_dir=tmp_path,
            max_age_seconds=24 * 60 * 60
        )

        assert cleaned == 0, "Should clean 0 sessions"
        assert preserved == 1, "Should preserve 1 recent session"
        assert recent_session.exists(), "Recent session should be preserved"
        assert (recent_session / "variant_1.wav").exists(), "Files should be preserved"

    def test_cleanup_orphan_sessions_handles_mixed_ages(self, tmp_path):
        """Test cleanup with mix of old and recent sessions."""
        from myvoice.utils.session_manager import SessionManager
        import time
        import os

        design_sessions = tmp_path / "design_sessions"
        design_sessions.mkdir()

        # Create old sessions
        old1 = design_sessions / "session_old1"
        old2 = design_sessions / "session_old2"
        old1.mkdir()
        old2.mkdir()

        # Create recent sessions
        recent1 = design_sessions / "session_recent1"
        recent2 = design_sessions / "session_recent2"
        recent1.mkdir()
        recent2.mkdir()

        # Set old sessions to 48 hours ago
        old_time = time.time() - (48 * 60 * 60)
        os.utime(old1, (old_time, old_time))
        os.utime(old2, (old_time, old_time))

        # Run cleanup
        cleaned, preserved = SessionManager.cleanup_orphan_sessions(
            base_dir=tmp_path,
            max_age_seconds=24 * 60 * 60
        )

        assert cleaned == 2, f"Should clean 2 old sessions, got {cleaned}"
        assert preserved == 2, f"Should preserve 2 recent sessions, got {preserved}"
        assert not old1.exists()
        assert not old2.exists()
        assert recent1.exists()
        assert recent2.exists()

    def test_cleanup_orphan_sessions_handles_no_sessions_dir(self, tmp_path):
        """Test cleanup when design_sessions directory doesn't exist."""
        from myvoice.utils.session_manager import SessionManager

        # Don't create design_sessions directory
        cleaned, preserved = SessionManager.cleanup_orphan_sessions(base_dir=tmp_path)

        assert cleaned == 0
        assert preserved == 0

    def test_cleanup_orphan_sessions_ignores_non_session_dirs(self, tmp_path):
        """Test that non-session directories are ignored."""
        from myvoice.utils.session_manager import SessionManager
        import time
        import os

        design_sessions = tmp_path / "design_sessions"
        design_sessions.mkdir()

        # Create non-session directory
        other_dir = design_sessions / "some_other_folder"
        other_dir.mkdir()

        # Create a file (not a directory)
        (design_sessions / "readme.txt").touch()

        # Set old time on other_dir
        old_time = time.time() - (48 * 60 * 60)
        os.utime(other_dir, (old_time, old_time))

        # Run cleanup
        cleaned, preserved = SessionManager.cleanup_orphan_sessions(
            base_dir=tmp_path,
            max_age_seconds=24 * 60 * 60
        )

        assert cleaned == 0, "Should not clean non-session directories"
        assert preserved == 0, "Should not count non-session directories"
        assert other_dir.exists(), "Non-session directory should be preserved"
        assert (design_sessions / "readme.txt").exists(), "Files should be preserved"

    def test_cleanup_orphan_sessions_custom_threshold(self, tmp_path):
        """Test cleanup with custom age threshold."""
        from myvoice.utils.session_manager import SessionManager
        import time
        import os

        design_sessions = tmp_path / "design_sessions"
        design_sessions.mkdir()

        # Create session that is 2 hours old
        session = design_sessions / "session_two-hours-old"
        session.mkdir()

        # Set modification time to 2 hours ago
        two_hours_ago = time.time() - (2 * 60 * 60)
        os.utime(session, (two_hours_ago, two_hours_ago))

        # With 24-hour threshold, should be preserved
        cleaned, preserved = SessionManager.cleanup_orphan_sessions(
            base_dir=tmp_path,
            max_age_seconds=24 * 60 * 60
        )
        assert preserved == 1, "Should preserve with 24h threshold"

        # With 1-hour threshold, should be cleaned
        cleaned, preserved = SessionManager.cleanup_orphan_sessions(
            base_dir=tmp_path,
            max_age_seconds=1 * 60 * 60
        )
        assert cleaned == 1, "Should clean with 1h threshold"

    def test_get_design_sessions_dir(self, tmp_path):
        """Test get_design_sessions_dir class method."""
        from myvoice.utils.session_manager import SessionManager

        result = SessionManager.get_design_sessions_dir(base_dir=tmp_path)

        assert result == tmp_path / "design_sessions"


class TestSessionManagerClearVariantFiles:
    """Tests for clear_variant_files() method (Story 4.1/1.9)."""

    def test_clear_variant_files_deletes_variant_pt_files(self, tmp_path):
        """Test that clear_variant_files deletes variant .pt files."""
        from myvoice.utils.session_manager import SessionManager

        manager = SessionManager(base_dir=tmp_path)
        session_dir = manager.session_dir

        # Create variant .pt files
        (session_dir / "variant_1.pt").touch()
        (session_dir / "variant_2.pt").touch()
        (session_dir / "variant_5.pt").touch()

        deleted = manager.clear_variant_files()

        assert deleted == 3, f"Should delete 3 .pt files, got {deleted}"
        assert not (session_dir / "variant_1.pt").exists()
        assert not (session_dir / "variant_2.pt").exists()
        assert not (session_dir / "variant_5.pt").exists()

        manager.cleanup()

    def test_clear_variant_files_deletes_variant_wav_files(self, tmp_path):
        """Test that clear_variant_files deletes variant .wav files."""
        from myvoice.utils.session_manager import SessionManager

        manager = SessionManager(base_dir=tmp_path)
        session_dir = manager.session_dir

        # Create variant .wav files
        (session_dir / "variant_1.wav").touch()
        (session_dir / "variant_3.wav").touch()

        deleted = manager.clear_variant_files()

        assert deleted == 2, f"Should delete 2 .wav files, got {deleted}"
        assert not (session_dir / "variant_1.wav").exists()
        assert not (session_dir / "variant_3.wav").exists()

        manager.cleanup()

    def test_clear_variant_files_preserves_session_directory(self, tmp_path):
        """Test that clear_variant_files keeps the session directory intact."""
        from myvoice.utils.session_manager import SessionManager

        manager = SessionManager(base_dir=tmp_path)
        session_dir = manager.session_dir

        # Create variant files
        (session_dir / "variant_1.pt").touch()
        (session_dir / "variant_1.wav").touch()

        manager.clear_variant_files()

        assert session_dir.exists(), "Session directory should still exist"
        assert session_dir.is_dir(), "Session directory should still be a directory"

        manager.cleanup()

    def test_clear_variant_files_preserves_other_files(self, tmp_path):
        """Test that clear_variant_files preserves non-variant files."""
        from myvoice.utils.session_manager import SessionManager

        manager = SessionManager(base_dir=tmp_path)
        session_dir = manager.session_dir

        # Create variant files and other files
        (session_dir / "variant_1.pt").touch()
        (session_dir / "variant_1.wav").touch()
        (session_dir / "metadata.json").touch()
        (session_dir / "preview.wav").touch()

        deleted = manager.clear_variant_files()

        assert deleted == 2, "Should only delete variant files"
        assert (session_dir / "metadata.json").exists(), "metadata.json should be preserved"
        assert (session_dir / "preview.wav").exists(), "preview.wav should be preserved"

        manager.cleanup()

    def test_clear_variant_files_returns_count(self, tmp_path):
        """Test that clear_variant_files returns correct deletion count."""
        from myvoice.utils.session_manager import SessionManager

        manager = SessionManager(base_dir=tmp_path)
        session_dir = manager.session_dir

        # Create 5 variant files (pt + wav for each)
        for i in range(1, 6):
            (session_dir / f"variant_{i}.pt").touch()
            (session_dir / f"variant_{i}.wav").touch()

        deleted = manager.clear_variant_files()

        assert deleted == 10, f"Should delete 10 files (5 .pt + 5 .wav), got {deleted}"

        manager.cleanup()

    def test_clear_variant_files_handles_empty_directory(self, tmp_path):
        """Test that clear_variant_files handles empty directory gracefully."""
        from myvoice.utils.session_manager import SessionManager

        manager = SessionManager(base_dir=tmp_path)

        deleted = manager.clear_variant_files()

        assert deleted == 0, "Should return 0 for empty directory"

        manager.cleanup()

    def test_clear_variant_files_can_be_called_multiple_times(self, tmp_path):
        """Test that clear_variant_files can be called multiple times safely."""
        from myvoice.utils.session_manager import SessionManager

        manager = SessionManager(base_dir=tmp_path)
        session_dir = manager.session_dir

        # Create files, clear, create again, clear again
        (session_dir / "variant_1.pt").touch()
        deleted1 = manager.clear_variant_files()
        assert deleted1 == 1

        (session_dir / "variant_2.pt").touch()
        (session_dir / "variant_2.wav").touch()
        deleted2 = manager.clear_variant_files()
        assert deleted2 == 2

        manager.cleanup()

    def test_clear_variant_files_not_allowed_after_cleanup(self, tmp_path):
        """Test that clear_variant_files returns 0 after full cleanup."""
        from myvoice.utils.session_manager import SessionManager

        manager = SessionManager(base_dir=tmp_path)
        manager.cleanup()

        # Should not fail, just return 0
        deleted = manager.clear_variant_files()
        assert deleted == 0, "Should return 0 after cleanup"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
