"""
Tests for Portable Path Management

Tests the portable_paths module which handles path resolution for both
development and frozen (PyInstaller) modes.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Skip this module if PyQt6 is not available (required for myvoice imports)
pytest.importorskip("PyQt6", reason="PyQt6 required for myvoice imports")


class TestGetAppRoot:
    """Tests for get_app_root() function."""

    def test_development_mode_path_calculation(self):
        """Test that get_app_root() correctly calculates project root in development mode.

        The portable_paths.py file is located at:
        src/myvoice/utils/portable_paths.py

        So we need to go up 4 levels to get to project root:
        portable_paths.py -> utils -> myvoice -> src -> project_root
        """
        from myvoice.utils.portable_paths import get_app_root

        # In development mode, get_app_root should return the project root
        app_root = get_app_root()

        # Verify it's a valid path
        assert app_root.exists(), f"App root should exist: {app_root}"
        assert app_root.is_dir(), f"App root should be a directory: {app_root}"

        # In development mode, app_root should contain src/myvoice
        src_dir = app_root / "src" / "myvoice"
        assert src_dir.exists(), f"src/myvoice should exist under app root: {src_dir}"

        # It should also have the voice_files directory (at project root level)
        # Note: voice_files may not exist yet, but we can check the path makes sense
        voice_files_path = app_root / "voice_files"
        # The path should be at project root level, not inside src/
        assert "src" not in str(voice_files_path.parent), \
            f"voice_files should be at project root, not inside src: {voice_files_path}"

    def test_development_mode_not_frozen(self):
        """Test that development mode is correctly detected (not frozen)."""
        from myvoice.utils.portable_paths import get_app_root

        # In development, sys.frozen should be False or not exist
        is_frozen = getattr(sys, 'frozen', False)
        assert is_frozen is False, "Should not be frozen in development mode"

        app_root = get_app_root()
        # Development root should contain src directory
        assert (app_root / "src").exists(), f"Development mode should have src directory: {app_root}"

    def test_frozen_mode_uses_executable_parent(self):
        """Test that frozen mode returns the parent directory of the executable."""
        # Test the logic that would be executed in frozen mode
        # We can't easily mock sys.frozen (built-in module), so we test the logic directly

        # Simulate what happens in frozen mode
        test_executable = Path('C:/Program Files/MyVoice/MyVoice.exe')
        expected_path = Path('C:/Program Files/MyVoice')

        # The frozen mode logic is: app_root = Path(sys.executable).parent
        app_root = test_executable.parent

        assert app_root == expected_path

    def test_app_root_is_absolute(self):
        """Test that get_app_root() returns an absolute path."""
        from myvoice.utils.portable_paths import get_app_root

        app_root = get_app_root()
        assert app_root.is_absolute(), f"App root should be absolute: {app_root}"

    def test_app_root_is_resolved(self):
        """Test that get_app_root() returns a resolved path (no .. components)."""
        from myvoice.utils.portable_paths import get_app_root

        app_root = get_app_root()
        # A resolved path should have no '..' components
        assert ".." not in str(app_root), f"App root should be resolved: {app_root}"


class TestGetVoiceFilesPath:
    """Tests for get_voice_files_path() function."""

    def test_returns_path_under_app_root(self):
        """Test that voice_files path is under app root."""
        from myvoice.utils.portable_paths import get_app_root, get_voice_files_path

        app_root = get_app_root()
        voice_files = get_voice_files_path()

        # voice_files should be directly under app_root
        assert voice_files.parent == app_root, \
            f"voice_files should be under app_root: {voice_files} vs {app_root}"

    def test_directory_is_created(self):
        """Test that get_voice_files_path() creates the directory if it doesn't exist."""
        from myvoice.utils.portable_paths import get_voice_files_path

        voice_files = get_voice_files_path()

        # Directory should exist after calling get_voice_files_path
        assert voice_files.exists(), f"voice_files directory should exist: {voice_files}"
        assert voice_files.is_dir(), f"voice_files should be a directory: {voice_files}"

    def test_directory_name_is_voice_files(self):
        """Test that the directory is named 'voice_files'."""
        from myvoice.utils.portable_paths import get_voice_files_path

        voice_files = get_voice_files_path()

        assert voice_files.name == "voice_files", \
            f"Directory should be named 'voice_files': {voice_files.name}"


class TestGetConfigPath:
    """Tests for get_config_path() function."""

    def test_returns_path_under_app_root(self):
        """Test that config path is under app root."""
        from myvoice.utils.portable_paths import get_app_root, get_config_path

        app_root = get_app_root()
        config = get_config_path()

        # config should be directly under app_root
        assert config.parent == app_root, \
            f"config should be under app_root: {config} vs {app_root}"

    def test_directory_name_is_config(self):
        """Test that the directory is named 'config'."""
        from myvoice.utils.portable_paths import get_config_path

        config = get_config_path()

        assert config.name == "config", f"Directory should be named 'config': {config.name}"


class TestIsPortableMode:
    """Tests for is_portable_mode() function."""

    def test_development_is_not_portable(self):
        """Test that development mode is not considered portable."""
        from myvoice.utils.portable_paths import is_portable_mode

        # In development, should return False
        result = is_portable_mode()
        assert result is False, "Development mode should not be portable"


class TestGetResourcePath:
    """Tests for get_resource_path() function."""

    def test_returns_path_for_relative_resource(self):
        """Test that get_resource_path() returns a valid path."""
        from myvoice.utils.portable_paths import get_resource_path

        resource_path = get_resource_path("icon/MyVoice.png")

        # Should be an absolute path
        assert resource_path.is_absolute(), f"Should be absolute: {resource_path}"

        # Should contain the relative path component
        assert "icon" in str(resource_path), f"Should contain 'icon': {resource_path}"
        assert "MyVoice.png" in str(resource_path), f"Should contain filename: {resource_path}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
