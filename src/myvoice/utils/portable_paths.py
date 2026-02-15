"""
Portable Path Management for MyVoice

This module provides utilities for managing application paths in a portable distribution.
All user data, settings, and voice files are stored relative to the
application directory, making the entire installation fully portable and self-contained.

Usage:
    from myvoice.utils.portable_paths import get_app_root, get_config_path, get_voice_files_path

    config = get_config_path()  # Returns Path to portable config directory
    voices = get_voice_files_path()  # Returns Path to portable voice_files directory
"""

import sys
import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_app_root() -> Path:
    """
    Get the root directory of the application.

    This function works correctly whether running:
    - From source (development)
    - As a PyInstaller executable (frozen)
    - In portable mode (zip distribution)

    Returns:
        Path: Absolute path to the application root directory

    Examples:
        # Development mode:
        # Returns: G:/MyVoicePublicInst/src

        # Frozen/Portable mode:
        # Returns: C:/Users/JohnDoe/MyVoice
    """
    if getattr(sys, 'frozen', False):
        # Running as compiled executable (PyInstaller)
        # sys._MEIPASS is the temporary folder where PyInstaller extracts files
        # We want the directory containing the .exe file
        if hasattr(sys, '_MEIPASS'):
            # Get the directory containing the executable
            app_root = Path(sys.executable).parent
        else:
            app_root = Path(sys.executable).parent
    else:
        # Running from source (development mode)
        # Go up from src/myvoice/utils/portable_paths.py to project root
        # Path: src/myvoice/utils/portable_paths.py -> project_root
        app_root = Path(__file__).parent.parent.parent.parent

    logger.debug(f"Application root directory: {app_root}")
    return app_root.resolve()


def get_data_root() -> Path:
    """
    Get the root directory for all user data in portable mode.

    In portable mode, all user data is stored in the same directory as the application
    executable, not in %LOCALAPPDATA% or other system directories.

    Returns:
        Path: Absolute path to the data root directory (same as app root in portable mode)
    """
    # In portable mode, data root is the same as app root
    return get_app_root()


def get_config_path() -> Path:
    """
    Get the path to the configuration directory for portable distribution.

    Returns:
        Path: Path to config directory (creates if doesn't exist)

    Example:
        Returns: C:/Users/JohnDoe/MyVoice/config
    """
    config_dir = get_data_root() / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Config directory: {config_dir}")
    return config_dir


def get_config_file_path() -> Path:
    """
    Get the full path to the settings.json file.

    Returns:
        Path: Full path to settings.json file

    Example:
        Returns: C:/Users/JohnDoe/MyVoice/config/settings.json
    """
    return get_config_path() / "settings.json"


def get_logs_path() -> Path:
    """
    Get the path to the logs directory for portable distribution.

    Returns:
        Path: Path to logs directory (creates if doesn't exist)

    Example:
        Returns: C:/Users/JohnDoe/MyVoice/logs
    """
    logs_dir = get_data_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Logs directory: {logs_dir}")
    return logs_dir


def get_voice_files_path() -> Path:
    """
    Get the path to the voice files directory for portable distribution.

    Returns:
        Path: Path to voice_files directory (creates if doesn't exist)

    Example:
        Returns: C:/Users/JohnDoe/MyVoice/voice_files
    """
    voice_files_dir = get_data_root() / "voice_files"
    voice_files_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Voice files directory: {voice_files_dir}")
    return voice_files_dir


def get_whisper_models_path() -> Path:
    """
    Get the path to the bundled Whisper models directory.

    Returns:
        Path: Path to whisper_models directory (creates if doesn't exist)

    Example:
        Returns: C:/Users/JohnDoe/MyVoice/whisper_models
    """
    whisper_models_dir = get_data_root() / "whisper_models"
    whisper_models_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Whisper models directory: {whisper_models_dir}")
    return whisper_models_dir


def get_templates_path() -> Path:
    """
    Get the path to the templates directory for custom categories.

    Returns:
        Path: Path to templates directory (creates if doesn't exist)

    Example:
        Returns: C:/Users/JohnDoe/MyVoice/config/templates
    """
    templates_dir = get_config_path() / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Templates directory: {templates_dir}")
    return templates_dir


def get_categories_file_path() -> Path:
    """
    Get the full path to the categories.json file.

    Returns:
        Path: Full path to categories.json file

    Example:
        Returns: C:/Users/JohnDoe/MyVoice/config/templates/categories.json
    """
    return get_templates_path() / "categories.json"


def get_ffmpeg_path() -> Optional[Path]:
    """
    Get the path to the bundled ffmpeg directory.

    Returns:
        Optional[Path]: Path to ffmpeg directory if it exists, None otherwise

    Example:
        Returns: C:/Users/JohnDoe/MyVoice/_internal/ffmpeg
    """
    # FFmpeg is bundled in _internal/ffmpeg by PyInstaller
    if getattr(sys, 'frozen', False):
        # Running as frozen executable
        internal_dir = get_app_root() / "_internal" / "ffmpeg"
        if internal_dir.exists():
            logger.debug(f"FFmpeg directory found: {internal_dir}")
            return internal_dir

    # Try app root level ffmpeg folder
    ffmpeg_dir = get_app_root() / "ffmpeg"
    if ffmpeg_dir.exists() and ffmpeg_dir.is_dir():
        logger.debug(f"FFmpeg directory found: {ffmpeg_dir}")
        return ffmpeg_dir

    logger.warning("FFmpeg directory not found")
    return None


def initialize_portable_directories() -> bool:
    """
    Initialize all required directories for portable distribution.

    This should be called early in the application startup to ensure all
    necessary directories exist.

    Returns:
        bool: True if all directories were created successfully, False otherwise
    """
    try:
        directories = [
            get_config_path(),
            get_logs_path(),
            get_voice_files_path(),
            get_whisper_models_path(),
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized directory: {directory}")

        # Check for optional directories
        ffmpeg = get_ffmpeg_path()
        if ffmpeg:
            logger.info(f"FFmpeg found at: {ffmpeg}")
        else:
            logger.warning("FFmpeg not found - audio processing may be limited")

        return True

    except Exception as e:
        logger.exception(f"Failed to initialize portable directories: {e}")
        return False


def is_portable_mode() -> bool:
    """
    Check if the application is running in portable mode.

    In portable mode:
    - Application is frozen (PyInstaller executable)
    - All data is stored relative to executable

    Returns:
        bool: True if running in portable mode, False otherwise
    """
    return getattr(sys, 'frozen', False)


def get_resource_path(relative_path: str) -> Path:
    """
    Get the absolute path to a bundled resource file.

    This works correctly whether running from source or as a frozen executable.

    Args:
        relative_path: Relative path to the resource from the app root

    Returns:
        Path: Absolute path to the resource

    Example:
        icon = get_resource_path("icon/MyVoice.png")
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running as frozen executable - resources are in _MEIPASS temp folder
        base_path = Path(sys._MEIPASS)
    else:
        # Running from source
        base_path = get_app_root()

    resource_path = base_path / relative_path
    logger.debug(f"Resource path for '{relative_path}': {resource_path}")
    return resource_path


def ensure_portable_compatibility():
    """
    Ensure the environment is properly configured for portable operation.

    This function:
    - Creates all necessary directories
    - Validates critical paths exist
    - Logs diagnostic information

    Returns:
        bool: True if environment is ready, False if critical issues found
    """
    try:
        logger.info("=== Portable Mode Diagnostics ===")
        logger.info(f"Portable mode: {is_portable_mode()}")
        logger.info(f"Application root: {get_app_root()}")
        logger.info(f"Data root: {get_data_root()}")
        logger.info(f"Config path: {get_config_path()}")
        logger.info(f"Logs path: {get_logs_path()}")
        logger.info(f"Voice files path: {get_voice_files_path()}")
        logger.info(f"FFmpeg path: {get_ffmpeg_path()}")
        logger.info("=== End Diagnostics ===")

        # Initialize all directories
        return initialize_portable_directories()

    except Exception as e:
        logger.exception(f"Failed to ensure portable compatibility: {e}")
        return False
