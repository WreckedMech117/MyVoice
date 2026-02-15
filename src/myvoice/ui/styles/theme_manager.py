"""
MyVoice Theme Manager

This module provides a centralized theme management system for consistent
styling across the MyVoice application. It handles loading, switching, and
applying different stylesheets.

Features:
- Theme loading and caching
- Dynamic theme switching
- Theme validation
- Custom property support
- Fallback handling
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, List
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import QObject, pyqtSignal

# Windows-specific imports for dark mode support
if sys.platform == "win32":
    try:
        import ctypes
        from ctypes import wintypes
        _windows_available = True
    except ImportError:
        _windows_available = False
else:
    _windows_available = False


def set_windows_dark_mode(widget: QWidget, enable: bool = True) -> bool:
    """
    Enable Windows dark mode for widget title bar on Windows 10/11.

    This function uses Windows APIs to set the title bar to dark mode,
    which affects the window frame and title bar appearance.

    Args:
        widget: The widget/window to apply dark mode to
        enable: Whether to enable (True) or disable (False) dark mode

    Returns:
        True if successful, False if not supported or failed
    """
    if not _windows_available or sys.platform != "win32":
        return False

    try:
        # Get the window handle
        hwnd = widget.winId()
        if not hwnd:
            return False

        # Windows 10 version 1903+ and Windows 11 support
        # DWMWA_USE_IMMERSIVE_DARK_MODE = 20 (Windows 11)
        # DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19 (Windows 10 1903-2004)

        # Try Windows 11 attribute first
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        value = ctypes.c_int(1 if enable else 0)

        result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
            ctypes.wintypes.HWND(hwnd),
            ctypes.wintypes.DWORD(DWMWA_USE_IMMERSIVE_DARK_MODE),
            ctypes.byref(value),
            ctypes.sizeof(value)
        )

        if result == 0:  # S_OK
            return True

        # Try older Windows 10 attribute as fallback
        DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19
        result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
            ctypes.wintypes.HWND(hwnd),
            ctypes.wintypes.DWORD(DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1),
            ctypes.byref(value),
            ctypes.sizeof(value)
        )

        return result == 0  # S_OK

    except Exception as e:
        logging.getLogger(__name__).debug(f"Failed to set Windows dark mode: {e}")
        return False


def is_windows_dark_mode_supported() -> bool:
    """
    Check if Windows dark mode is supported on this system.

    Returns:
        True if Windows dark mode APIs are available
    """
    if not _windows_available or sys.platform != "win32":
        return False

    try:
        # Check if we can access dwmapi functions
        return hasattr(ctypes.windll, 'dwmapi') and hasattr(ctypes.windll.dwmapi, 'DwmSetWindowAttribute')
    except Exception:
        return False


def is_high_contrast_mode() -> bool:
    """
    Check if Windows high contrast mode is enabled.

    Story 7.5: NFR20 - Support for high contrast mode for accessibility.

    Returns:
        True if high contrast mode is enabled
    """
    if not _windows_available or sys.platform != "win32":
        return False

    try:
        # SPI_GETHIGHCONTRAST = 0x0042
        # HIGHCONTRASTW structure has flags at offset 4 (HCF_HIGHCONTRASTON = 0x1)
        class HIGHCONTRASTW(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.c_uint),
                ("dwFlags", ctypes.c_uint),
                ("lpszDefaultScheme", ctypes.c_wchar_p)
            ]

        hc = HIGHCONTRASTW()
        hc.cbSize = ctypes.sizeof(HIGHCONTRASTW)

        result = ctypes.windll.user32.SystemParametersInfoW(
            0x0042,  # SPI_GETHIGHCONTRAST
            ctypes.sizeof(HIGHCONTRASTW),
            ctypes.byref(hc),
            0
        )

        if result:
            # HCF_HIGHCONTRASTON = 0x1
            return bool(hc.dwFlags & 0x1)
        return False

    except Exception as e:
        logging.getLogger(__name__).debug(f"Failed to check high contrast mode: {e}")
        return False


class ThemeManager(QObject):
    """
    Centralized theme management system.

    Handles loading, caching, and applying stylesheets across the application.
    Supports multiple theme types and dynamic switching.

    Signals:
        theme_changed: Emitted when theme is successfully changed
        theme_load_failed: Emitted when theme loading fails
    """

    # Signals
    theme_changed = pyqtSignal(str)  # theme_name
    theme_load_failed = pyqtSignal(str, str)  # theme_name, error_message

    def __init__(self):
        """Initialize the theme manager."""
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Theme cache and state
        self._theme_cache: Dict[str, str] = {}
        self._current_theme: Optional[str] = None
        self._styles_directory = Path(__file__).parent

        # Available themes configuration
        self._available_themes = {
            "dark": {
                "file": "minimal_dark.qss",
                "display_name": "Dark",
                "description": "Professional dark theme optimized for low-light environments"
            },
            "light": {
                "file": "minimal_light.qss",
                "display_name": "Light",
                "description": "Clean light theme with high contrast"
            }
        }

        self.logger.debug("ThemeManager initialized")

    def get_available_themes(self) -> List[Dict[str, str]]:
        """
        Get list of available themes.

        Returns:
            List of theme info dictionaries with keys: name, display_name, description
        """
        themes = []
        for theme_name, theme_info in self._available_themes.items():
            themes.append({
                "name": theme_name,
                "display_name": theme_info["display_name"],
                "description": theme_info["description"]
            })
        return themes

    def load_theme(self, theme_name: str, force_reload: bool = False) -> bool:
        """
        Load a theme stylesheet.

        Args:
            theme_name: Name of the theme to load
            force_reload: Whether to force reload from file (bypass cache)

        Returns:
            True if theme loaded successfully, False otherwise
        """
        try:
            # Check if theme exists
            if theme_name not in self._available_themes:
                self.logger.error(f"Theme '{theme_name}' not found in available themes")
                self.theme_load_failed.emit(theme_name, "Theme not found")
                return False

            # Check cache first (unless force reload)
            if not force_reload and theme_name in self._theme_cache:
                self.logger.debug(f"Using cached theme: {theme_name}")
                return True

            # Load single-file theme
            theme_config = self._available_themes[theme_name]
            theme_file = theme_config["file"]
            theme_path = self._styles_directory / theme_file

            if not theme_path.exists():
                error_msg = f"Theme file not found: {theme_path}"
                self.logger.error(error_msg)
                self.theme_load_failed.emit(theme_name, error_msg)
                return False

            # Read and validate stylesheet
            stylesheet_content = self._load_stylesheet_file(theme_path)
            if stylesheet_content is None:
                error_msg = f"Failed to read theme file: {theme_path}"
                self.logger.error(error_msg)
                self.theme_load_failed.emit(theme_name, error_msg)
                return False

            # Cache the theme
            self._theme_cache[theme_name] = stylesheet_content
            self.logger.info(f"Successfully loaded theme: {theme_name}")
            return True

        except Exception as e:
            error_msg = f"Error loading theme '{theme_name}': {str(e)}"
            self.logger.exception(error_msg)
            self.theme_load_failed.emit(theme_name, error_msg)
            return False

    def apply_theme(self, theme_name: str, target_widget: Optional[QWidget] = None) -> bool:
        """
        Apply a theme to the application or specific widget.

        Args:
            theme_name: Name of the theme to apply
            target_widget: Specific widget to apply theme to (None for application-wide)

        Returns:
            True if theme applied successfully, False otherwise
        """
        try:
            # Load theme if not cached
            if not self.load_theme(theme_name):
                return False

            # Get stylesheet content
            stylesheet = self._theme_cache[theme_name]

            # Apply theme
            if target_widget is not None:
                # Apply to specific widget
                target_widget.setStyleSheet(stylesheet)
                self.logger.debug(f"Applied theme '{theme_name}' to widget: {target_widget.__class__.__name__}")

                # Apply Windows dark mode for dark themes
                if theme_name == "dark" and is_windows_dark_mode_supported():
                    if set_windows_dark_mode(target_widget, True):
                        self.logger.debug(f"Applied Windows dark mode to {target_widget.__class__.__name__}")
                    else:
                        self.logger.debug(f"Failed to apply Windows dark mode to {target_widget.__class__.__name__}")

            else:
                # Apply application-wide
                app = QApplication.instance()
                if app:
                    app.setStyleSheet(stylesheet)
                    self.logger.debug(f"Applied theme '{theme_name}' application-wide")

                    # Apply Windows dark mode to all top-level widgets for dark themes
                    if theme_name == "dark" and is_windows_dark_mode_supported():
                        for widget in app.topLevelWidgets():
                            if widget.isWindow() and widget.isVisible():
                                if set_windows_dark_mode(widget, True):
                                    self.logger.debug(f"Applied Windows dark mode to {widget.__class__.__name__}")
                else:
                    self.logger.error("No QApplication instance found")
                    return False

            # Update current theme
            self._current_theme = theme_name
            self.theme_changed.emit(theme_name)
            return True

        except Exception as e:
            error_msg = f"Error applying theme '{theme_name}': {str(e)}"
            self.logger.exception(error_msg)
            self.theme_load_failed.emit(theme_name, error_msg)
            return False

    def switch_theme(self, theme_name: str, target_widget: Optional[QWidget] = None) -> bool:
        """
        Switch to a different theme.

        Args:
            theme_name: Name of the theme to switch to
            target_widget: Specific widget to apply theme to (None for application-wide)

        Returns:
            True if theme switch successful, False otherwise
        """
        if theme_name == self._current_theme:
            self.logger.debug(f"Theme '{theme_name}' is already active")
            return True

        self.logger.info(f"Switching theme from '{self._current_theme}' to '{theme_name}'")
        return self.apply_theme(theme_name, target_widget)

    def get_current_theme(self) -> Optional[str]:
        """
        Get the currently active theme name.

        Returns:
            Current theme name or None if no theme is active
        """
        return self._current_theme

    def reload_current_theme(self, target_widget: Optional[QWidget] = None) -> bool:
        """
        Reload the current theme from file.

        Args:
            target_widget: Specific widget to apply theme to (None for application-wide)

        Returns:
            True if reload successful, False otherwise
        """
        if self._current_theme is None:
            self.logger.warning("No current theme to reload")
            return False

        self.logger.info(f"Reloading current theme: {self._current_theme}")
        return self.apply_theme(self._current_theme, target_widget)

    def clear_cache(self):
        """Clear the theme cache."""
        self._theme_cache.clear()
        self.logger.debug("Theme cache cleared")

    def get_theme_info(self, theme_name: str) -> Optional[Dict[str, str]]:
        """
        Get information about a specific theme.

        Args:
            theme_name: Name of the theme

        Returns:
            Theme info dictionary or None if theme not found
        """
        if theme_name in self._available_themes:
            info = self._available_themes[theme_name].copy()
            info["name"] = theme_name
            info["is_loaded"] = theme_name in self._theme_cache
            info["is_current"] = theme_name == self._current_theme
            return info
        return None

    def _load_stylesheet_file(self, file_path: Path) -> Optional[str]:
        """
        Load stylesheet content from file with error handling.

        Args:
            file_path: Path to the stylesheet file

        Returns:
            Stylesheet content or None if loading failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Basic validation
            if not content.strip():
                self.logger.warning(f"Stylesheet file is empty: {file_path}")
                return ""

            # Log file size for debugging
            size_kb = len(content) / 1024
            self.logger.debug(f"Loaded stylesheet: {file_path.name} ({size_kb:.1f} KB)")

            return content

        except FileNotFoundError:
            self.logger.error(f"Stylesheet file not found: {file_path}")
            return None
        except PermissionError:
            self.logger.error(f"Permission denied reading stylesheet: {file_path}")
            return None
        except UnicodeDecodeError as e:
            self.logger.error(f"Unicode decode error in stylesheet {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.exception(f"Unexpected error loading stylesheet {file_path}: {e}")
            return None

    def validate_theme(self, theme_name: str) -> Dict[str, bool]:
        """
        Validate a theme and its dependencies.

        Args:
            theme_name: Name of the theme to validate

        Returns:
            Dictionary with validation results
        """
        results = {
            "exists": False,
            "file_readable": False,
            "content_valid": False,
            "can_load": False
        }

        try:
            # Check if theme exists in configuration
            if theme_name not in self._available_themes:
                return results

            results["exists"] = True

            # Check if theme file exists and is readable
            theme_config = self._available_themes[theme_name]
            theme_file = theme_config["file"]
            theme_path = self._styles_directory / theme_file

            if not theme_path.exists():
                return results

            results["file_readable"] = True

            # Try to load content and validate theme
            try:
                if self.load_theme(theme_name):
                    results["content_valid"] = True
                    results["can_load"] = True
            except Exception:
                # Loading failed, results remain False
                pass

        except Exception as e:
            self.logger.exception(f"Error validating theme '{theme_name}': {e}")

        return results


# Global theme manager instance
_theme_manager: Optional[ThemeManager] = None


def get_theme_manager() -> ThemeManager:
    """
    Get the global theme manager instance.

    Returns:
        ThemeManager instance (singleton)
    """
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager()
    return _theme_manager


def apply_theme_to_widget(widget: QWidget, theme_name: str) -> bool:
    """
    Convenience function to apply a theme to a specific widget.

    Args:
        widget: Widget to apply theme to
        theme_name: Name of the theme to apply

    Returns:
        True if successful, False otherwise
    """
    theme_manager = get_theme_manager()
    return theme_manager.apply_theme(theme_name, widget)


def set_application_theme(theme_name: str) -> bool:
    """
    Convenience function to set the application-wide theme.

    Args:
        theme_name: Name of the theme to apply

    Returns:
        True if successful, False otherwise
    """
    theme_manager = get_theme_manager()
    return theme_manager.apply_theme(theme_name)


def apply_windows_dark_mode_to_widget(widget: QWidget) -> bool:
    """
    Convenience function to apply Windows dark mode to a specific widget.

    Args:
        widget: Widget to apply Windows dark mode to

    Returns:
        True if successful, False if not supported or failed
    """
    return set_windows_dark_mode(widget, True)