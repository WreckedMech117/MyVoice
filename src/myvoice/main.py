#!/usr/bin/env python3
"""
MyVoice Application Entry Point (V2)

This module serves as the main entry point for the MyVoice desktop application.
It initializes the PyQt6 application framework and starts the application controller.

V2 uses embedded Qwen3-TTS - no external services required.

Usage:
    python -m myvoice.main
    or
    python main.py
"""

import sys
import os
import logging
import asyncio
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# CRITICAL: Register DLL directories BEFORE importing torch (Python 3.8+ Windows requirement)
# This is required for portable Python environments where DLL paths aren't in system PATH
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    # Add torch lib directory (contains c10.dll, torch_cpu.dll, etc.)
    _torch_lib = Path(__file__).parent.parent.parent / "python310" / "Lib" / "site-packages" / "torch" / "lib"
    if _torch_lib.exists():
        os.add_dll_directory(str(_torch_lib))

    # Add CUDA toolkit bin directories (required for GPU support)
    _cuda_paths = [
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp"),
    ]
    for _cuda_path in _cuda_paths:
        if _cuda_path.exists():
            os.add_dll_directory(str(_cuda_path))

# CRITICAL: Import torch BEFORE PyQt6 to avoid DLL loading conflicts
# PyTorch CUDA DLLs must be loaded before Qt's DLLs on Windows
try:
    import torch
except (ImportError, OSError):
    # torch not installed, DLL loading failed, or other OS-level error
    # Will be handled later when TTS service initializes
    pass

from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPixmap, QColor
from qasync import QEventLoop

from myvoice.app import MyVoiceApp
from myvoice.utils.error_handler import get_exception_handler


def setup_application() -> QApplication:
    """
    Initialize and configure the PyQt6 QApplication.

    Returns:
        QApplication: Configured PyQt6 application instance
    """
    # Enable high DPI support (PyQt6 automatic, but set policy explicitly)
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    # Create the application instance
    app = QApplication(sys.argv)

    # Set application metadata
    app.setApplicationName("MyVoice")
    app.setApplicationVersion("2.0.0")
    app.setApplicationDisplayName("MyVoice TTS Desktop")
    app.setOrganizationName("MyVoice")
    app.setOrganizationDomain("myvoice.local")

    # Set application icon if available
    # Try multiple icon locations
    icon_paths = [
        Path(__file__).parent.parent / "icon" / "MyVoice.png",  # src/icon/MyVoice.png
        Path(__file__).parent.parent.parent / "resources" / "icon.ico",  # resources/icon.ico
        Path(__file__).parent.parent.parent / "resources" / "icon.png",  # resources/icon.png
    ]

    for icon_path in icon_paths:
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
            break

    # Windows-specific: Set AppUserModelID for proper taskbar icon display
    if sys.platform == "win32":
        try:
            import ctypes
            app_id = "MyVoice.TTSDesktop.1.0"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to set AppUserModelID: {e}")

    return app


def create_splash_screen() -> QSplashScreen:
    """
    Create and display a splash screen with the MyVoice branding.

    Returns:
        QSplashScreen: The splash screen instance
    """
    import sys
    import os

    # Find the splash image - handle both development and PyInstaller paths
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        base_path = Path(sys._MEIPASS)
        splash_paths = [
            base_path / "icon" / "MyVoice_Splash.png",
        ]
    else:
        # Running in development
        splash_paths = [
            Path(__file__).parent.parent / "icon" / "MyVoice_Splash.png",
        ]

    splash_pixmap = None
    for splash_path in splash_paths:
        if splash_path.exists():
            splash_pixmap = QPixmap(str(splash_path))
            if not splash_pixmap.isNull():
                print(f"Splash screen loaded from: {splash_path}")
                break
            else:
                print(f"Failed to load splash image from: {splash_path}")
        else:
            print(f"Splash path does not exist: {splash_path}")

    # Fallback if splash image not found - create a simple colored splash
    if splash_pixmap is None or splash_pixmap.isNull():
        print("Using fallback colored splash screen")
        splash_pixmap = QPixmap(600, 400)
        splash_pixmap.fill(QColor("#2c3e50"))  # Dark blue-grey

    # Create splash screen with always-on-top flag
    splash = QSplashScreen(splash_pixmap, Qt.WindowType.WindowStaysOnTopHint)

    # Only set mask if image has transparency, otherwise skip to show full image
    if not splash_pixmap.mask().isNull():
        splash.setMask(splash_pixmap.mask())

    # Show the splash screen
    splash.show()

    return splash


def setup_logging():
    """Configure application logging for portable distribution."""
    from myvoice.utils.portable_paths import get_logs_path

    # Use portable logs directory (same folder as executable)
    log_dir = get_logs_path()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "myvoice.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )




def _setup_exception_handler_ui(main_window, logger: logging.Logger):
    """
    Set up UI callbacks for the global exception handler (Story 7.6).

    Connects the exception handler to display dialogs via the main window.

    Args:
        main_window: The main application window
        logger: Logger instance
    """
    from myvoice.ui.components.critical_error_dialog import CriticalErrorDialog

    exception_handler = get_exception_handler()

    def show_error_dialog(title: str, message: str, details: str):
        """Show a recoverable error dialog."""
        CriticalErrorDialog.show_error(
            title=title,
            message=message,
            details=details,
            allow_continue=True,
            parent=main_window
        )

    def show_critical_dialog(title: str, message: str, details: str):
        """Show a critical error dialog that may require exit."""
        user_wants_continue = CriticalErrorDialog.show_error(
            title=title,
            message=message,
            details=details,
            allow_continue=False,  # Critical errors: Exit button is default
            parent=main_window
        )
        if not user_wants_continue:
            logger.info("User chose to exit after critical error")
            # Application will exit via reject()

    exception_handler.set_error_callback(show_error_dialog)
    exception_handler.set_critical_callback(show_critical_dialog)

    logger.info("Exception handler UI callbacks configured")


async def async_main(qt_app: QApplication, logger: logging.Logger) -> int:
    """
    Async main application coroutine with qasync event loop.

    This coroutine runs the entire application lifecycle in a single shared
    event loop, preventing "Event loop is closed" errors from occurring when
    services are started in one loop and stopped in another.

    Args:
        qt_app: QApplication instance
        logger: Logger instance

    Returns:
        int: Application exit code (0 for success, non-zero for error)
    """
    # Setup shutdown event
    app_close = asyncio.Event()
    qt_app.aboutToQuit.connect(app_close.set)

    try:
        # Create and show splash screen
        splash = create_splash_screen()
        qt_app.processEvents()  # Process events to ensure splash is displayed

        # Update splash with initial status
        splash.showMessage("Initializing MyVoice...",
                          Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
                          QColor("white"))
        qt_app.processEvents()

        # Create the application controller
        myvoice_app = MyVoiceApp(qt_app)

        # Update splash with loading status
        splash.showMessage("Loading audio modules...",
                          Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
                          QColor("white"))
        qt_app.processEvents()

        # Initialize the application asynchronously (NEW - uses shared loop)
        success = await myvoice_app.initialize_async()
        if not success:
            logger.error("Failed to initialize MyVoice application")
            splash.close()
            return 1

        # Update splash with final status
        splash.showMessage("Starting application...",
                          Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
                          QColor("white"))
        qt_app.processEvents()

        # Close splash screen when main window is ready
        # The main window is shown by myvoice_app.initialize_async() -> _initialize_ui()
        if hasattr(myvoice_app, '_main_window') and myvoice_app._main_window:
            splash.finish(myvoice_app._main_window)

            # Story 7.6: Set up UI callbacks for global exception handler
            _setup_exception_handler_ui(myvoice_app._main_window, logger)
        else:
            splash.close()

        logger.info("MyVoice application initialized successfully")

        # Wait for application quit signal
        await app_close.wait()

        # QA Round 2 Item #8: Force exit failsafe BEFORE cleanup
        # Start timer first so it acts as hard timeout for entire cleanup process
        import threading
        import os as _os

        def force_exit():
            logger.warning("Force exit triggered - cleanup timed out after 10 seconds")
            _os._exit(0)

        # 10 second hard timeout for entire cleanup process
        exit_timer = threading.Timer(10.0, force_exit)
        exit_timer.daemon = True
        exit_timer.start()
        logger.info("Force exit failsafe armed (10s timeout)")

        # Cleanup in SAME event loop as initialization (CRITICAL FIX)
        logger.info("Application shutting down - cleaning up services...")
        try:
            await asyncio.wait_for(myvoice_app.cleanup_async(), timeout=8.0)
        except asyncio.TimeoutError:
            logger.warning("Cleanup timed out after 8 seconds, forcing exit")
            _os._exit(0)

        # Cancel the timer if we got here naturally
        exit_timer.cancel()

        logger.info("Cleanup complete, exiting normally...")
        return 0

    except Exception as e:
        logger.exception(f"Fatal error in async_main: {e}")
        return 1


def main() -> int:
    """
    Main application entry point.

    Returns:
        int: Application exit code (0 for success, non-zero for error)
    """
    # Initialize logger early so it's available in error handling
    logger = None

    try:
        # Setup logging first
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Starting MyVoice V2 application with qasync event loop")

        # Story 7.6: Install global exception handler early to catch all unhandled exceptions
        exception_handler = get_exception_handler()
        exception_handler.install()
        logger.info("Global exception handler installed")

        # Create and configure the PyQt6 application
        qt_app = setup_application()

        # Create qasync event loop (SINGLE SHARED LOOP for entire app lifecycle)
        loop = QEventLoop(qt_app)
        asyncio.set_event_loop(loop)

        logger.info("qasync event loop created and set as default")

        # Run async main coroutine in the qasync event loop
        with loop:
            exit_code = loop.run_until_complete(async_main(qt_app, logger))

        logger.info(f"MyVoice application exited with code: {exit_code}")
        return exit_code

    except Exception as e:
        print(f"Fatal error starting MyVoice: {e}", file=sys.stderr)
        if logger:
            logger.exception("Fatal error in main()")
        else:
            logging.exception("Fatal error in main()")
        return 1


if __name__ == "__main__":
    sys.exit(main())