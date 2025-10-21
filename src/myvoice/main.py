#!/usr/bin/env python3
"""
MyVoice Application Entry Point

This module serves as the main entry point for the MyVoice desktop application.
It initializes the PyQt6 application framework and starts the application controller.

Usage:
    python -m myvoice.main
    or
    python main.py
"""

import sys
import logging
import atexit
import asyncio
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtCore import Qt, QDir
from PyQt6.QtGui import QIcon, QPixmap, QColor
from qasync import QEventLoop

from myvoice.app import MyVoiceApp

# Windows-specific imports for Job Objects (guaranteed process cleanup)
if sys.platform == "win32":
    import win32job
    import win32process
    import win32api
    import win32con

# Global process and job references for GPT-SoVITS cleanup on shutdown
_gptsovits_process = None
_gptsovits_job = None  # Windows Job Object handle for guaranteed process tree termination


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
    app.setApplicationVersion("1.0.0")
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
    """Configure application logging."""
    # Use local logs directory relative to project root
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "myvoice.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def _start_gptsovits(logger: logging.Logger) -> bool:
    """
    Start GPT-SoVITS in the background if it's installed.

    Returns:
        bool: True if GPT-SoVITS was started (or already running), False if not installed
    """
    import subprocess
    import os

    # Find GPT-SoVITS installation directory
    # Check relative to MyVoice.exe location
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        base_path = Path(sys.executable).parent
    else:
        # Running in development
        base_path = Path(__file__).parent.parent.parent

    gptsovits_dir = base_path / "GPT-SoVITS"
    go_api_bat = gptsovits_dir / "go-api.bat"

    if not go_api_bat.exists():
        logger.info(f"GPT-SoVITS not found at {gptsovits_dir} - voice cloning disabled")
        return False

    logger.info(f"Found GPT-SoVITS at {gptsovits_dir}")

    # Patch GPT-SoVITS for CPU fallback if not already patched
    try:
        from myvoice.utils.patch_gptsovits_device import GPTSoVITSPatcher
        patcher = GPTSoVITSPatcher(gptsovits_dir)
        if not patcher.is_already_patched():
            logger.info("Patching GPT-SoVITS for CPU fallback support...")
            if patcher.patch_all():
                logger.info("GPT-SoVITS patched successfully for CPU fallback")
            else:
                logger.warning("GPT-SoVITS patching failed - may not work on non-GPU systems")
        else:
            logger.debug("GPT-SoVITS already patched for CPU fallback")
    except Exception as e:
        logger.warning(f"Error checking/applying GPT-SoVITS patches: {e}")

    # Check if already running
    import requests
    try:
        response = requests.get("http://localhost:9880", timeout=1)
        logger.info("GPT-SoVITS is already running")
        return True
    except:
        pass  # Not running, we'll start it

    # Start GPT-SoVITS in background with Windows Job Object for guaranteed cleanup
    global _gptsovits_process, _gptsovits_job
    try:
        logger.info("Starting GPT-SoVITS API server...")

        if sys.platform == "win32":
            # Create Windows Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE flag
            # This guarantees ALL child processes are terminated when the job handle is closed
            try:
                _gptsovits_job = win32job.CreateJobObject(None, "")

                # Query current job information
                extended_info = win32job.QueryInformationJobObject(
                    _gptsovits_job,
                    win32job.JobObjectExtendedLimitInformation
                )

                # Set the kill-on-close flag to ensure automatic termination
                extended_info['BasicLimitInformation']['LimitFlags'] = (
                    win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                )

                # Apply the updated limits to the job object
                win32job.SetInformationJobObject(
                    _gptsovits_job,
                    win32job.JobObjectExtendedLimitInformation,
                    extended_info
                )

                logger.info("Created Windows Job Object with kill-on-close flag")
            except Exception as e:
                logger.warning(f"Failed to create Job Object: {e} - falling back to manual cleanup")
                _gptsovits_job = None

            # Create hidden console window with no output redirection (preserves audio access)
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE

            # Use CREATE_NO_WINDOW instead of CREATE_NEW_CONSOLE to avoid Job inheritance issues
            # and CREATE_BREAKAWAY_FROM_JOB to allow us to assign to our custom job
            creation_flags = subprocess.CREATE_NO_WINDOW | win32process.CREATE_BREAKAWAY_FROM_JOB

            _gptsovits_process = subprocess.Popen(
                [str(go_api_bat)],
                cwd=str(gptsovits_dir),
                startupinfo=startupinfo,
                creationflags=creation_flags
                # DO NOT redirect stdout/stderr - this breaks audio!
            )

            # Assign the process to our Job Object for guaranteed cleanup
            if _gptsovits_job is not None:
                try:
                    # Get process handle with necessary permissions
                    perms = win32con.PROCESS_TERMINATE | win32con.PROCESS_SET_QUOTA
                    hProcess = win32api.OpenProcess(perms, False, _gptsovits_process.pid)

                    # Assign process to job object
                    win32job.AssignProcessToJobObject(_gptsovits_job, hProcess)

                    logger.info(f"Assigned GPT-SoVITS process (PID: {_gptsovits_process.pid}) to Job Object")
                except Exception as e:
                    logger.warning(f"Failed to assign process to Job Object: {e}")
                    # Continue anyway - process will still run, but may need manual cleanup
        else:
            # Unix: standard subprocess without job objects
            _gptsovits_process = subprocess.Popen(
                [str(go_api_bat)],
                cwd=str(gptsovits_dir)
                # DO NOT redirect stdout/stderr - this breaks audio!
            )

        logger.info("GPT-SoVITS API server started in background")
        return True
    except Exception as e:
        logger.error(f"Failed to start GPT-SoVITS: {e}")
        return False


def _stop_gptsovits(logger: logging.Logger) -> None:
    """
    Stop GPT-SoVITS process using Windows Job Object for guaranteed cleanup.

    On Windows, the Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE flag
    automatically terminates ALL child processes when the job handle is closed.
    This replaces all manual cleanup methods (taskkill, PowerShell, etc.).
    """
    global _gptsovits_process, _gptsovits_job

    if _gptsovits_process is None:
        logger.debug("GPT-SoVITS was not started by us, not terminating")
        return

    try:
        pid = _gptsovits_process.pid
        logger.info(f"Stopping GPT-SoVITS (PID: {pid})...")

        if _gptsovits_process.poll() is None:  # Process is still running
            # Optional: Try graceful shutdown via API first
            import requests
            try:
                logger.info("Attempting graceful shutdown via API...")
                response = requests.post(
                    "http://localhost:9880/control",
                    json={"command": "exit"},
                    timeout=2
                )
                logger.info(f"Sent exit command to GPT-SoVITS API (status: {response.status_code})")

                # Give it a brief moment to exit gracefully
                try:
                    _gptsovits_process.wait(timeout=2)
                    logger.info("GPT-SoVITS exited gracefully via API")
                except subprocess.TimeoutExpired:
                    logger.info("API shutdown timeout, proceeding to Job Object cleanup")
            except Exception as e:
                logger.debug(f"API shutdown failed: {e}, proceeding to Job Object cleanup")

            # Windows: Close Job Object handle - automatically kills ALL processes in the job
            if sys.platform == "win32" and _gptsovits_job is not None:
                try:
                    logger.info("Closing Job Object handle (automatic process tree termination)...")
                    _gptsovits_job.Close()
                    logger.info("Job Object closed - all GPT-SoVITS processes terminated automatically")

                    # Brief wait to confirm termination
                    try:
                        _gptsovits_process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass  # Job Object already killed it

                except Exception as e:
                    logger.warning(f"Job Object cleanup failed: {e}, falling back to terminate()")
                    _gptsovits_process.terminate()
                    try:
                        _gptsovits_process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        _gptsovits_process.kill()
            else:
                # Unix or no Job Object: use standard termination
                logger.info("Using standard process termination (no Job Object)")
                _gptsovits_process.terminate()
                try:
                    _gptsovits_process.wait(timeout=3)
                    logger.info("GPT-SoVITS process terminated")
                except subprocess.TimeoutExpired:
                    logger.warning("Process did not terminate, force killing...")
                    _gptsovits_process.kill()
                    _gptsovits_process.wait(timeout=2)
                    logger.info("GPT-SoVITS process force killed")
        else:
            logger.debug("GPT-SoVITS process already exited")

    except Exception as e:
        logger.error(f"Error stopping GPT-SoVITS: {e}")
    finally:
        _gptsovits_process = None
        _gptsovits_job = None


def _wait_for_gptsovits(qt_app: QApplication, splash: QSplashScreen, logger: logging.Logger, timeout: int = 30):
    """
    Wait for GPT-SoVITS to initialize (up to timeout seconds).

    This function waits regardless of whether GPT-SoVITS is installed via installer,
    giving users time to start their custom GPT-SoVITS installation manually.

    Args:
        qt_app: QApplication instance for processing events
        splash: Splash screen to update with progress
        logger: Logger instance
        timeout: Maximum seconds to wait (default: 30)
    """
    import time
    import requests
    from PyQt6.QtCore import QTimer

    gptsovits_url = "http://localhost:9880"
    start_time = time.time()
    elapsed = 0

    logger.info(f"Waiting for GPT-SoVITS at {gptsovits_url} (timeout: {timeout}s)")

    while elapsed < timeout:
        try:
            # Try to connect to GPT-SoVITS
            response = requests.get(gptsovits_url, timeout=1)
            # Any response (including 404) means server is running
            logger.info(f"GPT-SoVITS is running (HTTP {response.status_code})")
            splash.showMessage(f"✓ GPT-SoVITS connected!",
                              Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
                              QColor("lightgreen"))
            qt_app.processEvents()
            time.sleep(1)  # Show success message for 1 second
            return

        except requests.exceptions.HTTPError as e:
            # HTTP error means server responded - it's running
            if e.response is not None:
                logger.info(f"GPT-SoVITS is running (HTTP {e.response.status_code})")
                splash.showMessage(f"✓ GPT-SoVITS connected!",
                                  Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
                                  QColor("lightgreen"))
                qt_app.processEvents()
                time.sleep(1)
                return

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            # Server not ready yet, continue waiting
            pass

        except Exception as e:
            logger.debug(f"Error checking GPT-SoVITS: {e}")

        # Update progress with helpful message
        elapsed = int(time.time() - start_time)

        # Show different messages based on time elapsed
        if elapsed < 5:
            message = f"Waiting for GPT-SoVITS... ({elapsed}/{timeout}s)"
            color = QColor("white")
        elif elapsed < 15:
            message = f"Still waiting for GPT-SoVITS... ({elapsed}/{timeout}s) - Start it now if needed"
            color = QColor("yellow")
        else:
            message = f"GPT-SoVITS not detected ({elapsed}/{timeout}s) - You can start it manually"
            color = QColor("orange")

        splash.showMessage(message,
                          Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
                          color)
        qt_app.processEvents()
        time.sleep(1)

    # Timeout reached - but app will continue with auto-recovery capability
    logger.warning(f"GPT-SoVITS did not respond within {timeout} seconds")
    logger.info("App will continue - TTS will auto-connect when GPT-SoVITS becomes available")
    splash.showMessage("Continuing without GPT-SoVITS - Start it anytime for voice cloning",
                      Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
                      QColor("orange"))
    qt_app.processEvents()
    time.sleep(2)  # Give user time to read the message


def _emergency_cleanup():
    """
    Emergency cleanup handler for atexit - ensures GPT-SoVITS is stopped even on crash.

    Uses Job Object for guaranteed cleanup on Windows.
    """
    global _gptsovits_process, _gptsovits_job

    if _gptsovits_process is not None and _gptsovits_process.poll() is None:
        try:
            logger = logging.getLogger(__name__)
            logger.warning("Emergency cleanup: Stopping GPT-SoVITS")
            _stop_gptsovits(logger)
        except:
            # Last resort - close Job Object handle for automatic termination
            if sys.platform == "win32" and _gptsovits_job is not None:
                try:
                    _gptsovits_job.Close()
                except:
                    pass
            elif _gptsovits_process:
                try:
                    _gptsovits_process.kill()
                except:
                    pass


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

        # Start GPT-SoVITS if installed, then wait for it to initialize
        splash.showMessage("Starting GPT-SoVITS...",
                          Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
                          QColor("white"))
        qt_app.processEvents()

        gptsovits_started = _start_gptsovits(logger)

        # ALWAYS wait for GPT-SoVITS to initialize (up to 30 seconds)
        # This gives the user time to start their custom GPT-SoVITS installation manually
        if gptsovits_started:
            splash.showMessage("Waiting for GPT-SoVITS to initialize...",
                              Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
                              QColor("white"))
        else:
            # Not installed via installer - prompt user to start manually
            splash.showMessage("Waiting for GPT-SoVITS... (Please start GPT-SoVITS if needed)",
                              Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
                              QColor("yellow"))
        qt_app.processEvents()

        _wait_for_gptsovits(qt_app, splash, logger, timeout=30)

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
        else:
            splash.close()

        logger.info("MyVoice application initialized successfully")

        # Wait for application quit signal
        await app_close.wait()

        # Cleanup in SAME event loop as initialization (CRITICAL FIX)
        logger.info("Application shutting down - cleaning up services...")
        await myvoice_app.cleanup_async()

        return 0

    except Exception as e:
        logger.exception(f"Fatal error in async_main: {e}")
        return 1
    finally:
        # Stop GPT-SoVITS if we started it
        _stop_gptsovits(logger)


def main() -> int:
    """
    Main application entry point.

    Returns:
        int: Application exit code (0 for success, non-zero for error)
    """
    # Register emergency cleanup handler for crashes/abnormal exits
    atexit.register(_emergency_cleanup)

    # Initialize logger early so it's available in finally block
    logger = None

    try:
        # Setup logging first
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Starting MyVoice application with qasync event loop")

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