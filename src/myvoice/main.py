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
import multiprocessing
from pathlib import Path

# CRITICAL: PyInstaller multiprocessing guard. Story 18.4 / D-22 Branch B
# engages torch.compile, whose inductor backend spawns subprocess workers
# for parallel kernel compilation. In a PyInstaller-bundled app without
# this guard, each spawned subprocess re-execs the bundled exe, triggering
# the splash screen + Qt init + everything else at the top of main.py —
# the user sees the app launch twice. `freeze_support()` makes spawned
# subprocesses recognize themselves as workers and exit cleanly instead.
# Documented PyInstaller best practice; harmless in dev mode (no subprocess
# re-exec happens there).
multiprocessing.freeze_support()

# CRITICAL: Route Python's SSL trust through the OS certificate store BEFORE
# any HTTPS call happens. End-user machines running SSL-scanning antivirus
# (Norton, Kaspersky, Avast, Bitdefender, ESET, corporate MITM proxies, ...)
# re-sign outbound TLS handshakes with a private root that lives only in the
# Windows cert store — never in Python's bundled `certifi` cacert.pem. Without
# truststore, every huggingface_hub / requests call fails with
# `SSLCertVerificationError: unable to get local issuer certificate` and
# first-launch model download/resume hangs through a 23 s retry chain before
# failing outright (observed 2026-05-19 in production on a Norton-protected
# host). truststore replaces Python's SSL context with one that consults the
# OS trust store, where the AV's root IS trusted because the AV installed it
# there. Must run before any `requests.Session`, `huggingface_hub`, or
# `urllib3.PoolManager` is constructed; placing it here covers every
# downstream branch (--predownload-models, --reload-compile-repro, normal
# app launch). Best-effort: dev trees that haven't run `pip install
# truststore` yet still work over public CAs via the default certifi path.
try:
    import truststore as _truststore
    _truststore.inject_into_ssl()
except ImportError:
    pass

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Install-time model pre-download path (2026-05-17). Invoked from the
# Inno Setup installer's post-extraction phase via
# `MyVoice.exe --predownload-models --tier=<small|quality>` so the
# first user-facing generation doesn't have to wait for the multi-GB
# HuggingFace download. Routed BEFORE the heavy torch + PyQt6 + qasync
# imports below — pre-download only needs `huggingface_hub.snapshot_download`,
# not the full app graph. Exits immediately when done.
if "--predownload-models" in sys.argv:
    from myvoice.utils.predownload_models import run_predownload
    sys.exit(run_predownload(sys.argv))

# Reload-compile reproducer (compile-disengage-post-generation spec).
# Drives a deterministic load -> priming -> force-reload sequence to expose
# the post-generation torch.compile disengagement bug behind a switchable
# fix env var. Routed BEFORE the heavy torch + PyQt6 + qasync imports below
# (mirrors the --predownload-models precedent) so the reproducer doesn't
# re-trigger splash/Qt init.
if "--reload-compile-repro" in sys.argv:
    # Parse optional --fix=<value> from argv; defaults to the env var (or
    # "off" if neither). The env var MUST be set BEFORE any torch import
    # so the dispatch in `apply_reload_compile_fix(pre_load)` sees the
    # right value on the first registry load.
    _fix_value = os.environ.get("MYVOICE_RELOAD_COMPILE_FIX", "off")
    for _arg in sys.argv:
        if _arg.startswith("--fix="):
            _fix_value = _arg.split("=", 1)[1]
            break
    os.environ["MYVOICE_RELOAD_COMPILE_FIX"] = _fix_value
    # Suppress the natural compile warmup — its priming generation would
    # collide with the reproducer's own priming step and skew the
    # timing/cache state of the reload measurement.
    os.environ["MYVOICE_DISABLE_COMPILE_WARMUP"] = "1"
    # Force HuggingFace offline mode — the bundled Python's requests/HF cert
    # bundle does not validate against huggingface.co's certificate chain
    # (observed in the 2026-05-18 first verifier run: SSLCertVerificationError
    # on the model_info() API call). The models are already in the HF cache
    # via the installer's --predownload-models step, so offline mode short-
    # circuits the API check and `from_pretrained` reads cached weights
    # directly.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from myvoice.utils.reload_compile_reproducer import run_reproducer
    sys.exit(run_reproducer(_fix_value))

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

    # Set application metadata. Version is read from myvoice.__version__
    # (the single source of truth maintained by build_tools/version.py)
    # instead of being hardcoded here — previously stuck at "2.0.0"
    # across the 2.1 and 2.2 release cycles.
    from myvoice import __version__ as _myvoice_version
    app.setApplicationName("MyVoice")
    app.setApplicationVersion(_myvoice_version)
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
    # Setup shutdown event.
    #
    # We wake on `lastWindowClosed`, NOT `aboutToQuit`. Reason: qasync's
    # QEventLoop also connects to `aboutToQuit` and calls `loop.stop()`
    # there. If we use `aboutToQuit` to set the asyncio Event, async_main
    # races qasync's loop teardown — by the time async_main resumes from
    # `app_close.wait()` and reaches `asyncio.wait_for(cleanup_async(), …)`
    # the qasync loop is already stopped and `wait_for` raises
    # `RuntimeError: no running event loop`. The traceback was visible
    # on every exit (exit code 1) and `cleanup_async` never actually ran.
    #
    # Fix: `lastWindowClosed` fires BEFORE Qt's auto-quit cascade. Combined
    # with `setQuitOnLastWindowClosed(False)`, Qt does NOT auto-fire
    # `quit()` (and therefore not `aboutToQuit`/qasync's loop.stop) — so
    # async_main wakes with the loop fully healthy, completes cleanup_async,
    # returns; main() then proceeds to `_os._exit(0)` which terminates Qt.
    app_close = asyncio.Event()
    qt_app.setQuitOnLastWindowClosed(False)
    qt_app.lastWindowClosed.connect(app_close.set)

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
        except RuntimeError as runtime_err:
            # Belt-and-suspenders: if any unexpected shutdown path causes the
            # qasync loop to be torn down before this wait_for runs (the bug
            # the lastWindowClosed switch above already fixes for the normal
            # close path), surface a clear log line and force-exit cleanly
            # rather than letting `Fatal error in async_main: no running
            # event loop` escape with exit code 1.
            if "no running event loop" in str(runtime_err):
                logger.warning(
                    "Cleanup wait_for hit a torn-down event loop — forcing "
                    "clean exit. Some service-level cleanup did not run."
                )
                _os._exit(0)
            raise

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

        # Story 18.2: enable lossless TF32 + cuDNN benchmark autotune on
        # Ampere+ CUDA hosts (no-op on CPU / pre-Ampere). The probe and the
        # conditional logic live inside the helper; main.py just calls. Placed
        # after setup_logging() so the INFO breadcrumb lands in myvoice.log
        # rather than stderr-only, and before setup_application() so the
        # flags are global to the process before any qwen_tts work runs.
        # Guarded with the same (ImportError, OSError) discipline the torch
        # import block above uses (main.py:44-49) — a partial install or a
        # missing CUDA DLL must not abort startup; the speedup is opt-in and
        # absence is the V2 baseline. Other exception types from the helper
        # (e.g., a programming bug in metrics.record validation) are NOT
        # swallowed here — they surface to main()'s outer handler so a
        # genuine startup defect cannot hide behind this guard.
        try:
            from myvoice.services.tts_streaming.torch_runtime import (
                enable_tf32_and_cudnn_benchmark,
            )
            enable_tf32_and_cudnn_benchmark()
        except (ImportError, OSError) as _tf32_err:
            logger.warning(
                f"TF32 + cuDNN benchmark enable failed "
                f"(continuing without speedup): {_tf32_err}"
            )

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

        # Force terminate - PyAudio may leave non-daemon threads that prevent clean exit
        logger.info("Forcing process termination...")
        import os as _os
        _os._exit(exit_code)

    except Exception as e:
        print(f"Fatal error starting MyVoice: {e}", file=sys.stderr)
        if logger:
            logger.exception("Fatal error in main()")
        else:
            logging.exception("Fatal error in main()")
        return 1


if __name__ == "__main__":
    sys.exit(main())