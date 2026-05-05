# Runtime hook for PyTorch DLL loading
# This hook pre-loads critical DLLs before torch is imported to avoid CRT conflicts
#
# The issue: PyInstaller's bootloader initializes its own CRT, which can conflict
# with how PyTorch's c10.dll expects to initialize. By pre-loading the DLLs using
# ctypes with proper flags BEFORE any Python torch imports, we ensure the DLLs
# initialize in a clean state.

import os
import sys
import ctypes

def _preload_torch_dlls():
    """Pre-load PyTorch DLLs to avoid CRT initialization conflicts."""
    # Only run in frozen (PyInstaller) mode
    if not getattr(sys, 'frozen', False):
        return

    base_path = sys._MEIPASS
    torch_lib_path = os.path.join(base_path, 'torch', 'lib')

    # Debug logging
    debug_log = os.path.join(os.path.dirname(base_path), 'logs', 'rthook_debug.log')
    def log(msg):
        try:
            with open(debug_log, 'a') as f:
                f.write(msg + '\n')
        except Exception:
            pass

    log(f"=== Runtime Hook Starting ===")
    log(f"base_path: {base_path}")
    log(f"torch_lib_path: {torch_lib_path}")
    log(f"torch_lib exists: {os.path.exists(torch_lib_path)}")

    if not os.path.exists(torch_lib_path):
        log("ERROR: torch lib path does not exist!")
        return

    # Step 1: Add directories to DLL search path
    dll_dirs = [torch_lib_path, base_path]

    # Add PyQt6 bin if it exists
    pyqt6_bin = os.path.join(base_path, 'PyQt6', 'Qt6', 'bin')
    if os.path.exists(pyqt6_bin):
        dll_dirs.append(pyqt6_bin)

    # Add to PATH environment variable (prepend for priority)
    current_path = os.environ.get('PATH', '')
    new_paths = [p for p in dll_dirs if p not in current_path]
    if new_paths:
        os.environ['PATH'] = os.pathsep.join(new_paths) + os.pathsep + current_path
    log(f"Updated PATH with: {new_paths}")

    # Add using os.add_dll_directory (Windows 10+)
    if hasattr(os, 'add_dll_directory'):
        for dll_dir in dll_dirs:
            try:
                os.add_dll_directory(dll_dir)
                log(f"Added DLL directory: {dll_dir}")
            except OSError as e:
                log(f"Failed to add DLL directory {dll_dir}: {e}")

    # Step 2: Pre-load critical DLLs using ctypes
    # Load order matters - dependencies must be loaded first
    critical_dlls = [
        # VC++ Runtime (usually already loaded, but ensure they're available)
        'vcruntime140.dll',
        'vcruntime140_1.dll',
        'msvcp140.dll',
        # Intel OpenMP (required by c10.dll)
        'libiomp5md.dll',
        # Core PyTorch DLLs in dependency order
        'c10.dll',
        'torch_cpu.dll',
        'c10_cuda.dll',
        'torch_cuda.dll',
        'torch.dll',
    ]

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.LoadLibraryW.restype = ctypes.c_void_p

    loaded_dlls = []
    failed_dlls = []

    for dll_name in critical_dlls:
        dll_path = os.path.join(torch_lib_path, dll_name)

        # Some DLLs might be in _internal root instead of torch/lib
        if not os.path.exists(dll_path):
            dll_path = os.path.join(base_path, dll_name)

        if not os.path.exists(dll_path):
            # Try loading from system/PATH
            dll_path = dll_name

        try:
            # Use LoadLibraryW which searches PATH
            handle = kernel32.LoadLibraryW(dll_path)
            if handle:
                loaded_dlls.append(dll_name)
                log(f"Pre-loaded: {dll_name} -> {handle}")
            else:
                error = ctypes.get_last_error()
                failed_dlls.append((dll_name, error))
                log(f"Failed to load {dll_name}: WinError {error}")
        except Exception as e:
            failed_dlls.append((dll_name, str(e)))
            log(f"Exception loading {dll_name}: {e}")

    log(f"Pre-loaded {len(loaded_dlls)} DLLs successfully")
    if failed_dlls:
        log(f"Failed to load: {failed_dlls}")
    log("=== Runtime Hook Complete ===")

# Run immediately when this hook is loaded
_preload_torch_dlls()
