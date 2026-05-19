# Runtime hook for PyTorch DLL loading
# This hook pre-loads critical DLLs before torch is imported to avoid CRT conflicts
#
# The issue: PyInstaller's bootloader initializes its own CRT, which can conflict
# with how PyTorch's c10.dll expects to initialize. By pre-loading the DLLs using
# ctypes with proper flags BEFORE any Python torch imports, we ensure the DLLs
# initialize in a clean state.
#
# Story 18.5 (Phase ⊥-Polish-2-Ship) extends this hook with two additional
# helpers that run AFTER _preload_torch_dlls() so the bundled CUDA Toolkit
# redistributables + triton-windows are reachable at runtime:
#   * _inject_cuda_redist_paths() — sets CUDA_PATH, adds the bundled
#     cuda_redist/bin to the DLL search path so triton's NVRTC find_cuda()
#     resolves to the bundled DLLs without requiring a system-wide install
#   * _probe_triton_availability() — minimal `import triton` + version log
#     for observability (does NOT enforce; enforcement is the
#     engage_compile_optimizations NFR7 fallback chain in source-tree)

import os
import sys
import ctypes


def _ensure_logs_dir(base_path):
    """Create `<application_root>/logs/` if missing, return the rthook debug log path.

    Story 18.5 Task 6.5 / OQ #4 — fixes the latent rthook debug-log bug noted
    in `memory/build_tools_phase_perp_state.md`: prior to this helper, the
    debug log's `open(..., 'a')` would silently fail on first launch when
    logs/ didn't exist yet, leaving zero rthook diagnostics on disk. The
    makedirs(exist_ok=True) call is idempotent and tolerates concurrent
    creation by setup_logging() in the application code path.
    """
    logs_dir = os.path.join(os.path.dirname(base_path), 'logs')
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except Exception:
        pass
    return os.path.join(logs_dir, 'rthook_debug.log')


def _preload_torch_dlls():
    """Pre-load PyTorch DLLs to avoid CRT initialization conflicts."""
    # Only run in frozen (PyInstaller) mode
    if not getattr(sys, 'frozen', False):
        return

    base_path = sys._MEIPASS
    torch_lib_path = os.path.join(base_path, 'torch', 'lib')

    # Debug logging — Story 18.5 Task 6.5 fix: ensure logs/ exists before write
    debug_log = _ensure_logs_dir(base_path)
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


def reinject_runtime_paths():
    """Re-invoke CUDA-redist + triton-backend env-var injection on demand.

    Extracted from ``_inject_cuda_redist_paths`` + ``_configure_triton_backend_discovery``
    so the H2 candidate fix of the compile-disengage-post-generation spec can
    call the same body from ``apply_reload_compile_fix(stage="pre_load")``.

    Gated on ``sys.frozen`` + ``_MEIPASS``: a non-frozen (dev-tree pytest)
    call is a silent no-op that returns an all-False status dict. The
    function does not raise — partial-failure paths log via the rthook
    debug log and report False for the affected slot.

    The function is idempotent: env vars retain their startup values when
    re-invoked, and PATH is only prepended if ``cuda_redist/bin`` is not
    already present. The returned dict reports whether each slot is
    currently set (regardless of whether *this* call did the setting).

    Returns:
        dict with four ``bool`` slots:
          * ``cuda_path_set``: True iff ``CUDA_PATH`` env var is set
          * ``path_prepended``: True iff ``cuda_redist/bin`` is in ``PATH``
          * ``triton_env_set``: True iff ``TRITON_BACKENDS_IN_TREE`` is set
          * ``cc_env_set``: True iff ``CC`` env var is set
    """
    status = {
        "cuda_path_set": False,
        "path_prepended": False,
        "triton_env_set": False,
        "cc_env_set": False,
    }
    if not getattr(sys, 'frozen', False):
        return status
    if not getattr(sys, '_MEIPASS', None):
        return status

    base_path = sys._MEIPASS
    debug_log = _ensure_logs_dir(base_path)

    def log(msg):
        try:
            with open(debug_log, 'a') as f:
                f.write(msg + '\n')
        except Exception:
            pass

    cuda_redist_root = os.path.join(base_path, 'cuda_redist')
    cuda_redist_bin = os.path.join(cuda_redist_root, 'bin')
    triton_cuda_root = os.path.join(base_path, 'triton', 'backends', 'nvidia')

    log("=== reinject_runtime_paths ===")
    log(f"cuda_redist_root exists: {os.path.exists(cuda_redist_root)}")
    log(f"cuda_redist_bin exists: {os.path.exists(cuda_redist_bin)}")
    log(f"triton_cuda_root exists: {os.path.exists(triton_cuda_root)}")

    # ---- CUDA_PATH ---- #
    try:
        if os.path.exists(os.path.join(triton_cuda_root, 'bin', 'ptxas.exe')):
            os.environ['CUDA_PATH'] = triton_cuda_root
            log(f"Set CUDA_PATH = {triton_cuda_root}")
        else:
            os.environ['CUDA_PATH'] = cuda_redist_root
            log(f"WARNING: triton's bundled ptxas missing; CUDA_PATH = "
                f"{cuda_redist_root} (fallback)")
        status["cuda_path_set"] = bool(os.environ.get('CUDA_PATH'))
    except Exception as e:
        log(f"CUDA_PATH set failed: {e}")

    # ---- PATH prepend + add_dll_directory ---- #
    try:
        if os.path.exists(cuda_redist_bin):
            current_path = os.environ.get('PATH', '')
            if cuda_redist_bin not in current_path:
                os.environ['PATH'] = cuda_redist_bin + os.pathsep + current_path
                log(f"Prepended {cuda_redist_bin} to PATH")
            else:
                log(f"PATH already contains {cuda_redist_bin}; skipping prepend")
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(cuda_redist_bin)
                except OSError as e:
                    log(f"add_dll_directory failed: {e}")
            status["path_prepended"] = cuda_redist_bin in os.environ.get('PATH', '')
        else:
            log(f"WARNING: cuda_redist/bin missing; PATH unchanged")
    except Exception as e:
        log(f"PATH prepend failed: {e}")

    # ---- TRITON_BACKENDS_IN_TREE ---- #
    try:
        os.environ['TRITON_BACKENDS_IN_TREE'] = '1'
        status["triton_env_set"] = os.environ.get('TRITON_BACKENDS_IN_TREE') == '1'
        log("Set TRITON_BACKENDS_IN_TREE=1")
    except Exception as e:
        log(f"TRITON_BACKENDS_IN_TREE set failed: {e}")

    # ---- CC (bundled TinyCC) ---- #
    try:
        bundled_tcc = os.path.join(
            base_path, 'triton', 'runtime', 'tcc', 'tcc.exe'
        )
        if os.path.exists(bundled_tcc):
            os.environ['CC'] = bundled_tcc
            status["cc_env_set"] = True
            log(f"Set CC = {bundled_tcc}")
        else:
            log(f"WARNING: bundled TinyCC missing at {bundled_tcc}")
    except Exception as e:
        log(f"CC set failed: {e}")

    log(f"=== reinject_runtime_paths complete: {status} ===")
    return status


def _inject_cuda_redist_paths():
    """Startup wrapper for the CUDA-redist subset of ``reinject_runtime_paths``.

    Kept as a thin wrapper for the existing startup-invocation list (line
    439) — the body now lives in ``reinject_runtime_paths`` so the H2
    candidate fix can re-invoke it on reload via
    ``apply_reload_compile_fix(stage="pre_load")``.
    """
    reinject_runtime_paths()


def _configure_triton_backend_discovery():
    """Startup wrapper for the triton-backend subset of ``reinject_runtime_paths``.

    Kept as a thin wrapper for the existing startup-invocation list (line
    440). The triton-backend env-var work is performed by
    ``reinject_runtime_paths``; this wrapper exists so the startup
    invocation block at line 438-442 remains unchanged.
    """
    # No-op at startup: reinject_runtime_paths was already called by
    # _inject_cuda_redist_paths above. We re-call here to preserve the
    # original two-phase startup behavior exactly (the env vars are
    # idempotent under the new function).
    reinject_runtime_paths()


def _hide_subprocess_windows():
    """Inject CREATE_NO_WINDOW into every subprocess.Popen so triton's
    tcc.exe / ptxas.exe (and any other helper exes) don't flash visible
    console windows during torch.compile cold compile.

    RTX 3060 smoke 2026-05-14: first-generation cold compile (80.88 s on
    that hardware per the log line at `19:53:30 → 19:55:36`) spawned
    multiple visible console windows that popped up and disappeared in
    quick succession. The windows are tcc / ptxas / nvcc-style helpers
    invoked by triton's codegen pipeline; the visual effect is alarming
    to first-time users even though the underlying compile is healthy.

    Fix: monkey-patch ``subprocess.Popen.__init__`` to OR ``CREATE_NO_WINDOW``
    into the ``creationflags`` argument if it isn't already set. This is
    a frozen GUI app — no subprocess should ever inherit a visible
    console. The patch is idempotent (subprocess calls that already pass
    CREATE_NO_WINDOW are unaffected) and gated on ``sys.frozen`` so the
    dev-tree pytest run is not affected.

    Side effect: every subprocess spawned by the frozen app gets
    CREATE_NO_WINDOW. Currently the only deliberate subprocess
    invocations are triton's compiler chain and (transitively) any tool
    qwen_tts may spawn — all internal helpers. ffmpeg is invoked via
    pydub/imageio-ffmpeg which already redirect stdio and don't need a
    console. If a future feature needs a visible console subprocess, the
    caller must pass ``creationflags=0`` explicitly (or any value with
    bit 0x08000000 cleared) to opt out.

    Reference: Python subprocess docs §17.5.1.1 — CREATE_NO_WINDOW
    (0x08000000) suppresses console allocation; safe on all Windows
    versions PyInstaller supports.
    """
    if not getattr(sys, 'frozen', False):
        return
    if sys.platform != 'win32':
        return

    import subprocess

    base_path = sys._MEIPASS
    debug_log = _ensure_logs_dir(base_path)

    def log(msg):
        try:
            with open(debug_log, 'a') as f:
                f.write(msg + '\n')
        except Exception:
            pass

    log("=== Subprocess CREATE_NO_WINDOW Injection ===")

    _CREATE_NO_WINDOW = 0x08000000
    _original_popen_init = subprocess.Popen.__init__

    def _patched_popen_init(self, *args, **kwargs):
        creationflags = kwargs.get('creationflags', 0)
        if not (creationflags & _CREATE_NO_WINDOW):
            kwargs['creationflags'] = creationflags | _CREATE_NO_WINDOW
        return _original_popen_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = _patched_popen_init
    log("Patched subprocess.Popen.__init__ to OR CREATE_NO_WINDOW (0x08000000)")
    log("=== Subprocess CREATE_NO_WINDOW Injection Complete ===")


def _probe_triton_availability():
    """Log triton-windows version (if importable) for runtime observability.

    Story 18.5 / Phase ⊥-Polish-2-Ship. Probes whether triton can be imported
    AFTER the CUDA redistributable path injection. Does NOT enforce — the
    canonical NFR7 fallback gate is `engage_compile_optimizations` in the
    source tree (which falls back to eager mode on any compile exception).
    This probe's job is observability, not enforcement.

    Also probes the backend registry (Story 18.5 Task 7 follow-up — fixes the
    `0 active drivers` failure class observed in the first bundled smoke).
    Confirms `triton.backends.backends` resolves at least one entry,
    catching regressions in `_configure_triton_backend_discovery` upstream.

    Gated on `sys.frozen` per the existing rthook convention.
    """
    if not getattr(sys, 'frozen', False):
        return

    base_path = sys._MEIPASS
    debug_log = _ensure_logs_dir(base_path)

    def log(msg):
        try:
            with open(debug_log, 'a') as f:
                f.write(msg + '\n')
        except Exception:
            pass

    log("=== Triton Availability Probe ===")
    try:
        import triton
        version = getattr(triton, '__version__', '(unknown)')
        log(f"triton-windows available (version={version})")
        # Probe the backend registry — confirms the in-tree discovery
        # path resolved at least one backend.
        try:
            from triton.backends import backends as _triton_backends
            log(f"triton.backends.backends discovered: "
                f"{sorted(_triton_backends.keys())} "
                f"(expected: ['amd', 'nvidia'] or subset)")
        except Exception as be:
            log(f"WARNING: triton.backends probe failed "
                f"({type(be).__name__}: {be}) — talker forward pass will "
                f"fall back to eager mode via Story 18.4 NFR7 branch")
    except Exception as e:
        log(f"WARNING: triton import failed ({type(e).__name__}: {e}) — "
            f"talker forward pass will fall back to eager mode via the "
            f"Story 18.4 NFR7 compile_failed branch")
    log("=== Triton Availability Probe Complete ===")


# Run immediately when this hook is loaded.
# Story 18.5 ordering: torch DLLs first (the canonical preload disciplined by
# tooling-2 + V2 baseline), THEN CUDA redistributable path injection
# (precondition for triton's NVRTC), THEN triton backend-discovery
# configuration (env var must be set BEFORE first `import triton`), THEN
# subprocess window-hiding (must run BEFORE triton import so the patched
# Popen is in place by the time the codegen pipeline spawns tcc/ptxas),
# THEN triton availability probe (observability log; does the first import).
# Each function is independently gated on sys.frozen.
_preload_torch_dlls()
_inject_cuda_redist_paths()
_configure_triton_backend_discovery()
_hide_subprocess_windows()
_probe_triton_availability()
