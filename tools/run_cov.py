"""Run pytest with --cov on Windows where torch's c10.dll must initialize before
coverage's C tracer.

The standard ``pytest --cov=...`` invocation fails on Windows with
``OSError [WinError 1114]`` (``c10.dll`` initialization routine failure)
because pytest-cov starts coverage in a ``tryfirst=True``
``pytest_load_initial_conftests`` hookwrapper, which runs *before*
``tests/conftest.py``'s torch-DLL preamble can fire. Once coverage's C tracer
is live, ``import torch`` cannot recover.

This wrapper performs the conftest preamble (DLL directory registration +
``import torch``) BEFORE invoking ``pytest.main``. pytest-cov then activates
inside ``pytest.main`` with torch's DLLs already loaded, so ``c10.dll`` is
satisfied and the gate runs cleanly.

Usage (mirrors the literal command in Story 12.3 AC #6):

    python tools/run_cov.py --cov=myvoice.ui.components.service_status_indicator \\
        --cov=myvoice.ui.main_window tests/ui/test_status_indicators.py

All arguments after the script name are passed verbatim to ``pytest.main``.
Returns pytest's exit code.

Background: Story 12.3 Dev Agent Record → "Coverage-tooling discovery";
memory note ``torch_before_coverage_dll_ordering.md``. The conftest's
plain torch-first preamble at ``tests/conftest.py:21-50`` covers regular
test runs but NOT ``--cov`` runs because of pytest-cov's hook-ordering.
"""

import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# Mirror tests/conftest.py:21-50 — DLL directories must be registered before
# importing torch on Windows. The bundled python310 env keeps torch's lib
# directory and the matching CUDA toolkit out of system PATH on purpose, so
# the explicit add_dll_directory calls are required.
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _torch_lib = _repo_root / "python310" / "Lib" / "site-packages" / "torch" / "lib"
    if _torch_lib.exists():
        os.add_dll_directory(str(_torch_lib))
    for _cuda in (
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp"),
    ):
        if _cuda.exists():
            os.add_dll_directory(str(_cuda))

# Failures are swallowed — a torch-less environment can still run tests that
# don't depend on torch (matches the conftest's contract).
try:
    import torch  # noqa: F401
except (ImportError, OSError):
    pass

import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv[1:]))
