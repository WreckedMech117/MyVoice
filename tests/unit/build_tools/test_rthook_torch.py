"""Tests for Story 18.5 Task 6.6 — `build_tools/hooks/rthook_torch.py` extensions.

Covers the two new helper functions added by Story 18.5:

  * ``_inject_cuda_redist_paths()`` — surfaces the bundled
    ``_internal/cuda_redist/`` subset to triton's NVRTC pipeline by setting
    ``CUDA_PATH``, prepending ``bin/`` to ``PATH``, and adding ``bin/`` via
    ``os.add_dll_directory``. Gated on ``sys.frozen`` to skip in dev-tree
    pytest.
  * ``_probe_triton_availability()`` — minimal ``import triton`` probe with a
    version log on success and a WARNING breadcrumb on failure. Gated on
    ``sys.frozen``; observability-only (no enforcement; the
    ``engage_compile_optimizations`` NFR7 fallback chain is the canonical gate).

Per `memory/code_review_regression_test_exact_class.md`, the bug class these
tests cover is "future maintainer breaks the env-var-injection logic or the
sys.frozen gate"; the bundled smoke (Task 7) is too slow to gate on every
commit. The tests exercise the functions' pure-Python behavior under
simulated ``sys.frozen`` state via ``monkeypatch``.

Loading discipline: the rthook is a PyInstaller runtime hook at
``build_tools/hooks/rthook_torch.py`` — not in ``src/`` and not on the
default Python path. Tests load it via ``importlib.util.spec_from_file_location``
to avoid polluting ``sys.path`` globally.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Module loader — load rthook_torch by file path so the test doesn't depend
# on build_tools/hooks/ being on sys.path.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rthook_module():
    """Load build_tools/hooks/rthook_torch.py as a module via importlib.

    On import, the module's three auto-invocation calls
    (``_preload_torch_dlls``, ``_inject_cuda_redist_paths``,
    ``_probe_triton_availability``) all execute — but each early-returns on
    ``not sys.frozen``, so the import is a clean no-op in the dev-env test
    runner where ``sys.frozen`` is False.
    """
    repo_root = Path(__file__).resolve().parents[3]
    hook_path = repo_root / "build_tools" / "hooks" / "rthook_torch.py"
    assert hook_path.exists(), f"rthook_torch.py missing at {hook_path}"

    spec = importlib.util.spec_from_file_location(
        "rthook_torch_under_test", str(hook_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def frozen_env(monkeypatch, tmp_path):
    """Simulate a frozen PyInstaller bundle.

    Sets ``sys.frozen = True`` and ``sys._MEIPASS = <tmp_path/meipass>``. The
    bundled production layout (post Story 18.5 Task 7 Fix #2) materializes:

      meipass/
      ├── cuda_redist/bin/         (NVRTC + CUDA Runtime DLLs at runtime)
      └── triton/
          ├── backends/nvidia/bin/ptxas.exe         (CUDA_PATH source)
          ├── backends/nvidia/include/cuda.h        (CUDA_PATH source)
          ├── backends/nvidia/lib/x64/cuda.lib      (CUDA_PATH source)
          └── runtime/tcc/tcc.exe                   (CC source for host C compile)

    The logs/ dir that ``_ensure_logs_dir`` writes to lands at
    ``tmp_path / 'logs/'`` (sibling of meipass; matches the production
    layout where logs/ is at application root and _MEIPASS is at
    application_root/_internal/).
    """
    meipass = tmp_path / "meipass"
    meipass.mkdir()
    (meipass / "cuda_redist" / "bin").mkdir(parents=True)
    # Plant minimal triton bundle structure so the rthook's "prefer triton's
    # bundled CUDA" branch and "set CC to bundled tcc" branch both trigger.
    triton_nv = meipass / "triton" / "backends" / "nvidia"
    (triton_nv / "bin").mkdir(parents=True)
    (triton_nv / "include").mkdir(parents=True)
    (triton_nv / "lib" / "x64").mkdir(parents=True)
    (triton_nv / "bin" / "ptxas.exe").write_bytes(b"fake-ptxas")
    (triton_nv / "include" / "cuda.h").write_bytes(b"fake-cuda-h")
    (triton_nv / "lib" / "x64" / "cuda.lib").write_bytes(b"fake-cuda-lib")
    tcc_dir = meipass / "triton" / "runtime" / "tcc"
    tcc_dir.mkdir(parents=True)
    (tcc_dir / "tcc.exe").write_bytes(b"fake-tcc")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)
    # Make PATH deterministic for the prepend test.
    monkeypatch.setenv("PATH", "C:\\original\\path")
    # Clear CUDA_PATH + CC if present so the test can detect the helper's set.
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.delenv("CC", raising=False)
    return {
        "tmp_path": tmp_path,
        "meipass": meipass,
        "cuda_redist_root": meipass / "cuda_redist",
        "cuda_redist_bin": meipass / "cuda_redist" / "bin",
        "triton_cuda_root": triton_nv,
        "bundled_tcc": tcc_dir / "tcc.exe",
    }


# ---------------------------------------------------------------------------
# _inject_cuda_redist_paths tests (4 rows)
# ---------------------------------------------------------------------------

class TestInjectCudaRedistPaths:
    """Story 18.5 Task 6.6 — `_inject_cuda_redist_paths` behavior.

    The function must (a) set CUDA_PATH to the bundled subset root, (b) prepend
    the bundled bin/ to PATH, (c) call ``os.add_dll_directory(bin)`` once, and
    (d) no-op when ``sys.frozen`` is False. Each test exercises exactly one of
    those contract points.
    """

    def test_sets_cuda_path_to_triton_bundled(self, rthook_module, frozen_env):
        """CUDA_PATH must equal triton's bundled `backends/nvidia/` root after injection.

        Story 18.5 Task 7 Fix #2 (2026-05-11): `find_cuda_env(CUDA_PATH)`
        checks for `ptxas.exe + cuda.h + cuda.lib` at the path — those live
        inside `triton/backends/nvidia/` in the bundle, not in `cuda_redist/`
        (which has only the redistributable DLLs). The rthook prefers triton's
        bundled path when ptxas.exe is present (the production case).
        """
        rthook_module._inject_cuda_redist_paths()
        expected = str(frozen_env["triton_cuda_root"])
        assert os.environ["CUDA_PATH"] == expected

    def test_falls_back_to_cuda_redist_when_triton_ptxas_missing(
        self, rthook_module, frozen_env
    ):
        """If triton's bundled ptxas is missing, fall back to cuda_redist root.

        Defensive: if a future bundle iteration breaks the triton/backends/nvidia/
        layout, the rthook should still set SOME CUDA_PATH so find_cuda's
        hardcoded fallback gets a chance to resolve a system CUDA Toolkit
        install (a degraded but not broken state on dev machines).
        """
        # Remove ptxas to simulate the missing-triton-cuda case.
        (frozen_env["triton_cuda_root"] / "bin" / "ptxas.exe").unlink()
        rthook_module._inject_cuda_redist_paths()
        expected = str(frozen_env["cuda_redist_root"])
        assert os.environ["CUDA_PATH"] == expected

    def test_prepends_bin_to_path(self, rthook_module, frozen_env):
        """The bundled cuda_redist/bin must be a prefix of PATH after injection.

        Tests the user-runtime invariant: name-only LoadLibraryW calls for
        cudart64_*.dll / nvrtc64_*.dll must resolve to the bundled DLLs, not
        to whatever happens to live in the user's PATH. Mirrors the existing
        torch-DLL PATH prepend at rthook_torch.py:48-:52.
        """
        rthook_module._inject_cuda_redist_paths()
        expected_prefix = str(frozen_env["cuda_redist_bin"])
        assert os.environ["PATH"].startswith(
            expected_prefix + os.pathsep
        ), f"PATH={os.environ['PATH']!r} does not start with {expected_prefix!r}"

    def test_adds_dll_directory(self, rthook_module, frozen_env, monkeypatch):
        """os.add_dll_directory must be called once with the bundled bin/ path.

        Spies on ``os.add_dll_directory`` to capture the call. The dev-env
        test runner is Windows so ``os.add_dll_directory`` exists (Win10+);
        this test does not cover the pre-Win10 platform branch (gated on
        ``hasattr(os, 'add_dll_directory')`` in the helper — production
        Windows ships Win10+, so the branch is effectively dead code on
        the supported platform set).
        """
        if not hasattr(os, "add_dll_directory"):
            pytest.skip("os.add_dll_directory not available (pre-Win10 platform)")

        calls = []

        # Capture the existing function so the spy can still chain through
        # to the real call (the real call also raises OSError if the dir is
        # outside Windows' DLL search whitelist — we want a working call here).
        real_add_dll_directory = os.add_dll_directory

        def spy_add_dll_directory(path):
            calls.append(path)
            return real_add_dll_directory(path)

        monkeypatch.setattr(os, "add_dll_directory", spy_add_dll_directory)

        rthook_module._inject_cuda_redist_paths()

        expected_bin = str(frozen_env["cuda_redist_bin"])
        assert expected_bin in calls, (
            f"expected {expected_bin!r} in add_dll_directory calls, got {calls!r}"
        )

    def test_is_noop_when_not_frozen(self, rthook_module, monkeypatch):
        """The helper must early-return when sys.frozen is False (or absent).

        Tests the dev-tree pytest isolation contract: importing this module
        in a dev environment (where sys.frozen is False) must not mutate
        CUDA_PATH, PATH, or the DLL search path. Without this gate the
        dev-env test runner would silently set CUDA_PATH to a meaningless
        value derived from PyInstaller's _MEIPASS sentinel — breaking every
        torch-dependent test downstream.
        """
        # Force sys.frozen to False (default in dev-env, but be explicit).
        monkeypatch.setattr(sys, "frozen", False, raising=False)
        # Snapshot CUDA_PATH state before the call.
        cuda_path_before = os.environ.get("CUDA_PATH", None)

        rthook_module._inject_cuda_redist_paths()

        cuda_path_after = os.environ.get("CUDA_PATH", None)
        assert cuda_path_before == cuda_path_after, (
            f"CUDA_PATH was mutated despite sys.frozen=False: "
            f"{cuda_path_before!r} -> {cuda_path_after!r}"
        )


# ---------------------------------------------------------------------------
# _configure_triton_backend_discovery tests (2 rows)
# ---------------------------------------------------------------------------

class TestConfigureTritonBackendDiscovery:
    """Story 18.5 Task 7 follow-up — `_configure_triton_backend_discovery` behavior.

    Fixes the `0 active drivers ([]). There should only be one.` failure
    class observed in the first bundled smoke 2026-05-11. Triton's default
    backend discovery uses `entry_points()` which needs dist-info metadata
    PyInstaller doesn't bundle; the fast-path env var switches discovery to
    `os.listdir(triton/backends/)` which works without dist-info.
    """

    def test_sets_triton_backends_in_tree(self, rthook_module, frozen_env):
        """TRITON_BACKENDS_IN_TREE must equal '1' after the helper runs."""
        # Clear any prior value to make the assertion deterministic.
        import os
        os.environ.pop('TRITON_BACKENDS_IN_TREE', None)

        rthook_module._configure_triton_backend_discovery()

        assert os.environ.get('TRITON_BACKENDS_IN_TREE') == '1'

    def test_is_noop_when_not_frozen(self, rthook_module, monkeypatch):
        """The helper must early-return when sys.frozen is False.

        The dev-env uses entry_points() discovery successfully (dist-info IS
        present in `python310/Lib/site-packages/`), so this function must
        not mutate the env var in dev-tree pytest. Otherwise downstream
        dev-env tests that import triton would silently switch to the
        filesystem-based discovery path — a behavior regression even
        though the dev-env happens to support both paths.
        """
        monkeypatch.setattr(sys, "frozen", False, raising=False)
        # Clear the env vars so we can detect a mutation.
        monkeypatch.delenv('TRITON_BACKENDS_IN_TREE', raising=False)
        monkeypatch.delenv('CC', raising=False)

        rthook_module._configure_triton_backend_discovery()

        assert 'TRITON_BACKENDS_IN_TREE' not in os.environ
        assert 'CC' not in os.environ

    def test_sets_cc_to_bundled_tcc(self, rthook_module, frozen_env):
        """CC must point at the bundled TinyCC after the helper runs.

        Story 18.5 Task 7 Fix #2 (2026-05-11): triton's `get_cc()` looks for
        the bundled tcc at `sysconfig.get_paths()["platlib"]/triton/runtime/
        tcc/tcc.exe` — but in a PyInstaller bundle `platlib` doesn't resolve
        to `_internal/`. Setting CC directly bypasses the sysconfig issue.
        """
        rthook_module._configure_triton_backend_discovery()
        assert os.environ['CC'] == str(frozen_env['bundled_tcc'])


# ---------------------------------------------------------------------------
# _probe_triton_availability tests (2 rows)
# ---------------------------------------------------------------------------

class TestProbeTritonAvailability:
    """Story 18.5 Task 6.6 — `_probe_triton_availability` behavior.

    The probe is observability-only: it logs the triton version on success
    and a WARNING breadcrumb on failure, but does NOT raise. Enforcement is
    the source-tree ``engage_compile_optimizations`` NFR7 fallback chain.
    """

    def test_succeeds_when_triton_importable(self, rthook_module, frozen_env):
        """The probe must complete without raising when triton is importable.

        The dev-env test runner has triton-windows installed (Story 18.5
        Task 1.6 prereq), so ``import triton`` succeeds. The probe writes a
        success line to the rthook debug log.
        """
        # Call must not raise.
        rthook_module._probe_triton_availability()

        # Verify the success line landed in rthook_debug.log.
        log_path = frozen_env["tmp_path"] / "logs" / "rthook_debug.log"
        assert log_path.exists(), f"rthook debug log not created at {log_path}"
        contents = log_path.read_text()
        assert "triton-windows available" in contents, (
            f"expected 'triton-windows available' in log, got:\n{contents}"
        )

    def test_logs_warning_on_import_failure(
        self, rthook_module, frozen_env, monkeypatch
    ):
        """The probe must log WARNING + not raise when triton import fails.

        Simulates ``import triton`` failing via ``sys.modules['triton'] = None``
        (Python's documented import-error injection mechanism). The probe
        catches the ImportError and writes a WARNING breadcrumb to the
        rthook debug log; the function returns normally.
        """
        # Force `import triton` to raise. Setting sys.modules[name] = None
        # causes Python's import machinery to raise ImportError on the next
        # `import name` statement (CPython documented behavior).
        monkeypatch.setitem(sys.modules, "triton", None)

        # Call must not raise.
        rthook_module._probe_triton_availability()

        # Verify the WARNING line landed.
        log_path = frozen_env["tmp_path"] / "logs" / "rthook_debug.log"
        assert log_path.exists(), f"rthook debug log not created at {log_path}"
        contents = log_path.read_text()
        assert "WARNING: triton import failed" in contents, (
            f"expected 'WARNING: triton import failed' in log, got:\n{contents}"
        )
