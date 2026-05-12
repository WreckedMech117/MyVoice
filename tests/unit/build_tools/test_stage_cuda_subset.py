"""Tests for Story 18.5 Task 2.2 — `build_tools/stage_cuda_subset.py`.

Per `memory/code_review_regression_test_exact_class.md`, these tests target
the EXACT bug class: "future maintainer adds NVCC (or any forbidden
developer-tool executable) to the staged CUDA subset and ships a NVIDIA EULA
§1.1.2 #4 license violation in the production bundle". The test exercises
the staging script's hard-reject path against a fake source tree containing
``nvcc.exe`` (and variants) — assert the script exits nonzero + raises
``LicenseViolationError`` with a clear message naming the offending file.

This is the lowest-cost defense against the highest-cost-of-failure bug
class in Story 18.5: a license-violating bundle could ship for weeks before
NVIDIA notices, with legal exposure scaling per-install. The unit test
exercises the script-level enforcement of EULA §1.1.2 #4; the staging script
itself adds a defense-in-depth post-stage audit.

Loading discipline: stage_cuda_subset.py is in ``build_tools/`` (not on
``sys.path``); tests load it via ``importlib.util.spec_from_file_location``
to avoid polluting sys.path globally.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def staging_module():
    """Load build_tools/stage_cuda_subset.py as a module."""
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "build_tools" / "stage_cuda_subset.py"
    assert script_path.exists(), f"stage_cuda_subset.py missing at {script_path}"

    spec = importlib.util.spec_from_file_location(
        "stage_cuda_subset_under_test", str(script_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_cuda_path(tmp_path):
    """Materialize a minimal valid CUDA_PATH layout under tmp_path.

    Includes the validation prereqs `_resolve_cuda_path` checks (bin/ dir,
    EULA.txt at root) so tests can route into the staging logic without
    tripping the upstream existence guards. Tests that want to inject
    forbidden files do so on top of this baseline.
    """
    cuda = tmp_path / "cuda"
    (cuda / "bin").mkdir(parents=True)
    (cuda / "include" / "crt").mkdir(parents=True)

    # Minimal legal redistributables so _stage_subset has something to copy.
    (cuda / "bin" / "cudart64_12.dll").write_bytes(b"fake-cudart")
    (cuda / "bin" / "nvrtc64_120_0.dll").write_bytes(b"fake-nvrtc")
    (cuda / "bin" / "nvrtc-builtins64_128.dll").write_bytes(b"fake-builtins")
    (cuda / "include" / "crt" / "device_functions.h").write_bytes(b"fake header")
    (cuda / "include" / "crt" / "mma.h").write_bytes(b"fake mma")

    # EULA — required by _resolve_cuda_path's existence guard.
    (cuda / "EULA.txt").write_text("FAKE EULA TEXT FOR TESTING\n", encoding="utf-8")
    return cuda


# ---------------------------------------------------------------------------
# LICENSE-VIOLATION REGRESSION TESTS (Task 2.2)
# ---------------------------------------------------------------------------
# These tests exercise the EXACT bug class per
# `memory/code_review_regression_test_exact_class.md`: "future maintainer
# adds NVCC to the bundle list and ships a license violation".
# ---------------------------------------------------------------------------

class TestLicenseViolationRejection:
    """Story 18.5 Task 2.2 — NVCC hard-reject regression coverage.

    Bug class per `memory/code_review_regression_test_exact_class.md`:
    "future maintainer broadens DLL_PATTERNS (or HEADER_PATTERN) so it
    overlaps with FORBIDDEN_PATTERNS, and the broadened glob silently
    copies nvcc.exe (or another developer-tool binary) into the staged
    subset". Real CUDA Toolkit installs ship nvcc.exe in `%CUDA_PATH%/bin/`
    by construction; that file's presence in the source tree is fine
    (the legitimate stage patterns don't match it). The violation is
    COPYING it into the staged subset. These tests simulate the bug
    class by monkey-patching DLL_PATTERNS to overlap with the forbidden
    pattern set, then assert the script catches the overlap.
    """

    def test_rejects_when_dll_patterns_would_copy_nvcc(
        self, staging_module, fake_cuda_path, tmp_path, monkeypatch
    ):
        """Simulate the bug: DLL_PATTERNS broadened to include nvcc.exe.

        Bug class: a future maintainer adds `'bin/nvcc.exe'` (or broadens
        an existing pattern to `'bin/*.exe'`) to DLL_PATTERNS. The
        `_reject_forbidden_in_candidates` check must catch this before
        any file is staged.
        """
        # Plant nvcc.exe in the fake source tree (mirrors a real CUDA install).
        (fake_cuda_path / "bin" / "nvcc.exe").write_bytes(b"fake nvcc binary")

        # Simulate the bug: future maintainer broadens DLL_PATTERNS to include nvcc.
        monkeypatch.setattr(
            staging_module,
            "DLL_PATTERNS",
            (
                'bin/cudart64_*.dll',
                'bin/nvrtc64_*.dll',
                'bin/nvrtc-builtins64_*.dll',
                'bin/nvcc.exe',  # ← the bug: never bundle this
            ),
        )

        monkeypatch.setenv("CUDA_PATH", str(fake_cuda_path))
        target = tmp_path / "cuda_toolkit_subset_test"

        exit_code = staging_module.main(["--target", str(target)])
        assert exit_code != 0, (
            "stage_cuda_subset.py exited 0 despite DLL_PATTERNS including "
            "'bin/nvcc.exe' — license-violation regression: NVCC is not "
            "redistributable per NVIDIA EULA §1.1.2 #4."
        )

        # nvcc must NOT have been staged.
        if target.exists():
            staged_nvcc = list(target.rglob("nvcc.exe"))
            assert not staged_nvcc, (
                f"nvcc.exe staged to target tree despite pre-stage audit: "
                f"{[str(p) for p in staged_nvcc]}"
            )

    def test_rejects_when_dll_patterns_use_broad_glob(
        self, staging_module, fake_cuda_path, tmp_path, monkeypatch
    ):
        """Simulate the bug: DLL_PATTERNS broadened to `bin/*.exe`.

        A future maintainer could refactor DLL_PATTERNS to be more
        inclusive (e.g., catch new NVRTC variants automatically). A
        `'bin/*.exe'` glob would absorb nvcc.exe + __nvcc_device_query.exe
        + any other dev-tool binaries. The pre-stage audit must catch this.
        """
        (fake_cuda_path / "bin" / "nvcc.exe").write_bytes(b"fake")
        (fake_cuda_path / "bin" / "cudart64_12.exe").write_bytes(b"fake")  # legit-ish

        monkeypatch.setattr(
            staging_module,
            "DLL_PATTERNS",
            (
                'bin/*.exe',  # ← the bug: too broad
            ),
        )

        monkeypatch.setenv("CUDA_PATH", str(fake_cuda_path))
        target = tmp_path / "cuda_toolkit_subset_test"

        exit_code = staging_module.main(["--target", str(target)])
        assert exit_code != 0, "DLL_PATTERNS='bin/*.exe' must be hard-rejected"

    def test_rejects_nvcc_variant(
        self, staging_module, fake_cuda_path, tmp_path, monkeypatch
    ):
        """Reject `bin/nvcc-*.exe` variants when included in DLL_PATTERNS."""
        (fake_cuda_path / "bin" / "nvcc-experimental.exe").write_bytes(b"fake")

        monkeypatch.setattr(
            staging_module,
            "DLL_PATTERNS",
            (
                'bin/cudart64_*.dll',
                'bin/nvrtc64_*.dll',
                'bin/nvrtc-builtins64_*.dll',
                'bin/nvcc-*.exe',  # ← the bug
            ),
        )
        monkeypatch.setenv("CUDA_PATH", str(fake_cuda_path))
        target = tmp_path / "cuda_toolkit_subset_test"

        exit_code = staging_module.main(["--target", str(target)])
        assert exit_code != 0, "nvcc-*.exe variant must be hard-rejected"

    def test_rejects_nvcc_device_query_helper(
        self, staging_module, fake_cuda_path, tmp_path, monkeypatch
    ):
        """Reject `bin/__nvcc_device_query.exe` (real CUDA 12.8 install).

        Per Task 1.5 inventory at evidence file §"Build-host CUDA Toolkit
        inventory", real CUDA Toolkit 12.8 ships this file in bin/ alongside
        nvcc.exe. Treating it as part of the nvcc dev-tool toolchain per
        EULA §1.1.2 #4, reject it if a future maintainer broadens
        DLL_PATTERNS to include it.
        """
        (fake_cuda_path / "bin" / "__nvcc_device_query.exe").write_bytes(b"fake")

        monkeypatch.setattr(
            staging_module,
            "DLL_PATTERNS",
            (
                'bin/cudart64_*.dll',
                'bin/nvrtc64_*.dll',
                'bin/nvrtc-builtins64_*.dll',
                'bin/__nvcc_device_query.exe',  # ← the bug
            ),
        )
        monkeypatch.setenv("CUDA_PATH", str(fake_cuda_path))
        target = tmp_path / "cuda_toolkit_subset_test"

        exit_code = staging_module.main(["--target", str(target)])
        assert exit_code != 0, "__nvcc_device_query.exe must be hard-rejected"

    def test_rejects_via_LicenseViolationError_class(
        self, staging_module, fake_cuda_path, monkeypatch
    ):
        """The rejection path raises `LicenseViolationError` specifically.

        Pin the exception class (not just any RuntimeError) so future
        refactors that broaden the error type get caught in code review.
        Exception message must name the offending pattern + file so the
        maintainer can audit without re-reading the source.
        """
        (fake_cuda_path / "bin" / "nvcc.exe").write_bytes(b"fake")

        # Synthesize a candidate list that includes nvcc.exe (simulates a
        # broadened DLL_PATTERNS having matched it).
        candidates = [(fake_cuda_path / "bin" / "nvcc.exe", "bin")]

        with pytest.raises(staging_module.LicenseViolationError) as exc_info:
            staging_module._reject_forbidden_in_candidates(candidates, fake_cuda_path)

        msg = str(exc_info.value)
        assert "LICENSE VIOLATION" in msg
        assert "nvcc.exe" in msg
        assert "EULA" in msg

    def test_clean_source_with_nvcc_present_still_passes(
        self, staging_module, fake_cuda_path, tmp_path, monkeypatch
    ):
        """Real CUDA Toolkit installs ship nvcc.exe in bin/ — that's fine.

        Negative-of-the-positive: confirms the corrected contract — the
        check is on STAGE CANDIDATES (what would be copied), NOT on the
        source tree at large. A real CUDA Toolkit install has nvcc.exe in
        source by construction; the script must succeed (only copies the
        legitimate cudart/nvrtc/headers).
        """
        # Plant nvcc.exe in source — simulates a real CUDA Toolkit install.
        (fake_cuda_path / "bin" / "nvcc.exe").write_bytes(b"fake nvcc")
        monkeypatch.setenv("CUDA_PATH", str(fake_cuda_path))
        target = tmp_path / "cuda_toolkit_subset_test"

        # DLL_PATTERNS unchanged (legitimate); script must succeed.
        exit_code = staging_module.main(["--target", str(target)])
        assert exit_code == 0, (
            f"script failed with exit={exit_code} despite legitimate DLL_PATTERNS — "
            "the corrected contract is to check STAGE CANDIDATES, not source tree. "
            "Real CUDA installs have nvcc.exe in source by construction."
        )

        # nvcc.exe must NOT be in the staged output.
        assert not (target / "bin" / "nvcc.exe").exists(), (
            "nvcc.exe was staged despite legitimate DLL_PATTERNS — staging logic regressed."
        )


# ---------------------------------------------------------------------------
# HAPPY-PATH SMOKE (sanity check that the script can stage a clean tree)
# ---------------------------------------------------------------------------

class TestStageSubsetHappyPath:
    """Sanity smoke that the staging logic produces the expected tree."""

    def test_stages_dlls_headers_eula_with_hash(
        self, staging_module, fake_cuda_path, tmp_path, monkeypatch
    ):
        """End-to-end: clean source produces a complete staged tree."""
        monkeypatch.setenv("CUDA_PATH", str(fake_cuda_path))
        target = tmp_path / "cuda_toolkit_subset_test"

        exit_code = staging_module.main(["--target", str(target)])
        assert exit_code == 0, f"main exited {exit_code} on clean source tree"

        # DLLs staged.
        assert (target / "bin" / "cudart64_12.dll").exists()
        assert (target / "bin" / "nvrtc64_120_0.dll").exists()
        assert (target / "bin" / "nvrtc-builtins64_128.dll").exists()

        # Headers staged.
        assert (target / "include" / "crt" / "device_functions.h").exists()
        assert (target / "include" / "crt" / "mma.h").exists()

        # EULA + hash file.
        assert (target / "EULA.txt").exists()
        assert (target / "EULA.txt.sha256").exists()

        # Hash file content is a 64-char hex string + newline.
        hash_content = (target / "EULA.txt.sha256").read_text().strip()
        assert len(hash_content) == 64
        assert all(c in "0123456789abcdef" for c in hash_content)

    def test_idempotent_restage(
        self, staging_module, fake_cuda_path, tmp_path, monkeypatch
    ):
        """Re-running the script wipes + re-stages cleanly.

        Story 18.5 Task 2.1 contract: "Idempotent — re-running the script
        re-stages cleanly (deletes the target dir first if it exists)".
        """
        monkeypatch.setenv("CUDA_PATH", str(fake_cuda_path))
        target = tmp_path / "cuda_toolkit_subset_test"

        # First stage.
        assert staging_module.main(["--target", str(target)]) == 0

        # Plant a stale file in the target.
        (target / "stale.txt").write_text("should be removed")

        # Re-stage.
        assert staging_module.main(["--target", str(target)]) == 0

        # Stale file should be gone.
        assert not (target / "stale.txt").exists()
        # But the real files should be back.
        assert (target / "bin" / "cudart64_12.dll").exists()

    def test_missing_cuda_path_env_var_fails_cleanly(
        self, staging_module, tmp_path, monkeypatch
    ):
        """Unset CUDA_PATH must fail with a clear message, not a stack trace."""
        monkeypatch.delenv("CUDA_PATH", raising=False)
        target = tmp_path / "cuda_toolkit_subset_test"

        exit_code = staging_module.main(["--target", str(target)])
        assert exit_code != 0, "unset CUDA_PATH must fail"

    def test_missing_eula_fails(
        self, staging_module, fake_cuda_path, tmp_path, monkeypatch
    ):
        """Missing EULA.txt must fail per NVIDIA EULA §1.1.2 #5.

        Bundling redistributable files without the accompanying EULA is a
        license violation — the script must refuse to proceed.
        """
        (fake_cuda_path / "EULA.txt").unlink()
        monkeypatch.setenv("CUDA_PATH", str(fake_cuda_path))
        target = tmp_path / "cuda_toolkit_subset_test"

        exit_code = staging_module.main(["--target", str(target)])
        assert exit_code != 0, "missing EULA.txt must fail"
