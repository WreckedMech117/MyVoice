"""stage_cuda_subset.py — Story 18.5 / Phase ⊥-Polish-2-Ship.

Stages the NVIDIA-Attachment-A-redistributable subset of CUDA Toolkit 12.x
from ``%CUDA_PATH%`` to ``build_tools/cuda_toolkit_subset/``. This is the
canonical recipe + the canonical defense against a fresh-clone-can't-build
trap: the script source is git-tracked, its output (binary DLLs + headers)
is gitignored.

NVIDIA license discipline (load-bearing):

  * Bundle ONLY files explicitly authorized by NVIDIA CUDA Toolkit EULA
    Attachment A: ``cudart.dll``, ``nvrtc.dll``, ``nvrtc-builtins.dll``
    (and their version-suffix variations per the §2.6 preamble), plus the
    device-side headers from ``%CUDA_PATH%/include/crt/`` required by
    NVRTC at compile time, plus the EULA itself.
  * EXPLICITLY EXCLUDE ``nvcc.exe``, ``nvcc-*.exe``, and any developer-
    tool executable per EULA §1.1.2 #4 ("Unless a developer tool is
    identified in this Agreement as distributable, it is delivered for
    your internal use only"). Triton's NVRTC compilation path uses runtime
    compilation via NVRTC DLL APIs, NOT the nvcc compiler-driver toolchain.
  * The script HARD-REJECTS any source-path glob that would match
    ``nvcc*`` — even if ``%CUDA_PATH%`` is otherwise correct, the script
    must refuse to stage NVCC. This is the script-level enforcement of
    EULA §1.1.2 #4 + the Task 2.2 regression-test contract.
  * Compute SHA-256 of the staged ``EULA.txt`` after copy + write alongside
    as ``EULA.txt.sha256`` so the next build verifies EULA-version stability.

License-clearance memo: ``_bmad-output/implementation-artifacts/
18-5-cuda-toolkit-triton-bundling-evidence.md §"NVIDIA license clearance"``
(Story 18.5 Task 1.8; gated on Commander sign-off before this script runs
against a real ``%CUDA_PATH%``).

Reference: ``build_tools/verify_qwen_tts_pin.py`` (~140 lines; same
single-purpose-build-script idiom from Story tooling-2).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import shutil
import sys
from pathlib import Path

# File-list contract per Story 18.5 AC #2.
#
# These are glob patterns relative to %CUDA_PATH%. The script captures the
# actual filenames on the build host (which encode the CUDA Toolkit version,
# e.g. ``cudart64_12.dll`` for Toolkit 12.x) rather than hard-coding them.
DLL_PATTERNS = (
    'bin/cudart64_*.dll',
    'bin/nvrtc64_*.dll',           # absorbs nvrtc64_120_0.dll AND .alt variants
    'bin/nvrtc-builtins64_*.dll',
)

# Headers from %CUDA_PATH%/include/crt/ — required by NVRTC at compile time.
# Glob captures .h + .hpp + any new variant CUDA Toolkit ships.
HEADER_PATTERN = 'include/crt/*.h*'

# License-violation guard: any source-tree path matching these patterns is
# REJECTED, even if a future maintainer accidentally adds nvcc to the bundle
# list. This is the regression-test contract per Task 2.2.
FORBIDDEN_PATTERNS = (
    'bin/nvcc.exe',
    'bin/nvcc-*.exe',
    'bin/nvcc',          # POSIX path variant — defensive
    'bin/__nvcc_*',      # device_query helper bundled alongside nvcc
)


class LicenseViolationError(RuntimeError):
    """Raised when a forbidden file would be (or has been) staged.

    The CI / build-pipeline gate against bundling NVCC. The exception
    message names the offending file so the maintainer can audit + remove.

    Story 18.5 Task 2.1 contract: the script must refuse to stage any file
    matching FORBIDDEN_PATTERNS. The check operates on the STAGING SCOPE
    (files that would be / have been copied), NOT on the source tree at
    large — real CUDA Toolkit installs ship `nvcc.exe` in `%CUDA_PATH%/bin/`
    by construction, and that file's mere presence in the source tree is
    not a violation; the violation is COPYING it into the bundle.
    """


def _resolve_cuda_path() -> Path:
    """Read %CUDA_PATH% and validate the install layout."""
    cuda_path = os.environ.get('CUDA_PATH')
    if not cuda_path:
        raise RuntimeError(
            'CUDA_PATH is not set. Install CUDA Toolkit 12.x from '
            'https://developer.nvidia.com/cuda-toolkit and re-open the shell, '
            'OR set CUDA_PATH manually to point at the install root.'
        )
    cuda_dir = Path(cuda_path)
    if not cuda_dir.exists():
        raise RuntimeError(
            f'CUDA_PATH={cuda_path!r} does not exist. Verify the CUDA Toolkit '
            f'install or unset CUDA_PATH.'
        )
    if not (cuda_dir / 'bin').is_dir():
        raise RuntimeError(
            f'CUDA_PATH={cuda_path!r} missing bin/ subdirectory; this does not '
            f'look like a valid CUDA Toolkit install root.'
        )
    if not (cuda_dir / 'EULA.txt').is_file():
        raise RuntimeError(
            f'CUDA_PATH={cuda_path!r} missing EULA.txt at the install root. '
            f'Cannot redistribute without the EULA per NVIDIA EULA §1.1.2 #5.'
        )
    return cuda_dir


def _gather_stage_candidates(source_root: Path) -> list[tuple[Path, str]]:
    """Return the (src, dest_subdir) tuples that _stage_subset would copy.

    Used by both the pre-stage audit (validates DLL_PATTERNS + HEADER_PATTERN
    don't overlap with FORBIDDEN_PATTERNS) and _stage_subset itself (single
    source of truth for what gets copied).
    """
    candidates: list[tuple[Path, str]] = []
    for pattern in DLL_PATTERNS:
        for src in source_root.glob(pattern):
            candidates.append((src, 'bin'))
    for src in source_root.glob(HEADER_PATTERN):
        candidates.append((src, 'include/crt'))
    return candidates


def _reject_forbidden_in_candidates(
    candidates: list[tuple[Path, str]], source_root: Path
) -> None:
    """Hard-reject if any STAGE CANDIDATE matches a forbidden pattern.

    Story 18.5 AC #2 + Task 2.2 regression-test contract: the script must
    refuse to stage NVCC. Catches the bug class "future maintainer broadens
    DLL_PATTERNS (e.g., from `bin/cudart64_*.dll` to `bin/*.dll`) and the
    new pattern would copy nvcc-adjacent DLLs". A genuine CUDA Toolkit
    install contains nvcc.exe in `bin/` — its mere presence in the source
    tree is fine; copying it into the staged subset is the violation.

    Raises LicenseViolationError on the first forbidden match.
    """
    for src, _ in candidates:
        rel = src.relative_to(source_root).as_posix()
        for pattern in FORBIDDEN_PATTERNS:
            # FORBIDDEN_PATTERNS use POSIX-style globs (e.g., 'bin/nvcc.exe',
            # 'bin/nvcc-*.exe'). fnmatch translates these to regex; we
            # compare against the source-relative POSIX path.
            import fnmatch
            if fnmatch.fnmatch(rel, pattern):
                raise LicenseViolationError(
                    f'LICENSE VIOLATION: stage candidate {rel!r} matches '
                    f'forbidden pattern {pattern!r}. NVCC is a developer '
                    f'tool per NVIDIA CUDA Toolkit EULA §1.1.2 #4 and is '
                    f'NOT redistributable. Refusing to stage. This catches '
                    f'the bug class "future maintainer broadened a stage '
                    f'pattern (DLL_PATTERNS or HEADER_PATTERN) so it now '
                    f'overlaps with FORBIDDEN_PATTERNS". Audit the stage '
                    f'patterns in build_tools/stage_cuda_subset.py and '
                    f'tighten them before bundling.'
                )


def _stage_subset(source_root: Path, target_root: Path) -> dict:
    """Copy the AC #2 file set from source_root to target_root.

    Returns a summary dict with per-category counts + the EULA SHA-256.
    """
    summary = {
        'dlls': [],
        'headers': [],
        'eula_sha256': None,
    }

    # Idempotent re-stage: wipe target_root if it exists.
    if target_root.exists():
        shutil.rmtree(target_root)

    (target_root / 'bin').mkdir(parents=True)
    (target_root / 'include' / 'crt').mkdir(parents=True)

    # DLLs
    for pattern in DLL_PATTERNS:
        for src in source_root.glob(pattern):
            dst = target_root / 'bin' / src.name
            shutil.copy2(src, dst)
            summary['dlls'].append(src.name)
            print(f'  [DLL]    {src.name}  ({src.stat().st_size / (1024 * 1024):.2f} MB)')

    # Headers
    for src in source_root.glob(HEADER_PATTERN):
        dst = target_root / 'include' / 'crt' / src.name
        shutil.copy2(src, dst)
        summary['headers'].append(src.name)
    print(f'  [HEADERS] {len(summary["headers"])} files (~'
          f'{sum((source_root / "include" / "crt" / h).stat().st_size for h in summary["headers"]) / 1024:.1f} KB combined)')

    # EULA
    eula_src = source_root / 'EULA.txt'
    eula_dst = target_root / 'EULA.txt'
    shutil.copy2(eula_src, eula_dst)
    eula_sha = hashlib.sha256(eula_dst.read_bytes()).hexdigest()
    (target_root / 'EULA.txt.sha256').write_text(eula_sha + '\n', encoding='ascii')
    summary['eula_sha256'] = eula_sha
    print(f'  [EULA]   EULA.txt  (sha256={eula_sha})')

    return summary


def _post_stage_audit(target_root: Path) -> None:
    """Re-verify that no forbidden file slipped into the staged tree.

    Defense-in-depth: even though _reject_forbidden was checked against the
    source tree, re-verify against the target tree post-copy so the staging
    contract is observably airtight in the script's output.
    """
    for pattern in FORBIDDEN_PATTERNS:
        matches = list(target_root.glob(pattern))
        if matches:
            raise LicenseViolationError(
                f'POST-STAGE LICENSE VIOLATION: forbidden file matched '
                f'pattern {pattern!r} in staged tree {target_root}: '
                f'{[str(m.relative_to(target_root)) for m in matches]}. '
                f'This should be impossible — the source-tree pre-check '
                f'should have rejected first. Investigate before bundling.'
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Stage the NVIDIA-Attachment-A-redistributable subset of '
                    'CUDA Toolkit for the Story 18.5 PyInstaller bundle.'
    )
    parser.add_argument(
        '--target',
        type=str,
        default=None,
        help='Override target directory (default: '
             '<repo_root>/build_tools/cuda_toolkit_subset/).',
    )
    args = parser.parse_args(argv)

    try:
        source_root = _resolve_cuda_path()
    except RuntimeError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2

    if args.target:
        target_root = Path(args.target).resolve()
    else:
        script_dir = Path(__file__).resolve().parent
        target_root = script_dir / 'cuda_toolkit_subset'

    print(f'CUDA_PATH source: {source_root}')
    print(f'Staging target:   {target_root}')
    print()

    # Pre-stage audit: validate the STAGE CANDIDATES don't overlap with
    # FORBIDDEN_PATTERNS. (A real CUDA Toolkit install ships nvcc.exe in
    # bin/ by construction; that's fine — the violation is COPYING it.)
    try:
        candidates = _gather_stage_candidates(source_root)
        _reject_forbidden_in_candidates(candidates, source_root)
    except LicenseViolationError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 3

    # Stage.
    try:
        summary = _stage_subset(source_root, target_root)
    except Exception as exc:
        print(f'ERROR: staging failed: {exc}', file=sys.stderr)
        return 4

    # Post-stage defense-in-depth.
    try:
        _post_stage_audit(target_root)
    except LicenseViolationError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 5

    # Validate: must have at least one cudart + one nvrtc DLL.
    if not any(d.startswith('cudart64_') for d in summary['dlls']):
        print('ERROR: staged tree missing cudart64_*.dll', file=sys.stderr)
        return 6
    if not any(d.startswith('nvrtc64_') for d in summary['dlls']):
        print('ERROR: staged tree missing nvrtc64_*.dll', file=sys.stderr)
        return 7

    # Compute uncompressed size for the evidence file.
    total_size = sum(
        f.stat().st_size for f in target_root.rglob('*') if f.is_file()
    )
    print()
    print(f'PASS — staged {len(summary["dlls"])} DLLs, '
          f'{len(summary["headers"])} headers, 1 EULA + sha256. '
          f'Total uncompressed size: {total_size / (1024 * 1024):.2f} MB.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
