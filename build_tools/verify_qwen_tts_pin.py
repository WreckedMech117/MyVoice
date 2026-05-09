"""verify_qwen_tts_pin.py — pre-build qwen-tts pin verification gate.

Confirms that ``python310/Lib/site-packages/qwen_tts/`` corresponds to the
pinned commit (``1ab0dd75353392f28a0d05d9ca960c9954b13c83`` per Story 16.1
/ D-12) by hashing three load-bearing source files and comparing against
the known-good hashes captured at pin time.

Story tooling-2 AC #2 / Subtask 2.2. Mirrors Story 16.1's runtime trip-wire
(``tests/test_qwen_tts_internals.py``) at build time rather than test time.

Mechanism choice (Subtask 2.1):
  (i)  ``qwen_tts/__init__.py.__version__`` check — REJECTED. The package
       declares ``__version__`` in ``__all__`` (line 24) but does not
       actually define the symbol, so an ``import qwen_tts; qwen_tts.__version__``
       would AttributeError. Upstream has not adopted a version-string
       discipline.
  (ii) SHA-256 hash of load-bearing files — CHOSEN. Fully automated; no
       clone-state assumption; resilient to non-load-bearing whitespace
       changes (because we hash only 3 specific files, not the whole tree).
  (iii) ``git -C <local-clone> rev-parse HEAD`` — REJECTED. Assumes the
       maintainer installs qwen-tts via ``pip install -e <local-clone>``;
       the actual install pattern is ``pip install git+https://...@<sha>``,
       which leaves no clone state on disk.

Failing the check halts ``build_release.bat`` at "Pre-Build Checks" via its
``if %ERRORLEVEL% NEQ 0`` guard.

Pin-bump regeneration (when upstream issues a needed update):
    python310\\python.exe build_tools\\verify_qwen_tts_pin.py --regenerate
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


PINNED_COMMIT = "1ab0dd75353392f28a0d05d9ca960c9954b13c83"

# Known-good SHA-256 hashes of the load-bearing qwen_tts files at the pinned
# commit. Captured 2026-05-08 from the maintainer's correctly-installed
# python310/Lib/site-packages/qwen_tts/ on Windows 11 (file-byte ordering
# preserved as installed by pip; no LF/CRLF normalization).
KNOWN_GOOD_HASHES = {
    "__init__.py":
        "2f2d51d7c65be2afa47675760dafb57f0f8cf48d4db3f4aa337b3bb4561004b5",
    "inference/qwen3_tts_model.py":
        "8498559de22a9e152d1fef70d046eb0c7c5fba0dfcfb9592d3c662e3b15d87e8",
    "core/models/modeling_qwen3_tts.py":
        "2f4b6c451195b94b61b210ef840d2194ff64d20459ded55ef9abf5025c05bedd",
}


def _qwen_tts_pkg_path() -> Path:
    """Resolve qwen_tts package path: ../python310/Lib/site-packages/qwen_tts/."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent / "python310" / "Lib" / "site-packages" / "qwen_tts"


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _print_regenerated_hashes(pkg: Path) -> int:
    print("Regenerated KNOWN_GOOD_HASHES (paste into this script):")
    print("KNOWN_GOOD_HASHES = {")
    for relpath in KNOWN_GOOD_HASHES.keys():
        file_path = pkg / relpath
        if file_path.exists():
            h = _hash_file(file_path)
            print(f'    "{relpath}":')
            print(f'        "{h}",')
        else:
            print(f"    # MISSING: {relpath}")
    print("}")
    print()
    print(f"Update PINNED_COMMIT to the new hash too.")
    return 0


def _verify(pkg: Path) -> int:
    mismatches = []
    for relpath, expected in KNOWN_GOOD_HASHES.items():
        file_path = pkg / relpath
        if not file_path.exists():
            mismatches.append((relpath, "MISSING", expected))
            continue
        actual = _hash_file(file_path)
        if actual != expected:
            mismatches.append((relpath, actual, expected))

    if mismatches:
        print("=" * 76)
        print("ERROR: qwen_tts pin verification FAILED")
        print("=" * 76)
        print()
        print(f"Pinned commit (per Story 16.1 / D-12):")
        print(f"  {PINNED_COMMIT}")
        print()
        print("File-hash mismatches:")
        for relpath, actual, expected in mismatches:
            print(f"  {relpath}")
            print(f"    expected: {expected}")
            print(f"    actual:   {actual}")
        print()
        print("Likely cause: qwen-tts in python310/ has drifted from the pinned")
        print("commit (e.g., a debugging session reinstalled from upstream HEAD).")
        print()
        print("To restore the pinned commit:")
        print(f"  python310\\python.exe -m pip install --force-reinstall \\")
        print(f'    "qwen-tts @ git+https://github.com/QwenLM/Qwen3-TTS.git@{PINNED_COMMIT}"')
        print()
        print("To bump the pin (only if upstream issued a needed update):")
        print(f"  1. Update requirements.txt and build_tools/requirements-production.txt")
        print(f"     to point to the new commit hash.")
        print(f"  2. Run: python310\\python.exe build_tools\\verify_qwen_tts_pin.py --regenerate")
        print(f"  3. Paste the regenerated KNOWN_GOOD_HASHES dict into this script.")
        print(f"  4. Update PINNED_COMMIT to the new hash.")
        print(f"  5. Run tests/test_qwen_tts_internals.py to confirm runtime")
        print(f"     contracts still hold (Story 16.1 trip-wire).")
        return 1

    print(f"+ qwen_tts pin verified (commit {PINNED_COMMIT[:10]}...)")
    return 0


def main(argv: list[str]) -> int:
    pkg = _qwen_tts_pkg_path()

    if not pkg.exists():
        print(f"ERROR: qwen_tts package not found at {pkg}")
        print(f"  Install with: python310\\python.exe -m pip install -r requirements.txt")
        return 2

    regenerate = "--regenerate" in argv
    if regenerate:
        return _print_regenerated_hashes(pkg)
    return _verify(pkg)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
