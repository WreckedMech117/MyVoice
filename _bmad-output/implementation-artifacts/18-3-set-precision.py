"""Story 18.3 helper — programmatically set AppSettings.tts_precision.

Resolves settings.json via the same in-tree path MyVoice's
``portable_paths.get_config_file_path()`` returns in dev mode
(``<repo_root>/config/settings.json``). The helper deliberately does NOT
import any ``myvoice.*`` module so it stays viable in a bare-Python
context — the DLL-ordering invariant in ``memory/torch_pyqt6_dll_ordering.md``
requires torch to be imported before PyQt6, and ``main.py`` /
``conftest.py`` are the only enforcers of that order. A bare-import
helper that pulls in ``myvoice`` would transitively load torch under
the wrong DLL-search-path setup and raise ``OSError [WinError 1114]``
on Windows.

Sets ``tts_precision`` to the supplied value (one of ``auto`` / ``bf16``
/ ``fp32``) and writes back. Used by the Story 18.3 NFR1 measurement
batch files (Tasks 7.1 + 7.2) to flip between the bf16 and fp32
branches without manual JSON editing.

Usage (from repo root):
    python310/python.exe _bmad-output/implementation-artifacts/18-3-set-precision.py auto
    python310/python.exe _bmad-output/implementation-artifacts/18-3-set-precision.py bf16
    python310/python.exe _bmad-output/implementation-artifacts/18-3-set-precision.py fp32

If settings.json does not exist yet, this script creates one with just
the ``tts_precision`` key — MyVoice's ConfigurationService merges
defaults at load time.

Working file — gitignored under ``_bmad-output/``; not committed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


VALID_PRECISIONS = ("auto", "bf16", "fp32")


def _resolve_settings_path() -> Path:
    """Resolve the dev-tree settings.json path WITHOUT importing myvoice.

    Mirrors ``myvoice.utils.portable_paths.get_config_file_path()``'s
    dev-mode return value (``<repo_root>/config/settings.json``)
    without triggering the torch DLL load that any ``myvoice`` import
    would chain into. The script lives at
    ``<repo_root>/_bmad-output/implementation-artifacts/18-3-set-precision.py``
    so its grandparent's parent is the repo root.
    """
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    return repo_root / "config" / "settings.json"


def main(precision: str) -> int:
    if precision not in VALID_PRECISIONS:
        print(
            f"FATAL: precision must be one of {VALID_PRECISIONS}; got {precision!r}",
            file=sys.stderr,
        )
        return 2

    path = _resolve_settings_path()
    print(f"settings.json path: {path}")

    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"FATAL: could not parse existing settings.json: {e}", file=sys.stderr)
            return 2
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}

    old = data.get("tts_precision", "<absent>")
    data["tts_precision"] = precision
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  tts_precision: {old} -> {precision}")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <auto|bf16|fp32>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
