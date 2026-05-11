"""Story 18.4 helper — programmatically set AppSettings.tts_precision AND tts_compile.

Adapts Story 18.3's ``18-3-set-precision.py`` to toggle BOTH fields in a
single invocation. Resolves settings.json via the same in-tree path
MyVoice's ``portable_paths.get_config_file_path()`` returns in dev mode
(``<repo_root>/config/settings.json``).

The helper deliberately does NOT import any ``myvoice.*`` module so it
stays viable in a bare-Python context — the DLL-ordering invariant in
``memory/torch_pyqt6_dll_ordering.md`` requires torch to be imported
before PyQt6, and ``main.py`` / ``conftest.py`` are the only enforcers of
that order. A bare-import helper that pulls in ``myvoice`` would
transitively load torch under the wrong DLL-search-path setup and raise
``OSError [WinError 1114]`` on Windows.

Usage (from repo root):
    python310/python.exe _bmad-output/implementation-artifacts/18-4-set-precision-and-compile.py <precision> <compile>

Examples:
    # bf16 + compile_on (the production target — branch A)
    python310/python.exe _bmad-output/implementation-artifacts/18-4-set-precision-and-compile.py bf16 on

    # bf16 + compile_off (the bf16-only baseline — branch B; same as Story 18.3)
    python310/python.exe _bmad-output/implementation-artifacts/18-4-set-precision-and-compile.py bf16 off

    # fp32 + compile_off (the strict-eager baseline — branch C)
    python310/python.exe _bmad-output/implementation-artifacts/18-4-set-precision-and-compile.py fp32 off

If settings.json does not exist yet, this script creates one with just
the two keys — MyVoice's ConfigurationService merges defaults at load
time.

Working file — gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


VALID_PRECISIONS = ("auto", "bf16", "fp32")
VALID_COMPILE_VALUES = ("auto", "on", "off")


def _resolve_settings_path() -> Path:
    """Resolve the dev-tree settings.json path WITHOUT importing myvoice."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    return repo_root / "config" / "settings.json"


def main(precision: str, compile_mode: str) -> int:
    if precision not in VALID_PRECISIONS:
        print(
            f"FATAL: precision must be one of {VALID_PRECISIONS}; got {precision!r}",
            file=sys.stderr,
        )
        return 2
    if compile_mode not in VALID_COMPILE_VALUES:
        print(
            f"FATAL: compile_mode must be one of {VALID_COMPILE_VALUES}; got {compile_mode!r}",
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

    old_precision = data.get("tts_precision", "<absent>")
    old_compile = data.get("tts_compile", "<absent>")
    data["tts_precision"] = precision
    data["tts_compile"] = compile_mode
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  tts_precision: {old_precision} -> {precision}")
    print(f"  tts_compile:   {old_compile} -> {compile_mode}")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            f"Usage: python {sys.argv[0]} <auto|bf16|fp32> <auto|on|off>",
            file=sys.stderr,
        )
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
