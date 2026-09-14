"""Story 20.8 AC #3 — launch the SHIPPED GUI with chunk_size forced back to 25.

WHY THIS EXISTS
---------------
The retune's whole justification is a TTFA win, and Phase 1 measured that win
headlessly. Story 20.6 §12 established that a headless number and a GUI number
are not interchangeable and — more importantly — that a **cross-session**
comparison already produced one false conclusion in this epic. So confirming
the shipped TTFA needs a `cs25` GUI arm captured in the SAME sitting as the
`cs10` one, not Story 20.6's `cs25` figure quoted from a different month.

After the retune ships, the committed constant is 10, so the reference arm has
to be produced some other way. This launcher rebinds
``codec_token_streamer.DEFAULT_CHUNK_SIZE`` (and the ``__init__`` defaults it
was bound into at class-definition time) **in-process, before the app starts**,
then runs ``myvoice/main.py``'s ``main()``.

That is the identical mechanism Story 20.1 §5.1 used for its sweep and Story
20.4 used for its round-4 fixture: exactly equivalent to the module-constant
edit the class docstring documents as the tuning path, and it **leaves no
source-tree edit to revert** — which matters a great deal more here, because
the alternative is asking an operator to hand-edit a shipping constant between
launches and hand-edit it back.

DLL ORDERING
------------
``torch`` is imported on the first line, before anything that can reach PyQt6
(``memory/torch_pyqt6_dll_ordering.md``). ``main.py`` enforces the same
invariant; honouring it here as well is free and keeps this file safe to run
directly.

PROVENANCE
----------
Writes ``20-8-cs25-manifest.json`` recording the geometry this process actually
resolved, so the comparison cannot be scored against a mis-declared arm —
Story 20.6's ``_manifest_lookahead`` check exists because a command-line flag
is not provenance.

Usage (normally invoked by ``17_Story_20.8_CS25_Baseline.bat``):
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-8-run-myvoice-at-cs25.py

Working file — gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import torch  # noqa: F401  — MUST precede anything that can pull in PyQt6

import json
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

REFERENCE_CS = 25


def _rebind(chunk_size: int) -> None:
    from myvoice.services.tts_streaming import codec_token_streamer as cts

    cts.DEFAULT_CHUNK_SIZE = chunk_size
    # ``CodecTokenStreamer()`` is constructed with no arguments on the dispatch
    # path, so the geometry comes from the __init__ defaults Python bound at
    # class-definition time. Rebinding the module constant alone would move
    # resolve_streamer_geometry() (and therefore the compile-cache key) while
    # leaving the streamer itself at 10 — the exact D-25 split this epic spent
    # two stories closing. Both have to move.
    cts.CodecTokenStreamer.__init__.__defaults__ = (
        chunk_size,
        cts.DEFAULT_LOOKAHEAD,
        cts.DEFAULT_QUEUE_MAX_FACTOR,
        None,
    )


def main() -> int:
    _rebind(REFERENCE_CS)

    from myvoice.services.tts_streaming import resolve_streamer_geometry
    from myvoice.services.tts_streaming import CodecTokenStreamer

    geom = resolve_streamer_geometry()
    probe = CodecTokenStreamer()
    if geom != (REFERENCE_CS, 0) or probe.chunk_size != REFERENCE_CS:
        print(
            "FATAL: rebind did not take. resolve_streamer_geometry()={} and a "
            "freshly constructed streamer reports chunk_size={}. Refusing to "
            "capture an arm whose geometry is not what it claims."
            .format(geom, probe.chunk_size), file=sys.stderr)
        return 2

    manifest = {
        "arm": "reference",
        "resolved_chunk_size": geom[0],
        "resolved_lookahead": geom[1],
        "decode_window_frames": sum(geom),
        "streamer_probe_chunk_size": probe.chunk_size,
        "mechanism": "in-process __defaults__ rebind; no source-tree edit",
    }
    (Path(__file__).resolve().parent / "20-8-cs25-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 66)
    print("Story 20.8 reference arm — chunk_size forced to {}".format(
        REFERENCE_CS))
    print("  resolve_streamer_geometry() = {}  (decode_window_frames = {})"
          .format(geom, sum(geom)))
    print("  this is the SAME-SITTING control for the cs10 capture.")
    print("=" * 66, flush=True)

    # Run main.py as __main__ so its own DLL-ordering preamble and its
    # ``if __name__ == '__main__'`` entry both behave exactly as they do on a
    # normal launch.
    runpy.run_path(str(REPO_ROOT / "src" / "myvoice" / "main.py"),
                   run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
