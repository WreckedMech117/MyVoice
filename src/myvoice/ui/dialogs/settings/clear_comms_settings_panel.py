"""PRELOADED audio source loader (Story 15.1, Phase 5 of D-20).

OFR-B Clear Comms — first story of Epic 15. Adds the WAV-loading
primitive that the Clear Comms button (Story 15.2) and the Settings
panel (Story 15.3) will both consume.

This module currently houses **only** the loader function plus its
exception class and the file-dialog filter constant. Story 15.3 will
append the ``ClearCommsSettingsPanel`` ``QWidget`` class to this same
file when the panel UI lands.

Architecture decisions activated here:

  - **D-17** (line 282-283): "Audio loader for Clear Comms (OFR-B).
    ``soundfile.read()``, WAV-only in v1." → AC #1, #5, #6.
  - **D-5** (line 247-248): PRELOADED-clone exclusion is enforced
    upstream by Story 14.1's saveable-lifecycle policy. This module
    has no responsibility to assert it; the integration test in the
    test file verifies the loader's output composes correctly with
    the existing ``clone_for_replay(source=PRELOADED)`` zero-copy
    path (D-6).
  - **D-6** (line 249-250): zero-copy clone semantics — the audio
    array returned by this loader is the source-of-truth buffer that
    later replay-clones share by reference.
  - **Architecture file map line 704:** "PRELOADED-source loader inline
    in the panel." Interpreted here as "in the same module as the
    panel class" — top-level function rather than a method, so Story
    15.2's button click path can call it without instantiating a
    panel and so unit tests do not need a ``QApplication``.

Module-boundary discipline (architecture lines 689-694): ``ui/*`` may
import ``sessions.session_registry`` and ``sessions.generation_session``
only. For 15.1 specifically, neither is needed — the loader returns a
plain ``(audio, sample_rate)`` tuple. **No** PyQt6 imports in this
story; the panel UI lands in Story 15.3.

Sample-rate policy (resolves the architecture's "consistent rate must
be guaranteed — decision deferred to tech-spec"): **preserve the
file's native sample rate.** The downstream playback pipeline is
already rate-parameterized end-to-end —
``AudioCoordinator.start_streaming_sessions(sample_rate=...)``
propagates an arbitrary rate, the virtual-mic service auto-resamples
to its 48 kHz comms target, and the monitor service opens its stream
at the source rate. Adding resampling in the loader would duplicate
work the pipeline already does. If a future integration test surfaces
a playback issue at an unusual rate, a follow-up story adds
resampling — gated on empirical evidence, not preemptive complexity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

WAV_FILE_DIALOG_FILTER = "WAV files (*.wav)"

_logger = logging.getLogger(__name__)


class PreloadedAudioLoadError(Exception):
    """Raised when ``load_preloaded_audio_source`` cannot produce a
    valid ``(audio, sample_rate)`` tuple.

    Carries a user-facing ``message`` attribute (suitable for direct
    display in a Settings panel error label) plus an optional
    ``cause`` chained via ``__cause__`` so the loader's logger
    preserves the original traceback.
    """

    def __init__(
        self,
        message: str,
        *,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if cause is not None:
            # Own the chain end-to-end: setting both __cause__ and
            # __suppress_context__ matches what `raise X from Y` does, so
            # call sites don't need to repeat `from exc` (which would set
            # __cause__ a second time and risk divergence).
            self.__cause__ = cause
            self.__suppress_context__ = True

    def __str__(self) -> str:
        return self.message


def load_preloaded_audio_source(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file into a ``(mono float32 audio array, sample_rate)``
    tuple compatible with ``GenerationSession.complete_audio`` and
    ``GenerationSession.sample_rate``.

    Reentrant; safe to call concurrently with distinct paths.

    Raises:
        PreloadedAudioLoadError: when the file is missing, unreadable,
            corrupt, non-WAV, or contains audio that cannot be coerced
            into the expected (1-D float32) shape. The error's
            ``message`` is short, user-facing, and free of stack-trace
            artifacts.
    """
    # Probe stat once; wrap so OS-level failures (PermissionError on a
    # protected path) surface as PreloadedAudioLoadError, not raw OSError
    # the panel won't catch.
    try:
        is_file = path.is_file()
        file_exists = True if is_file else path.exists()
    except OSError as exc:
        _logger.error(
            "could not stat preloaded audio source: path=%s, error=%s",
            path,
            exc,
        )
        raise PreloadedAudioLoadError(
            f"Could not access file: {path.name}",
            cause=exc,
        )

    if not is_file:
        if not file_exists:
            # Missing file: chain a synthetic FileNotFoundError per AC #5
            # so the logger preserves the canonical exception type.
            notfound = FileNotFoundError(str(path))
            _logger.error(
                "preloaded audio source not found: path=%s",
                path,
            )
            raise PreloadedAudioLoadError(
                f"File not found: {path.name}",
                cause=notfound,
            )
        # Exists but not a regular file (directory, special). No log,
        # no cause — the loader's own validation surfaced this.
        raise PreloadedAudioLoadError(f"File not found: {path.name}")

    if path.suffix.lower() != ".wav":
        raise PreloadedAudioLoadError(
            f"Only WAV files are supported in this version. "
            f"Cannot load: {path.name}"
        )
    try:
        audio, sr = sf.read(str(path), always_2d=False, dtype="float32")
    except (FileNotFoundError, PermissionError) as exc:
        # TOCTOU: file vanished or became inaccessible between the stat
        # probe and sf.read. Surface as access failure so the user is not
        # misdirected toward "corrupt file" troubleshooting.
        _logger.error(
            "preloaded audio source vanished between check and read: path=%s, error=%s",
            path,
            exc,
        )
        raise PreloadedAudioLoadError(
            f"Could not access file: {path.name}",
            cause=exc,
        )
    except Exception as exc:
        _logger.error(
            "failed to read preloaded audio source: path=%s, error=%s",
            path,
            exc,
        )
        raise PreloadedAudioLoadError(
            f"Could not read WAV file: {path.name} "
            f"- file may be corrupt or not a valid WAV",
            cause=exc,
        )
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    _logger.debug(
        "loaded preloaded audio source: path=%s, samples=%d, sr=%d, dtype=%s",
        path.name,
        len(audio),
        sr,
        audio.dtype,
    )
    return audio, int(sr)


__all__ = [
    "WAV_FILE_DIALOG_FILTER",
    "PreloadedAudioLoadError",
    "load_preloaded_audio_source",
]
