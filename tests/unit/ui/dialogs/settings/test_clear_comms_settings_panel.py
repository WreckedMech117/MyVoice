"""Unit tests for the PRELOADED audio source loader (Story 15.1).

Covers AC #2-#8 of story 15.1 plus the AC #1 shape/dtype contract via
the replay-clone integration test (test #9). The loader is signal-free
and Qt-free — none of these tests need a ``QApplication``; they use
``pytest``'s ``tmp_path`` fixture for filesystem isolation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from myvoice.services.sessions.generation_session import (
    GenerationSession,
    SessionSource,
    SessionState,
)
from myvoice.ui.dialogs.settings.clear_comms_settings_panel import (
    PreloadedAudioLoadError,
    WAV_FILE_DIALOG_FILTER,
    load_preloaded_audio_source,
)

_LOADER_LOGGER = "myvoice.ui.dialogs.settings.clear_comms_settings_panel"


def _write_mono_wav(path: Path, sr: int = 24_000, n_samples: int = 480) -> np.ndarray:
    """Write a small deterministic mono float32 WAV at ``path``; return
    the audio array that was written (so callers can compare values).
    """
    t = np.arange(n_samples, dtype=np.float32) / sr
    audio = (0.25 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return audio


def _write_multichannel_wav(
    path: Path, channels: int, sr: int = 24_000, n_samples: int = 480
) -> np.ndarray:
    """Write a multi-channel float32 WAV with distinct per-channel
    signals so the downmix mean is verifiable.
    """
    t = np.arange(n_samples, dtype=np.float32) / sr
    cols = [
        (0.1 * (i + 1) * np.sin(2 * np.pi * (220.0 * (i + 1)) * t)).astype(np.float32)
        for i in range(channels)
    ]
    audio = np.stack(cols, axis=1)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return audio


# ----- AC #2 + AC #7 — happy path + reentrancy ----------------------- #


def test_loads_mono_wav_returns_float32_1d_with_native_rate(tmp_path: Path) -> None:
    """AC #2 + AC #7: mono WAV → 1-D float32 array + native sample
    rate; two sequential loads return independent arrays.
    """
    target = tmp_path / "mono.wav"
    _write_mono_wav(target, sr=24_000, n_samples=480)

    audio_a, sr_a = load_preloaded_audio_source(target)
    assert audio_a.ndim == 1
    assert audio_a.dtype == np.float32
    assert audio_a.shape == (480,)
    assert sr_a == 24_000

    audio_b, sr_b = load_preloaded_audio_source(target)
    assert audio_a is not audio_b
    np.testing.assert_array_equal(audio_a, audio_b)
    assert sr_a == sr_b


# ----- AC #3 — multi-channel downmix --------------------------------- #


@pytest.mark.parametrize("channels", [2, 3])
def test_loads_multichannel_wav_downmixes_via_mean(
    tmp_path: Path, channels: int
) -> None:
    target = tmp_path / f"multi_{channels}ch.wav"
    written_2d = _write_multichannel_wav(target, channels=channels)

    # soundfile re-quantizes via PCM_16 on the round-trip; read the file
    # back through soundfile directly to compute the expected mean from
    # the same on-disk values, so the assertion isn't tripped by PCM_16
    # quantization noise rather than a real loader bug.
    expected_2d, _expected_sr = sf.read(
        str(target), always_2d=True, dtype="float32"
    )
    expected_mono = expected_2d.mean(axis=1).astype(np.float32)

    audio, sr = load_preloaded_audio_source(target)
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert audio.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(audio, expected_mono)

    # Sanity: the source had the requested channel count.
    assert written_2d.shape[1] == channels
    assert sr == 24_000


# ----- AC #4 — sample-rate preservation ------------------------------ #


@pytest.mark.parametrize("sr", [8_000, 22_050, 44_100, 48_000])
def test_native_sample_rate_preserved(tmp_path: Path, sr: int) -> None:
    target = tmp_path / f"rate_{sr}.wav"
    _write_mono_wav(target, sr=sr, n_samples=sr // 100)

    _audio, returned_sr = load_preloaded_audio_source(target)
    assert returned_sr == sr  # no silent coercion to 24 kHz


# ----- AC #5 — missing or directory path ----------------------------- #


def test_missing_path_raises_chains_file_not_found_and_logs_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    target = tmp_path / "does_not_exist.wav"
    with caplog.at_level(logging.ERROR, logger=_LOADER_LOGGER):
        with pytest.raises(PreloadedAudioLoadError) as exc_info:
            load_preloaded_audio_source(target)
    err = exc_info.value
    assert "File not found" in err.message
    assert isinstance(err.__cause__, FileNotFoundError)
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("not found" in r.getMessage() for r in error_records), (
        "expected an ERROR log naming the missing file; got: "
        f"{[r.getMessage() for r in error_records]}"
    )


def test_directory_path_raises_no_cause_no_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sub = tmp_path / "a_dir"
    sub.mkdir()
    with caplog.at_level(logging.ERROR, logger=_LOADER_LOGGER):
        with pytest.raises(PreloadedAudioLoadError) as exc_info:
            load_preloaded_audio_source(sub)
    err = exc_info.value
    assert "File not found" in err.message
    assert err.__cause__ is None
    # Directory case: validation is the loader's own; no underlying
    # exception, no ERROR log.
    assert not [r for r in caplog.records if r.levelno == logging.ERROR]


def test_permission_error_during_stat_wraps_as_could_not_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A PermissionError from ``Path.is_file()`` (e.g., locked Windows
    path) must be surfaced as ``PreloadedAudioLoadError`` so the panel's
    ``except PreloadedAudioLoadError`` catches it.
    """
    target = tmp_path / "protected.wav"
    target.write_bytes(b"placeholder")  # so absence isn't the failure mode

    def _denied(self: Path) -> bool:
        raise PermissionError(13, "Access is denied", str(self))

    monkeypatch.setattr(Path, "is_file", _denied)

    with caplog.at_level(logging.ERROR, logger=_LOADER_LOGGER):
        with pytest.raises(PreloadedAudioLoadError) as exc_info:
            load_preloaded_audio_source(target)
    err = exc_info.value
    assert "Could not access file" in err.message
    assert isinstance(err.__cause__, PermissionError)
    assert any(
        "could not stat" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.ERROR
    )


def test_toctou_file_vanished_between_check_and_read_reports_access_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If the file passes the stat probe but ``sf.read`` then raises
    FileNotFoundError (race window), the user should see an access
    error, not a misleading "corrupt file" message.
    """
    target = tmp_path / "vanishes.wav"
    _write_mono_wav(target, sr=24_000, n_samples=240)  # exists for the probe

    def _vanished(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", args[0])

    monkeypatch.setattr(
        "myvoice.ui.dialogs.settings.clear_comms_settings_panel.sf.read",
        _vanished,
    )

    with caplog.at_level(logging.ERROR, logger=_LOADER_LOGGER):
        with pytest.raises(PreloadedAudioLoadError) as exc_info:
            load_preloaded_audio_source(target)
    err = exc_info.value
    assert "Could not access file" in err.message
    assert "corrupt" not in err.message
    assert isinstance(err.__cause__, FileNotFoundError)
    assert any(
        "vanished" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.ERROR
    )


# ----- AC #6 — non-WAV extension layer ------------------------------- #


@pytest.mark.parametrize(
    "suffix",
    [".mp3", ".m4a", ".flac", ""],
)
def test_non_wav_extension_raises_wav_only_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    """AC #6 (extension layer): non-``.wav`` paths must raise
    *before* ``sf.read`` is reached. Patch ``sf.read`` to raise; if
    the loader called it, the test would see the patched exception
    bubble up as a corrupt-file error rather than the wav-only error.
    """
    target = tmp_path / f"audio{suffix}"
    target.write_bytes(b"some non-wav content")

    def _no_read(*args, **kwargs):
        raise AssertionError("sf.read should not be called for non-WAV path")

    monkeypatch.setattr(
        "myvoice.ui.dialogs.settings.clear_comms_settings_panel.sf.read",
        _no_read,
    )

    with pytest.raises(PreloadedAudioLoadError) as exc_info:
        load_preloaded_audio_source(target)
    err = exc_info.value
    assert "Only WAV files are supported" in err.message
    assert err.__cause__ is None


# ----- AC #5 + AC #6 — corrupt / spoofed WAV content ----------------- #


@pytest.mark.parametrize(
    "content",
    [b"NOT A WAV HEADER\x00\x00\x00\x00", b"hello world"],
)
def test_corrupt_or_spoofed_wav_raises_corrupt_message(
    tmp_path: Path, content: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    target = tmp_path / "spoofed.wav"
    target.write_bytes(content)

    with caplog.at_level(logging.ERROR, logger=_LOADER_LOGGER):
        with pytest.raises(PreloadedAudioLoadError) as exc_info:
            load_preloaded_audio_source(target)
    err = exc_info.value
    assert "corrupt" in err.message or "not a valid WAV" in err.message
    # The original soundfile exception is chained as __cause__.
    assert err.__cause__ is not None
    assert isinstance(err.__cause__, BaseException)
    assert any(
        "failed to read" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.ERROR
    )


# ----- AC #5 / AC #7 — user-facing message hygiene -------------------- #


def test_load_error_message_is_user_facing(tmp_path: Path) -> None:
    """For every error path: ≤80 chars, ASCII-only, no stack traces,
    and ``str(error) == error.message``.
    """
    cases: list[tuple[Path, type[Exception] | None]] = []

    # Missing path.
    cases.append((tmp_path / "missing.wav", None))

    # Directory path.
    a_dir = tmp_path / "a_dir2"
    a_dir.mkdir()
    cases.append((a_dir, None))

    # Non-WAV extension (file exists).
    bad_ext = tmp_path / "audio.mp3"
    bad_ext.write_bytes(b"fake")
    cases.append((bad_ext, None))

    # Corrupt .wav (sf.read fails).
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"NOT A WAV HEADER\x00\x00\x00\x00")
    cases.append((corrupt, None))

    for path, _ in cases:
        with pytest.raises(PreloadedAudioLoadError) as exc_info:
            load_preloaded_audio_source(path)
        err = exc_info.value
        assert len(err.message) <= 80, (
            f"Error message exceeds 80 chars ({len(err.message)}): {err.message!r}"
        )
        # ASCII-only.
        err.message.encode("ascii")
        # No Python type names or traceback artifacts.
        assert "Traceback" not in err.message
        assert "PreloadedAudioLoadError" not in err.message
        assert "RuntimeError" not in err.message
        # str(error) yields a clean string identical to .message.
        assert str(err) == err.message


# ----- AC #6 — file-dialog filter constant --------------------------- #


def test_wav_file_dialog_filter_constant() -> None:
    assert WAV_FILE_DIALOG_FILTER == "WAV files (*.wav)"


# ----- AC #1 / AC #8 #9 — replay-clone shape/dtype compatibility ---- #


def test_loaded_audio_constructs_valid_replay_clone_session(
    tmp_path: Path,
) -> None:
    """The loader's output must compose with the existing
    ``GenerationSession`` replay-clone path (state=READY_TO_PLAY +
    complete_audio set), and a subsequent ``clone_for_replay`` must
    share the buffer by reference (D-6 zero-copy).
    """
    target = tmp_path / "preloaded.wav"
    _write_mono_wav(target, sr=24_000, n_samples=240)

    audio, sr = load_preloaded_audio_source(target)

    session = GenerationSession(
        text="(preloaded)",
        voice="preloaded",
        source=SessionSource.PRELOADED,
        state=SessionState.READY_TO_PLAY,
        complete_audio=audio,
        sample_rate=sr,
    )
    assert session.state == SessionState.READY_TO_PLAY
    assert session.complete_audio is audio
    assert session.sample_rate == sr

    clone = session.clone_for_replay()
    assert clone.session_id != session.session_id
    assert clone.state == SessionState.READY_TO_PLAY
    # D-6: zero-copy clone shares the buffer by reference.
    assert clone.complete_audio is session.complete_audio
    assert clone.sample_rate == sr
