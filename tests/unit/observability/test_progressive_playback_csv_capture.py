"""Story 18.1 Task 1.4 — env-var-gated progressive-playback CSV capture tests.

Covers the public surface of
``myvoice.observability.progressive_playback_csv_capture``:

  * ``_resolve_path``: env-value → CSV path resolution (default / absolute /
    relative-to-logs / disabled cases).
  * ``enable_csv_capture``: header is written immediately; only the three
    Story 18.1 metric names are captured; the listener flushes per-record;
    ``stop()`` is idempotent and removes the listener.
  * ``maybe_enable_from_env``: respects the env-var; failure to open the
    file logs but does not raise.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from myvoice.observability import metrics
from myvoice.observability.progressive_playback_csv_capture import (
    _resolve_path,
    enable_csv_capture,
    maybe_enable_from_env,
)


@pytest.fixture
def tmp_logs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "logs"
    d.mkdir()
    return d


class TestResolvePath:
    def test_default_path_for_value_1(self, tmp_logs_dir):
        assert _resolve_path("1", tmp_logs_dir) == (
            tmp_logs_dir / "18-1-instrumentation-rtx5090-longform.csv"
        )

    def test_disabled_for_value_0(self, tmp_logs_dir):
        assert _resolve_path("0", tmp_logs_dir) is None

    def test_disabled_for_empty_or_whitespace(self, tmp_logs_dir):
        assert _resolve_path("", tmp_logs_dir) is None
        assert _resolve_path("   ", tmp_logs_dir) is None

    def test_absolute_path_used_verbatim(self, tmp_path, tmp_logs_dir):
        target = tmp_path / "elsewhere" / "x.csv"
        assert _resolve_path(str(target), tmp_logs_dir) == target

    def test_relative_path_anchored_at_logs_dir(self, tmp_logs_dir):
        resolved = _resolve_path("subdir/x.csv", tmp_logs_dir)
        assert resolved == tmp_logs_dir / "subdir" / "x.csv"


class TestEnableCsvCapture:
    def test_header_written_immediately(self, tmp_logs_dir):
        path = tmp_logs_dir / "out.csv"
        stop = enable_csv_capture(path)
        try:
            with path.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
            assert header == [
                "metric_name",
                "value",
                "session_id",
                "chunk_index",
                "is_final",
                "audio_data_size",
            ]
        finally:
            stop()

    def test_only_targeted_metrics_are_captured(self, tmp_logs_dir):
        path = tmp_logs_dir / "out.csv"
        stop = enable_csv_capture(path)
        try:
            metrics.record(
                "progressive_chunk_emit_ms",
                1.5,
                session_id="s",
                chunk_index=0,
            )
            # Off-target metrics must NOT land in the CSV — they would
            # clutter the Task 1.4 ratio analysis with unrelated rows.
            metrics.record("first_chunk_latency_ms", 4500.0, session_id="s")
            metrics.record("queue_depth", 3)
            metrics.record(
                "progressive_chunk_audio_duration_ms",
                100.0,
                chunk_index=0,
            )
            metrics.record(
                "progressive_chunk_playback_arrival_ms",
                12345.6,
                chunk_index=0,
                is_final=False,
                audio_data_size=2400,
            )
        finally:
            stop()

        rows = list(csv.reader(path.open("r", encoding="utf-8")))
        # Header + 3 captured rows (first_chunk_latency_ms / queue_depth filtered out).
        assert len(rows) == 1 + 3, f"unexpected rows: {rows}"
        captured_names = {r[0] for r in rows[1:]}
        assert captured_names == {
            "progressive_chunk_emit_ms",
            "progressive_chunk_audio_duration_ms",
            "progressive_chunk_playback_arrival_ms",
        }

    def test_row_columns_match_header_for_all_three_metrics(self, tmp_logs_dir):
        path = tmp_logs_dir / "out.csv"
        stop = enable_csv_capture(path)
        try:
            metrics.record(
                "progressive_chunk_emit_ms",
                1.5,
                session_id="abc",
                chunk_index=2,
            )
            metrics.record(
                "progressive_chunk_playback_arrival_ms",
                42.0,
                chunk_index=2,
                is_final=True,
                audio_data_size=4096,
            )
            metrics.record(
                "progressive_chunk_audio_duration_ms",
                170.0,
                chunk_index=2,
            )
        finally:
            stop()

        rows = list(csv.reader(path.open("r", encoding="utf-8")))[1:]
        emit_row = next(r for r in rows if r[0] == "progressive_chunk_emit_ms")
        assert emit_row == ["progressive_chunk_emit_ms", "1.5", "abc", "2", "", ""]

        arrival_row = next(
            r for r in rows if r[0] == "progressive_chunk_playback_arrival_ms"
        )
        assert arrival_row == [
            "progressive_chunk_playback_arrival_ms",
            "42.0",
            "",  # session_id absent on consumer-side metric
            "2",
            "True",
            "4096",
        ]

        duration_row = next(
            r for r in rows if r[0] == "progressive_chunk_audio_duration_ms"
        )
        assert duration_row == [
            "progressive_chunk_audio_duration_ms",
            "170.0",
            "",
            "2",
            "",
            "",
        ]

    def test_stop_is_idempotent_and_removes_listener(self, tmp_logs_dir):
        path = tmp_logs_dir / "out.csv"
        stop = enable_csv_capture(path)
        # Idempotent: two stop calls must not raise.
        stop()
        stop()
        # After stop, further records must NOT land in the CSV.
        metrics.record(
            "progressive_chunk_emit_ms",
            99.9,
            session_id="post-stop",
            chunk_index=0,
        )
        rows = list(csv.reader(path.open("r", encoding="utf-8")))
        # Only the header should be present.
        assert len(rows) == 1


class TestMaybeEnableFromEnv:
    def test_no_env_var_returns_none(self, tmp_logs_dir, monkeypatch):
        monkeypatch.delenv("MYVOICE_PROGRESSIVE_PLAYBACK_CSV", raising=False)
        assert maybe_enable_from_env(tmp_logs_dir) is None

    def test_env_var_zero_returns_none(self, tmp_logs_dir, monkeypatch):
        monkeypatch.setenv("MYVOICE_PROGRESSIVE_PLAYBACK_CSV", "0")
        assert maybe_enable_from_env(tmp_logs_dir) is None

    def test_env_var_one_creates_default_file_and_returns_stop(
        self, tmp_logs_dir, monkeypatch
    ):
        monkeypatch.setenv("MYVOICE_PROGRESSIVE_PLAYBACK_CSV", "1")
        stop = maybe_enable_from_env(tmp_logs_dir)
        try:
            assert stop is not None
            target = (
                tmp_logs_dir / "18-1-instrumentation-rtx5090-longform.csv"
            )
            assert target.exists(), (
                "default-path file was not created when env-var=1"
            )
        finally:
            if stop is not None:
                stop()

    def test_env_var_open_failure_returns_none_does_not_raise(
        self, tmp_logs_dir, monkeypatch
    ):
        # Point at a path whose parent is a regular file — mkdir(parents=True)
        # cannot create a directory under a file, so open() will fail.
        bogus_parent = tmp_logs_dir / "not_a_dir"
        bogus_parent.write_text("blocker", encoding="utf-8")
        monkeypatch.setenv(
            "MYVOICE_PROGRESSIVE_PLAYBACK_CSV",
            str(bogus_parent / "child" / "x.csv"),
        )
        assert maybe_enable_from_env(tmp_logs_dir) is None
