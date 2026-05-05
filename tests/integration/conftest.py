"""
Shared fixtures for integration tests (Story 11.4).

Provides:
- ``qapp``: module-scoped QApplication (pytest-qt-free, project convention)
- ``signal_records``: factory that records the four D-13 SessionRegistry
    signals (lifted from Story 11.2's test pattern)
- ``LatencyCapture`` / ``latency_capture``: logging.Handler that captures
    every ``first_chunk_latency_ms`` record on the metric stream (lifted
    from Story 11.3's existing handler in
    ``test_qwen_tts_metrics_migration.py``)
- ``fake_generate_sync``: monkeypatch helper to swap
    ``QwenTTSService._generate_sync`` for a deterministic stub. Avoids
    loading a real Qwen3-TTS model in the integration suite.
- ``qwen_service_with_registry``: yields a fully-wired
    ``(service, registry, signal_records, latency_capture)`` tuple. Tear
    down via ``service.stop()`` so the Story 11.3 unsubscribe path runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pytest


# --------------------------------------------------------------------------- #
# QApplication fixture (no pytest-qt — project convention)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication (project convention; no pytest-qt)."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# --------------------------------------------------------------------------- #
# signal_records — capture the four D-13 SessionRegistry signals
# --------------------------------------------------------------------------- #


@pytest.fixture
def signal_records():
    """Factory: ``records = signal_records(registry)`` returns a list of
    ``(signal_name, payload)`` tuples accumulated as the registry emits."""

    def _connect(registry) -> "List[Tuple[str, Any]]":
        captured: "List[Tuple[str, Any]]" = []
        registry.session_state_changed.connect(
            lambda sid, state: captured.append(("session_state_changed", (sid, state)))
        )
        registry.current_session_changed.connect(
            lambda focal: captured.append(("current_session_changed", focal))
        )
        registry.playback_queue_depth_changed.connect(
            lambda depth: captured.append(("playback_queue_depth_changed", depth))
        )
        registry.saveable_session_changed.connect(
            lambda payload: captured.append(("saveable_session_changed", payload))
        )
        return captured

    return _connect


# --------------------------------------------------------------------------- #
# LatencyCapture — capture metric records on the myvoice.metrics logger
# --------------------------------------------------------------------------- #


class LatencyCapture(logging.Handler):
    """Capture every ``first_chunk_latency_ms`` LogRecord on myvoice.metrics.

    Records ``(metric_name, value, tags)`` tuples where ``tags`` carries
    ``session_id`` alongside the other tags so Story 11.4 tests can assert
    on the registry-issued session id.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: "List[Tuple[str, float, Dict[str, Any]]]" = []

    def emit(self, log_record: logging.LogRecord) -> None:
        from myvoice.observability import METRIC_LOGGER_NAME

        if (
            log_record.name == METRIC_LOGGER_NAME
            and getattr(log_record, "metric_name", None) == "first_chunk_latency_ms"
        ):
            tags_with_sid = dict(log_record.tags)
            # Story 11.4: surface session_id alongside other tags so the
            # test assertions don't have to peer at LogRecord internals.
            tags_with_sid["session_id"] = getattr(log_record, "session_id", None)
            self.records.append(
                (
                    log_record.metric_name,
                    log_record.value,
                    tags_with_sid,
                )
            )


@pytest.fixture
def latency_capture(request):
    from myvoice.observability import METRIC_LOGGER_NAME

    logger = logging.getLogger(METRIC_LOGGER_NAME)
    handler = LatencyCapture()
    prior_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    def _teardown() -> None:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)

    request.addfinalizer(_teardown)
    return handler
