"""Story 17.2 AC #4 — unit tests for the ServiceStatusIndicator's inline
preparing-voice label behavior.

The first dev-story iteration shipped tooltip-only rendering, which the
Commander's bundled-smoke run on 2026-05-08 21:17 confirmed was too subtle
(the message arrived but only on hover). This iteration adds an inline
label change so the message is unmissable during the cold-cache precompute
window.

Invariants verified:
  - Steady-state label shows the service_name (e.g. "TTS").
  - When ``ServiceStatusInfo.preparing_voice_message`` is set, the label
    text changes to the message (truncated past 24 chars) AND becomes bold.
  - When the message clears (None on exit), label reverts to service_name
    AND bold reverts to non-bold.
  - Tooltip carries the full message regardless of label truncation.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from PyQt6.QtWidgets import QApplication

from myvoice.models.service_enums import ServiceStatus
from myvoice.models.ui_state import ServiceHealthStatus, ServiceStatusInfo
from myvoice.ui.components.service_status_indicator import ServiceStatusIndicator


@pytest.fixture(scope="module")
def qapp():
    """QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def indicator(qapp):
    """A fresh ServiceStatusIndicator for the TTS service."""
    widget = ServiceStatusIndicator("TTS")
    yield widget
    widget.cleanup()
    widget.deleteLater()


def _healthy_status_with_message(message=None):
    return ServiceStatusInfo(
        service_name="TTS",
        status=ServiceStatus.RUNNING,
        health_status=ServiceHealthStatus.HEALTHY,
        last_check=datetime.now(),
        preparing_voice_message=message,
    )


class TestPreparingVoiceLabel:
    def test_steady_state_label_is_service_name(self, indicator):
        indicator.update_status(_healthy_status_with_message(None))
        assert indicator._status_label.text() == "TTS"
        assert indicator._status_label.font().bold() is False

    def test_message_set_changes_label_and_bolds_it(self, indicator):
        indicator.update_status(
            _healthy_status_with_message("Preparing voice for streaming…")
        )
        # Long message gets truncated past 24 chars (with ellipsis).
        text = indicator._status_label.text()
        assert text != "TTS"
        assert "Preparing voice" in text
        assert len(text) <= 24
        assert indicator._status_label.font().bold() is True

    def test_short_message_passes_through_untruncated(self, indicator):
        indicator.update_status(_healthy_status_with_message("Loading…"))
        assert indicator._status_label.text() == "Loading…"

    def test_message_cleared_restores_service_name_and_unbold(self, indicator):
        # Set then clear.
        indicator.update_status(
            _healthy_status_with_message("Preparing voice for streaming…")
        )
        assert indicator._status_label.font().bold() is True

        indicator.update_status(_healthy_status_with_message(None))
        assert indicator._status_label.text() == "TTS"
        assert indicator._status_label.font().bold() is False

    def test_tooltip_carries_full_message_even_if_label_truncated(self, indicator):
        full_message = "Preparing voice for streaming…"
        indicator.update_status(_healthy_status_with_message(full_message))
        # Tooltip is built in _update_tooltip via the status_info; check it
        # contains the full (untruncated) message.
        tooltip = indicator.toolTip()
        assert full_message in tooltip
