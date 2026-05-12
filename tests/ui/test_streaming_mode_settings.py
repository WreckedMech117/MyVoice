"""Story 16.6 — UI tests for ``StreamingSettingsPanel`` (AC #5).

Verifies:
  - The dropdown reflects the persisted ``AppSettings.streaming_mode_override``
    on load.
  - Changing the selection writes back to ``app_settings`` immediately.
  - The "Auto" label round-trips to ``None`` (and vice-versa) per Story
    16.2's already-shipped serializer test (``test_streaming_mode.py``).

Module-boundary discipline: this file imports the panel and AppSettings
only — no QwenTTSService import (would couple UI tests to the dispatcher).
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from myvoice.models.app_settings import AppSettings
from myvoice.ui.dialogs.settings import StreamingSettingsPanel


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# --------------------------------------------------------------------------- #
# AC #5 — TestStreamingModeDropdownInitialState
# --------------------------------------------------------------------------- #


class TestStreamingModeDropdownInitialState:
    """The dropdown reflects the AppSettings value on construction."""

    def test_auto_label_shown_when_override_is_none(self, qapp):
        settings = AppSettings(streaming_mode_override=None)
        panel = StreamingSettingsPanel(app_settings=settings)
        assert "Auto" in panel._mode_combo.currentText()
        assert panel.current_value is None

    def test_true_stream_label_shown_when_override_is_true_stream(self, qapp):
        settings = AppSettings(streaming_mode_override="true_stream")
        panel = StreamingSettingsPanel(app_settings=settings)
        assert "True Stream" in panel._mode_combo.currentText()
        assert panel.current_value == "true_stream"

    def test_sentence_stream_label_shown_when_override_is_sentence_stream(self, qapp):
        settings = AppSettings(streaming_mode_override="sentence_stream")
        panel = StreamingSettingsPanel(app_settings=settings)
        assert "Sentence Stream" in panel._mode_combo.currentText()
        assert panel.current_value == "sentence_stream"

    def test_batch_label_shown_when_override_is_batch(self, qapp):
        settings = AppSettings(streaming_mode_override="batch")
        panel = StreamingSettingsPanel(app_settings=settings)
        assert "Batch" in panel._mode_combo.currentText()
        assert panel.current_value == "batch"


# --------------------------------------------------------------------------- #
# AC #5 — TestStreamingModeDropdownPersistence
# --------------------------------------------------------------------------- #


class TestStreamingModeDropdownPersistence:
    """Changes to the dropdown write back to AppSettings."""

    def test_setting_to_true_stream_updates_app_settings(self, qapp):
        settings = AppSettings(streaming_mode_override=None)
        panel = StreamingSettingsPanel(app_settings=settings)

        # Find and select the True Stream option.
        combo = panel._mode_combo
        for i in range(combo.count()):
            if "True Stream" in combo.itemText(i):
                combo.setCurrentIndex(i)
                break

        assert settings.streaming_mode_override == "true_stream"

    def test_setting_to_sentence_stream_updates_app_settings(self, qapp):
        settings = AppSettings(streaming_mode_override=None)
        panel = StreamingSettingsPanel(app_settings=settings)

        combo = panel._mode_combo
        for i in range(combo.count()):
            if "Sentence Stream" in combo.itemText(i):
                combo.setCurrentIndex(i)
                break

        assert settings.streaming_mode_override == "sentence_stream"

    def test_setting_to_batch_updates_app_settings(self, qapp):
        settings = AppSettings(streaming_mode_override=None)
        panel = StreamingSettingsPanel(app_settings=settings)

        combo = panel._mode_combo
        for i in range(combo.count()):
            if "Batch" in combo.itemText(i):
                combo.setCurrentIndex(i)
                break

        assert settings.streaming_mode_override == "batch"

    def test_setting_back_to_auto_clears_override_to_none(self, qapp):
        settings = AppSettings(streaming_mode_override="true_stream")
        panel = StreamingSettingsPanel(app_settings=settings)
        assert settings.streaming_mode_override == "true_stream"

        combo = panel._mode_combo
        for i in range(combo.count()):
            if "Auto" in combo.itemText(i):
                combo.setCurrentIndex(i)
                break

        assert settings.streaming_mode_override is None


# --------------------------------------------------------------------------- #
# AC #5 — TestStreamingModeDropdownSignals
# --------------------------------------------------------------------------- #


class TestStreamingModeDropdownSignals:
    """The streaming_mode_changed signal fires on user change but not on
    panel construction (which loads state silently)."""

    def test_signal_does_not_fire_during_initial_load(self, qapp):
        settings = AppSettings(streaming_mode_override="true_stream")
        emitted: list = []

        # Connect BEFORE construction is impossible — connect right after,
        # then call load_state to verify no spurious signal there.
        panel = StreamingSettingsPanel(app_settings=settings)
        panel.streaming_mode_changed.connect(lambda val: emitted.append(val))

        # Re-load state — should NOT emit.
        panel.load_state(AppSettings(streaming_mode_override="batch"))
        assert emitted == [], (
            "load_state emitted streaming_mode_changed; "
            "AC #5 requires it to be silent"
        )

    def test_signal_fires_with_new_value_on_user_selection(self, qapp):
        settings = AppSettings(streaming_mode_override=None)
        panel = StreamingSettingsPanel(app_settings=settings)
        emitted: list = []
        panel.streaming_mode_changed.connect(lambda val: emitted.append(val))

        combo = panel._mode_combo
        for i in range(combo.count()):
            if "True Stream" in combo.itemText(i):
                combo.setCurrentIndex(i)
                break

        assert emitted == ["true_stream"]

    def test_signal_emits_none_when_user_picks_auto(self, qapp):
        settings = AppSettings(streaming_mode_override="batch")
        panel = StreamingSettingsPanel(app_settings=settings)
        emitted: list = []
        panel.streaming_mode_changed.connect(lambda val: emitted.append(val))

        combo = panel._mode_combo
        for i in range(combo.count()):
            if "Auto" in combo.itemText(i):
                combo.setCurrentIndex(i)
                break

        assert emitted == [None]


# --------------------------------------------------------------------------- #
# AC #5 — TestSaveStateRoundTrip
# --------------------------------------------------------------------------- #


class TestSaveStateRoundTrip:
    """``save_state`` writes the current dropdown value to a target settings
    object — covers the "write to settings on dialog OK" parent-dialog flow.
    """

    @pytest.mark.parametrize(
        "starting_override,target_label,expected_value",
        [
            (None, "Auto", None),
            (None, "True Stream", "true_stream"),
            (None, "Sentence Stream", "sentence_stream"),
            (None, "Batch", "batch"),
            ("true_stream", "Auto", None),
            ("batch", "True Stream", "true_stream"),
        ],
    )
    def test_save_state_writes_current_selection(
        self, qapp, starting_override, target_label, expected_value
    ):
        # Original settings — initial state for the panel.
        starting_settings = AppSettings(
            streaming_mode_override=starting_override
        )
        panel = StreamingSettingsPanel(app_settings=starting_settings)

        # User picks a different option.
        combo = panel._mode_combo
        for i in range(combo.count()):
            if target_label in combo.itemText(i):
                combo.setCurrentIndex(i)
                break

        # Different target settings object — proves save_state writes to
        # the *passed* settings, not just the construction-time one.
        target_settings = AppSettings(streaming_mode_override="batch")
        panel.save_state(target_settings)

        assert target_settings.streaming_mode_override == expected_value

    def test_save_state_round_trips_through_to_dict_from_dict(self, qapp):
        """Story 16.2's serializer covers the AppSettings round-trip; this
        test verifies the panel's save_state output composes with that
        round-trip without surprises.
        """
        settings = AppSettings(streaming_mode_override=None)
        panel = StreamingSettingsPanel(app_settings=settings)

        combo = panel._mode_combo
        for i in range(combo.count()):
            if "Sentence Stream" in combo.itemText(i):
                combo.setCurrentIndex(i)
                break

        # Round-trip through the persistence layer.
        target = AppSettings()
        panel.save_state(target)
        as_dict = target.to_dict()
        assert as_dict["streaming_mode_override"] == "sentence_stream"

        rebuilt = AppSettings.from_dict(as_dict)
        assert rebuilt.streaming_mode_override == "sentence_stream"
