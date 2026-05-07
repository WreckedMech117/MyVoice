"""Tests for Story 15.2 AC #12 — AppSettings Clear Comms fields.

Verifies the three new persisted fields land with the right defaults,
serialize round-trip cleanly, validate with auto-correct on unknown
``source_kind``, and reset to defaults via ``reset_to_defaults``.
"""

from __future__ import annotations

import pytest

from myvoice.models.app_settings import AppSettings
from myvoice.models.validation import ValidationStatus


class TestAppSettingsClearCommsDefaults:
    """Default values for the three Clear Comms fields per AC #12."""

    def test_default_source_kind_is_last_generation(self):
        settings = AppSettings()
        assert settings.clear_comms_source_kind == "last_generation"

    def test_default_file_path_is_none(self):
        settings = AppSettings()
        assert settings.clear_comms_file_path is None

    def test_default_queue_mode_is_false(self):
        settings = AppSettings()
        assert settings.clear_comms_queue_mode is False


class TestAppSettingsClearCommsRoundTrip:
    """to_dict / from_dict preserves all three Clear Comms fields."""

    def test_to_dict_includes_clear_comms_keys(self):
        settings = AppSettings()
        data = settings.to_dict()
        assert "clear_comms_source_kind" in data
        assert "clear_comms_file_path" in data
        assert "clear_comms_queue_mode" in data
        assert data["clear_comms_source_kind"] == "last_generation"
        assert data["clear_comms_file_path"] is None
        assert data["clear_comms_queue_mode"] is False

    def test_to_dict_from_dict_round_trip(self):
        original = AppSettings()
        original.clear_comms_source_kind = "file"
        original.clear_comms_file_path = "/path/to/clip.wav"
        original.clear_comms_queue_mode = True
        data = original.to_dict()
        restored = AppSettings.from_dict(data)
        assert restored.clear_comms_source_kind == "file"
        assert restored.clear_comms_file_path == "/path/to/clip.wav"
        assert restored.clear_comms_queue_mode is True

    def test_from_dict_missing_keys_returns_defaults(self):
        # Existing settings JSON without the 15.2 keys still loads.
        data = AppSettings().to_dict()
        for k in (
            "clear_comms_source_kind",
            "clear_comms_file_path",
            "clear_comms_queue_mode",
        ):
            data.pop(k, None)
        restored = AppSettings.from_dict(data)
        assert restored.clear_comms_source_kind == "last_generation"
        assert restored.clear_comms_file_path is None
        assert restored.clear_comms_queue_mode is False


class TestAppSettingsClearCommsValidation:
    """``validate()`` warns and auto-corrects unknown ``source_kind``."""

    def test_validate_unknown_source_kind_warns_and_corrects(self):
        settings = AppSettings()
        settings.clear_comms_source_kind = "bogus"
        result = settings.validate()
        # Auto-correct happened.
        assert settings.clear_comms_source_kind == "last_generation"
        # WARNING with the right code present.
        codes = [issue.code for issue in result.warnings]
        assert "UNKNOWN_CLEAR_COMMS_SOURCE" in codes
        # Severity is WARNING (not INVALID).
        for issue in result.warnings:
            if issue.code == "UNKNOWN_CLEAR_COMMS_SOURCE":
                assert issue.severity == ValidationStatus.WARNING

    def test_validate_known_source_kind_does_not_warn(self):
        settings = AppSettings()
        settings.clear_comms_source_kind = "file"
        result = settings.validate()
        codes = [issue.code for issue in result.warnings]
        assert "UNKNOWN_CLEAR_COMMS_SOURCE" not in codes
        assert settings.clear_comms_source_kind == "file"


class TestAppSettingsClearCommsResetToDefaults:
    """``reset_to_defaults`` restores the three fields."""

    def test_reset_restores_all_three_fields(self):
        settings = AppSettings()
        settings.clear_comms_source_kind = "file"
        settings.clear_comms_file_path = "/some/path.wav"
        settings.clear_comms_queue_mode = True
        settings.reset_to_defaults()
        assert settings.clear_comms_source_kind == "last_generation"
        assert settings.clear_comms_file_path is None
        assert settings.clear_comms_queue_mode is False
