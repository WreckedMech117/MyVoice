"""Tests for Story 18.3 AC #2 — AppSettings.tts_precision field.

Verifies the new field lands with the right default ("auto"), accepts the
three valid values ("auto", "bf16", "fp32"), serializes round-trip cleanly
through ``to_dict`` / ``from_dict``, validates with auto-correct on unknown
value (UNKNOWN_TTS_PRECISION ValidationIssue → reset to "auto"), and resets
to defaults via ``reset_to_defaults``.

Per `memory/code_review_regression_test_exact_class.md`, the bug class
"validation drift between two near-identical fields" (tts_precision vs
streaming_mode_override) is the highest-risk regression class for this
field — the test row structure mirrors the streaming_mode_override
discipline at ``tests/unit/services/tts_streaming/test_streaming_mode.py``
(lines 102–186) verbatim.
"""

from __future__ import annotations

import json

import pytest

from myvoice.models.app_settings import AppSettings


class TestAppSettingsTtsPrecisionDefaults:
    """Default value per Story 18.3 AC #2."""

    def test_default_tts_precision_is_auto(self):
        settings = AppSettings()
        assert settings.tts_precision == "auto"


class TestAppSettingsTtsPrecisionAcceptedValues:
    """The three valid values land as-is via ``__post_init__``'s validator."""

    @pytest.mark.parametrize("value", ["auto", "bf16", "fp32"])
    def test_accepted_value_passes_validation_unchanged(self, value):
        settings = AppSettings(tts_precision=value)
        assert settings.tts_precision == value


class TestAppSettingsTtsPrecisionRoundTrip:
    """to_dict / from_dict round-trips the new field through the production
    serializer path. Mirrors test_streaming_mode.py:107–142 exactly."""

    def test_to_dict_persists_tts_precision_value(self):
        # AC #2 violated if ``to_dict`` silently drops the new key —
        # ConfigurationService.save_settings would lose user overrides.
        settings = AppSettings(tts_precision="bf16")
        payload = settings.to_dict()
        assert payload["tts_precision"] == "bf16", (
            "AC #2 violated: to_dict() must persist tts_precision; "
            "dropping the key silently breaks ConfigurationService.save_settings."
        )

    def test_to_dict_default_persists_auto(self):
        settings = AppSettings()
        payload = settings.to_dict()
        assert payload["tts_precision"] == "auto"

    def test_round_trip_via_real_json_preserves_value(self):
        # Mirror the on-disk path via real JSON — the production
        # ConfigurationService writes a JSON file, not a dict.
        settings = AppSettings(tts_precision="fp32")
        raw = json.dumps(settings.to_dict())
        restored = AppSettings.from_dict(json.loads(raw))
        assert restored.tts_precision == "fp32", (
            "AC #2 violated: from_dict() must read tts_precision; "
            "dropping the key silently breaks ConfigurationService.load_settings."
        )

    def test_missing_key_in_payload_defaults_to_auto(self):
        # Simulates a pre-Story-18.3 settings.json on disk: the key is
        # absent entirely. ``from_dict`` must default to "auto" (the new
        # default) rather than raising or producing None.
        payload = AppSettings().to_dict()
        del payload["tts_precision"]
        restored = AppSettings.from_dict(payload)
        assert restored.tts_precision == "auto"


class TestAppSettingsTtsPrecisionValidation:
    """Unknown values must surface a UNKNOWN_TTS_PRECISION ValidationIssue
    and auto-correct to "auto"."""

    def test_invalid_value_resets_to_auto_in_post_init(self):
        # __post_init__ runs validate(), which resets unknown values to "auto"
        # so the runtime then resolves cleanly via the hardware probe.
        settings = AppSettings(tts_precision="experimental_unknown_precision")
        assert settings.tts_precision == "auto"

    def test_invalid_value_emits_unknown_warning_code(self):
        # validate() must append a ValidationIssue with
        # code="UNKNOWN_TTS_PRECISION" for non-allowed strings.
        # Mirroring the streaming_mode_override test discipline at
        # test_streaming_mode.py:152–173: mutate the field after
        # __post_init__ so the next validate() call observes the bad value
        # before auto-correcting it. The bug class is "warning never
        # asserted" per `memory/code_review_regression_test_exact_class.md`;
        # inspect validate()'s returned warnings list directly.
        settings = AppSettings()  # valid defaults; __post_init__ runs cleanly.
        settings.tts_precision = "experimental_unknown_precision"
        result = settings.validate()
        warning_codes = [issue.code for issue in result.warnings]
        assert "UNKNOWN_TTS_PRECISION" in warning_codes, (
            f"AC #2 violated: validate() must emit UNKNOWN_TTS_PRECISION "
            f"warning for unknown precision; got warning_codes={warning_codes}"
        )
        # And the auto-correct side effect must have fired.
        assert settings.tts_precision == "auto"


class TestAppSettingsTtsPrecisionReset:
    """``reset_to_defaults()`` must reset tts_precision to "auto"."""

    def test_reset_to_defaults_clears_tts_precision_override(self):
        # Code-review regression: reset_to_defaults() enumerates field names
        # by hand; a new field that is not added to that list silently
        # survives a "Reset to Defaults" click, leaking a stale override
        # across the reset. Same bug class as the streaming_mode_override
        # reset coverage at test_streaming_mode.py:176–186.
        settings = AppSettings(tts_precision="fp32")
        assert settings.tts_precision == "fp32"
        settings.reset_to_defaults()
        assert settings.tts_precision == "auto", (
            "reset_to_defaults() must reset tts_precision; missing it from "
            "the reset field-list leaks user overrides past a reset action."
        )
