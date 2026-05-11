"""Tests for Story 18.4 AC #6 — AppSettings.tts_compile field.

Verifies the new field lands with the right default ("auto"), accepts the
three valid values ("auto", "on", "off"), serializes round-trip cleanly
through ``to_dict`` / ``from_dict``, validates with auto-correct on unknown
value (UNKNOWN_TTS_COMPILE ValidationIssue → reset to "auto"), and resets
to defaults via ``reset_to_defaults``.

Per `memory/code_review_regression_test_exact_class.md`, the bug class
"validation drift between two near-identical fields" (tts_compile vs
tts_precision) is the highest-risk regression class for this field — the
test row structure mirrors the ``tts_precision`` discipline at
``tests/unit/models/test_app_settings_tts_precision.py`` verbatim, so any
post-Story-18.4 change to ``tts_precision``'s validation contract that
fails to land in ``tts_compile`` (or vice versa) fails this file.
"""

from __future__ import annotations

import json

import pytest

from myvoice.models.app_settings import AppSettings


class TestAppSettingsTtsCompileDefaults:
    """Default value per Story 18.4 AC #6 + the 2026-05-10 bundled-smoke
    architecture amendment that flipped the default from "auto" to "off"
    pending the triton-on-Windows toolchain resolution (see field-level
    docstring in app_settings.py for rationale)."""

    def test_default_tts_compile_is_off(self):
        settings = AppSettings()
        assert settings.tts_compile == "off"


class TestAppSettingsTtsCompileAcceptedValues:
    """The three valid values land as-is via ``__post_init__``'s validator."""

    @pytest.mark.parametrize("value", ["auto", "on", "off"])
    def test_accepted_value_passes_validation_unchanged(self, value):
        settings = AppSettings(tts_compile=value)
        assert settings.tts_compile == value


class TestAppSettingsTtsCompileRoundTrip:
    """to_dict / from_dict round-trips the new field through the production
    serializer path. Mirrors test_app_settings_tts_precision.py exactly."""

    def test_to_dict_persists_tts_compile_value(self):
        # AC #6 violated if ``to_dict`` silently drops the new key —
        # ConfigurationService.save_settings would lose user overrides.
        settings = AppSettings(tts_compile="on")
        payload = settings.to_dict()
        assert payload["tts_compile"] == "on", (
            "AC #6 violated: to_dict() must persist tts_compile; "
            "dropping the key silently breaks ConfigurationService.save_settings."
        )

    def test_to_dict_default_persists_off(self):
        settings = AppSettings()
        payload = settings.to_dict()
        assert payload["tts_compile"] == "off"

    def test_round_trip_via_real_json_preserves_value(self):
        # Mirror the on-disk path via real JSON — the production
        # ConfigurationService writes a JSON file, not a dict.
        settings = AppSettings(tts_compile="off")
        raw = json.dumps(settings.to_dict())
        restored = AppSettings.from_dict(json.loads(raw))
        assert restored.tts_compile == "off", (
            "AC #6 violated: from_dict() must read tts_compile; "
            "dropping the key silently breaks ConfigurationService.load_settings."
        )

    def test_missing_key_in_payload_defaults_to_off(self):
        # Simulates a pre-Story-18.4 settings.json on disk: the key is
        # absent entirely. ``from_dict`` must default to "off" (the
        # Story 18.4 bundled-smoke amendment default) rather than
        # raising or producing None.
        payload = AppSettings().to_dict()
        del payload["tts_compile"]
        restored = AppSettings.from_dict(payload)
        assert restored.tts_compile == "off"


class TestAppSettingsTtsCompileValidation:
    """Unknown values must surface a UNKNOWN_TTS_COMPILE ValidationIssue
    and auto-correct to "off" (the field's declared default).

    Code-review H3 (Story 18.4 review pass): the validator's reset
    target must match the field-declaration default. A user who typed
    an invalid value gets safe degradation (eager-mode) rather than
    silently re-engaging the broken Windows triton toolchain the "off"
    default was designed to avoid. Per
    `memory/code_review_regression_test_exact_class.md`, the bug class is
    "validation drift between two near-identical fields" — these tests
    pin the symmetry between declaration default and validator reset.
    """

    def test_invalid_value_resets_to_off_in_post_init(self):
        # __post_init__ runs validate(), which resets unknown values to
        # "off" — the same value as the field's declared default at
        # ``app_settings.py:133``. NFR7 graceful-degradation discipline:
        # an invalid input lands at the safe fallback, NOT the engaged
        # path. The previous behavior (reset → "auto") silently
        # re-activated compile on Ampere+ hosts where the bundled
        # triton toolchain is non-functional (Story 18.4 bundled-smoke
        # Fix #4 outcome).
        settings = AppSettings(tts_compile="experimental_unknown_compile")
        assert settings.tts_compile == "off"

    def test_invalid_value_emits_unknown_warning_code(self):
        # validate() must append a ValidationIssue with
        # code="UNKNOWN_TTS_COMPILE" for non-allowed strings.
        # Mirroring the tts_precision test discipline: mutate the field
        # after __post_init__ so the next validate() call observes the
        # bad value before auto-correcting it. The bug class is "warning
        # never asserted" per `memory/code_review_regression_test_exact_class.md`;
        # inspect validate()'s returned warnings list directly.
        settings = AppSettings()  # valid defaults; __post_init__ runs cleanly.
        settings.tts_compile = "experimental_unknown_compile"
        result = settings.validate()
        warning_codes = [issue.code for issue in result.warnings]
        assert "UNKNOWN_TTS_COMPILE" in warning_codes, (
            f"AC #6 violated: validate() must emit UNKNOWN_TTS_COMPILE "
            f"warning for unknown compile mode; got warning_codes={warning_codes}"
        )
        # And the auto-correct side effect must have fired — reset
        # target matches the declared default ("off"), not "auto".
        assert settings.tts_compile == "off"


class TestAppSettingsTtsCompileReset:
    """``reset_to_defaults()`` must reset tts_compile to "auto"."""

    def test_reset_to_defaults_clears_tts_compile_override(self):
        # Code-review regression: reset_to_defaults() enumerates field names
        # by hand; a new field that is not added to that list silently
        # survives a "Reset to Defaults" click, leaking a stale override
        # across the reset. Same bug class as the tts_precision reset
        # coverage; if this row is added but ``tts_compile`` is not in the
        # field-name list, this test catches it loudly.
        # Note: the field's declared default is "off" per the Story 18.4
        # bundled-smoke amendment, so reset_to_defaults() should land on
        # "off" (not "auto"). The user-supplied "on" override is what's
        # being cleared by the reset action.
        settings = AppSettings(tts_compile="on")
        assert settings.tts_compile == "on"
        settings.reset_to_defaults()
        assert settings.tts_compile == "off", (
            "reset_to_defaults() must reset tts_compile to its declared "
            "default ('off' per Story 18.4 bundled-smoke amendment); "
            "missing it from the reset field-list leaks user overrides past a reset."
        )
