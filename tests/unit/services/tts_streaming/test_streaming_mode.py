"""Tests for StreamingMode + probe + resolver (Story 16.2).

Verifies D-9 (architecture-optimization-pass.md:257) and NFR12 (line 75):
  - GPU host (cuda.is_available == True) -> default = TRUE_STREAM (FR2).
  - CPU host (cuda.is_available == False) -> default = SENTENCE_STREAM (NFR12).
  - User override takes precedence over the hardware default.
  - AppSettings.streaming_mode_override round-trips through JSON, validates,
    and falls back to None on unknown strings.
"""

import json

import pytest

from myvoice.models.app_settings import AppSettings
from myvoice.services.tts_streaming import (
    StreamingMode,
    default_streaming_mode_for_hardware,
    effective_streaming_mode,
)


# -- Enum shape (AC #1) ---------------------------------------------------- #


def test_streaming_mode_has_exactly_three_members():
    assert [m.name for m in StreamingMode] == [
        "BATCH",
        "SENTENCE_STREAM",
        "TRUE_STREAM",
    ]


def test_streaming_mode_values_are_lowercase_strings():
    assert StreamingMode.BATCH.value == "batch"
    assert StreamingMode.SENTENCE_STREAM.value == "sentence_stream"
    assert StreamingMode.TRUE_STREAM.value == "true_stream"


def test_streaming_mode_round_trips_from_string():
    assert StreamingMode("batch") is StreamingMode.BATCH
    assert StreamingMode("sentence_stream") is StreamingMode.SENTENCE_STREAM
    assert StreamingMode("true_stream") is StreamingMode.TRUE_STREAM


# -- Hardware probe (AC #2, #3) -------------------------------------------- #


def test_default_for_hardware_returns_true_stream_when_cuda_available(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert default_streaming_mode_for_hardware() is StreamingMode.TRUE_STREAM


def test_default_for_hardware_returns_sentence_stream_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    result = default_streaming_mode_for_hardware()
    assert result is StreamingMode.SENTENCE_STREAM, (
        f"NFR12 invariant violated - CPU-only host must default to "
        f"SENTENCE_STREAM, got {result.value}"
    )


# -- Resolver (AC #4) ------------------------------------------------------ #


def test_effective_mode_with_none_override_delegates_to_probe_gpu(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert effective_streaming_mode(None) is StreamingMode.TRUE_STREAM


def test_effective_mode_with_none_override_delegates_to_probe_cpu(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert effective_streaming_mode(None) is StreamingMode.SENTENCE_STREAM


def test_effective_mode_with_override_returns_override_verbatim(monkeypatch):
    # Override wins regardless of hardware. Pin cuda.is_available to True
    # so we can verify the resolver does NOT silently flip just because
    # the GPU happens to be available.
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert effective_streaming_mode(StreamingMode.BATCH) is StreamingMode.BATCH
    assert effective_streaming_mode(StreamingMode.SENTENCE_STREAM) is StreamingMode.SENTENCE_STREAM


def test_effective_mode_with_true_stream_override_on_cpu_returns_true_stream(monkeypatch):
    # The resolver does NOT second-guess the user. "TRUE_STREAM picked but
    # CPU-only" is a Story 16.6 dispatch-chain concern, not a decision-layer
    # concern. If a future maintainer is tempted to "fix" the resolver to
    # return SENTENCE_STREAM here, they should re-read D-9 + AC #4 first.
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert effective_streaming_mode(StreamingMode.TRUE_STREAM) is StreamingMode.TRUE_STREAM


def test_effective_mode_rejects_string_argument():
    with pytest.raises(TypeError, match="StreamingMode or None"):
        effective_streaming_mode("true_stream")


# -- AppSettings field round-trip (AC #5, #6) ------------------------------ #


def test_app_settings_default_streaming_mode_override_is_none():
    s = AppSettings()
    assert s.streaming_mode_override is None


def test_app_settings_streaming_mode_override_round_trips_via_to_dict_from_dict():
    # AC #5: round-trip through the *production* serializer path
    # (ConfigurationService uses AppSettings.to_dict() / AppSettings.from_dict()).
    # Code review of the original test caught it bypassing this path via
    # dataclasses.asdict + **kwargs, which masked a bug where to_dict/from_dict
    # silently dropped the new field. Encode the production path here so any
    # future regression that drops streaming_mode_override from the serializer
    # is caught immediately.
    s = AppSettings(streaming_mode_override="true_stream")
    payload = s.to_dict()
    assert payload["streaming_mode_override"] == "true_stream", (
        "AC #5 violated: to_dict() must persist streaming_mode_override; "
        "dropping the key silently breaks ConfigurationService.save_settings."
    )
    # Mirror the on-disk path via real JSON.
    raw = json.dumps(payload)
    restored = AppSettings.from_dict(json.loads(raw))
    assert restored.streaming_mode_override == "true_stream", (
        "AC #5 violated: from_dict() must read streaming_mode_override; "
        "dropping the key silently breaks ConfigurationService.load_settings."
    )
    # And confirm the string can be converted back to the enum at the
    # call-site contract Story 16.6 will rely on.
    assert StreamingMode(restored.streaming_mode_override) is StreamingMode.TRUE_STREAM


def test_app_settings_default_override_round_trips_as_none_via_to_dict_from_dict():
    # AC #5 (last given/when/then): a settings.json that omits the key entirely
    # OR carries null must reload as None. Verify both shapes via the
    # production path.
    payload_with_null = AppSettings().to_dict()
    assert payload_with_null["streaming_mode_override"] is None
    assert AppSettings.from_dict(payload_with_null).streaming_mode_override is None
    # Now drop the key entirely (simulates a pre-Story-16.2 settings.json on disk).
    del payload_with_null["streaming_mode_override"]
    assert AppSettings.from_dict(payload_with_null).streaming_mode_override is None


def test_app_settings_invalid_override_falls_back_to_none_in_post_init():
    # AC #6 part (4): __post_init__ runs validate(), which resets unknown
    # values to None so the runtime then uses the hardware-probe default.
    s = AppSettings(streaming_mode_override="experimental_unknown_mode")
    assert s.streaming_mode_override is None


def test_app_settings_invalid_override_emits_unknown_warning_code():
    # AC #6 part (3): validate() must append a ValidationIssue with
    # code="UNKNOWN_STREAMING_MODE_OVERRIDE" for non-allowed strings.
    # The original test asserted `<code> in warnings OR override is None`,
    # which short-circuited to True via the right-hand clause and never
    # actually exercised the warning emission. Per user-memory
    # `code_review_regression_test_exact_class.md`, the regression test
    # must mirror the exact bug class — here, "warning never asserted" —
    # by inspecting validate()'s returned warnings list directly, BEFORE
    # __post_init__'s auto-correction has cleared the bad value.
    s = AppSettings()  # valid defaults; __post_init__ runs cleanly.
    # Mutate the field after __post_init__ so the next validate() call
    # observes the bad value before auto-correcting it.
    s.streaming_mode_override = "experimental_unknown_mode"
    result = s.validate()
    warning_codes = [issue.code for issue in result.warnings]
    assert "UNKNOWN_STREAMING_MODE_OVERRIDE" in warning_codes, (
        f"AC #6 violated: validate() must emit UNKNOWN_STREAMING_MODE_OVERRIDE "
        f"warning for unknown override; got warning_codes={warning_codes}"
    )
    # And the auto-correct side effect must have fired.
    assert s.streaming_mode_override is None


def test_reset_to_defaults_clears_streaming_mode_override():
    # Code-review regression: reset_to_defaults() enumerates field names by
    # hand; a new field that is not added to that list silently survives a
    # "Reset to Defaults" click, leaking a stale override across the reset.
    s = AppSettings(streaming_mode_override="true_stream")
    assert s.streaming_mode_override == "true_stream"
    s.reset_to_defaults()
    assert s.streaming_mode_override is None, (
        "reset_to_defaults() must reset streaming_mode_override; missing it "
        "from the reset field-list leaks user overrides past a reset action."
    )
