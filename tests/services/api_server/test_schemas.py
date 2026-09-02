"""Schema validation + voice-mapping tests (tech-spec Task 12 / F5, F11)."""

import pytest
from pydantic import ValidationError

from myvoice.services.api_server.schemas import (
    MODEL_ID,
    SpeechRequest,
    build_models_response,
    build_voices_response,
    resolve_voice,
)


def test_valid_request_defaults():
    req = SpeechRequest(input="hello", voice="Ryan")
    assert req.model == MODEL_ID
    assert req.response_format == "mp3"
    assert req.speed == 1.0
    assert req.stream is False


def test_empty_input_rejected():
    with pytest.raises(ValidationError):
        SpeechRequest(input="", voice="Ryan")


def test_oversize_input_rejected():
    with pytest.raises(ValidationError):
        SpeechRequest(input="x" * 4097, voice="Ryan")


def test_max_length_input_accepted():
    req = SpeechRequest(input="x" * 4096, voice="Ryan")
    assert len(req.input) == 4096


@pytest.mark.parametrize("speed", [0.24, 4.01, 0.0, -1.0])
def test_speed_out_of_range_rejected(speed):
    with pytest.raises(ValidationError):
        SpeechRequest(input="hi", voice="Ryan", speed=speed)


def test_speed_in_range_non_one_accepted():
    # 2.0 is accepted (rendered at 1.0x downstream, not rejected) so OpenAI
    # clients that always send speed still work (F5).
    req = SpeechRequest(input="hi", voice="Ryan", speed=2.0)
    assert req.speed == 2.0


def test_bad_response_format_rejected():
    with pytest.raises(ValidationError):
        SpeechRequest(input="hi", voice="Ryan", response_format="ogg")


# --- voice mapping ---------------------------------------------------------


class _FakeProfile:
    def __init__(self, voice_type="bundled", description=None):
        self.voice_type = voice_type
        self.description = description


class _FakeVoiceManager:
    def __init__(self, names):
        self._profiles = {n: _FakeProfile() for n in names}

    def get_valid_profiles(self):
        return self._profiles


def test_resolve_voice_known():
    vm = _FakeVoiceManager(["Ryan", "Vivian"])
    assert resolve_voice(vm, "Ryan") == "Ryan"


def test_resolve_voice_unknown_returns_none():
    vm = _FakeVoiceManager(["Ryan"])
    assert resolve_voice(vm, "Nobody") is None
    assert resolve_voice(vm, "") is None


def test_build_voices_response_lists_valid_profiles():
    vm = _FakeVoiceManager(["Ryan", "Vivian"])
    resp = build_voices_response(vm)
    names = {v.name for v in resp.voices}
    assert names == {"Ryan", "Vivian"}


def test_build_models_response_advertises_model_id():
    resp = build_models_response()
    assert [m.id for m in resp.data] == [MODEL_ID]
