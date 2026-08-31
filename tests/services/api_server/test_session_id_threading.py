"""Task 6.5 / AC19 — caller-supplied session_id threads through create_session.

The emit sites in QwenTTSService stamp ``AudioChunk.session_id`` from the id
returned by ``SessionRegistry.create_session``; this test verifies the
registry seam honors a supplied id and otherwise auto-generates (preserving
the existing GUI behavior).
"""

import pytest

from myvoice.services.sessions.session_registry import SessionRegistry


def test_create_session_honors_supplied_id(qapp):
    reg = SessionRegistry(parent=qapp)
    sid = reg.create_session(
        text="hi", voice="Ryan", model_type="CustomVoice", session_id="abc123"
    )
    assert sid == "abc123"
    assert reg.get("abc123").session_id == "abc123"


def test_create_session_auto_generates_when_none(qapp):
    reg = SessionRegistry(parent=qapp)
    sid = reg.create_session(text="hi", voice="Ryan", model_type="CustomVoice")
    assert sid
    assert reg.get(sid) is not None
    # Default factory is uuid4 -> not the literal we'd pass.
    assert sid != "abc123"


def test_request_dataclass_carries_session_id():
    # Lightweight check that QwenTTSRequest stores the id without invoking the
    # GPU engine. Imported lazily so the heavy module only loads if reached.
    from myvoice.services.qwen_tts_service import QwenTTSRequest

    req = QwenTTSRequest(text="hi", session_id="zzz")
    assert req.session_id == "zzz"
    assert QwenTTSRequest(text="hi").session_id is None
