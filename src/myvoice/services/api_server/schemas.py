"""Request/response schemas + voice mapping for the local TTS API.

Pydantic models for the OpenAI ``/v1/audio/speech`` surface plus the
metadata routes. Validation is intentionally OpenAI-client-friendly:

- ``input`` length 1..4096 -> empty/oversize yields 422 automatically.
- ``response_format`` constrained to the three encoders we support.
- ``speed`` accepted in 0.25..4.0 (out-of-range -> 422) but only ``1.0`` is
  honored in v1 (an in-range non-1.0 value is logged and rendered at 1.0x,
  NOT rejected, so stock OpenAI clients that always send ``speed`` work).

See tech-spec Task 3 (F5, F11).
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# Advertised model id (tech-spec: GET /v1/models -> "myvoice-1").
MODEL_ID = "myvoice-1"


class SpeechRequest(BaseModel):
    """Body for ``POST /v1/audio/speech`` (OpenAI-compatible)."""

    model: str = MODEL_ID
    input: str = Field(..., min_length=1, max_length=4096)
    voice: str
    response_format: Literal["mp3", "wav", "pcm"] = "mp3"
    # Accepted for client-compat; only 1.0 applied in v1 (see resolve note).
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    stream: bool = False


class VoiceItem(BaseModel):
    """One entry in ``GET /v1/voices``."""

    name: str
    type: str
    description: Optional[str] = None


class VoicesResponse(BaseModel):
    voices: List[VoiceItem]


class ModelItem(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "myvoice"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelItem]


def resolve_voice(voice_manager, name: str) -> Optional[str]:
    """Map an OpenAI ``voice`` string to a MyVoice profile name.

    The ``voice`` string is the dict **key** in ``get_valid_profiles()`` and
    is passed verbatim as ``speaker=`` to ``generate_custom_voice``. Returns
    the canonical name on a match, else ``None`` (route -> 400).

    Backed by ``get_valid_profiles()`` (bundled + valid clones) so we never
    advertise or accept a broken profile (tech-spec F3).
    """
    if not name:
        return None
    profiles = voice_manager.get_valid_profiles()
    if name in profiles:
        return name
    return None


def build_voices_response(voice_manager) -> VoicesResponse:
    """Build the ``GET /v1/voices`` payload from ``get_valid_profiles()``."""
    items: List[VoiceItem] = []
    for profile_name, profile in voice_manager.get_valid_profiles().items():
        voice_type = getattr(profile, "voice_type", None)
        type_str = getattr(voice_type, "value", str(voice_type)) if voice_type else "unknown"
        items.append(
            VoiceItem(
                name=profile_name,
                type=type_str,
                description=getattr(profile, "description", None),
            )
        )
    return VoicesResponse(voices=items)


def build_models_response() -> ModelsResponse:
    """Build the ``GET /v1/models`` payload advertising ``myvoice-1``."""
    return ModelsResponse(data=[ModelItem(id=MODEL_ID)])
