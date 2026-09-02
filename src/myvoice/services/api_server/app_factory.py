"""FastAPI application factory for the local TTS API.

Builds the ASGI app, stashes the runtime collaborators on ``app.state`` (read
by the route handlers and security guards), installs the restrictive CORS
policy, and mounts the v1 + metadata routers.

Kept import-light at module top so importing the package doesn't pull FastAPI
until the server is actually built (the API is off by default).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from .routes import build_routers
from .security import install_cors

logger = logging.getLogger(__name__)


def build_app(
    tts_service,
    voice_manager,
    app_ref,
    settings_provider: Callable,
    controller: Optional[object] = None,
):
    """Construct the FastAPI app wired to MyVoice's live services.

    Args:
        tts_service: ``QwenTTSService`` to await generations on.
        voice_manager: ``VoiceProfileManager`` for voice enumeration/mapping.
        app_ref: ``MyVoiceApp`` owning ``_stream_hub`` + ``_api_origin_sessions``.
        settings_provider: zero-arg callable returning the current AppSettings
            (read live so key/port changes take effect without a restart).
        controller: ``ApiServerController`` exposing ``_active_stream_tasks``.
    """
    from fastapi import FastAPI

    app = FastAPI(
        title="MyVoice Local TTS API",
        description="OpenAI-compatible /v1/audio/speech over MyVoice",
        version="1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    app.state.tts_service = tts_service
    app.state.voice_manager = voice_manager
    app.state.app_ref = app_ref
    app.state.settings_provider = settings_provider
    app.state.controller = controller

    install_cors(app)

    v1_router, meta_router = build_routers()
    app.include_router(v1_router)
    app.include_router(meta_router)

    logger.debug("Local TTS API FastAPI app built")
    return app
