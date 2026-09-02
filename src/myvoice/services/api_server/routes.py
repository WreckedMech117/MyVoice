"""HTTP routes for the local TTS API.

OpenAI ``/v1/audio/speech`` (buffered + chunked streaming) plus the metadata
routes ``/v1/voices``, ``/v1/models``, and ``/health``.

Runtime collaborators are read from ``request.app.state`` (populated by
:func:`app_factory.build_app`):

- ``tts_service``      -> QwenTTSService (awaited directly on the qasync loop)
- ``voice_manager``    -> VoiceProfileManager (voice enumeration/mapping)
- ``app_ref``          -> MyVoiceApp (``_stream_hub`` + ``_api_origin_sessions``)
- ``controller``       -> ApiServerController (``_active_stream_tasks`` registry)
- ``settings_provider``-> callable returning the live AppSettings

All generation is awaited on the same loop that serves HTTP, so the existing
``QwenTTSService`` request-semaphore auto-serializes GUI vs API generations
(tech-spec AC13).
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from .audio_encode import StreamEncoder, encode_buffered
from .schemas import (
    SpeechRequest,
    build_models_response,
    build_voices_response,
    resolve_voice,
)
from .security import verify_auth, verify_host

logger = logging.getLogger(__name__)

_STREAM_HEADERS = {
    "X-Accel-Buffering": "no",
    "Cache-Control": "no-cache",
}


def build_routers():
    """Build the (v1, meta) routers with their dependency guards attached.

    ``/v1/*`` gets both the Host guard and Bearer auth; ``/health`` gets the
    Host guard only (it is the Settings-UI liveness probe, not a generation
    surface).
    """
    v1 = APIRouter(prefix="/v1", dependencies=[Depends(verify_host), Depends(verify_auth)])
    meta = APIRouter(dependencies=[Depends(verify_host)])

    @v1.post("/audio/speech")
    async def create_speech(req: SpeechRequest, request: Request):
        state = request.app.state
        mapped = resolve_voice(state.voice_manager, req.voice)
        if mapped is None:
            raise HTTPException(status_code=400, detail=f"Unknown voice: {req.voice!r}")

        # v1 honors only speed=1.0; accept-and-ignore otherwise (F5) so stock
        # OpenAI clients that always send speed are not rejected.
        if req.speed != 1.0:
            logger.warning(
                "speed=%.3f requested but unsupported in v1; rendering at 1.0x",
                req.speed,
            )

        if req.stream:
            # Streamed wav can't declare a length up front -> buffered-only (G11).
            if req.response_format == "wav":
                raise HTTPException(
                    status_code=400,
                    detail="response_format 'wav' is not supported with stream=true",
                )
            return _streaming_response(request, req, mapped)

        return await _buffered_response(state, req, mapped)

    @v1.get("/voices")
    async def list_voices(request: Request):
        return build_voices_response(request.app.state.voice_manager)

    @v1.get("/models")
    async def list_models():
        return build_models_response()

    @meta.get("/health")
    async def health():
        return {"status": "ok"}

    return v1, meta


async def _buffered_response(state, req: SpeechRequest, mapped: str) -> Response:
    """Whole-clip path: generate (BATCH), encode off-loop, return bytes."""
    try:
        resp = await state.tts_service.generate_custom_voice(
            text=req.input,
            speaker=mapped,
            streaming=False,
        )
    except Exception as exc:  # noqa: BLE001 - surface any engine failure as 500
        logger.exception("Buffered generation raised")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    if not getattr(resp, "success", False) or resp.audio_data is None:
        detail = getattr(resp, "error_message", None) or "Generation failed"
        raise HTTPException(status_code=500, detail=detail)

    loop = asyncio.get_running_loop()
    payload, media_type = await loop.run_in_executor(
        None, encode_buffered, resp.audio_data, resp.sample_rate, req.response_format
    )
    return Response(content=payload, media_type=media_type)


def _streaming_response(request: Request, req: SpeechRequest, mapped: str) -> StreamingResponse:
    """Chunked path: subscribe to the StreamHub and fan the API session out.

    Registers an API-origin ``session_id`` so the progressive-chunk handler
    routes those chunks to the HTTP queue and never opens the desktop audio
    device (tech-spec Task 7 / AC12b).
    """
    state = request.app.state
    app_ref = state.app_ref
    controller = state.controller

    sid = secrets.token_hex(8)
    app_ref._api_origin_sessions.add(sid)
    queue = app_ref._stream_hub.subscribe(sid)

    gen_task = asyncio.ensure_future(
        state.tts_service.generate_custom_voice(
            text=req.input,
            speaker=mapped,
            streaming=True,
            session_id=sid,
        )
    )
    if controller is not None:
        controller._active_stream_tasks.add(gen_task)

    encoder = StreamEncoder(req.response_format)  # mp3 or pcm (wav rejected above)

    async def gen():
        loop = asyncio.get_running_loop()
        # Dedicated single-thread executor (F8): the StreamEncoder holds a
        # stateful lameenc.Encoder C object; pinning every encode_chunk/flush to
        # ONE worker thread guarantees calls never hop threads in the shared
        # default pool. Torn down in the finally below.
        encode_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ltts-enc")
        published = 0
        try:
            while True:
                if await request.is_disconnected():
                    logger.info("Streaming client disconnected (sid=%s)", sid)
                    break

                getter = asyncio.ensure_future(queue.get())
                await asyncio.wait(
                    {getter, gen_task}, return_when=asyncio.FIRST_COMPLETED
                )

                if getter.done() and not getter.cancelled():
                    audio_bytes, is_final = getter.result()
                    published += 1
                    encoded = await loop.run_in_executor(
                        encode_pool, encoder.encode_chunk, audio_bytes
                    )
                    if encoded:
                        yield encoded
                    if is_final:
                        tail = encoder.flush()
                        if tail:
                            yield tail
                        return
                    continue

                # gen_task completed first; getter is still pending -> cancel it
                # (queue was empty, so no item is lost).
                getter.cancel()

                if gen_task.done():
                    # Drain anything the publisher left behind.
                    while not queue.empty():
                        audio_bytes, is_final = queue.get_nowait()
                        published += 1
                        encoded = await loop.run_in_executor(
                            encode_pool, encoder.encode_chunk, audio_bytes
                        )
                        if encoded:
                            yield encoded
                        if is_final:
                            tail = encoder.flush()
                            if tail:
                                yield tail
                            return

                    if published == 0:
                        # Zero progressive chunks (BATCH-resolved or non-
                        # progressive hardware): gracefully degrade to the
                        # buffered result encoded as a single body (G4/G5).
                        async for body in _degrade_to_buffered(
                            gen_task, req, loop, encode_pool
                        ):
                            yield body
                    else:
                        tail = encoder.flush()
                        if tail:
                            yield tail
                    return
        finally:
            app_ref._stream_hub.unsubscribe(queue)
            app_ref._api_origin_sessions.discard(sid)
            if controller is not None:
                controller._active_stream_tasks.discard(gen_task)
            if not gen_task.done():
                gen_task.cancel()
            encode_pool.shutdown(wait=False)

    return StreamingResponse(
        gen(),
        media_type=encoder.media_type,
        headers=dict(_STREAM_HEADERS),
    )


async def _degrade_to_buffered(gen_task, req: SpeechRequest, loop, executor):
    """Yield the buffered audio body when no progressive chunks were produced."""
    if gen_task.cancelled():
        return
    exc = gen_task.exception()
    if exc is not None:
        logger.warning("Streaming generation failed before any chunk: %s", exc)
        return  # client sees a truncated/empty body
    resp = gen_task.result()
    if not getattr(resp, "success", False) or resp.audio_data is None:
        logger.warning("Streaming generation produced no audio; ending stream")
        return
    payload, _media_type = await loop.run_in_executor(
        executor, encode_buffered, resp.audio_data, resp.sample_rate, req.response_format
    )
    if payload:
        yield payload
