"""Session-keyed broadcaster for HTTP streaming fan-out.

The TTS engine delivers progressive audio through a single global
fire-and-forget callback (``set_audio_chunk_ready_callback`` — one slot, owned
by the GUI). To let HTTP clients also receive those chunks without disturbing
the desktop path, :class:`StreamHub` is a NEW app-level fan-out keyed on
``AudioChunk.session_id``.

This is **not** ``StreamingChunkBuffer`` (that is an internal watermark/
crossfade smoothing buffer inside ``AudioCoordinator`` with no subscribe API).

Threading: ``publish`` is called from ``_handle_progressive_chunk_async``,
which already runs ON the qasync loop, so it is a plain on-loop
``queue.put_nowait`` — no cross-thread marshalling (tech-spec G3). Queues are
bounded and drop-oldest on overflow so a slow/disconnected HTTP consumer can
never block the loop (F8).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Per-subscriber queue depth. At 24 kHz mono int16, chunks are small; ~64
# buffered chunks is generous headroom before drop-oldest kicks in.
DEFAULT_QUEUE_MAXSIZE = 64

# Queue item: (int16_bytes, is_final)
StreamItem = Tuple[bytes, bool]


class StreamHub:
    """Fan-out of int16 audio chunks to HTTP subscribers, keyed by session id."""

    def __init__(self, queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE):
        self._queue_maxsize = queue_maxsize
        # session_id -> list of subscriber queues (supports >1 listener/session)
        self._subscribers: Dict[str, List["asyncio.Queue[StreamItem]"]] = {}

    def subscribe(self, session_id: str) -> "asyncio.Queue[StreamItem]":
        """Register a bounded queue for ``session_id`` and return it."""
        queue: "asyncio.Queue[StreamItem]" = asyncio.Queue(maxsize=self._queue_maxsize)
        self._subscribers.setdefault(session_id, []).append(queue)
        return queue

    def publish(self, session_id: str, int16_bytes: bytes, is_final: bool) -> None:
        """Deliver a chunk to every subscriber of ``session_id`` (on-loop).

        On a full queue, drops the oldest item to make room (never blocks the
        event loop). No-op if the session has no subscribers.
        """
        queues = self._subscribers.get(session_id)
        if not queues:
            return
        item: StreamItem = (int16_bytes, is_final)
        for queue in queues:
            if queue.full():
                # Make room by dropping the oldest item — but NEVER evict a
                # terminal (is_final) sentinel that a slow consumer hasn't read
                # yet, or the stream would lose its end-of-stream marker. If the
                # oldest is a final and the incoming chunk is not, drop the
                # incoming instead (keep the sentinel). (F1)
                try:
                    oldest = queue.get_nowait()
                except asyncio.QueueEmpty:
                    oldest = None
                if oldest is not None and oldest[1] and not is_final:
                    try:
                        queue.put_nowait(oldest)  # restore the sentinel
                    except asyncio.QueueFull:  # pragma: no cover - defensive
                        pass
                    logger.debug(
                        "StreamHub queue full for session %s; dropped incoming "
                        "non-final chunk to preserve the final sentinel",
                        session_id,
                    )
                    continue
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                # Racing producers in the same loop turn; safe to drop.
                logger.debug("StreamHub queue full for session %s; dropping chunk", session_id)

    def unsubscribe(self, queue: "asyncio.Queue[StreamItem]") -> None:
        """Remove a previously subscribed queue; prunes empty session entries."""
        for session_id, queues in list(self._subscribers.items()):
            if queue in queues:
                queues.remove(queue)
                if not queues:
                    del self._subscribers[session_id]
                return
