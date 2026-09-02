"""StreamHub fan-out + bounded-queue tests (tech-spec Task 12 / F8)."""

import asyncio

import pytest

from myvoice.services.api_server.stream_hub import StreamHub


@pytest.mark.asyncio
async def test_publish_reaches_subscriber():
    hub = StreamHub()
    q = hub.subscribe("sid-1")
    hub.publish("sid-1", b"\x01\x02", False)
    item = await asyncio.wait_for(q.get(), timeout=1.0)
    assert item == (b"\x01\x02", False)


@pytest.mark.asyncio
async def test_publish_fans_out_to_all_subscribers_of_session():
    hub = StreamHub()
    q1 = hub.subscribe("sid-1")
    q2 = hub.subscribe("sid-1")
    hub.publish("sid-1", b"data", True)
    assert (await q1.get()) == (b"data", True)
    assert (await q2.get()) == (b"data", True)


@pytest.mark.asyncio
async def test_publish_to_unknown_session_is_noop():
    hub = StreamHub()
    q = hub.subscribe("sid-1")
    hub.publish("other", b"x", False)
    assert q.empty()


@pytest.mark.asyncio
async def test_bounded_queue_drops_oldest_on_overflow():
    hub = StreamHub(queue_maxsize=2)
    q = hub.subscribe("sid-1")
    hub.publish("sid-1", b"a", False)
    hub.publish("sid-1", b"b", False)
    hub.publish("sid-1", b"c", False)  # overflow -> drop oldest ("a")
    drained = []
    while not q.empty():
        drained.append(q.get_nowait()[0])
    assert drained == [b"b", b"c"]


@pytest.mark.asyncio
async def test_overflow_drops_non_final_but_keeps_final():
    # Final sentinel must survive an overflow that evicts older audio (F1).
    hub = StreamHub(queue_maxsize=2)
    q = hub.subscribe("s")
    hub.publish("s", b"a", False)
    hub.publish("s", b"b", True)   # terminal sentinel
    hub.publish("s", b"c", False)  # overflow -> drop oldest non-final ("a")
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    assert (b"b", True) in items          # final preserved
    assert (b"a", False) not in items     # oldest non-final dropped


@pytest.mark.asyncio
async def test_incoming_dropped_to_preserve_oldest_final():
    # When the oldest queued item IS the final and the queue is full, the
    # incoming non-final chunk is dropped instead of the sentinel (F1).
    hub = StreamHub(queue_maxsize=1)
    q = hub.subscribe("s")
    hub.publish("s", b"final", True)
    hub.publish("s", b"late", False)
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    assert items == [(b"final", True)]


@pytest.mark.asyncio
async def test_unsubscribe_removes_queue_and_prunes_session():
    hub = StreamHub()
    q = hub.subscribe("sid-1")
    hub.unsubscribe(q)
    # Subsequent publish must be a no-op (session pruned).
    hub.publish("sid-1", b"x", False)
    assert q.empty()
    assert "sid-1" not in hub._subscribers
