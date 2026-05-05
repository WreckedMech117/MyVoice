"""
Static-scan enforcement of the Story 11.3 migration (AC #14).

Reads ``services/qwen_tts_service.py`` source directly (no runtime import)
and asserts that:

1. The pre-migration inline running-average arithmetic
   (``_avg_first_chunk_latency = (self._avg_first_chunk_latency * (...) ...``)
   is GONE from the streaming-completion path.
2. The new aggregator class ``_FirstChunkLatencyAggregator`` is present.
3. ``metrics.record("first_chunk_latency_ms"`` is present (the migration
   target).
4. The human-readable ``"First chunk latency: ..."`` log line at line 1919
   is preserved (AC #16 — separate from the structured metric stream).

Static reading is sufficient — and is required where the production module
cannot be imported in CI environments (QwenTTSService transitively imports
torch, which has DLL dependencies on Windows builds).
"""

import re
from pathlib import Path

import pytest


SERVICE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "myvoice"
    / "services"
    / "qwen_tts_service.py"
)


@pytest.fixture(scope="module")
def service_source() -> str:
    assert SERVICE_PATH.exists(), f"Expected service file at {SERVICE_PATH}"
    return SERVICE_PATH.read_text(encoding="utf-8")


class TestNoInlineMetricMath:
    """The inline running-average arithmetic must not survive migration."""

    INLINE_MATH_PATTERN = re.compile(
        r"_avg_first_chunk_latency\s*=\s*\(\s*\(\s*self\._avg_first_chunk_latency\s*\*"
    )

    def test_inline_running_average_pattern_absent(self, service_source: str):
        # Pre-migration shape: ``self._avg_first_chunk_latency = (
        #   (self._avg_first_chunk_latency * (self._streaming_requests - 1)
        #     + first_chunk_time)
        #   / self._streaming_requests
        # )``
        # The post-migration code path emits via metrics.record() and the
        # aggregator — there must be ZERO occurrences of that arithmetic
        # in the service source.
        matches = self.INLINE_MATH_PATTERN.findall(service_source)
        assert len(matches) == 0, (
            f"Found {len(matches)} occurrence(s) of inline running-average math: "
            f"{matches!r}. Migration AC #14 requires this arithmetic to be removed "
            f"from the streaming-completion path (it now lives in "
            f"_FirstChunkLatencyAggregator)."
        )

    def test_aggregator_class_present(self, service_source: str):
        assert "_FirstChunkLatencyAggregator" in service_source, (
            "Expected _FirstChunkLatencyAggregator class in qwen_tts_service.py"
        )

    def test_aggregator_running_average_inside_class(self, service_source: str):
        # The running-average update math has not vanished — it has moved
        # into the aggregator. We expect to see the formula expressed in
        # terms of the aggregator's local ``prior``/``n``/``value_seconds``
        # variables (the rename also serves as a marker that the math is
        # in the right place, not duplicated at the call site).
        assert re.search(
            r"prior\s*\*\s*\(\s*n\s*-\s*1\s*\)\s*\+\s*value_seconds",
            service_source,
        ), "Aggregator running-average update arithmetic not found"

    def test_metrics_record_call_present(self, service_source: str):
        assert 'metrics.record(' in service_source
        assert '"first_chunk_latency_ms"' in service_source, (
            "Expected metrics.record(\"first_chunk_latency_ms\", ...) call site"
        )

    def test_human_readable_first_chunk_log_preserved(self, service_source: str):
        # AC #16: the existing operator-readable log line stays.
        assert "First chunk latency:" in service_source

    def test_streaming_requests_increment_preserved(self, service_source: str):
        # AC #15: the denominator counter increment must remain in service
        # state (NOT migrated into the metric stream).
        assert "self._streaming_requests += 1" in service_source

    def test_observability_import_present(self, service_source: str):
        # The migration must import the chokepoint helper.
        assert (
            "from myvoice.observability import" in service_source
            and "metrics" in service_source
        ), "Expected: from myvoice.observability import metrics, MetricRecord"
