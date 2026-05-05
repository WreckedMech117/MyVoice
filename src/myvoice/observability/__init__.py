"""
Observability subpackage — single-chokepoint metric stream (Story 11.3).

Architecture rule (P-9): every metric in the codebase flows through
``metrics.record(...)``. This package owns that chokepoint so future stories
(13.1 queue depth, 14.1 saveable transitions, 16.4 decode latency, 16.7
streaming validation gates) emit through one uniform path. The module imports
nothing from ``myvoice.*`` — making it the dependency-inversion anchor for
telemetry: everything may import metrics, metrics imports nothing.
"""

from myvoice.observability.metrics import (
    METRIC_LOGGER_NAME,
    MetricRecord,
    add_listener,
    record,
    remove_listener,
)

__all__ = [
    "METRIC_LOGGER_NAME",
    "MetricRecord",
    "add_listener",
    "record",
    "remove_listener",
]
