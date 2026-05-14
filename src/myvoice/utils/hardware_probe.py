"""Hardware probes used at first-launch to pick reasonable defaults.

Currently exposes ``detect_recommended_tier()`` — a CUDA-VRAM-based
heuristic for choosing between the 0.6B "small" and 1.7B "quality"
TTS model tiers. Used by ConfigurationManager on first launch only
(when no settings file exists). Existing installs keep whatever the
user chose; only fresh installs benefit from auto-detection.

Empirical motivation (2026-05-12 RTX 3060 12GB test): the 1.7B
quality tier on an Ampere RTX 3060 produces audio at roughly 0.5×
realtime in steady state, which is audibly stuttery. The 0.6B small
tier comfortably runs at ~3× realtime on the same card. The fix
class is "default to small on lower-throughput cards"; this module
implements the detection side. See `epic18_producer_bottleneck_finding`.
"""

from __future__ import annotations

import logging


_logger = logging.getLogger(__name__)


# 16 GiB threshold catches the RTX 30xx Ampere line that ships with
# 8–12 GiB and lets RTX 3090 (24 GiB), RTX 4080+ (16 GiB+), and the
# RTX 5090 (32 GiB) through to the higher-quality tier. VRAM is an
# imperfect proxy for compute throughput — a 3080 10GB has similar
# compute to a 3090 24GB, and they should arguably get different
# defaults — but VRAM is what `torch.cuda.get_device_properties`
# exposes at zero cost. A more accurate signal would require a
# benchmark warm-up at first launch, which is too expensive for an
# install-time decision.
_LOW_VRAM_THRESHOLD_BYTES = 16 * 1024 ** 3


def detect_recommended_tier(
    low_vram_threshold_bytes: int = _LOW_VRAM_THRESHOLD_BYTES,
) -> str:
    """Return ``"small"`` or ``"quality"`` based on detected VRAM.

    Returns ``"small"`` if the primary CUDA device's total VRAM is
    below ``low_vram_threshold_bytes``. Returns ``"quality"`` in every
    other case (no CUDA, probe failure, or VRAM at/above threshold) —
    so the existing default is preserved for any path where the probe
    can't make a confident decision.

    Logs the decision at INFO so the diagnostic is in `myvoice.log`.
    """
    try:
        import torch  # type: ignore[import]
    except (ImportError, OSError) as exc:
        _logger.info(
            "Hardware tier probe: torch import failed (%s); defaulting to 'quality'",
            exc,
        )
        return "quality"

    try:
        if not torch.cuda.is_available():
            _logger.info(
                "Hardware tier probe: CUDA not available; defaulting to 'quality'"
            )
            return "quality"

        props = torch.cuda.get_device_properties(0)
        total_vram_bytes = int(props.total_memory)
        total_vram_gib = total_vram_bytes / (1024 ** 3)

        if total_vram_bytes < low_vram_threshold_bytes:
            tier = "small"
        else:
            tier = "quality"

        _logger.info(
            "Hardware tier probe: device='%s' total_vram=%.1f GiB → tier='%s' "
            "(threshold=%.1f GiB)",
            getattr(props, "name", "unknown"),
            total_vram_gib,
            tier,
            low_vram_threshold_bytes / (1024 ** 3),
        )
        return tier
    except Exception as exc:
        _logger.info(
            "Hardware tier probe: CUDA query raised (%s); defaulting to 'quality'",
            exc,
        )
        return "quality"
