"""Unit tests for hardware_probe.detect_recommended_tier.

Pure mocked tests — no real CUDA or GPU access required. The probe
itself imports torch; the tests stub torch.cuda and never instantiate
real device handles.

Originated from the 2026-05-12 RTX 3060 finding (1.7B quality tier
audibly stuttery on 12GB Ampere). The probe wires the tier auto-
default into ConfigurationManager's first-launch path; these tests
pin the heuristic at the unit level.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from myvoice.utils import hardware_probe
from myvoice.utils.hardware_probe import detect_recommended_tier


def _stub_cuda(total_memory_bytes: int, name: str = "Stubbed GPU"):
    """Return a torch-like stub whose cuda module reports the given VRAM."""
    cuda_stub = MagicMock()
    cuda_stub.is_available.return_value = True
    props = MagicMock()
    props.total_memory = total_memory_bytes
    props.name = name
    cuda_stub.get_device_properties.return_value = props
    torch_stub = MagicMock()
    torch_stub.cuda = cuda_stub
    return torch_stub


# --------------------------------------------------------------------------- #
# Threshold semantics
# --------------------------------------------------------------------------- #


class TestThreshold:
    def test_rtx_3060_12gb_returns_small(self):
        torch_stub = _stub_cuda(12 * 1024 ** 3, name="NVIDIA GeForce RTX 3060")
        with patch.dict("sys.modules", {"torch": torch_stub}):
            assert detect_recommended_tier() == "small"

    def test_rtx_3070_8gb_returns_small(self):
        torch_stub = _stub_cuda(8 * 1024 ** 3, name="NVIDIA GeForce RTX 3070")
        with patch.dict("sys.modules", {"torch": torch_stub}):
            assert detect_recommended_tier() == "small"

    def test_rtx_3080_10gb_returns_small(self):
        torch_stub = _stub_cuda(10 * 1024 ** 3, name="NVIDIA GeForce RTX 3080")
        with patch.dict("sys.modules", {"torch": torch_stub}):
            assert detect_recommended_tier() == "small"

    def test_rtx_3090_24gb_returns_quality(self):
        torch_stub = _stub_cuda(24 * 1024 ** 3, name="NVIDIA GeForce RTX 3090")
        with patch.dict("sys.modules", {"torch": torch_stub}):
            assert detect_recommended_tier() == "quality"

    def test_rtx_4080_16gb_returns_quality(self):
        # Exactly at threshold → goes to quality (the "<" comparison).
        torch_stub = _stub_cuda(16 * 1024 ** 3, name="NVIDIA GeForce RTX 4080")
        with patch.dict("sys.modules", {"torch": torch_stub}):
            assert detect_recommended_tier() == "quality"

    def test_rtx_5090_32gb_returns_quality(self):
        torch_stub = _stub_cuda(32 * 1024 ** 3, name="NVIDIA GeForce RTX 5090")
        with patch.dict("sys.modules", {"torch": torch_stub}):
            assert detect_recommended_tier() == "quality"

    def test_just_below_threshold_returns_small(self):
        # 16GB - 1 byte → small. The "<" boundary semantics.
        torch_stub = _stub_cuda(16 * 1024 ** 3 - 1)
        with patch.dict("sys.modules", {"torch": torch_stub}):
            assert detect_recommended_tier() == "small"

    def test_custom_threshold_overrides_default(self):
        # 12GB card with a 24GB threshold → small.
        torch_stub = _stub_cuda(12 * 1024 ** 3)
        with patch.dict("sys.modules", {"torch": torch_stub}):
            assert detect_recommended_tier(low_vram_threshold_bytes=24 * 1024 ** 3) == "small"
        # 12GB card with a 8GB threshold → quality.
        torch_stub2 = _stub_cuda(12 * 1024 ** 3)
        with patch.dict("sys.modules", {"torch": torch_stub2}):
            assert detect_recommended_tier(low_vram_threshold_bytes=8 * 1024 ** 3) == "quality"


# --------------------------------------------------------------------------- #
# Fail-safe behavior — every error path returns "quality"
# --------------------------------------------------------------------------- #


class TestFailSafe:
    def test_no_cuda_returns_quality(self):
        torch_stub = MagicMock()
        torch_stub.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": torch_stub}):
            assert detect_recommended_tier() == "quality"

    def test_cuda_query_exception_returns_quality(self):
        torch_stub = MagicMock()
        torch_stub.cuda.is_available.return_value = True
        torch_stub.cuda.get_device_properties.side_effect = RuntimeError("CUDA error")
        with patch.dict("sys.modules", {"torch": torch_stub}):
            assert detect_recommended_tier() == "quality"

    def test_cuda_is_available_exception_returns_quality(self):
        torch_stub = MagicMock()
        torch_stub.cuda.is_available.side_effect = RuntimeError("driver error")
        with patch.dict("sys.modules", {"torch": torch_stub}):
            assert detect_recommended_tier() == "quality"

    def test_torch_import_failure_returns_quality(self):
        # Force the import to raise by removing torch from sys.modules and
        # patching the import system. Use a MagicMock that raises on attr
        # access — simpler than full meta-path manipulation.
        original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def _failing_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("torch not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_failing_import):
            assert detect_recommended_tier() == "quality"
