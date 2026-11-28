"""tooling-11 — the bundle ships without cuDNN's sub-libraries.

The two halves must never drift apart: dropping the files while cuDNN is
still enabled aborts the process on the first convolution, and disabling
cuDNN while still shipping the files wastes 956 MB.
"""
import re
from pathlib import Path

import torch

from myvoice.services.tts_streaming import torch_runtime

REPO = Path(__file__).resolve().parents[2]


def test_disable_cudnn_turns_the_backend_off():
    saved = torch.backends.cudnn.enabled
    try:
        torch.backends.cudnn.enabled = True
        assert torch_runtime.disable_cudnn() is True
        assert torch.backends.cudnn.enabled is False
    finally:
        torch.backends.cudnn.enabled = saved


def test_spec_drops_exactly_the_libraries_the_app_stops_using():
    spec = (REPO / "build_tools" / "myvoice.spec").read_text(encoding="utf-8")
    dropped = set(re.findall(r"'(cudnn[\w.]*\.dll)'", spec))
    assert dropped == set(torch_runtime.CUDNN_LIBRARIES_NOT_SHIPPED)


def test_the_two_libraries_torch_needs_to_import_are_kept():
    assert "cudnn64_9.dll" not in torch_runtime.CUDNN_LIBRARIES_NOT_SHIPPED
    assert "cudnn_graph64_9.dll" not in torch_runtime.CUDNN_LIBRARIES_NOT_SHIPPED


def test_app_startup_disables_cudnn():
    """Source invariant: main.py must call disable_cudnn, or a build made
    from this tree aborts on its first convolution."""
    source = (REPO / "src" / "myvoice" / "main.py").read_text(encoding="utf-8")
    code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    assert "disable_cudnn()" in code
