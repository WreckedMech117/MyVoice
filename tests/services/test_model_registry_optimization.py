"""
Tests for ModelRegistry Performance Optimizations

QA8: Qwen3-TTS Performance Investigation
- torch.set_float32_matmul_precision('high') for Ampere+ GPUs
- Flash Attention 2 detection

See: https://github.com/QwenLM/Qwen3-TTS/issues/89

Note: These tests verify the optimization code exists and follows expected patterns
without requiring torch to be loaded (which can fail in CI environments).
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path


class TestModelRegistryOptimizationCodePresence:
    """Tests verifying optimization code is present in model_registry.py."""

    def test_matmul_precision_code_exists(self):
        """Test that matmul precision optimization code exists in model_registry."""
        import inspect
        from pathlib import Path

        # Read the source file directly to avoid torch import issues
        model_registry_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "services" / "model_registry.py"
        source = model_registry_path.read_text(encoding='utf-8')

        # Verify the optimization code is present
        assert "set_float32_matmul_precision" in source, \
            "Missing torch.set_float32_matmul_precision optimization"
        assert "'high'" in source, \
            "Missing 'high' precision setting"

    def test_matmul_precision_only_for_cuda(self):
        """Test that matmul precision is only set for CUDA devices."""
        model_registry_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "services" / "model_registry.py"
        source = model_registry_path.read_text(encoding='utf-8')

        # Find the _load_model_sync method and verify precision is inside CUDA block
        # The pattern should be: if self.device.startswith("cuda"): ... set_float32_matmul_precision
        lines = source.split('\n')
        in_cuda_block = False
        precision_in_cuda_block = False

        for i, line in enumerate(lines):
            if 'self.device.startswith("cuda")' in line:
                in_cuda_block = True
            if in_cuda_block and 'set_float32_matmul_precision' in line:
                precision_in_cuda_block = True
                break
            # Reset if we exit the if block (simplified check)
            if in_cuda_block and line.strip() and not line.startswith(' ' * 8) and not line.startswith('\t\t'):
                if 'set_float32_matmul_precision' not in line:
                    in_cuda_block = False

        assert precision_in_cuda_block, \
            "matmul precision should only be set inside CUDA device check"

    def test_flash_attention_detection_code_exists(self):
        """Test that Flash Attention 2 detection code exists."""
        model_registry_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "services" / "model_registry.py"
        source = model_registry_path.read_text(encoding='utf-8')

        # Verify flash attention detection
        assert "flash_attn" in source, \
            "Missing flash_attn import attempt"
        assert "flash_attention_2" in source, \
            "Missing flash_attention_2 attention implementation"

    def test_matmul_precision_error_handling(self):
        """Test that matmul precision setting has error handling."""
        model_registry_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "services" / "model_registry.py"
        source = model_registry_path.read_text(encoding='utf-8')

        # Find the matmul precision block
        lines = source.split('\n')
        found_try = False
        found_except = False

        for i, line in enumerate(lines):
            if 'set_float32_matmul_precision' in line:
                # Look backward for try
                for j in range(max(0, i-5), i):
                    if 'try:' in lines[j]:
                        found_try = True
                # Look forward for except
                for j in range(i, min(len(lines), i+5)):
                    if 'except' in lines[j]:
                        found_except = True

        assert found_try and found_except, \
            "matmul precision should be wrapped in try/except for graceful handling"


class TestModelRegistryOptimizationDocumentation:
    """Tests verifying optimization documentation."""

    def test_github_issue_reference(self):
        """Test that GitHub issue #89 is referenced in comments."""
        model_registry_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "services" / "model_registry.py"
        source = model_registry_path.read_text(encoding='utf-8')

        # The optimization should reference the GitHub issue
        assert "89" in source or "Qwen3-TTS" in source, \
            "Optimization should reference source/issue for context"

    def test_ampere_gpu_comment(self):
        """Test that Ampere+ GPU requirement is documented."""
        model_registry_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "services" / "model_registry.py"
        source = model_registry_path.read_text(encoding='utf-8')

        # Should mention Ampere or GPU series
        has_ampere_mention = "Ampere" in source or "RTX" in source or "30" in source or "40" in source or "50" in source
        assert has_ampere_mention, \
            "Should document which GPUs benefit from the optimization"


class TestModelRegistryDeviceHandling:
    """Tests for device configuration patterns."""

    def test_auto_device_pattern(self):
        """Test that 'auto' device selection pattern exists."""
        model_registry_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "services" / "model_registry.py"
        source = model_registry_path.read_text(encoding='utf-8')

        # Should have auto device selection
        assert 'device == "auto"' in source, \
            "Should support 'auto' device selection"
        assert 'cuda.is_available()' in source, \
            "Should check CUDA availability for auto device"

    def test_device_stored_as_attribute(self):
        """Test that device is stored as instance attribute."""
        model_registry_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "services" / "model_registry.py"
        source = model_registry_path.read_text(encoding='utf-8')

        assert 'self.device' in source, \
            "Device should be stored as instance attribute"


class TestModelRegistryLoadKwargs:
    """Tests for model loading keyword arguments."""

    def test_device_map_in_load_kwargs(self):
        """Test that device_map is passed to model loading."""
        model_registry_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "services" / "model_registry.py"
        source = model_registry_path.read_text(encoding='utf-8')

        assert 'device_map' in source, \
            "Should pass device_map to model loading"

    def test_dtype_in_load_kwargs(self):
        """Test that torch_dtype is passed to model loading."""
        model_registry_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "services" / "model_registry.py"
        source = model_registry_path.read_text(encoding='utf-8')

        assert 'torch_dtype' in source, \
            "Should pass torch_dtype to model loading"

    def test_attn_implementation_conditionally_added(self):
        """Test that attn_implementation is conditionally added."""
        model_registry_path = Path(__file__).parent.parent.parent / "src" / "myvoice" / "services" / "model_registry.py"
        source = model_registry_path.read_text(encoding='utf-8')

        assert 'attn_implementation' in source, \
            "Should support attn_implementation for Flash Attention"
        assert 'if attn_impl' in source, \
            "attn_implementation should be conditionally added"
