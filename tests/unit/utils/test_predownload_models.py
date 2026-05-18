"""Unit tests for the install-time model predownload entry point."""
from __future__ import annotations

import logging
import sys
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from myvoice.utils.predownload_models import (
    _ALLOW_PATTERNS,
    _parse_argv,
    _resolve_model_ids_for_tier,
    run_predownload,
)


class TestResolveModelIdsForTier:
    def test_quality_tier_returns_three_1_7b_models(self):
        ids = _resolve_model_ids_for_tier("quality")
        assert ids == [
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        ]

    def test_small_tier_returns_two_0_6b_models_voicedesign_dropped(self):
        # VoiceDesign is QUALITY-only — Small tier drops it so the
        # installer doesn't try to download a non-existent 0.6B variant.
        ids = _resolve_model_ids_for_tier("small")
        assert ids == [
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        ]

    def test_tier_is_case_insensitive_and_strip_tolerant(self):
        assert _resolve_model_ids_for_tier("QUALITY") == _resolve_model_ids_for_tier("quality")
        assert _resolve_model_ids_for_tier("  small  ") == _resolve_model_ids_for_tier("small")

    def test_unknown_tier_raises(self):
        with pytest.raises(ValueError, match="Unknown tier"):
            _resolve_model_ids_for_tier("medium")


class TestParseArgv:
    def test_returns_none_when_flag_absent(self):
        assert _parse_argv(["MyVoice.exe"]) is None
        assert _parse_argv(["MyVoice.exe", "--other-flag"]) is None

    def test_returns_tier_when_flag_present(self):
        argv = ["MyVoice.exe", "--predownload-models", "--tier=quality"]
        assert _parse_argv(argv) == "quality"

    def test_raises_when_flag_present_but_tier_missing(self):
        with pytest.raises(ValueError, match="--tier="):
            _parse_argv(["MyVoice.exe", "--predownload-models"])

    def test_raises_when_tier_empty(self):
        with pytest.raises(ValueError, match="--tier="):
            _parse_argv(["MyVoice.exe", "--predownload-models", "--tier="])


class TestRunPredownload:
    def test_returns_1_when_flag_missing(self, caplog):
        # Caller error: run_predownload was invoked but argv doesn't
        # contain --predownload-models. Returns 1 with an error.
        rc = run_predownload(["MyVoice.exe"])
        assert rc == 1

    def test_returns_1_when_tier_missing(self):
        rc = run_predownload(["MyVoice.exe", "--predownload-models"])
        assert rc == 1

    def test_returns_1_when_tier_unknown(self):
        rc = run_predownload([
            "MyVoice.exe", "--predownload-models", "--tier=medium",
        ])
        assert rc == 1

    def test_quality_tier_downloads_three_models(self):
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            mock_dl.return_value = "/fake/cache/path"
            rc = run_predownload([
                "MyVoice.exe", "--predownload-models", "--tier=quality",
            ])
        assert rc == 0
        assert mock_dl.call_count == 3
        called_ids = [call.kwargs["repo_id"] for call in mock_dl.call_args_list]
        assert called_ids == [
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        ]
        # `resume_download` is intentionally NOT passed: it's deprecated
        # in huggingface_hub >= 0.30, the default behavior already
        # resumes interrupted downloads, and emitting the
        # DeprecationWarning was the most likely cause of "exit code 2"
        # despite a successful download on the 2026-05-17 3060 smoke.
        for call in mock_dl.call_args_list:
            assert "resume_download" not in call.kwargs
        # `allow_patterns` must be passed so snapshot_download doesn't
        # pull the entire repo (which includes demo audio, README
        # media, etc.). Without this, the install-time download takes
        # 30+ minutes vs. 1-2 minutes for the equivalent from_pretrained.
        for call in mock_dl.call_args_list:
            allow = call.kwargs.get("allow_patterns")
            assert allow is _ALLOW_PATTERNS, (
                "snapshot_download must be called with allow_patterns="
                "_ALLOW_PATTERNS to restrict to the from_pretrained file set"
            )
        # Sanity-check the allow-list itself: it must include the file
        # types from_pretrained needs and the speech_tokenizer subfolder.
        assert "*.safetensors" in _ALLOW_PATTERNS
        assert "*.json" in _ALLOW_PATTERNS
        assert "speech_tokenizer/*.safetensors" in _ALLOW_PATTERNS
        assert "speech_tokenizer/*.json" in _ALLOW_PATTERNS

    def test_small_tier_downloads_two_models(self):
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            mock_dl.return_value = "/fake/cache/path"
            rc = run_predownload([
                "MyVoice.exe", "--predownload-models", "--tier=small",
            ])
        assert rc == 0
        assert mock_dl.call_count == 2

    def test_partial_failure_returns_2_and_continues_remaining(self):
        # Network drops mid-install: first model succeeds, second
        # raises, third still attempted. Exit code 2 (not 0) so the
        # installer can warn the user; the app will fall back to
        # downloading the missing models on first launch.
        call_history: List[str] = []

        def side_effect(*, repo_id, **_kwargs):
            call_history.append(repo_id)
            if "VoiceDesign" in repo_id:
                raise OSError("network died")
            return "/fake/cache/path"

        with patch(
            "huggingface_hub.snapshot_download", side_effect=side_effect
        ):
            rc = run_predownload([
                "MyVoice.exe", "--predownload-models", "--tier=quality",
            ])
        assert rc == 2
        # Even though the second call failed, the third was still
        # attempted — partial cache population is valuable.
        assert call_history == [
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        ]

    def test_huggingface_hub_import_failure_returns_2(self):
        # If for some reason huggingface_hub is missing from the bundle,
        # we surface a graceful failure rather than crashing — the
        # installer treats this as "download on first launch" fallback.
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            rc = run_predownload([
                "MyVoice.exe", "--predownload-models", "--tier=quality",
            ])
        assert rc == 2
