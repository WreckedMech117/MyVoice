"""
Tests for Emotion Variants Feature in VoiceProfile

QA5: Voice Design Studio Refinements - Emotion Variants
Tests for EMBEDDING voices with multiple emotion-specific embeddings.

Features tested:
- available_emotions field and validation
- detect_available_emotions() static method
- get_available_emotions() instance method
- has_emotion() method for EMBEDDING voices
- Metadata JSON parsing with emotion variants
- Serialization/deserialization with emotion variants
"""

import pytest
import json
from pathlib import Path
from datetime import datetime

from myvoice.models.voice_profile import VoiceProfile, VoiceType, VALID_EMOTIONS


class TestValidEmotions:
    """Tests for VALID_EMOTIONS constant."""

    def test_valid_emotions_count(self):
        """Test there are exactly 5 valid emotions."""
        assert len(VALID_EMOTIONS) == 5

    def test_valid_emotions_values(self):
        """Test valid emotion values match expected list."""
        expected = ["neutral", "happy", "sad", "angry", "flirtatious"]
        assert VALID_EMOTIONS == expected


class TestAvailableEmotionsField:
    """Tests for available_emotions field in VoiceProfile."""

    def test_default_available_emotions(self, tmp_path):
        """Test default available_emotions is ['neutral']."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file
        )

        assert profile.available_emotions == ["neutral"]

    def test_custom_available_emotions(self, tmp_path):
        """Test setting custom available_emotions."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file,
            available_emotions=["neutral", "happy", "sad"]
        )

        assert profile.available_emotions == ["neutral", "happy", "sad"]

    def test_all_emotions_available(self, tmp_path):
        """Test voice with all 5 emotions available."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        all_emotions = ["neutral", "happy", "sad", "angry", "flirtatious"]
        profile = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file,
            available_emotions=all_emotions
        )

        assert profile.available_emotions == all_emotions
        assert len(profile.available_emotions) == 5

    def test_invalid_emotions_filtered(self, tmp_path):
        """Test invalid emotions are filtered out during validation."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile(
            file_path=Path("embedding://TestVoice"),
            name="TestVoice",
            checkpoint_path=embedding_file,
            available_emotions=["neutral", "invalid_emotion", "happy", "unknown"]
        )

        # Only valid emotions should remain
        assert "invalid_emotion" not in profile.available_emotions
        assert "unknown" not in profile.available_emotions
        assert "neutral" in profile.available_emotions
        assert "happy" in profile.available_emotions

    def test_all_invalid_defaults_to_neutral(self, tmp_path):
        """Test that all invalid emotions defaults to ['neutral']."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile(
            file_path=Path("embedding://TestVoice"),
            name="TestVoice",
            checkpoint_path=embedding_file,
            available_emotions=["invalid1", "invalid2"]
        )

        # Should default to neutral
        assert profile.available_emotions == ["neutral"]


class TestDetectAvailableEmotions:
    """Tests for VoiceProfile.detect_available_emotions() static method."""

    def test_detect_empty_directory(self, tmp_path):
        """Test detection in empty directory returns empty list."""
        voice_dir = tmp_path / "empty_voice"
        voice_dir.mkdir()

        emotions = VoiceProfile.detect_available_emotions(voice_dir)
        assert emotions == []

    def test_detect_legacy_structure(self, tmp_path):
        """Test detection of legacy single embedding (root embedding.pt)."""
        voice_dir = tmp_path / "legacy_voice"
        voice_dir.mkdir()
        (voice_dir / "embedding.pt").write_bytes(b"PK_MOCK")

        emotions = VoiceProfile.detect_available_emotions(voice_dir)
        assert emotions == ["neutral"]

    def test_detect_single_emotion_subfolder(self, tmp_path):
        """Test detection of single emotion subfolder."""
        voice_dir = tmp_path / "single_emotion_voice"
        voice_dir.mkdir()
        neutral_dir = voice_dir / "neutral"
        neutral_dir.mkdir()
        (neutral_dir / "embedding.pt").write_bytes(b"PK_MOCK")

        emotions = VoiceProfile.detect_available_emotions(voice_dir)
        assert emotions == ["neutral"]

    def test_detect_multiple_emotion_subfolders(self, tmp_path):
        """Test detection of multiple emotion subfolders."""
        voice_dir = tmp_path / "multi_emotion_voice"
        voice_dir.mkdir()

        # Create emotion subfolders
        for emotion in ["neutral", "happy", "sad"]:
            emotion_dir = voice_dir / emotion
            emotion_dir.mkdir()
            (emotion_dir / "embedding.pt").write_bytes(b"PK_MOCK")

        emotions = VoiceProfile.detect_available_emotions(voice_dir)

        assert "neutral" in emotions
        assert "happy" in emotions
        assert "sad" in emotions
        assert len(emotions) == 3

    def test_detect_all_emotions(self, tmp_path):
        """Test detection of all 5 emotion subfolders."""
        voice_dir = tmp_path / "full_emotion_voice"
        voice_dir.mkdir()

        # Create all emotion subfolders
        for emotion in VALID_EMOTIONS:
            emotion_dir = voice_dir / emotion
            emotion_dir.mkdir()
            (emotion_dir / "embedding.pt").write_bytes(b"PK_MOCK")

        emotions = VoiceProfile.detect_available_emotions(voice_dir)

        assert len(emotions) == 5
        for emotion in VALID_EMOTIONS:
            assert emotion in emotions

    def test_detect_ignores_empty_subfolders(self, tmp_path):
        """Test that empty subfolders (no embedding.pt) are ignored."""
        voice_dir = tmp_path / "partial_voice"
        voice_dir.mkdir()

        # Create neutral with embedding
        neutral_dir = voice_dir / "neutral"
        neutral_dir.mkdir()
        (neutral_dir / "embedding.pt").write_bytes(b"PK_MOCK")

        # Create happy without embedding (empty)
        happy_dir = voice_dir / "happy"
        happy_dir.mkdir()

        emotions = VoiceProfile.detect_available_emotions(voice_dir)

        assert "neutral" in emotions
        assert "happy" not in emotions
        assert len(emotions) == 1

    def test_detect_ignores_invalid_folders(self, tmp_path):
        """Test that non-emotion folders are ignored."""
        voice_dir = tmp_path / "mixed_voice"
        voice_dir.mkdir()

        # Create valid emotion
        neutral_dir = voice_dir / "neutral"
        neutral_dir.mkdir()
        (neutral_dir / "embedding.pt").write_bytes(b"PK_MOCK")

        # Create invalid folder with embedding
        invalid_dir = voice_dir / "excited"  # Not a valid emotion
        invalid_dir.mkdir()
        (invalid_dir / "embedding.pt").write_bytes(b"PK_MOCK")

        emotions = VoiceProfile.detect_available_emotions(voice_dir)

        assert "neutral" in emotions
        assert "excited" not in emotions
        assert len(emotions) == 1

    def test_detect_preserves_order(self, tmp_path):
        """Test that detected emotions follow VALID_EMOTIONS order."""
        voice_dir = tmp_path / "ordered_voice"
        voice_dir.mkdir()

        # Create in reverse order
        for emotion in reversed(["neutral", "happy", "sad", "angry"]):
            emotion_dir = voice_dir / emotion
            emotion_dir.mkdir()
            (emotion_dir / "embedding.pt").write_bytes(b"PK_MOCK")

        emotions = VoiceProfile.detect_available_emotions(voice_dir)

        # Should be in VALID_EMOTIONS order, not creation order
        expected_order = ["neutral", "happy", "sad", "angry"]
        assert emotions == expected_order

    def test_detect_nonexistent_directory(self, tmp_path):
        """Test detection on nonexistent directory returns neutral (default fallback)."""
        nonexistent = tmp_path / "nonexistent"

        emotions = VoiceProfile.detect_available_emotions(nonexistent)
        # Implementation returns ["neutral"] for nonexistent directories as safe default
        assert emotions == ["neutral"]


class TestGetAvailableEmotions:
    """Tests for VoiceProfile.get_available_emotions() instance method."""

    def test_embedding_voice_returns_available_emotions(self, tmp_path):
        """Test EMBEDDING voice returns its available_emotions list."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file,
            available_emotions=["neutral", "happy"]
        )

        emotions = profile.get_available_emotions()
        assert emotions == ["neutral", "happy"]

    def test_embedding_voice_returns_copy(self, tmp_path):
        """Test that get_available_emotions returns a copy, not the original list."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file,
            available_emotions=["neutral", "happy"]
        )

        emotions = profile.get_available_emotions()
        emotions.append("sad")  # Modify the returned list

        # Original should be unchanged
        assert profile.available_emotions == ["neutral", "happy"]

    def test_bundled_voice_returns_all_emotions(self):
        """Test BUNDLED voice returns all valid emotions."""
        profile = VoiceProfile.create_bundled_profile("Ryan")

        emotions = profile.get_available_emotions()
        assert emotions == VALID_EMOTIONS

    def test_cloned_voice_returns_neutral_only(self, tmp_path):
        """Test CLONED voice returns only neutral."""
        # Create a mock audio file
        audio_file = tmp_path / "voice.wav"
        audio_file.write_bytes(b"RIFF_MOCK")

        profile = VoiceProfile(
            file_path=audio_file,
            name="ClonedVoice",
            voice_type=VoiceType.CLONED,
            is_valid=True
        )

        emotions = profile.get_available_emotions()
        assert emotions == ["neutral"]


class TestHasEmotion:
    """Tests for VoiceProfile.has_emotion() method."""

    def test_embedding_voice_has_available_emotion(self, tmp_path):
        """Test EMBEDDING voice has_emotion returns True for available emotions."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file,
            available_emotions=["neutral", "happy", "sad"]
        )

        assert profile.has_emotion("neutral") is True
        assert profile.has_emotion("happy") is True
        assert profile.has_emotion("sad") is True

    def test_embedding_voice_lacks_unavailable_emotion(self, tmp_path):
        """Test EMBEDDING voice has_emotion returns False for unavailable emotions."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file,
            available_emotions=["neutral", "happy"]
        )

        assert profile.has_emotion("sad") is False
        assert profile.has_emotion("angry") is False
        assert profile.has_emotion("flirtatious") is False

    def test_bundled_voice_has_all_emotions(self):
        """Test BUNDLED voice has all valid emotions."""
        profile = VoiceProfile.create_bundled_profile("Ryan")

        for emotion in VALID_EMOTIONS:
            assert profile.has_emotion(emotion) is True

    def test_cloned_voice_only_has_neutral(self, tmp_path):
        """Test CLONED voice only has neutral emotion."""
        audio_file = tmp_path / "voice.wav"
        audio_file.write_bytes(b"RIFF_MOCK")

        profile = VoiceProfile(
            file_path=audio_file,
            name="ClonedVoice",
            voice_type=VoiceType.CLONED,
            is_valid=True
        )

        assert profile.has_emotion("neutral") is True
        assert profile.has_emotion("happy") is False
        assert profile.has_emotion("sad") is False


class TestMetadataEmotionVariants:
    """Tests for metadata JSON parsing with emotion variants."""

    def test_parse_metadata_v2_with_emotions(self, tmp_path):
        """Test parsing v2.0 metadata with available_emotions."""
        metadata_path = tmp_path / "metadata.json"
        metadata = {
            "version": "2.0",
            "name": "TestVoice",
            "description": "A test voice",
            "available_emotions": ["neutral", "happy", "sad"]
        }
        metadata_path.write_text(json.dumps(metadata))

        parsed = VoiceProfile.parse_embedding_metadata(metadata_path)

        assert parsed["version"] == "2.0"
        assert parsed["available_emotions"] == ["neutral", "happy", "sad"]

    def test_parse_metadata_v1_defaults_to_neutral(self, tmp_path):
        """Test parsing v1.0 metadata defaults to neutral only."""
        metadata_path = tmp_path / "metadata.json"
        metadata = {
            "version": "1.0",
            "name": "LegacyVoice",
            "description": "A legacy voice"
        }
        metadata_path.write_text(json.dumps(metadata))

        parsed = VoiceProfile.parse_embedding_metadata(metadata_path)

        assert parsed["available_emotions"] == ["neutral"]

    def test_parse_metadata_missing_file_defaults(self, tmp_path):
        """Test parsing nonexistent metadata defaults correctly."""
        metadata_path = tmp_path / "nonexistent.json"

        parsed = VoiceProfile.parse_embedding_metadata(metadata_path)

        assert parsed["version"] == "1.0"
        assert parsed["available_emotions"] == ["neutral"]

    def test_parse_metadata_filters_invalid_emotions(self, tmp_path):
        """Test that invalid emotions are filtered from metadata."""
        metadata_path = tmp_path / "metadata.json"
        metadata = {
            "version": "2.0",
            "name": "TestVoice",
            "available_emotions": ["neutral", "invalid", "happy", "unknown"]
        }
        metadata_path.write_text(json.dumps(metadata))

        parsed = VoiceProfile.parse_embedding_metadata(metadata_path)

        assert "invalid" not in parsed["available_emotions"]
        assert "unknown" not in parsed["available_emotions"]
        assert "neutral" in parsed["available_emotions"]
        assert "happy" in parsed["available_emotions"]

    def test_to_embedding_metadata(self, tmp_path):
        """Test generating metadata dict from profile."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file,
            description="A test voice",
            available_emotions=["neutral", "happy", "sad"]
        )

        metadata = profile.to_embedding_metadata()

        # 2026-05-06: schema version bumped from "2.0" to "3.0"
        # (voice_profile.py:1212, 1225 — "Multi-tier schema version").
        # The test was orphaned by the unanchored .gitignore `models/`
        # rule and never caught the bump.
        assert metadata["version"] == "3.0"
        assert metadata["name"] == "TestVoice"
        assert metadata["description"] == "A test voice"
        assert metadata["available_emotions"] == ["neutral", "happy", "sad"]


class TestEmotionVariantsSerialization:
    """Tests for emotion variants in serialization/deserialization."""

    def test_to_dict_includes_available_emotions(self, tmp_path):
        """Test to_dict includes available_emotions field."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file,
            available_emotions=["neutral", "happy", "angry"]
        )

        data = profile.to_dict()

        assert "available_emotions" in data
        assert data["available_emotions"] == ["neutral", "happy", "angry"]

    def test_from_dict_parses_available_emotions(self, tmp_path):
        """Test from_dict correctly parses available_emotions."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        data = {
            "file_path": "embedding://TestVoice",
            "name": "TestVoice",
            "voice_type": "embedding",
            "checkpoint_path": str(embedding_file),
            "is_valid": True,
            "available_emotions": ["neutral", "sad", "flirtatious"]
        }

        profile = VoiceProfile.from_dict(data)

        assert profile.available_emotions == ["neutral", "sad", "flirtatious"]

    def test_from_dict_defaults_available_emotions(self, tmp_path):
        """Test from_dict defaults available_emotions when not present."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        data = {
            "file_path": "embedding://TestVoice",
            "name": "TestVoice",
            "voice_type": "embedding",
            "checkpoint_path": str(embedding_file),
            "is_valid": True
            # No available_emotions field
        }

        profile = VoiceProfile.from_dict(data)

        assert profile.available_emotions == ["neutral"]

    def test_roundtrip_serialization(self, tmp_path):
        """Test full roundtrip: profile -> dict -> profile preserves emotions."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        original = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file,
            available_emotions=["neutral", "happy", "sad", "angry", "flirtatious"]
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = VoiceProfile.from_dict(data)

        assert restored.available_emotions == original.available_emotions
        assert restored.name == original.name
        assert restored.voice_type == original.voice_type
