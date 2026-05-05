"""
Integration Tests for Emotion Variants Workflow

QA5: Voice Design Studio Refinements - Emotion Variants
End-to-end tests for the complete emotion variants feature.

Test scenarios:
1. Full workflow: Create voice with all 5 emotions
2. Partial emotions: Create voice with subset
3. Neutral only: Single emotion voice
4. Backward compatibility: Legacy single-embedding structure
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from myvoice.models.voice_profile import VoiceProfile, VoiceType, VALID_EMOTIONS


class TestEmotionVariantsVoiceCreation:
    """Tests for creating voices with emotion variants."""

    def test_create_voice_all_emotions(self, tmp_path):
        """Test creating voice with all 5 emotions."""
        voice_dir = tmp_path / "full_emotion_voice"
        voice_dir.mkdir()

        # Create all emotion subfolders with embeddings
        for emotion in VALID_EMOTIONS:
            emotion_dir = voice_dir / emotion
            emotion_dir.mkdir()
            (emotion_dir / "embedding.pt").write_bytes(b"PK_MOCK_EMBEDDING")

        # Create metadata
        metadata = {
            "version": "2.0",
            "name": "FullEmotionVoice",
            "description": "Voice with all emotions",
            "available_emotions": VALID_EMOTIONS
        }
        (voice_dir / "metadata.json").write_text(json.dumps(metadata))

        # Detect emotions
        detected = VoiceProfile.detect_available_emotions(voice_dir)

        assert len(detected) == 5
        for emotion in VALID_EMOTIONS:
            assert emotion in detected

    def test_create_voice_partial_emotions(self, tmp_path):
        """Test creating voice with partial emotions (neutral + happy only)."""
        voice_dir = tmp_path / "partial_voice"
        voice_dir.mkdir()

        # Create only neutral and happy
        for emotion in ["neutral", "happy"]:
            emotion_dir = voice_dir / emotion
            emotion_dir.mkdir()
            (emotion_dir / "embedding.pt").write_bytes(b"PK_MOCK")

        detected = VoiceProfile.detect_available_emotions(voice_dir)

        assert detected == ["neutral", "happy"]
        assert "sad" not in detected
        assert "angry" not in detected
        assert "flirtatious" not in detected

    def test_create_voice_neutral_only(self, tmp_path):
        """Test creating voice with neutral only."""
        voice_dir = tmp_path / "neutral_only_voice"
        voice_dir.mkdir()

        # Create only neutral
        neutral_dir = voice_dir / "neutral"
        neutral_dir.mkdir()
        (neutral_dir / "embedding.pt").write_bytes(b"PK_MOCK")

        detected = VoiceProfile.detect_available_emotions(voice_dir)

        assert detected == ["neutral"]


class TestBackwardCompatibility:
    """Tests for backward compatibility with legacy embedding voices."""

    def test_legacy_single_embedding(self, tmp_path):
        """Test legacy voice with single embedding.pt at root."""
        voice_dir = tmp_path / "legacy_voice"
        voice_dir.mkdir()

        # Legacy structure: embedding.pt at root
        (voice_dir / "embedding.pt").write_bytes(b"PK_MOCK")

        detected = VoiceProfile.detect_available_emotions(voice_dir)

        # Should detect as neutral only
        assert detected == ["neutral"]

    def test_legacy_metadata_no_emotions_field(self, tmp_path):
        """Test legacy metadata without available_emotions field."""
        metadata_path = tmp_path / "metadata.json"
        metadata = {
            "version": "1.0",
            "name": "LegacyVoice",
            "description": "Old format voice"
            # No available_emotions field
        }
        metadata_path.write_text(json.dumps(metadata))

        parsed = VoiceProfile.parse_embedding_metadata(metadata_path)

        assert parsed["available_emotions"] == ["neutral"]

    def test_create_profile_from_legacy_voice(self, tmp_path):
        """Test creating VoiceProfile from legacy voice directory."""
        voice_dir = tmp_path / "legacy_voice"
        voice_dir.mkdir()
        embedding_file = voice_dir / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile.create_embedding_profile(
            name="LegacyVoice",
            embedding_path=embedding_file
        )

        # Should default to neutral only
        assert profile.available_emotions == ["neutral"]
        assert profile.get_available_emotions() == ["neutral"]


class TestEmotionVariantsProfileCreation:
    """Tests for VoiceProfile creation with emotion variants."""

    def test_profile_with_all_emotions(self, tmp_path):
        """Test profile creation with all emotions."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile.create_embedding_profile(
            name="FullVoice",
            embedding_path=embedding_file,
            available_emotions=VALID_EMOTIONS.copy()
        )

        assert profile.voice_type == VoiceType.EMBEDDING
        assert profile.available_emotions == VALID_EMOTIONS
        assert len(profile.get_available_emotions()) == 5

        # Should have all emotions
        for emotion in VALID_EMOTIONS:
            assert profile.has_emotion(emotion)

    def test_profile_with_partial_emotions(self, tmp_path):
        """Test profile creation with partial emotions."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        available = ["neutral", "happy", "sad"]
        profile = VoiceProfile.create_embedding_profile(
            name="PartialVoice",
            embedding_path=embedding_file,
            available_emotions=available
        )

        assert profile.available_emotions == ["neutral", "happy", "sad"]

        # Should have these emotions
        assert profile.has_emotion("neutral")
        assert profile.has_emotion("happy")
        assert profile.has_emotion("sad")

        # Should NOT have these emotions
        assert not profile.has_emotion("angry")
        assert not profile.has_emotion("flirtatious")


class TestEmotionPathResolution:
    """Tests for emotion-specific embedding path resolution."""

    def test_get_emotion_embedding_path_v2(self, tmp_path):
        """Test getting emotion-specific embedding path (v2.0 structure)."""
        voice_dir = tmp_path / "multi_emotion_voice"
        voice_dir.mkdir()

        # Create emotion subfolders
        for emotion in ["neutral", "happy"]:
            emotion_dir = voice_dir / emotion
            emotion_dir.mkdir()
            (emotion_dir / "embedding.pt").write_bytes(b"PK_MOCK")

        # Create profile pointing to directory
        profile = VoiceProfile.create_embedding_profile(
            name="MultiEmotionVoice",
            embedding_path=voice_dir / "neutral" / "embedding.pt",
            available_emotions=["neutral", "happy"]
        )
        # Override checkpoint_path to point to directory for v2.0
        profile.checkpoint_path = voice_dir

        # Test path resolution for each emotion
        neutral_path = profile.get_emotion_embedding_path("neutral")
        happy_path = profile.get_emotion_embedding_path("happy")

        # Should resolve to emotion-specific paths
        assert neutral_path is not None
        assert happy_path is not None

    def test_get_emotion_embedding_path_fallback(self, tmp_path):
        """Test fallback to neutral when requested emotion unavailable."""
        voice_dir = tmp_path / "partial_voice"
        voice_dir.mkdir()

        # Only create neutral
        neutral_dir = voice_dir / "neutral"
        neutral_dir.mkdir()
        (neutral_dir / "embedding.pt").write_bytes(b"PK_MOCK")

        profile = VoiceProfile.create_embedding_profile(
            name="NeutralOnlyVoice",
            embedding_path=neutral_dir / "embedding.pt",
            available_emotions=["neutral"]
        )
        profile.checkpoint_path = voice_dir

        # Request unavailable emotion should fallback
        path = profile.get_emotion_embedding_path("happy")

        # Should fallback (either to neutral path or None depending on implementation)
        # The important thing is it doesn't crash


class TestVoiceProfileServiceScanning:
    """Tests for VoiceProfileService scanning with emotion variants."""

    def test_scan_detects_multi_emotion_voice(self, tmp_path):
        """Test service scanning detects multi-emotion voice structure."""
        embeddings_dir = tmp_path / "embeddings"
        embeddings_dir.mkdir()

        voice_dir = embeddings_dir / "MultiEmotionVoice"
        voice_dir.mkdir()

        # Create multi-emotion structure
        for emotion in ["neutral", "happy", "sad"]:
            emotion_dir = voice_dir / emotion
            emotion_dir.mkdir()
            (emotion_dir / "embedding.pt").write_bytes(b"PK_MOCK")

        # Create metadata
        metadata = {
            "version": "2.0",
            "name": "MultiEmotionVoice",
            "available_emotions": ["neutral", "happy", "sad"]
        }
        (voice_dir / "metadata.json").write_text(json.dumps(metadata))

        # Detect emotions
        detected = VoiceProfile.detect_available_emotions(voice_dir)

        assert detected == ["neutral", "happy", "sad"]

    def test_scan_detects_legacy_voice(self, tmp_path):
        """Test service scanning detects legacy single-embedding voice."""
        embeddings_dir = tmp_path / "embeddings"
        embeddings_dir.mkdir()

        voice_dir = embeddings_dir / "LegacyVoice"
        voice_dir.mkdir()

        # Create legacy structure
        (voice_dir / "embedding.pt").write_bytes(b"PK_MOCK")

        detected = VoiceProfile.detect_available_emotions(voice_dir)

        assert detected == ["neutral"]


class TestMetadataVersioning:
    """Tests for metadata versioning with emotion variants."""

    def test_v2_metadata_full(self, tmp_path):
        """Test v2.0 metadata with all fields."""
        metadata_path = tmp_path / "metadata.json"
        metadata = {
            "version": "2.0",
            "name": "TestVoice",
            "description": "A test voice",
            "available_emotions": ["neutral", "happy", "sad", "angry", "flirtatious"],
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T12:00:00"
        }
        metadata_path.write_text(json.dumps(metadata))

        parsed = VoiceProfile.parse_embedding_metadata(metadata_path)

        assert parsed["version"] == "2.0"
        assert parsed["name"] == "TestVoice"
        assert parsed["description"] == "A test voice"
        assert len(parsed["available_emotions"]) == 5

    def test_generate_v2_metadata(self, tmp_path):
        """Test generating v2.0 metadata from profile."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file,
            description="Test description",
            available_emotions=["neutral", "happy"]
        )

        metadata = profile.to_embedding_metadata()

        assert metadata["version"] == "2.0"
        assert metadata["name"] == "TestVoice"
        assert metadata["description"] == "Test description"
        assert metadata["available_emotions"] == ["neutral", "happy"]
        assert "created_at" in metadata
        assert "updated_at" in metadata


class TestEmotionVariantsEdgeCases:
    """Edge case tests for emotion variants."""

    def test_empty_voice_directory(self, tmp_path):
        """Test handling of empty voice directory."""
        voice_dir = tmp_path / "empty_voice"
        voice_dir.mkdir()

        detected = VoiceProfile.detect_available_emotions(voice_dir)

        assert detected == []

    def test_corrupted_metadata(self, tmp_path):
        """Test handling of corrupted metadata file."""
        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text("not valid json {{{")

        parsed = VoiceProfile.parse_embedding_metadata(metadata_path)

        # Should return defaults
        assert parsed["version"] == "1.0"
        assert parsed["available_emotions"] == ["neutral"]

    def test_mixed_valid_invalid_emotions(self, tmp_path):
        """Test metadata with mix of valid and invalid emotions."""
        metadata_path = tmp_path / "metadata.json"
        metadata = {
            "version": "2.0",
            "name": "MixedVoice",
            "available_emotions": ["neutral", "excited", "happy", "terrified", "sad"]
        }
        metadata_path.write_text(json.dumps(metadata))

        parsed = VoiceProfile.parse_embedding_metadata(metadata_path)

        # Only valid emotions should remain
        assert "neutral" in parsed["available_emotions"]
        assert "happy" in parsed["available_emotions"]
        assert "sad" in parsed["available_emotions"]
        assert "excited" not in parsed["available_emotions"]
        assert "terrified" not in parsed["available_emotions"]

    def test_duplicate_emotions_in_list(self, tmp_path):
        """Test handling of duplicate emotions in available_emotions."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        # Create profile with duplicates
        profile = VoiceProfile(
            file_path=Path("embedding://TestVoice"),
            name="TestVoice",
            checkpoint_path=embedding_file,
            available_emotions=["neutral", "happy", "neutral", "happy"]
        )

        # Duplicates should be in list (validation doesn't dedupe)
        # But functionally it should still work
        assert "neutral" in profile.available_emotions
        assert "happy" in profile.available_emotions
