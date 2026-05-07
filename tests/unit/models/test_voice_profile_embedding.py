"""
Tests for Embedding Voice Profile Support

Story 1.6: Emotion Control for Saved Voice
- Tests VoiceType.EMBEDDING enum value and properties
- Tests VoiceProfile.create_embedding_profile factory method
- Tests VoiceProfile.__post_init__ handling of embedding:// virtual paths
- Tests embedding voice emotion capability
"""

import pytest
from pathlib import Path
from datetime import datetime

from myvoice.models.voice_profile import VoiceProfile, VoiceType, TranscriptionStatus


class TestVoiceTypeEmbedding:
    """Tests for VoiceType.EMBEDDING enum value and properties."""

    def test_embedding_type_exists(self):
        """Test EMBEDDING is a valid VoiceType value."""
        assert VoiceType.EMBEDDING is not None
        assert VoiceType.EMBEDDING.value == "embedding"

    def test_embedding_icon(self):
        """Test EMBEDDING has correct icon (DNA/gene symbol)."""
        assert VoiceType.EMBEDDING.icon == "🧬"

    def test_embedding_display_name(self):
        """Test EMBEDDING has correct display name."""
        assert VoiceType.EMBEDDING.display_name == "Embedding"

    def test_embedding_supports_emotion(self):
        """Test EMBEDDING voice type supports emotion control."""
        assert VoiceType.EMBEDDING.supports_emotion is True

    def test_embedding_sort_order(self):
        """Test EMBEDDING sorts after bundled voices."""
        assert VoiceType.EMBEDDING.sort_order == 1
        assert VoiceType.BUNDLED.sort_order < VoiceType.EMBEDDING.sort_order
        assert VoiceType.EMBEDDING.sort_order < VoiceType.DESIGNED.sort_order

    def test_embedding_required_model(self):
        """Test EMBEDDING uses Base model with voice_clone_prompt.

        2026-05-06: assertion updated from VOICE_DESIGN to BASE to match
        the current voice_profile implementation (line 119) — the test
        was orphaned by the unanchored .gitignore `models/` rule and
        never caught the EMBEDDING → Base model refactor. See
        voice_profile.py:107 docstring for the rationale.
        """
        from myvoice.models.service_enums import QwenModelType
        assert VoiceType.EMBEDDING.required_model == QwenModelType.BASE


class TestVoiceProfileCreateEmbeddingProfile:
    """Tests for VoiceProfile.create_embedding_profile factory method."""

    def test_create_embedding_profile_basic(self, tmp_path):
        """Test basic embedding profile creation."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK_EMBEDDING")

        profile = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file
        )

        assert profile is not None
        assert profile.name == "TestVoice"
        assert profile.is_valid is True
        assert profile.voice_type == VoiceType.EMBEDDING

    def test_create_embedding_profile_with_description(self, tmp_path):
        """Test embedding profile with description."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file,
            description="A warm, friendly voice"
        )

        assert profile.description == "A warm, friendly voice"

    def test_create_embedding_profile_emotion_capable(self, tmp_path):
        """Test embedding profile has emotion capability."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file
        )

        assert profile.emotion_capable is True

    def test_create_embedding_profile_transcription_skipped(self, tmp_path):
        """Test embedding profile has transcription skipped."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file
        )

        assert profile.transcription_status == TranscriptionStatus.SKIPPED

    def test_create_embedding_profile_stores_embedding_path(self, tmp_path):
        """Test embedding path is stored in checkpoint_path field."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file
        )

        assert profile.checkpoint_path == embedding_file

    def test_create_embedding_profile_virtual_path(self, tmp_path):
        """Test embedding profile has embedding:// virtual path."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file
        )

        file_path_str = str(profile.file_path)
        # On Windows, Path("embedding://X") becomes "embedding:\X"
        assert (
            file_path_str.startswith("embedding://") or
            file_path_str.startswith("embedding:\\") or
            file_path_str.startswith("embedding:")
        )

    def test_create_embedding_profile_nonexistent_path_raises(self, tmp_path):
        """Test creating profile with nonexistent embedding path raises error."""
        nonexistent = tmp_path / "nonexistent.pt"

        with pytest.raises(ValueError, match="does not exist"):
            VoiceProfile.create_embedding_profile(
                name="TestVoice",
                embedding_path=nonexistent
            )

    def test_create_embedding_profile_with_preview_audio(self, tmp_path):
        """Test embedding profile with preview audio extracts duration."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        # Create a simple WAV file (header only for duration extraction)
        preview_audio = tmp_path / "preview.wav"
        # Minimal WAV header: RIFF + fmt + data chunks
        # 44 byte header + 22050 bytes of data = ~0.5 seconds at 44100Hz mono 16-bit
        import wave
        with wave.open(str(preview_audio), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(44100)
            wav.writeframes(b'\x00' * 22050)  # ~0.25 seconds

        profile = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file,
            preview_audio_path=preview_audio
        )

        assert profile.duration is not None
        assert profile.duration > 0


class TestVoiceProfileEmbeddingPostInit:
    """Tests for VoiceProfile.__post_init__ handling of embedding:// paths."""

    def test_embedding_path_detection(self, tmp_path):
        """Test embedding:// virtual path is detected in __post_init__."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile(
            file_path=Path("embedding://TestVoice"),
            name="TestVoice",
            checkpoint_path=embedding_file
        )

        assert profile.voice_type == VoiceType.EMBEDDING
        assert profile.is_valid is True

    def test_embedding_path_invalid_without_checkpoint(self):
        """Test embedding:// path is invalid without checkpoint_path."""
        profile = VoiceProfile(
            file_path=Path("embedding://TestVoice"),
            name="TestVoice",
            checkpoint_path=None
        )

        assert profile.is_valid is False

    def test_embedding_path_invalid_with_nonexistent_checkpoint(self, tmp_path):
        """Test embedding:// path is invalid with nonexistent checkpoint."""
        nonexistent = tmp_path / "nonexistent.pt"

        profile = VoiceProfile(
            file_path=Path("embedding://TestVoice"),
            name="TestVoice",
            checkpoint_path=nonexistent
        )

        assert profile.is_valid is False

    def test_embedding_path_sets_emotion_capable(self, tmp_path):
        """Test embedding:// path sets emotion_capable to True."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile(
            file_path=Path("embedding://TestVoice"),
            name="TestVoice",
            checkpoint_path=embedding_file
        )

        assert profile.emotion_capable is True

    def test_embedding_path_sets_transcription_skipped(self, tmp_path):
        """Test embedding:// path sets transcription status to skipped."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile(
            file_path=Path("embedding://TestVoice"),
            name="TestVoice",
            checkpoint_path=embedding_file
        )

        assert profile.transcription_status == TranscriptionStatus.SKIPPED


class TestVoiceProfileEmbeddingHelpers:
    """Tests for embedding-related helper methods."""

    def test_is_embedding_true_for_embedding_type(self, tmp_path):
        """Test is_embedding returns True for EMBEDDING type."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file
        )

        assert profile.is_embedding() is True

    def test_is_embedding_false_for_other_types(self):
        """Test is_embedding returns False for non-EMBEDDING types."""
        bundled = VoiceProfile.create_bundled_profile("Ryan")
        assert bundled.is_embedding() is False

    def test_get_embedding_path_returns_path(self, tmp_path):
        """Test get_embedding_path returns the embedding path."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file
        )

        assert profile.get_embedding_path() == embedding_file

    def test_get_embedding_path_none_for_other_types(self):
        """Test get_embedding_path returns None for non-EMBEDDING types."""
        bundled = VoiceProfile.create_bundled_profile("Ryan")
        assert bundled.get_embedding_path() is None


class TestVoiceProfileSerialization:
    """Tests for embedding voice serialization/deserialization."""

    def test_embedding_voice_to_dict(self, tmp_path):
        """Test embedding voice serializes correctly."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        profile = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file,
            description="Test description"
        )

        data = profile.to_dict()

        assert data['name'] == "TestVoice"
        assert data['voice_type'] == "embedding"
        assert data['emotion_capable'] is True
        assert data['checkpoint_path'] == str(embedding_file)
        assert data['description'] == "Test description"

    def test_embedding_voice_from_dict(self, tmp_path):
        """Test embedding voice deserializes correctly."""
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        data = {
            'file_path': 'embedding://TestVoice',
            'name': 'TestVoice',
            'voice_type': 'embedding',
            'emotion_capable': True,
            'checkpoint_path': str(embedding_file),
            'description': 'Test description',
            'is_valid': True,
        }

        profile = VoiceProfile.from_dict(data)

        assert profile.name == "TestVoice"
        assert profile.voice_type == VoiceType.EMBEDDING
        assert profile.emotion_capable is True
        assert profile.checkpoint_path == embedding_file
        assert profile.description == "Test description"
