"""
Tests for VoiceProfileManager Embedding Support

Story 1.6: Emotion Control for Saved Voice
- Tests VoiceProfileManager._scan_embeddings_directory
- Tests VoiceProfileManager.get_embedding_profiles
- Tests embedding voice loading from cache
"""

import pytest
import json
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

from myvoice.models.voice_profile import VoiceProfile, VoiceType
from myvoice.services.voice_profile_service import VoiceProfileManager


@pytest.fixture
def manager():
    """Create a VoiceProfileManager instance."""
    manager = VoiceProfileManager(auto_scan=False)
    yield manager


@pytest.fixture
def embeddings_dir(tmp_path):
    """Create a temporary embeddings directory structure."""
    voice_files = tmp_path / "voice_files"
    embeddings = voice_files / "embeddings"
    embeddings.mkdir(parents=True)
    return embeddings


class TestScanEmbeddingsDirectory:
    """Tests for VoiceProfileManager._scan_embeddings_directory."""

    def test_scan_empty_directory(self, manager, tmp_path):
        """Test scanning empty embeddings directory returns 0."""
        manager.voice_directory = tmp_path / "voice_files"
        (manager.voice_directory / "embeddings").mkdir(parents=True)

        count = manager._scan_embeddings_directory()

        assert count == 0

    def test_scan_nonexistent_directory(self, manager, tmp_path):
        """Test scanning nonexistent embeddings directory returns 0."""
        manager.voice_directory = tmp_path / "nonexistent"

        count = manager._scan_embeddings_directory()

        assert count == 0

    def test_scan_finds_embedding_voice(self, manager, embeddings_dir, tmp_path):
        """Test scanning finds voice with embedding.pt file."""
        manager.voice_directory = tmp_path / "voice_files"

        # Create voice directory with embedding.pt
        voice_dir = embeddings_dir / "TestVoice"
        voice_dir.mkdir()
        (voice_dir / "embedding.pt").write_bytes(b"PK_MOCK")

        count = manager._scan_embeddings_directory()

        assert count == 1
        assert "TestVoice" in manager._profiles
        assert manager._profiles["TestVoice"].voice_type == VoiceType.EMBEDDING

    def test_scan_skips_existing_profile(self, manager, embeddings_dir, tmp_path):
        """Test scanning skips voices already in profiles."""
        manager.voice_directory = tmp_path / "voice_files"

        # Create voice directory with embedding.pt
        voice_dir = embeddings_dir / "TestVoice"
        voice_dir.mkdir()
        (voice_dir / "embedding.pt").write_bytes(b"PK_MOCK")

        # Pre-add profile
        manager._profiles["TestVoice"] = MagicMock()

        count = manager._scan_embeddings_directory()

        assert count == 0  # Should skip because already exists

    def test_scan_skips_directory_without_embedding(self, manager, embeddings_dir, tmp_path):
        """Test scanning skips directories without embedding.pt."""
        manager.voice_directory = tmp_path / "voice_files"

        # Create voice directory without embedding.pt
        voice_dir = embeddings_dir / "InvalidVoice"
        voice_dir.mkdir()
        (voice_dir / "metadata.json").write_text("{}")

        count = manager._scan_embeddings_directory()

        assert count == 0
        assert "InvalidVoice" not in manager._profiles

    def test_scan_loads_metadata(self, manager, embeddings_dir, tmp_path):
        """Test scanning loads description from metadata.json."""
        manager.voice_directory = tmp_path / "voice_files"

        # Create voice directory with embedding and metadata
        voice_dir = embeddings_dir / "TestVoice"
        voice_dir.mkdir()
        (voice_dir / "embedding.pt").write_bytes(b"PK_MOCK")
        (voice_dir / "metadata.json").write_text(json.dumps({
            "name": "TestVoice",
            "description": "A warm, friendly voice"
        }))

        count = manager._scan_embeddings_directory()

        assert count == 1
        assert manager._profiles["TestVoice"].description == "A warm, friendly voice"

    def test_scan_uses_metadata_name(self, manager, embeddings_dir, tmp_path):
        """Test scanning uses name from metadata if different from folder."""
        manager.voice_directory = tmp_path / "voice_files"

        # Create voice directory with different folder name
        voice_dir = embeddings_dir / "folder_name"
        voice_dir.mkdir()
        (voice_dir / "embedding.pt").write_bytes(b"PK_MOCK")
        (voice_dir / "metadata.json").write_text(json.dumps({
            "name": "Real Voice Name",
            "description": "Description"
        }))

        count = manager._scan_embeddings_directory()

        assert count == 1
        assert "Real Voice Name" in manager._profiles

    def test_scan_multiple_voices(self, manager, embeddings_dir, tmp_path):
        """Test scanning finds multiple embedding voices."""
        manager.voice_directory = tmp_path / "voice_files"

        # Create multiple voice directories
        for name in ["Voice1", "Voice2", "Voice3"]:
            voice_dir = embeddings_dir / name
            voice_dir.mkdir()
            (voice_dir / "embedding.pt").write_bytes(b"PK_MOCK")

        count = manager._scan_embeddings_directory()

        assert count == 3
        assert "Voice1" in manager._profiles
        assert "Voice2" in manager._profiles
        assert "Voice3" in manager._profiles


class TestGetEmbeddingProfiles:
    """Tests for VoiceProfileManager.get_embedding_profiles."""

    def test_returns_only_embedding_profiles(self, manager, tmp_path):
        """Test get_embedding_profiles returns only EMBEDDING type."""
        # Add a bundled profile
        bundled = VoiceProfile.create_bundled_profile("Ryan")
        manager._profiles["Ryan"] = bundled

        # Add an embedding profile
        embedding_file = tmp_path / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")
        embedding = VoiceProfile.create_embedding_profile(
            name="TestVoice",
            embedding_path=embedding_file
        )
        manager._profiles["TestVoice"] = embedding

        result = manager.get_embedding_profiles()

        assert len(result) == 1
        assert "TestVoice" in result
        assert "Ryan" not in result

    def test_returns_empty_if_no_embeddings(self, manager):
        """Test get_embedding_profiles returns empty dict if no embeddings."""
        bundled = VoiceProfile.create_bundled_profile("Ryan")
        manager._profiles["Ryan"] = bundled

        result = manager.get_embedding_profiles()

        assert len(result) == 0


class TestEmbeddingCacheLoading:
    """Tests for loading embedding voices from cache."""

    @pytest.mark.asyncio
    async def test_cache_loads_embedding_voice(self, manager, tmp_path):
        """Test embedding voice is loaded from cache correctly."""
        manager.cache_file = tmp_path / "cache.json"
        manager.voice_directory = tmp_path / "voice_files"

        # Create embedding file
        embeddings_dir = manager.voice_directory / "embeddings" / "TestVoice"
        embeddings_dir.mkdir(parents=True)
        embedding_file = embeddings_dir / "embedding.pt"
        embedding_file.write_bytes(b"PK_MOCK")

        # Create cache with embedding voice
        cache_data = {
            "cache_version": "1.0",
            "profiles": [{
                "file_path": "embedding://TestVoice",
                "name": "TestVoice",
                "voice_type": "embedding",
                "checkpoint_path": str(embedding_file),
                "description": "Test description",
                "is_valid": True,
                "emotion_capable": True,
                "duration": None,
            }],
            "active_profile": None,
            "last_updated": None
        }
        manager.cache_file.write_text(json.dumps(cache_data))

        await manager._load_profile_cache()

        assert "TestVoice" in manager._profiles
        profile = manager._profiles["TestVoice"]
        assert profile.voice_type == VoiceType.EMBEDDING
        assert profile.description == "Test description"
        assert profile.is_valid is True

    @pytest.mark.asyncio
    async def test_cache_invalidates_missing_embedding(self, manager, tmp_path):
        """Test embedding voice with missing file is not loaded."""
        manager.cache_file = tmp_path / "cache.json"
        manager.voice_directory = tmp_path / "voice_files"

        # Create cache with embedding voice but no actual file
        cache_data = {
            "cache_version": "1.0",
            "profiles": [{
                "file_path": "embedding://TestVoice",
                "name": "TestVoice",
                "voice_type": "embedding",
                "checkpoint_path": str(tmp_path / "nonexistent.pt"),
                "description": "Test description",
                "is_valid": True,
            }],
            "active_profile": None,
            "last_updated": None
        }
        manager.cache_file.write_text(json.dumps(cache_data))

        await manager._load_profile_cache()

        assert "TestVoice" not in manager._profiles


class TestEmbeddingServiceStart:
    """Tests for embedding scanning during service start."""

    @pytest.mark.asyncio
    async def test_start_scans_embeddings(self, manager, tmp_path):
        """Test service start scans embeddings directory."""
        manager.voice_directory = tmp_path / "voice_files"
        manager.cache_file = tmp_path / "cache.json"
        manager.auto_scan = False  # Disable regular scan

        # Create embeddings directory with voice
        embeddings_dir = manager.voice_directory / "embeddings"
        voice_dir = embeddings_dir / "TestVoice"
        voice_dir.mkdir(parents=True)
        (voice_dir / "embedding.pt").write_bytes(b"PK_MOCK")

        await manager.start()

        try:
            assert "TestVoice" in manager._profiles
            assert manager._profiles["TestVoice"].voice_type == VoiceType.EMBEDDING
        finally:
            await manager.stop()
