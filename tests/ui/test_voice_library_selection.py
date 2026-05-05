"""
Tests for Voice Library & Selection

Story 4.1: Voice Library & Selection
Covers: FR11
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
import os

# Skip if PyQt6 not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from myvoice.models.voice_profile import VoiceProfile, VoiceType, TranscriptionStatus, BUNDLED_SPEAKERS


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestVoiceType:
    """Tests for VoiceType enum (Story 4.1: FR11)."""

    def test_voice_type_count(self):
        """Test that there are exactly 5 voice types."""
        types = list(VoiceType)
        assert len(types) == 5

    def test_voice_type_values(self):
        """Test voice type enum values."""
        assert VoiceType.BUNDLED.value == "bundled"
        assert VoiceType.DESIGNED.value == "designed"
        assert VoiceType.CLONED.value == "cloned"
        assert VoiceType.OPTIMIZED.value == "optimized"
        assert VoiceType.EMBEDDING.value == "embedding"

    def test_voice_type_icons(self):
        """Test voice type icons match spec: 📦 bundled, 🎭 designed, 🎤 cloned."""
        assert VoiceType.BUNDLED.icon == "📦"
        assert VoiceType.DESIGNED.icon == "🎭"
        assert VoiceType.CLONED.icon == "🎤"
        assert VoiceType.OPTIMIZED.icon == "🔧"
        assert VoiceType.EMBEDDING.icon == "🧬"

    def test_voice_type_display_names(self):
        """Test voice type display names."""
        assert VoiceType.BUNDLED.display_name == "Bundled"
        assert VoiceType.DESIGNED.display_name == "Designed"
        assert VoiceType.CLONED.display_name == "Cloned"
        assert VoiceType.OPTIMIZED.display_name == "Optimized"
        assert VoiceType.EMBEDDING.display_name == "Embedding"

    def test_voice_type_emotion_support(self):
        """Test emotion support by voice type (FR8a)."""
        # Bundled, Designed, Optimized, Embedding support emotion
        assert VoiceType.BUNDLED.supports_emotion is True
        assert VoiceType.DESIGNED.supports_emotion is True
        assert VoiceType.OPTIMIZED.supports_emotion is True
        assert VoiceType.EMBEDDING.supports_emotion is True
        # Cloned does not support emotion
        assert VoiceType.CLONED.supports_emotion is False

    def test_voice_type_sort_order(self):
        """Test sort order for grouping (bundled first, then embedding, designed, optimized, cloned)."""
        assert VoiceType.BUNDLED.sort_order == 0
        assert VoiceType.EMBEDDING.sort_order == 1
        assert VoiceType.DESIGNED.sort_order == 2
        assert VoiceType.OPTIMIZED.sort_order == 3
        assert VoiceType.CLONED.sort_order == 4

    def test_voice_types_sorted_by_order(self):
        """Test that sorting by sort_order gives correct order."""
        types = list(VoiceType)
        sorted_types = sorted(types, key=lambda t: t.sort_order)

        assert sorted_types[0] == VoiceType.BUNDLED
        assert sorted_types[1] == VoiceType.EMBEDDING
        assert sorted_types[2] == VoiceType.DESIGNED
        assert sorted_types[3] == VoiceType.OPTIMIZED
        assert sorted_types[4] == VoiceType.CLONED


class TestVoiceProfileWithType:
    """Tests for VoiceProfile with voice_type field."""

    @pytest.fixture
    def temp_wav_file(self):
        """Create a temporary WAV file for testing."""
        import wave
        import struct

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        # Create a valid WAV file (1 second of silence at 16kHz)
        with wave.open(wav_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            # Write 1 second of silence
            data = struct.pack('<' + 'h' * 16000, *([0] * 16000))
            wav_file.writeframes(data)

        yield Path(wav_path)

        # Cleanup
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    def test_default_voice_type_is_cloned(self, temp_wav_file):
        """Test that default voice type is CLONED."""
        profile = VoiceProfile(file_path=temp_wav_file, name="Test Voice")
        assert profile.voice_type == VoiceType.CLONED

    def test_bundled_voice_type(self, temp_wav_file):
        """Test creating a bundled voice profile."""
        profile = VoiceProfile(
            file_path=temp_wav_file,
            name="Bundled Voice",
            voice_type=VoiceType.BUNDLED
        )
        assert profile.voice_type == VoiceType.BUNDLED
        assert profile.voice_type.icon == "📦"

    def test_designed_voice_type(self, temp_wav_file):
        """Test creating a designed voice profile."""
        profile = VoiceProfile(
            file_path=temp_wav_file,
            name="Designed Voice",
            voice_type=VoiceType.DESIGNED
        )
        assert profile.voice_type == VoiceType.DESIGNED
        assert profile.voice_type.icon == "🎭"

    def test_cloned_voice_type(self, temp_wav_file):
        """Test creating a cloned voice profile."""
        profile = VoiceProfile(
            file_path=temp_wav_file,
            name="Cloned Voice",
            voice_type=VoiceType.CLONED
        )
        assert profile.voice_type == VoiceType.CLONED
        assert profile.voice_type.icon == "🎤"

    def test_emotion_capable_auto_sync_bundled(self, temp_wav_file):
        """Test emotion_capable is auto-set for bundled voices."""
        profile = VoiceProfile(
            file_path=temp_wav_file,
            name="Bundled Voice",
            voice_type=VoiceType.BUNDLED
        )
        assert profile.emotion_capable is True

    def test_emotion_capable_auto_sync_designed(self, temp_wav_file):
        """Test emotion_capable is auto-set for designed voices."""
        profile = VoiceProfile(
            file_path=temp_wav_file,
            name="Designed Voice",
            voice_type=VoiceType.DESIGNED
        )
        assert profile.emotion_capable is True

    def test_emotion_capable_auto_sync_cloned(self, temp_wav_file):
        """Test emotion_capable is auto-set to False for cloned voices."""
        profile = VoiceProfile(
            file_path=temp_wav_file,
            name="Cloned Voice",
            voice_type=VoiceType.CLONED
        )
        assert profile.emotion_capable is False

    def test_to_dict_includes_voice_type(self, temp_wav_file):
        """Test to_dict includes voice_type field."""
        profile = VoiceProfile(
            file_path=temp_wav_file,
            name="Test Voice",
            voice_type=VoiceType.DESIGNED
        )
        data = profile.to_dict()
        assert "voice_type" in data
        assert data["voice_type"] == "designed"

    def test_from_dict_parses_voice_type(self, temp_wav_file):
        """Test from_dict correctly parses voice_type."""
        data = {
            "file_path": str(temp_wav_file),
            "name": "Test Voice",
            "voice_type": "bundled",
            "is_valid": True
        }
        profile = VoiceProfile.from_dict(data)
        assert profile.voice_type == VoiceType.BUNDLED
        assert profile.emotion_capable is True  # Auto-synced

    def test_from_dict_defaults_to_cloned(self, temp_wav_file):
        """Test from_dict defaults to CLONED for legacy profiles."""
        data = {
            "file_path": str(temp_wav_file),
            "name": "Legacy Voice",
            "is_valid": True
            # No voice_type field (legacy profile)
        }
        profile = VoiceProfile.from_dict(data)
        assert profile.voice_type == VoiceType.CLONED
        assert profile.emotion_capable is False


class TestVoiceSelectorWithTypes:
    """Tests for VoiceSelector with voice type support."""

    @pytest.fixture
    def mock_voice_manager(self):
        """Create a mock VoiceProfileManager."""
        manager = Mock()
        manager.get_valid_profiles.return_value = {}
        manager.get_active_profile.return_value = None
        return manager

    @pytest.fixture
    def voice_selector(self, qapp, mock_voice_manager):
        """Create a VoiceSelector instance."""
        from myvoice.ui.components.voice_selector import VoiceSelector
        selector = VoiceSelector(voice_manager=mock_voice_manager)
        yield selector
        selector.deleteLater()

    def test_format_profile_includes_icon(self, voice_selector, qapp):
        """Test _format_profile_display includes voice type icon."""
        from myvoice.models.voice_profile import VoiceProfile, VoiceType

        # Create a mock profile with cloned type (has duration)
        profile = Mock(spec=VoiceProfile)
        profile.name = "Test Voice"
        profile.duration = 5.0
        profile.voice_type = VoiceType.CLONED

        display_text = voice_selector._format_profile_display(profile)

        assert "🎤" in display_text
        assert "Test Voice" in display_text
        assert "5.0s" in display_text

    def test_format_profile_designed_icon(self, voice_selector):
        """Test designed voice shows correct icon."""
        from myvoice.models.voice_profile import VoiceProfile, VoiceType

        profile = Mock(spec=VoiceProfile)
        profile.name = "Designed Voice"
        profile.duration = 3.5
        profile.voice_type = VoiceType.DESIGNED

        display_text = voice_selector._format_profile_display(profile)

        assert "🎭" in display_text

    def test_format_profile_cloned_icon(self, voice_selector):
        """Test cloned voice shows correct icon."""
        from myvoice.models.voice_profile import VoiceProfile, VoiceType

        profile = Mock(spec=VoiceProfile)
        profile.name = "Cloned Voice"
        profile.duration = 4.0
        profile.voice_type = VoiceType.CLONED

        display_text = voice_selector._format_profile_display(profile)

        assert "🎤" in display_text

    def test_populate_sorts_bundled_first(self, voice_selector, mock_voice_manager):
        """Test populate_voices sorts bundled voices first."""
        from myvoice.models.voice_profile import VoiceProfile, VoiceType

        # Create mock profiles of different types
        bundled = Mock(spec=VoiceProfile)
        bundled.name = "Bundled Voice"
        bundled.duration = 5.0
        bundled.voice_type = VoiceType.BUNDLED

        cloned = Mock(spec=VoiceProfile)
        cloned.name = "Cloned Voice"
        cloned.duration = 4.0
        cloned.voice_type = VoiceType.CLONED

        designed = Mock(spec=VoiceProfile)
        designed.name = "Designed Voice"
        designed.duration = 3.0
        designed.voice_type = VoiceType.DESIGNED

        # Return in wrong order (cloned, designed, bundled)
        mock_voice_manager.get_valid_profiles.return_value = {
            "Cloned Voice": cloned,
            "Designed Voice": designed,
            "Bundled Voice": bundled
        }

        voice_selector.populate_voices()

        # Check the order in the combo box
        combo = voice_selector._voice_combo
        assert combo.count() == 3

        # First item should be bundled (📦)
        assert "📦" in combo.itemText(0)
        assert "Bundled" in combo.itemText(0)

        # Second should be designed (🎭)
        assert "🎭" in combo.itemText(1)
        assert "Designed" in combo.itemText(1)

        # Third should be cloned (🎤)
        assert "🎤" in combo.itemText(2)
        assert "Cloned" in combo.itemText(2)

    def test_populate_sorts_within_type_alphabetically(self, voice_selector, mock_voice_manager):
        """Test that voices are sorted alphabetically within each type group."""
        from myvoice.models.voice_profile import VoiceProfile, VoiceType

        # Create two bundled voices
        bundled_z = Mock(spec=VoiceProfile)
        bundled_z.name = "Zeta Voice"
        bundled_z.duration = 5.0
        bundled_z.voice_type = VoiceType.BUNDLED

        bundled_a = Mock(spec=VoiceProfile)
        bundled_a.name = "Alpha Voice"
        bundled_a.duration = 5.0
        bundled_a.voice_type = VoiceType.BUNDLED

        mock_voice_manager.get_valid_profiles.return_value = {
            "Zeta Voice": bundled_z,
            "Alpha Voice": bundled_a
        }

        voice_selector.populate_voices()

        combo = voice_selector._voice_combo
        assert combo.count() == 2

        # Alpha should be first (alphabetically)
        assert "Alpha" in combo.itemText(0)
        assert "Zeta" in combo.itemText(1)


class TestVoiceEmotionIntegration:
    """Tests for voice type → emotion_enabled integration."""

    def test_bundled_voice_enables_emotion(self):
        """Test selecting bundled voice enables emotion controls."""
        # VoiceType.BUNDLED.supports_emotion should be True
        assert VoiceType.BUNDLED.supports_emotion is True

    def test_designed_voice_enables_emotion(self):
        """Test selecting designed voice enables emotion controls."""
        assert VoiceType.DESIGNED.supports_emotion is True

    def test_cloned_voice_disables_emotion(self):
        """Test selecting cloned voice disables emotion controls."""
        assert VoiceType.CLONED.supports_emotion is False

    def test_voice_profile_emotion_matches_type(self):
        """Test VoiceProfile.emotion_capable matches voice_type.supports_emotion."""
        import tempfile
        import wave
        import struct

        # Create a temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            with wave.open(wav_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(struct.pack('<' + 'h' * 16000, *([0] * 16000)))

            for voice_type in VoiceType:
                profile = VoiceProfile(
                    file_path=Path(wav_path),
                    name=f"Test {voice_type.value}",
                    voice_type=voice_type
                )
                assert profile.emotion_capable == voice_type.supports_emotion, \
                    f"Mismatch for {voice_type}: emotion_capable={profile.emotion_capable}, " \
                    f"supports_emotion={voice_type.supports_emotion}"
        finally:
            os.unlink(wav_path)


class TestBundledSpeakers:
    """Tests for the 9 pre-trained CustomVoice timbres.

    Task: Integrate 9 Template Timbres into Voice Library
    Acceptance Criteria:
    - VoiceProfileService creates VoiceProfile entries for all 9 timbres
    - Each profile has voice_type=VoiceType.BUNDLED and emotion_capable=True
    - VoiceSelector shows bundled voices at top (📦 icon)
    """

    def test_bundled_speakers_count(self):
        """Test that there are exactly 9 bundled speakers."""
        assert len(BUNDLED_SPEAKERS) == 9

    def test_bundled_speakers_names(self):
        """Test all 9 bundled speaker names are present."""
        expected_speakers = [
            "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
            "Ryan", "Aiden", "Ono_Anna", "Sohee"
        ]
        for speaker in expected_speakers:
            assert speaker in BUNDLED_SPEAKERS, f"Missing bundled speaker: {speaker}"

    def test_bundled_speakers_have_metadata(self):
        """Test each bundled speaker has required metadata."""
        for speaker_name, info in BUNDLED_SPEAKERS.items():
            assert "description" in info, f"{speaker_name} missing description"
            assert "language" in info, f"{speaker_name} missing language"
            assert "gender" in info, f"{speaker_name} missing gender"

    def test_bundled_speakers_languages(self):
        """Test bundled speakers cover expected languages."""
        languages = {info["language"] for info in BUNDLED_SPEAKERS.values()}
        assert "Chinese" in languages
        assert "English" in languages
        assert "Japanese" in languages
        assert "Korean" in languages

    def test_bundled_speakers_genders(self):
        """Test bundled speakers include both male and female voices."""
        genders = {info["gender"] for info in BUNDLED_SPEAKERS.values()}
        assert "male" in genders
        assert "female" in genders


class TestCreateBundledProfile:
    """Tests for VoiceProfile.create_bundled_profile factory method."""

    def test_create_bundled_profile_success(self):
        """Test creating bundled profile for valid speaker."""
        profile = VoiceProfile.create_bundled_profile("Ryan")

        assert profile.name == "Ryan"
        assert profile.voice_type == VoiceType.BUNDLED
        assert profile.emotion_capable is True
        assert profile.is_valid is True
        assert profile.transcription_status == TranscriptionStatus.SKIPPED

    def test_create_bundled_profile_all_speakers(self):
        """Test creating bundled profiles for all 9 speakers."""
        for speaker_name in BUNDLED_SPEAKERS:
            profile = VoiceProfile.create_bundled_profile(speaker_name)

            assert profile.name == speaker_name
            assert profile.voice_type == VoiceType.BUNDLED
            assert profile.emotion_capable is True
            assert profile.is_valid is True

    def test_create_bundled_profile_invalid_speaker(self):
        """Test creating bundled profile for invalid speaker raises error."""
        with pytest.raises(ValueError) as exc_info:
            VoiceProfile.create_bundled_profile("InvalidSpeaker")

        assert "Unknown bundled speaker" in str(exc_info.value)
        assert "InvalidSpeaker" in str(exc_info.value)

    def test_create_bundled_profile_virtual_path(self):
        """Test bundled profile has virtual path marker."""
        profile = VoiceProfile.create_bundled_profile("Vivian")

        # Path may be "bundled://Vivian" or "bundled:\Vivian" depending on OS
        path_str = str(profile.file_path)
        assert "bundled" in path_str.lower()
        assert "Vivian" in path_str

    def test_bundled_profile_no_duration(self):
        """Test bundled profile has no duration (no audio file)."""
        profile = VoiceProfile.create_bundled_profile("Ryan")
        assert profile.duration is None

    def test_bundled_profile_no_transcription(self):
        """Test bundled profile has no transcription."""
        profile = VoiceProfile.create_bundled_profile("Ryan")
        assert profile.transcription is None


class TestIsBundled:
    """Tests for VoiceProfile.is_bundled() method."""

    def test_is_bundled_true_for_bundled_voice(self):
        """Test is_bundled returns True for bundled voices."""
        profile = VoiceProfile.create_bundled_profile("Ryan")
        assert profile.is_bundled() is True

    def test_is_bundled_false_for_cloned_voice(self):
        """Test is_bundled returns False for cloned voices."""
        import tempfile
        import wave
        import struct

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            with wave.open(wav_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(struct.pack('<' + 'h' * 16000, *([0] * 16000)))

            profile = VoiceProfile(file_path=Path(wav_path), name="Cloned")
            assert profile.is_bundled() is False
        finally:
            os.unlink(wav_path)


class TestVoiceSelectorBundledDisplay:
    """Tests for VoiceSelector display of bundled voices."""

    @pytest.fixture
    def mock_voice_manager(self):
        """Create a mock VoiceProfileManager."""
        manager = Mock()
        manager.get_valid_profiles.return_value = {}
        manager.get_active_profile.return_value = None
        return manager

    @pytest.fixture
    def voice_selector(self, qapp, mock_voice_manager):
        """Create a VoiceSelector instance."""
        from myvoice.ui.components.voice_selector import VoiceSelector
        selector = VoiceSelector(voice_manager=mock_voice_manager)
        yield selector
        selector.deleteLater()

    def test_bundled_voice_shows_bundled_label(self, voice_selector):
        """Test bundled voice shows '(bundled)' instead of duration."""
        profile = VoiceProfile.create_bundled_profile("Ryan")
        display_text = voice_selector._format_profile_display(profile)

        assert "📦" in display_text
        assert "Ryan" in display_text
        assert "(bundled)" in display_text
        # Should NOT show duration for bundled voices
        assert "s)" not in display_text  # No "5.0s)" pattern

    def test_bundled_voices_sorted_first(self, voice_selector, mock_voice_manager):
        """Test bundled voices appear before cloned voices."""
        import tempfile
        import wave
        import struct

        # Create bundled profile
        bundled = VoiceProfile.create_bundled_profile("Ryan")

        # Create cloned profile (needs temp WAV)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            with wave.open(wav_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(struct.pack('<' + 'h' * 16000, *([0] * 16000)))

            cloned = VoiceProfile(file_path=Path(wav_path), name="MyClone")

            # Return cloned first (wrong order)
            mock_voice_manager.get_valid_profiles.return_value = {
                "MyClone": cloned,
                "Ryan": bundled
            }

            voice_selector.populate_voices()

            combo = voice_selector._voice_combo
            assert combo.count() == 2

            # Bundled should be first (📦)
            assert "📦" in combo.itemText(0)
            assert "Ryan" in combo.itemText(0)

            # Cloned should be second (🎤)
            assert "🎤" in combo.itemText(1)
            assert "MyClone" in combo.itemText(1)

        finally:
            os.unlink(wav_path)

    def test_set_selected_voice_works_with_bundled(self, voice_selector, mock_voice_manager):
        """Test set_selected_voice works with bundled voice icon prefix."""
        bundled = VoiceProfile.create_bundled_profile("Vivian")

        mock_voice_manager.get_valid_profiles.return_value = {
            "Vivian": bundled
        }

        voice_selector.populate_voices()

        # Should be able to select by name
        voice_selector.set_selected_voice("Vivian")
        assert voice_selector.get_selected_profile_name() == "Vivian"
