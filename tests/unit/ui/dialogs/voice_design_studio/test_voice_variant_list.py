"""
Tests for VoiceVariantList Widget

Story 1.7: Generate 5 Variations with Progressive Reveal
Story 1.8: Audition and Select Variant
Story 1.9: Regenerate All Variations
"""

import pytest
from pathlib import Path

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from myvoice.ui.dialogs.voice_design_studio.voice_variant_list import VoiceVariantList
from myvoice.ui.dialogs.voice_design_studio.voice_variant_row import VariantState


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def variant_list(qapp):
    """Create a VoiceVariantList instance."""
    vlist = VoiceVariantList()
    yield vlist
    vlist.deleteLater()


class TestVoiceVariantListInit:
    """Tests for VoiceVariantList initialization."""

    def test_list_creation(self, variant_list):
        """Test list creates successfully."""
        assert variant_list is not None
        assert isinstance(variant_list, VoiceVariantList)

    def test_list_has_5_rows(self, variant_list):
        """Test list has exactly 5 variant rows."""
        for i in range(5):
            row = variant_list.get_variant_row(i)
            assert row is not None

    def test_list_has_status_label(self, variant_list):
        """Test list has status label."""
        assert variant_list.status_label is not None

    def test_list_has_regenerate_button(self, variant_list):
        """Test list has regenerate button."""
        assert variant_list.regenerate_button is not None

    def test_regenerate_button_initially_hidden(self, variant_list):
        """Test regenerate button is hidden initially."""
        assert not variant_list.regenerate_button.isVisible()

    def test_no_initial_selection(self, variant_list):
        """Test no variant is selected initially."""
        assert variant_list.get_selected_index() is None
        assert not variant_list.has_selection()


class TestVoiceVariantListStartGeneration:
    """Tests for starting generation."""

    def test_start_generation_sets_all_waiting(self, variant_list):
        """Test start_generation sets all rows to waiting."""
        variant_list.start_generation()

        for i in range(5):
            row = variant_list.get_variant_row(i)
            assert row.get_state() == VariantState.WAITING

    def test_start_generation_clears_selection(self, variant_list, tmp_path):
        """Test start_generation clears any existing selection."""
        # Setup: complete a variant and select it
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")
        variant_list.set_variant_complete(0, audio, embedding)
        variant_list.get_variant_row(0).set_selected(True)

        variant_list.start_generation()

        assert not variant_list.has_selection()

    def test_is_generating_returns_true(self, variant_list):
        """Test is_generating returns True after start."""
        variant_list.start_generation()

        assert variant_list.is_generating()


class TestVoiceVariantListProgressiveReveal:
    """Tests for progressive reveal during generation."""

    def test_set_variant_generating(self, variant_list):
        """Test setting a variant to generating state."""
        variant_list.start_generation()
        variant_list.set_variant_generating(0)

        row = variant_list.get_variant_row(0)
        assert row.get_state() == VariantState.GENERATING

    def test_other_rows_remain_waiting(self, variant_list):
        """Test other rows remain in waiting when one is generating."""
        variant_list.start_generation()
        variant_list.set_variant_generating(0)

        for i in range(1, 5):
            row = variant_list.get_variant_row(i)
            assert row.get_state() == VariantState.WAITING

    def test_set_variant_complete(self, variant_list, tmp_path):
        """Test setting a variant to complete state."""
        variant_list.start_generation()
        variant_list.set_variant_generating(0)

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")

        variant_list.set_variant_complete(0, audio, embedding, duration=5.0)

        row = variant_list.get_variant_row(0)
        assert row.get_state() == VariantState.COMPLETE

    def test_complete_count_updates(self, variant_list, tmp_path):
        """Test complete count updates as variants complete."""
        variant_list.start_generation()

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")

        variant_list.set_variant_complete(0, audio, embedding)
        assert variant_list.get_complete_count() == 1

        variant_list.set_variant_complete(1, audio, embedding)
        assert variant_list.get_complete_count() == 2


class TestVoiceVariantListAllComplete:
    """Tests for all variants complete."""

    def test_all_complete_signal_emitted(self, variant_list, tmp_path):
        """Test all_complete signal emits when all 5 complete."""
        variant_list.start_generation()

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")

        signal_received = []
        variant_list.all_complete.connect(lambda: signal_received.append(True))

        for i in range(5):
            variant_list.set_variant_complete(i, audio, embedding)

        assert len(signal_received) == 1

    def test_is_generating_false_when_all_complete(self, variant_list, tmp_path):
        """Test is_generating is False when all complete."""
        variant_list.start_generation()

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")

        for i in range(5):
            variant_list.set_variant_complete(i, audio, embedding)

        assert not variant_list.is_generating()

    def test_regenerate_button_shown_when_complete(self, variant_list, tmp_path):
        """Test regenerate button shown when variants complete."""
        variant_list.start_generation()

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")

        for i in range(5):
            variant_list.set_variant_complete(i, audio, embedding)

        # Use isHidden() instead of isVisible() because parent isn't shown
        assert not variant_list.regenerate_button.isHidden()

    def test_status_shows_5_ready(self, variant_list, tmp_path):
        """Test status shows '5 variations ready' when all complete."""
        variant_list.start_generation()

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")

        for i in range(5):
            variant_list.set_variant_complete(i, audio, embedding)

        assert "5" in variant_list.status_label.text()
        assert "ready" in variant_list.status_label.text().lower()


class TestVoiceVariantListSelection:
    """Tests for variant selection."""

    def test_variant_selected_signal(self, variant_list, tmp_path):
        """Test variant_selected signal emits on selection."""
        variant_list.start_generation()

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")

        variant_list.set_variant_complete(0, audio, embedding)

        signal_received = []
        variant_list.variant_selected.connect(lambda idx: signal_received.append(idx))

        variant_list.get_variant_row(0).set_selected(True)

        assert len(signal_received) == 1
        assert signal_received[0] == 0

    def test_get_selected_index(self, variant_list, tmp_path):
        """Test get_selected_index returns correct index."""
        variant_list.start_generation()

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")

        variant_list.set_variant_complete(2, audio, embedding)
        variant_list.get_variant_row(2).set_selected(True)

        assert variant_list.get_selected_index() == 2

    def test_get_selected_variant_returns_paths(self, variant_list, tmp_path):
        """Test get_selected_variant returns paths."""
        variant_list.start_generation()

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")

        variant_list.set_variant_complete(0, audio, embedding)
        variant_list.get_variant_row(0).set_selected(True)

        result = variant_list.get_selected_variant()

        assert result is not None
        assert result[0] == audio
        assert result[1] == embedding

    def test_only_one_selection_at_a_time(self, variant_list, tmp_path):
        """Test only one variant can be selected at a time."""
        variant_list.start_generation()

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")

        for i in range(5):
            variant_list.set_variant_complete(i, audio, embedding)

        # Select variant 0
        variant_list.get_variant_row(0).set_selected(True)
        # Select variant 2
        variant_list.get_variant_row(2).set_selected(True)

        # Only variant 2 should be selected
        assert variant_list.get_selected_index() == 2
        assert not variant_list.get_variant_row(0).is_selected()


class TestVoiceVariantListError:
    """Tests for error handling."""

    def test_set_variant_error(self, variant_list):
        """Test setting a variant to error state."""
        variant_list.start_generation()
        variant_list.set_variant_error(0, "Test error")

        row = variant_list.get_variant_row(0)
        assert row.get_state() == VariantState.ERROR

    def test_partial_success(self, variant_list, tmp_path):
        """Test partial success when some variants fail."""
        variant_list.start_generation()

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")

        # 3 succeed, 2 fail
        for i in range(3):
            variant_list.set_variant_complete(i, audio, embedding)
        for i in range(3, 5):
            variant_list.set_variant_error(i, "Failed")

        assert variant_list.get_complete_count() == 3
        assert variant_list.has_complete_variants()


class TestVoiceVariantListClear:
    """Tests for clear functionality."""

    def test_clear_resets_all_rows(self, variant_list, tmp_path):
        """Test clear resets all rows to initial state."""
        variant_list.start_generation()

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")

        for i in range(5):
            variant_list.set_variant_complete(i, audio, embedding)

        variant_list.clear()

        for i in range(5):
            row = variant_list.get_variant_row(i)
            assert row.get_state() == VariantState.WAITING

    def test_clear_clears_selection(self, variant_list, tmp_path):
        """Test clear removes selection."""
        variant_list.start_generation()

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 40)
        embedding = tmp_path / "embedding.pt"
        embedding.write_bytes(b"PK")

        variant_list.set_variant_complete(0, audio, embedding)
        variant_list.get_variant_row(0).set_selected(True)

        variant_list.clear()

        assert not variant_list.has_selection()


class TestVoiceVariantListSignals:
    """Tests for signals."""

    def test_variant_selected_signal_exists(self, variant_list):
        """Test variant_selected signal exists."""
        assert hasattr(variant_list, 'variant_selected')

    def test_variant_deselected_signal_exists(self, variant_list):
        """Test variant_deselected signal exists."""
        assert hasattr(variant_list, 'variant_deselected')

    def test_all_complete_signal_exists(self, variant_list):
        """Test all_complete signal exists."""
        assert hasattr(variant_list, 'all_complete')

    def test_generation_progress_signal_exists(self, variant_list):
        """Test generation_progress signal exists."""
        assert hasattr(variant_list, 'generation_progress')


class TestVoiceVariantListAccessibility:
    """Tests for accessibility."""

    def test_list_has_accessible_name(self, variant_list):
        """Test list has accessible name."""
        assert variant_list.accessibleName() != ""

    def test_regenerate_button_has_accessible_name(self, variant_list):
        """Test regenerate button has accessible name."""
        assert variant_list.regenerate_button.accessibleName() != ""
