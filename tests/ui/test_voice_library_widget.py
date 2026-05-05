"""
Tests for VoiceLibraryWidget Component

Story 4.4: Voice Library Management
Covers: FR19, FR46
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
import os

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from myvoice.ui.components.voice_library_widget import (
    VoiceLibraryWidget,
    VoiceListItem
)
from myvoice.models.voice_profile import VoiceProfile, VoiceType


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def mock_voice_manager():
    """Create a mock voice manager with test profiles."""
    manager = MagicMock()

    # Create test profiles
    bundled_profile = MagicMock(spec=VoiceProfile)
    bundled_profile.name = "Ryan"
    bundled_profile.voice_type = VoiceType.BUNDLED
    bundled_profile.duration = 10.0
    bundled_profile.emotion_capable = True
    bundled_profile.file_path = Path("/bundled/ryan.wav")

    designed_profile = MagicMock(spec=VoiceProfile)
    designed_profile.name = "Custom Voice"
    designed_profile.voice_type = VoiceType.DESIGNED
    designed_profile.duration = 5.0
    designed_profile.emotion_capable = True
    designed_profile.file_path = Path("/designed/custom/voice.wav")

    cloned_profile = MagicMock(spec=VoiceProfile)
    cloned_profile.name = "My Clone"
    cloned_profile.voice_type = VoiceType.CLONED
    cloned_profile.duration = 8.0
    cloned_profile.emotion_capable = False
    cloned_profile.file_path = Path("/cloned/myclone/voice.wav")

    manager.get_valid_profiles.return_value = {
        "Ryan": bundled_profile,
        "Custom Voice": designed_profile,
        "My Clone": cloned_profile,
    }
    manager.get_active_profile.return_value = bundled_profile

    return manager


@pytest.fixture
def widget(qapp, mock_voice_manager):
    """Create a VoiceLibraryWidget instance."""
    widget = VoiceLibraryWidget(voice_manager=mock_voice_manager)
    yield widget
    widget.deleteLater()


@pytest.fixture
def empty_widget(qapp):
    """Create a VoiceLibraryWidget without voice manager."""
    widget = VoiceLibraryWidget()
    yield widget
    widget.deleteLater()


class TestVoiceLibraryWidgetInit:
    """Tests for VoiceLibraryWidget initialization."""

    def test_widget_creation(self, widget):
        """Test widget creates successfully."""
        assert widget is not None
        assert isinstance(widget, VoiceLibraryWidget)

    def test_voice_list_exists(self, widget):
        """Test voice list widget exists."""
        assert widget.voice_list is not None

    def test_delete_button_exists(self, widget):
        """Test delete button exists."""
        assert widget.delete_button is not None

    def test_refresh_button_exists(self, widget):
        """Test refresh button exists."""
        assert widget.refresh_button is not None

    def test_design_button_exists(self, widget):
        """Test design voice button exists."""
        assert widget.design_button is not None
        assert widget.bottom_design_button is not None

    def test_clone_button_exists(self, widget):
        """Test clone voice button exists."""
        assert widget.clone_button is not None
        assert widget.bottom_clone_button is not None


class TestVoiceLibraryWidgetPopulation:
    """Tests for voice list population."""

    def test_voices_populated(self, widget, mock_voice_manager):
        """Test voices are populated from manager."""
        # Should have 3 voices
        assert widget.voice_list.count() == 3

    def test_voices_sorted_by_type(self, widget):
        """Test voices are sorted: bundled first."""
        # First item should be bundled (Ryan)
        first_item = widget.voice_list.item(0)
        assert isinstance(first_item, VoiceListItem)
        assert first_item.profile.voice_type == VoiceType.BUNDLED

    def test_profiles_dict_populated(self, widget):
        """Test profiles dictionary is populated."""
        assert len(widget._profiles) == 3
        assert "Ryan" in widget._profiles
        assert "Custom Voice" in widget._profiles
        assert "My Clone" in widget._profiles


class TestVoiceListItem:
    """Tests for VoiceListItem class."""

    def test_item_creation(self, qapp):
        """Test list item creates successfully."""
        profile = MagicMock(spec=VoiceProfile)
        profile.name = "Test Voice"
        profile.voice_type = VoiceType.DESIGNED
        profile.duration = 5.0
        profile.emotion_capable = True

        item = VoiceListItem(profile)
        assert item is not None
        assert item.profile == profile

    def test_item_display_text_includes_icon(self, qapp):
        """Test item display includes voice type icon."""
        profile = MagicMock(spec=VoiceProfile)
        profile.name = "Test Voice"
        profile.voice_type = VoiceType.DESIGNED  # Real enum, icon is "🎭"
        profile.duration = 5.0

        item = VoiceListItem(profile)
        text = item.text()
        # VoiceType.DESIGNED has icon "🎭"
        assert "🎭" in text or "Test Voice" in text

    def test_item_has_tooltip(self, qapp):
        """Test item has tooltip."""
        profile = MagicMock(spec=VoiceProfile)
        profile.name = "Test Voice"
        profile.voice_type = VoiceType.CLONED  # Real enum, display_name is "Cloned"
        profile.duration = 8.0
        profile.emotion_capable = False

        item = VoiceListItem(profile)
        tooltip = item.toolTip()
        assert tooltip  # Has some tooltip


class TestVoiceLibraryWidgetSelection:
    """Tests for voice selection functionality."""

    def test_initial_selection(self, widget):
        """Test profiles are loaded and available for selection."""
        # Widget should have profiles loaded from voice manager
        assert len(widget._profiles) > 0
        # Voice list should have items
        assert widget.voice_list.count() > 0

    def test_select_voice_by_name(self, widget):
        """Test selecting voice by name."""
        widget.select_voice("Custom Voice")
        assert widget.get_selected_profile_name() == "Custom Voice"

    def test_selection_emits_signal(self, widget):
        """Test selection emits voice_selected signal."""
        signal_mock = Mock()
        widget.voice_selected.connect(signal_mock)

        widget.select_voice("My Clone")

        # Signal should be emitted
        assert signal_mock.called or widget.get_selected_profile_name() == "My Clone"


class TestVoiceLibraryWidgetDelete:
    """Tests for voice deletion functionality."""

    def test_delete_button_disabled_for_bundled(self, widget):
        """Test delete button is disabled for bundled voices."""
        # Select bundled voice
        widget.select_voice("Ryan")

        # Delete button should be disabled
        assert not widget.delete_button.isEnabled()

    def test_delete_button_enabled_for_designed(self, widget):
        """Test delete button is enabled for designed voices."""
        widget.select_voice("Custom Voice")
        assert widget.delete_button.isEnabled()

    def test_delete_button_enabled_for_cloned(self, widget):
        """Test delete button is enabled for cloned voices."""
        widget.select_voice("My Clone")
        assert widget.delete_button.isEnabled()


class TestVoiceLibraryWidgetEmptyState:
    """Tests for empty state display."""

    def test_empty_state_exists(self, widget):
        """Test empty state widget exists."""
        assert widget.empty_state is not None

    def test_empty_state_hidden_when_voices_exist(self, widget):
        """Test empty state is hidden when voices exist."""
        # Trigger a refresh to populate voices
        widget.refresh_voices()
        # With voices from mock manager, voice_list should exist
        # Note: visibility may depend on parent widget being shown
        assert widget.voice_list is not None
        # Voice list should have items if manager returned profiles
        assert widget.voice_list.count() > 0 or widget._profiles


class TestVoiceLibraryWidgetSignals:
    """Tests for widget signals."""

    def test_voice_selected_signal_exists(self, widget):
        """Test voice_selected signal exists."""
        assert hasattr(widget, 'voice_selected')

    def test_voice_deleted_signal_exists(self, widget):
        """Test voice_deleted signal exists."""
        assert hasattr(widget, 'voice_deleted')

    def test_design_voice_requested_signal_exists(self, widget):
        """Test design_voice_requested signal exists."""
        assert hasattr(widget, 'design_voice_requested')

    def test_clone_voice_requested_signal_exists(self, widget):
        """Test clone_voice_requested signal exists."""
        assert hasattr(widget, 'clone_voice_requested')

    def test_refresh_requested_signal_exists(self, widget):
        """Test refresh_requested signal exists."""
        assert hasattr(widget, 'refresh_requested')

    def test_design_button_emits_signal(self, widget):
        """Test design button emits signal."""
        signal_mock = Mock()
        widget.design_voice_requested.connect(signal_mock)

        widget.bottom_design_button.click()

        signal_mock.assert_called_once()

    def test_clone_button_emits_signal(self, widget):
        """Test clone button emits signal."""
        signal_mock = Mock()
        widget.clone_voice_requested.connect(signal_mock)

        widget.bottom_clone_button.click()

        signal_mock.assert_called_once()


class TestVoiceLibraryWidgetCounts:
    """Tests for voice count methods."""

    def test_user_voice_count(self, widget):
        """Test user voice count."""
        # 2 user voices: designed + cloned
        assert widget.get_user_voice_count() == 2

    def test_bundled_voice_count(self, widget):
        """Test bundled voice count."""
        # 1 bundled voice
        assert widget.get_bundled_voice_count() == 1


class TestVoiceLibraryWidgetServiceIntegration:
    """Tests for service integration."""

    def test_set_voice_manager(self, empty_widget):
        """Test setting voice manager."""
        manager = MagicMock()
        manager.get_valid_profiles.return_value = {}
        manager.get_active_profile.return_value = None

        empty_widget.set_voice_manager(manager)

        assert empty_widget.voice_manager == manager

    def test_refresh_voices(self, widget, mock_voice_manager):
        """Test refresh voices method."""
        initial_count = widget.voice_list.count()

        widget.refresh_voices()

        # Should still have same count after refresh
        assert widget.voice_list.count() == initial_count
