"""
Tests for ModelLoadingIndicator Widget

Story 4.5: Model Lazy Loading for Voice Types
"""

import pytest
from unittest.mock import Mock, MagicMock

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from myvoice.ui.components.model_loading_indicator import (
    ModelLoadingIndicator,
    ModelLoadingOverlay
)


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def indicator(qapp):
    """Create a ModelLoadingIndicator instance."""
    indicator = ModelLoadingIndicator()
    yield indicator
    indicator.deleteLater()


@pytest.fixture
def overlay(qapp):
    """Create a ModelLoadingOverlay instance."""
    overlay = ModelLoadingOverlay()
    yield overlay
    overlay.deleteLater()


class TestModelLoadingIndicatorInit:
    """Tests for ModelLoadingIndicator initialization."""

    def test_indicator_creation(self, indicator):
        """Test indicator creates successfully."""
        assert indicator is not None
        assert isinstance(indicator, ModelLoadingIndicator)

    def test_initially_hidden(self, indicator):
        """Test indicator is hidden initially."""
        assert not indicator.isVisible()

    def test_has_message_label(self, indicator):
        """Test indicator has message label."""
        assert indicator.message_label is not None

    def test_has_progress_bar(self, indicator):
        """Test indicator has progress bar."""
        assert indicator.progress_bar is not None

    def test_has_icon_label(self, indicator):
        """Test indicator has icon label."""
        assert indicator.icon_label is not None


class TestModelLoadingIndicatorLoadingState:
    """Tests for loading state handling."""

    def test_on_loading_started_shows_indicator(self, indicator):
        """Test indicator shows on loading started."""
        indicator.on_loading_started("VoiceDesign", "Loading voice model...")

        assert indicator.isVisible()
        assert indicator.is_loading()

    def test_on_loading_started_sets_message(self, indicator):
        """Test loading message is set."""
        indicator.on_loading_started("VoiceDesign", "Loading voice model...")

        assert "Loading" in indicator.message_label.text()

    def test_on_loading_progress_updates(self, indicator):
        """Test progress updates work."""
        indicator.on_loading_started("VoiceDesign", "Loading...")
        indicator.on_loading_progress(50.0, "50% complete")

        # Progress bar should be updated
        assert indicator.progress_bar.value() == 50


class TestModelLoadingIndicatorComplete:
    """Tests for loading completion."""

    def test_on_loading_complete_success(self, indicator):
        """Test completion with success."""
        indicator.on_loading_started("VoiceDesign", "Loading...")
        indicator.on_loading_complete(True, "Model ready")

        assert not indicator.is_loading()
        assert "✓" in indicator.icon_label.text()

    def test_on_loading_complete_failure(self, indicator):
        """Test completion with failure."""
        indicator.on_loading_started("VoiceDesign", "Loading...")
        indicator.on_loading_complete(False, "Load failed")

        assert not indicator.is_loading()
        assert "✗" in indicator.icon_label.text()


class TestModelLoadingIndicatorManualControl:
    """Tests for manual control methods."""

    def test_show_loading(self, indicator):
        """Test manual show_loading."""
        indicator.show_loading("Custom message")

        assert indicator.isVisible()
        assert "Custom message" in indicator.message_label.text()

    def test_show_ready(self, indicator):
        """Test manual show_ready."""
        indicator.show_ready("Ready!")

        # Should auto-hide after delay, but initially visible
        assert "✓" in indicator.icon_label.text()

    def test_show_error(self, indicator):
        """Test manual show_error."""
        indicator.show_error("Something went wrong")

        assert "✗" in indicator.icon_label.text()
        assert "Failed" in indicator.message_label.text()

    def test_hide_indicator(self, indicator):
        """Test manual hide."""
        indicator.show_loading("Loading...")
        indicator.hide_indicator()

        assert not indicator.isVisible()


class TestModelLoadingOverlayInit:
    """Tests for ModelLoadingOverlay initialization."""

    def test_overlay_creation(self, overlay):
        """Test overlay creates successfully."""
        assert overlay is not None
        assert isinstance(overlay, ModelLoadingOverlay)

    def test_initially_hidden(self, overlay):
        """Test overlay is hidden initially."""
        assert not overlay.isVisible()

    def test_has_message_label(self, overlay):
        """Test overlay has message label."""
        assert overlay.message_label is not None

    def test_has_progress_bar(self, overlay):
        """Test overlay has progress bar."""
        assert overlay.progress_bar is not None


class TestModelLoadingOverlayControl:
    """Tests for overlay control."""

    def test_on_loading_started_shows(self, overlay):
        """Test overlay shows on loading started."""
        overlay.on_loading_started("VoiceDesign", "Loading model...")

        assert overlay.isVisible()

    def test_on_loading_complete_hides(self, overlay):
        """Test overlay hides on loading complete."""
        overlay.on_loading_started("VoiceDesign", "Loading...")
        overlay.on_loading_complete(True, "Ready")

        assert not overlay.isVisible()

    def test_show_loading(self, overlay):
        """Test manual show_loading."""
        overlay.show_loading("Please wait...")

        assert overlay.isVisible()
        assert "Please wait" in overlay.message_label.text()

    def test_hide_overlay(self, overlay):
        """Test manual hide."""
        overlay.show_loading("Loading...")
        overlay.hide_overlay()

        assert not overlay.isVisible()
