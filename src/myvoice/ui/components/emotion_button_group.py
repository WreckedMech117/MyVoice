"""
Emotion Button Group Component

This module implements the EmotionButtonGroup with 5 preset emotion buttons
for Qwen3-TTS emotion control. The buttons display emoji icons and support
keyboard shortcuts (F1-F5).

Story 3.1: Emotion Button Group UI
Covers: FR6, FR10
"""

import logging
from typing import Optional
from enum import Enum

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QButtonGroup, QLabel
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont


class EmotionPreset(Enum):
    """
    Enumeration of emotion presets for Qwen3-TTS.

    Each preset maps to an instruct parameter for TTS generation.
    """
    NEUTRAL = ("neutral", "😐", "Neutral", "F1",
               "Speak in a calm, balanced, and matter-of-fact tone with steady pacing and even intonation, "
               "conveying composed professionalism without emotional inflection")
    HAPPY = ("happy", "😄", "Happy", "F2",
             "Speak with bright, joyful enthusiasm and warmth, voice lifted with genuine delight, "
             "occasional light laughter in the breath, upbeat tempo with melodic rises expressing pure happiness")
    SAD = ("sad", "😢", "Sad", "F3",
           "Speak with deep sorrow and heaviness, voice trembling with grief, shaky breaths between phrases, "
           "slow wavering delivery as if holding back tears, pitch dropping with melancholy and despair")
    ANGRY = ("angry", "😠", "Angry", "F4",
             "Speak with fierce intensity and sharp edges, voice tight with barely contained rage, "
             "forceful emphasis on words, clipped aggressive pacing, seething undertone building to explosive bursts")
    FLIRTATIOUS = ("flirtatious", "😏", "Flirtatious", "F5",
                   "Speak with playful, teasing allure and a coy smile in the voice, breathy whispered hints, "
                   "drawn-out vowels with suggestive pauses, warm intimate tone dripping with charming mischief")

    def __init__(self, id: str, emoji: str, display_name: str, shortcut: str, instruct: str):
        self.id = id
        self.emoji = emoji
        self.display_name = display_name
        self.shortcut = shortcut
        self.instruct = instruct

    @classmethod
    def from_id(cls, id: str) -> Optional['EmotionPreset']:
        """Get preset by ID string."""
        for preset in cls:
            if preset.id == id:
                return preset
        return None

    @classmethod
    def from_index(cls, index: int) -> Optional['EmotionPreset']:
        """Get preset by index (0-4)."""
        presets = list(cls)
        if 0 <= index < len(presets):
            return presets[index]
        return None


class EmotionButton(QPushButton):
    """
    Individual emotion button with emoji icon.

    Sized at 28x28px with emoji display and tooltip showing
    emotion name and keyboard shortcut.
    """

    def __init__(self, preset: EmotionPreset, parent: Optional[QWidget] = None):
        """
        Initialize emotion button.

        Args:
            preset: The emotion preset this button represents
            parent: Parent widget
        """
        super().__init__(parent)
        self.preset = preset

        # Set button properties
        self.setText(preset.emoji)
        self.setToolTip(f"{preset.display_name} ({preset.shortcut})")
        self.setCheckable(True)
        self.setObjectName(f"emotion_button_{preset.id}")

        # Fixed size: 28x28px per UX spec
        self.setFixedSize(QSize(28, 28))

        # Set font for emoji display
        font = QFont()
        font.setPointSize(12)
        font.setFamily("Segoe UI Emoji")
        self.setFont(font)

        # Accessibility
        self.setAccessibleName(f"{preset.display_name} emotion")
        self.setAccessibleDescription(f"Select {preset.display_name.lower()} emotion for speech synthesis. Keyboard shortcut: {preset.shortcut}")


class EmotionButtonGroup(QWidget):
    """
    Widget containing 5 emotion preset buttons with radio selection behavior.

    Features:
    - 5 emoji buttons (28x28px, 4px spacing): 😐 😄 😢 😠 😏
    - Tooltips show emotion name + shortcut (F1-F5)
    - Neutral selected by default
    - Accent border on selected button
    - "Custom..." link opens Settings
    - Enable/disable support for cloned voices
    - Custom emotion support (Story 3.4)

    Signals:
        emotion_changed: Emitted when emotion selection changes
        custom_emotion_requested: Emitted when "Custom..." is clicked
        custom_emotion_cleared: Emitted when custom emotion is cleared (preset selected)
    """

    # Signals
    emotion_changed = pyqtSignal(EmotionPreset)  # Emitted with selected preset
    custom_emotion_requested = pyqtSignal()  # Emitted when Custom... clicked
    custom_emotion_cleared = pyqtSignal()  # Emitted when custom emotion is replaced by preset

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the emotion button group.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        # State
        self._current_preset: EmotionPreset = EmotionPreset.NEUTRAL
        self._enabled = True
        self._buttons: dict[EmotionPreset, EmotionButton] = {}
        self._custom_emotion_text: Optional[str] = None  # Story 3.4: Custom emotion state
        self._is_custom_active: bool = False  # Story 3.4: Whether custom emotion is active

        # Create UI
        self._create_ui()

        # Select neutral by default (FR9)
        self._select_preset(EmotionPreset.NEUTRAL, emit_signal=False)

        self.logger.debug("EmotionButtonGroup initialized")

    def _create_ui(self):
        """Create the emotion button group UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)  # 4px spacing per UX spec

        # Create button group for radio behavior
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)

        # Create emotion buttons
        for i, preset in enumerate(EmotionPreset):
            button = EmotionButton(preset, self)
            self._buttons[preset] = button
            self.button_group.addButton(button, i)
            layout.addWidget(button)

            # Connect button click
            button.clicked.connect(lambda checked, p=preset: self._on_button_clicked(p))

        # Add "Custom..." link button
        self.custom_button = QPushButton("Custom...")
        self.custom_button.setObjectName("emotion_custom_button")
        self.custom_button.setToolTip("Open Settings to configure custom emotions")
        self.custom_button.setFixedHeight(28)
        self.custom_button.clicked.connect(self._on_custom_clicked)
        layout.addWidget(self.custom_button)

        # Add stretch to push buttons left
        layout.addStretch()

    def _on_button_clicked(self, preset: EmotionPreset):
        """
        Handle emotion button click.

        Args:
            preset: The clicked emotion preset
        """
        if not self._enabled:
            return

        self._select_preset(preset, emit_signal=True)

    def _select_preset(self, preset: EmotionPreset, emit_signal: bool = True):
        """
        Select an emotion preset.

        Args:
            preset: The preset to select
            emit_signal: Whether to emit the emotion_changed signal
        """
        # Update internal state
        old_preset = self._current_preset
        self._current_preset = preset

        # Clear custom emotion when preset is selected (Story 3.4)
        was_custom_active = self._is_custom_active
        if self._is_custom_active:
            self._is_custom_active = False
            self._update_custom_button_state()
            if emit_signal:
                self.custom_emotion_cleared.emit()

        # Update button states
        for p, button in self._buttons.items():
            button.setChecked(p == preset)

        self.logger.debug(f"Emotion changed: {old_preset.display_name} -> {preset.display_name}")

        # Emit signal if requested
        if emit_signal and (old_preset != preset or was_custom_active):
            self.emotion_changed.emit(preset)

    def _on_custom_clicked(self):
        """Handle Custom... button click."""
        self.logger.debug("Custom emotion requested")
        self.custom_emotion_requested.emit()

    # Public API

    def get_current_preset(self) -> EmotionPreset:
        """
        Get the currently selected emotion preset.

        Returns:
            The currently selected EmotionPreset
        """
        return self._current_preset

    def get_current_instruct(self) -> str:
        """
        Get the instruct parameter for the current emotion.

        Returns:
            The instruct string for TTS generation.
            Returns custom emotion text if custom emotion is active (Story 3.4).
        """
        if self._is_custom_active and self._custom_emotion_text:
            return self._custom_emotion_text
        return self._current_preset.instruct

    def set_preset(self, preset: EmotionPreset):
        """
        Set the selected emotion preset programmatically.

        Args:
            preset: The preset to select
        """
        self._select_preset(preset, emit_signal=True)

    def set_preset_by_id(self, id: str) -> bool:
        """
        Set the selected emotion by ID string.

        Args:
            id: The emotion ID (e.g., "happy", "sad")

        Returns:
            True if preset was found and set, False otherwise
        """
        preset = EmotionPreset.from_id(id)
        if preset:
            self._select_preset(preset, emit_signal=True)
            return True
        return False

    def set_preset_by_index(self, index: int) -> bool:
        """
        Set the selected emotion by index (0-4).

        Args:
            index: The emotion index (0=Neutral, 1=Happy, etc.)

        Returns:
            True if index was valid and preset set, False otherwise
        """
        preset = EmotionPreset.from_index(index)
        if preset:
            self._select_preset(preset, emit_signal=True)
            return True
        return False

    def set_emotion_enabled(self, enabled: bool):
        """
        Enable or disable emotion selection.

        When disabled, buttons are grayed out and show a tooltip
        explaining that emotion control is not available for cloned voices.

        Args:
            enabled: Whether emotion selection should be enabled
        """
        self._enabled = enabled

        # Update all emotion buttons
        for preset, button in self._buttons.items():
            button.setEnabled(enabled)
            if enabled:
                button.setToolTip(f"{preset.display_name} ({preset.shortcut})")
            else:
                button.setToolTip("Emotion control not available for cloned voices")

        # Custom button follows same state
        self.custom_button.setEnabled(enabled)

        self.logger.debug(f"Emotion buttons {'enabled' if enabled else 'disabled'}")

    def is_emotion_enabled(self) -> bool:
        """
        Check if emotion selection is enabled.

        Returns:
            True if emotion buttons are enabled
        """
        return self._enabled

    def reset_to_default(self):
        """Reset to default emotion (Neutral)."""
        self._select_preset(EmotionPreset.NEUTRAL, emit_signal=True)

    def get_preset_button(self, preset: EmotionPreset) -> Optional[EmotionButton]:
        """
        Get the button for a specific preset.

        Args:
            preset: The emotion preset

        Returns:
            The EmotionButton for the preset, or None if not found
        """
        return self._buttons.get(preset)

    def get_emotion_buttons(self) -> list[EmotionButton]:
        """
        Get all emotion buttons in order for tab navigation.

        Story 7.5: Used for setting up keyboard tab order (NFR18).

        Returns:
            List of EmotionButton widgets in display order (Neutral → Flirtatious)
        """
        # Return buttons in enum order (which matches display order)
        return [self._buttons[preset] for preset in EmotionPreset if preset in self._buttons]

    # =========================================================================
    # Custom Emotion Methods (Story 3.4: FR7)
    # =========================================================================

    def set_custom_emotion(self, custom_text: str):
        """
        Set a custom emotion as active.

        Args:
            custom_text: The custom emotion instruction text
        """
        if not custom_text or not custom_text.strip():
            self.logger.warning("Cannot set empty custom emotion text")
            return

        self._custom_emotion_text = custom_text.strip()
        self._is_custom_active = True

        # Temporarily disable exclusive mode to allow unchecking all buttons
        self.button_group.setExclusive(False)
        for button in self._buttons.values():
            button.setChecked(False)
        self.button_group.setExclusive(True)

        # Update custom button to show active state
        self._update_custom_button_state()

        self.logger.info(f"Custom emotion set: {self._custom_emotion_text}")

    def clear_custom_emotion(self):
        """Clear the custom emotion and revert to current preset."""
        was_custom = self._is_custom_active
        self._is_custom_active = False
        self._custom_emotion_text = None

        # Re-select the current preset button
        if self._current_preset in self._buttons:
            self._buttons[self._current_preset].setChecked(True)

        self._update_custom_button_state()

        if was_custom:
            self.logger.debug("Custom emotion cleared")

    def is_custom_emotion_active(self) -> bool:
        """
        Check if a custom emotion is currently active.

        Returns:
            True if custom emotion is active
        """
        return self._is_custom_active

    def get_custom_emotion_text(self) -> Optional[str]:
        """
        Get the current custom emotion text.

        Returns:
            The custom emotion text, or None if not set
        """
        return self._custom_emotion_text if self._is_custom_active else None

    def _update_custom_button_state(self):
        """Update the Custom... button to reflect active state."""
        if self._is_custom_active:
            self.custom_button.setText("Custom ✓")
            self.custom_button.setToolTip(f"Custom emotion active: {self._custom_emotion_text}")
            self.custom_button.setProperty("active", True)
        else:
            self.custom_button.setText("Custom...")
            self.custom_button.setToolTip("Open Settings to configure custom emotions")
            self.custom_button.setProperty("active", False)

        # Force style refresh
        self.custom_button.style().unpolish(self.custom_button)
        self.custom_button.style().polish(self.custom_button)

    # =========================================================================
    # Emotion Variants: Per-emotion enable/disable for EMBEDDING voices
    # =========================================================================

    def update_available_emotions(self, available_emotions: list, voice_name: str = ""):
        """
        Update which emotion buttons are enabled based on available emotions.

        Emotion Variants: EMBEDDING voices have specific emotions available.
        This method enables only the buttons for available emotions and
        disables others with an explanatory tooltip.

        Args:
            available_emotions: List of emotion IDs that are available
                               (e.g., ["neutral", "happy", "sad"])
            voice_name: Voice name for tooltip (optional)
        """
        self._enabled = True  # Enable overall but manage per-button

        for preset, button in self._buttons.items():
            emotion_id = preset.id  # e.g., "neutral", "happy"
            is_available = emotion_id in available_emotions

            button.setEnabled(is_available)

            if is_available:
                # Normal tooltip
                button.setToolTip(f"{preset.display_name} ({preset.shortcut})")
            else:
                # Explanatory tooltip for unavailable emotion
                if voice_name:
                    button.setToolTip(
                        f"{preset.display_name} emotion was not created for '{voice_name}'"
                    )
                else:
                    button.setToolTip(
                        f"{preset.display_name} emotion is not available for this voice"
                    )

        # Custom button follows overall state
        self.custom_button.setEnabled(True)

        # If current preset is no longer available, switch to first available
        if self._current_preset.id not in available_emotions:
            for emotion_id in available_emotions:
                preset = EmotionPreset.from_id(emotion_id)
                if preset:
                    self._select_preset(preset, emit_signal=True)
                    break

        self.logger.debug(f"Updated available emotions: {available_emotions}")
