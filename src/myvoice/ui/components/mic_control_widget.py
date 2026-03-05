"""
Microphone Control Widget

Provides compact mic controls for the main window:
- Mute/unmute toggle button
- Volume slider
- Visual feedback for mic state
"""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QSlider, QLabel
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QIcon


class MicControlWidget(QWidget):
    """
    Compact microphone control widget for main window.

    Provides quick access to mic mute/unmute and volume control
    when microphone mixing is enabled.

    Signals:
        mute_toggled: Emitted when mute state changes (bool: is_muted)
        volume_changed: Emitted when volume changes (float: 0.0-1.0)
    """

    mute_toggled = pyqtSignal(bool)  # is_muted
    volume_changed = pyqtSignal(float)  # volume 0.0-1.0

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        self._is_muted = False
        self._volume = 1.0
        self._device_name = "No microphone"

        self._setup_ui()
        self._update_mute_button_state()

    def _setup_ui(self):
        """Create the compact UI layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        # Mic mute/unmute toggle button
        self.mute_button = QPushButton()
        self.mute_button.setFixedSize(QSize(28, 28))
        self.mute_button.setObjectName("mic_mute_button")
        self.mute_button.setCheckable(True)
        self.mute_button.clicked.connect(self._on_mute_clicked)
        self.mute_button.setToolTip("Toggle microphone mute (Ctrl+M)")
        layout.addWidget(self.mute_button)

        # Volume slider (compact)
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(100)
        self.volume_slider.setFixedWidth(80)
        self.volume_slider.setObjectName("mic_volume_slider")
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        self.volume_slider.setToolTip("Microphone volume")
        layout.addWidget(self.volume_slider)

        # Volume percentage label
        self.volume_label = QLabel("100%")
        self.volume_label.setFixedWidth(35)
        self.volume_label.setObjectName("mic_volume_label")
        layout.addWidget(self.volume_label)

        # Add stretch to keep controls compact on left
        layout.addStretch()

    def _on_mute_clicked(self):
        """Handle mute button click."""
        self._is_muted = self.mute_button.isChecked()
        self._update_mute_button_state()
        self.mute_toggled.emit(self._is_muted)
        self.logger.debug(f"Mic {'muted' if self._is_muted else 'unmuted'}")

    def _on_volume_changed(self, value: int):
        """Handle volume slider change."""
        self._volume = value / 100.0
        self.volume_label.setText(f"{value}%")
        self.volume_changed.emit(self._volume)

    def _update_mute_button_state(self):
        """Update mute button appearance based on state."""
        if self._is_muted:
            # Muted state - show mic-off icon or text
            self.mute_button.setText("MIC")
            self.mute_button.setStyleSheet("""
                QPushButton {
                    background-color: #dc3545;
                    color: white;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 9px;
                }
                QPushButton:hover {
                    background-color: #c82333;
                }
            """)
            self.mute_button.setToolTip(f"Unmute microphone\n{self._device_name}")
        else:
            # Active state - show mic-on icon or text
            self.mute_button.setText("MIC")
            self.mute_button.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 9px;
                }
                QPushButton:hover {
                    background-color: #218838;
                }
            """)
            self.mute_button.setToolTip(f"Mute microphone\n{self._device_name}")

    def set_muted(self, muted: bool):
        """
        Set mute state programmatically.

        Args:
            muted: True to mute, False to unmute
        """
        self._is_muted = muted
        self.mute_button.setChecked(muted)
        self._update_mute_button_state()

    def set_volume(self, volume: float):
        """
        Set volume level programmatically.

        Args:
            volume: Volume level 0.0 to 1.0
        """
        self._volume = max(0.0, min(1.0, volume))
        self.volume_slider.setValue(int(self._volume * 100))

    def set_device_name(self, name: str):
        """
        Set the device name for tooltip display.

        Args:
            name: Microphone device name
        """
        self._device_name = name
        self._update_mute_button_state()

    def is_muted(self) -> bool:
        """Get current mute state."""
        return self._is_muted

    def get_volume(self) -> float:
        """Get current volume level (0.0 to 1.0)."""
        return self._volume

    def set_enabled_state(self, enabled: bool):
        """
        Enable or disable all controls.

        Args:
            enabled: True to enable, False to disable
        """
        self.mute_button.setEnabled(enabled)
        self.volume_slider.setEnabled(enabled)

        if not enabled:
            self.mute_button.setStyleSheet("""
                QPushButton {
                    background-color: #6c757d;
                    color: white;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 9px;
                }
            """)
