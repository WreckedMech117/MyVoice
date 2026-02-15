"""
Virtual Microphone Setup Dialog

Story 7.4: Provides setup guidance when virtual microphone is not detected (FR44).
Includes instructions for VB-Audio Cable and Voicemeeter configuration.
"""

import logging
import webbrowser
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QScrollArea, QWidget, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class VirtualMicSetupDialog(QDialog):
    """
    Dialog providing setup guidance for virtual microphone configuration.

    Story 7.4 (FR44): Displayed when user clicks on virtual mic warning indicator.
    Provides step-by-step instructions for VB-Audio Cable and Voicemeeter setup.
    """

    # Download URLs
    VB_CABLE_URL = "https://vb-audio.com/Cable/"
    VOICEMEETER_URL = "https://vb-audio.com/Voicemeeter/"

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the virtual microphone setup dialog.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.setWindowTitle("Virtual Microphone Setup")
        self.setMinimumSize(500, 450)
        self.setModal(True)

        self._create_ui()

        self.logger.debug("VirtualMicSetupDialog created")

    def _create_ui(self):
        """Create the dialog UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header
        header_label = QLabel("Virtual Microphone Not Detected")
        header_font = QFont()
        header_font.setPointSize(12)
        header_font.setBold(True)
        header_label.setFont(header_font)
        layout.addWidget(header_label)

        # Description
        desc_label = QLabel(
            "A virtual microphone allows MyVoice to send generated speech directly to "
            "voice chat applications like Discord, Zoom, and Skype. Without it, your "
            "friends won't hear your generated speech.\n\n"
            "Choose one of the options below to set up a virtual microphone:"
        )
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        # Scrollable content area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(16)

        # VB-Audio Cable section
        vb_cable_group = self._create_vb_cable_section()
        scroll_layout.addWidget(vb_cable_group)

        # Voicemeeter section
        voicemeeter_group = self._create_voicemeeter_section()
        scroll_layout.addWidget(voicemeeter_group)

        # MyVoice configuration section
        myvoice_config_group = self._create_myvoice_config_section()
        scroll_layout.addWidget(myvoice_config_group)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area, 1)

        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

    def _create_vb_cable_section(self) -> QGroupBox:
        """Create the VB-Audio Cable instructions section."""
        group = QGroupBox("Option 1: VB-Audio Cable (Recommended - Free)")
        layout = QVBoxLayout(group)

        # Description
        desc = QLabel(
            "VB-Audio Cable is a free virtual audio cable that creates a pair of "
            "virtual audio devices. It's simple to install and works with most applications."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Steps
        steps_text = """
<b>Installation Steps:</b>
<ol>
<li>Download VB-Cable from the link below</li>
<li>Run the installer as Administrator</li>
<li><b>Restart your computer</b> (required for the driver to load)</li>
<li>Open Windows Sound settings and verify "CABLE Input" and "CABLE Output" appear</li>
</ol>

<b>Voice Chat Configuration:</b>
<ul>
<li>In Discord/Zoom/Skype, set your microphone to "CABLE Output (VB-Audio Virtual Cable)"</li>
<li>In MyVoice Settings, set Virtual Microphone to "CABLE Input (VB-Audio Virtual Cable)"</li>
</ul>
"""
        steps_label = QLabel(steps_text)
        steps_label.setWordWrap(True)
        steps_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(steps_label)

        # Download button
        download_button = QPushButton("Download VB-Audio Cable")
        download_button.clicked.connect(lambda: self._open_url(self.VB_CABLE_URL))
        layout.addWidget(download_button)

        return group

    def _create_voicemeeter_section(self) -> QGroupBox:
        """Create the Voicemeeter instructions section."""
        group = QGroupBox("Option 2: Voicemeeter (Advanced - Free)")
        layout = QVBoxLayout(group)

        # Description
        desc = QLabel(
            "Voicemeeter is a more advanced virtual audio mixer that includes virtual "
            "audio cables plus mixing capabilities. Recommended if you need more control "
            "over audio routing or want to mix multiple audio sources."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Steps
        steps_text = """
<b>Installation Steps:</b>
<ol>
<li>Download Voicemeeter (or Voicemeeter Banana for more features) from the link below</li>
<li>Run the installer as Administrator</li>
<li><b>Restart your computer</b> (required for the driver to load)</li>
<li>Open Voicemeeter and configure your audio routing</li>
</ol>

<b>Voice Chat Configuration:</b>
<ul>
<li>In Discord/Zoom/Skype, set your microphone to "VoiceMeeter Output"</li>
<li>In MyVoice Settings, set Virtual Microphone to "VoiceMeeter Input"</li>
</ul>
"""
        steps_label = QLabel(steps_text)
        steps_label.setWordWrap(True)
        steps_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(steps_label)

        # Download button
        download_button = QPushButton("Download Voicemeeter")
        download_button.clicked.connect(lambda: self._open_url(self.VOICEMEETER_URL))
        layout.addWidget(download_button)

        return group

    def _create_myvoice_config_section(self) -> QGroupBox:
        """Create the MyVoice configuration section."""
        group = QGroupBox("MyVoice Configuration")
        layout = QVBoxLayout(group)

        config_text = """
<b>After installing a virtual audio driver:</b>
<ol>
<li>Open MyVoice Settings (click the gear icon or press Ctrl+S)</li>
<li>Go to the <b>Audio</b> tab</li>
<li>In the "Virtual Microphone" dropdown, select your virtual device:
    <ul>
    <li>For VB-Cable: Select "CABLE Input (VB-Audio Virtual Cable)"</li>
    <li>For Voicemeeter: Select "VoiceMeeter Input"</li>
    </ul>
</li>
<li>Click "Test" to verify audio is routed correctly</li>
<li>In your voice chat app, select the corresponding output as your microphone</li>
</ol>

<b>Troubleshooting:</b>
<ul>
<li>If devices don't appear, restart your computer after driver installation</li>
<li>Make sure to select "Input" in MyVoice and "Output" in your voice chat app</li>
<li>Check that the virtual audio driver is not muted in Windows Sound settings</li>
</ul>
"""
        config_label = QLabel(config_text)
        config_label.setWordWrap(True)
        config_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(config_label)

        return group

    def _open_url(self, url: str):
        """
        Open a URL in the default web browser.

        Args:
            url: URL to open
        """
        try:
            webbrowser.open(url)
            self.logger.info(f"Opened URL: {url}")
        except Exception as e:
            self.logger.error(f"Failed to open URL {url}: {e}")
