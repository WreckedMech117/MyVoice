"""Local TTS API (OpenAI-compatible) settings panel.

Exposes the three ``AppSettings`` fields the local HTTP API uses:

- ``enable_http_api`` — master toggle (checkbox).
- ``http_api_port``   — bind port (QSpinBox, default 7778).
- ``http_api_key``    — Bearer key (QLineEdit + Generate button).

Security-by-default (tech-spec F9 / G12): on the *first* enable with an empty
key, a high-entropy key is auto-populated so the API never comes up open. The
user may clear the key to run keyless, in which case a warning is shown.

Follows the ``StreamingSettingsPanel`` / ``ClearCommsSettingsPanel`` pattern:
``load_state(app_settings)`` (block signals while loading) and
``save_state(app_settings)`` (mutate the AppSettings). The parent dialog owns
the actual persistence on OK.

Module-boundary discipline: ``ui/*`` imports ``models.app_settings`` and
``services.api_server.security`` (the key generator only) — it does NOT import
the server/uvicorn machinery.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from myvoice.models.app_settings import AppSettings

_logger = logging.getLogger(__name__)

_PORT_MIN = 1024
_PORT_MAX = 65535


class APIAccessSettingsPanel(QWidget):
    """Settings widget for the local TTS API (enable / port / key)."""

    api_settings_changed = pyqtSignal()

    def __init__(
        self,
        app_settings: Optional[AppSettings] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._app_settings = app_settings

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        group = QGroupBox("Local API (OpenAI-compatible)")
        form = QFormLayout(group)

        # Enable toggle ----------------------------------------------------
        self._enable_checkbox = QCheckBox("Enable local API")
        self._enable_checkbox.setObjectName("api_enable_checkbox")
        self._enable_checkbox.setAccessibleName("Enable local TTS API")
        self._enable_checkbox.setAccessibleDescription(
            "Starts a localhost-only HTTP server exposing an OpenAI-compatible "
            "/v1/audio/speech API over MyVoice's voices."
        )
        form.addRow(self._enable_checkbox)

        # Port -------------------------------------------------------------
        self._port_spin = QSpinBox()
        self._port_spin.setObjectName("api_port_spin")
        self._port_spin.setRange(_PORT_MIN, _PORT_MAX)
        self._port_spin.setValue(7778)
        self._port_spin.setAccessibleName("Local API port")
        form.addRow(QLabel("Port:"), self._port_spin)

        # API key + Generate ----------------------------------------------
        key_row = QHBoxLayout()
        self._key_edit = QLineEdit()
        self._key_edit.setObjectName("api_key_edit")
        self._key_edit.setPlaceholderText("(required — Generate a key)")
        self._key_edit.setAccessibleName("Local API key")
        self._generate_button = QPushButton("Generate")
        self._generate_button.setObjectName("api_generate_key_button")
        self._generate_button.setAccessibleName("Generate API key")
        key_row.addWidget(self._key_edit)
        key_row.addWidget(self._generate_button)
        form.addRow(QLabel("API key:"), key_row)

        # Live status / URL ------------------------------------------------
        self._status_label = QLabel()
        self._status_label.setObjectName("api_status_label")
        self._status_label.setWordWrap(True)
        form.addRow(QLabel("Status:"), self._status_label)

        # Warning (keyless / localhost-only) -------------------------------
        self._warning_label = QLabel()
        self._warning_label.setObjectName("api_warning_label")
        self._warning_label.setWordWrap(True)
        self._warning_label.setStyleSheet("color: #c0392b;")
        form.addRow(self._warning_label)

        note = QLabel(
            "The API binds to 127.0.0.1 (this machine only). No LAN exposure "
            "in this version."
        )
        note.setWordWrap(True)
        form.addRow(note)

        layout.addWidget(group)
        layout.addStretch()

        # Signals
        self._enable_checkbox.toggled.connect(self._on_enable_toggled)
        self._port_spin.valueChanged.connect(self._on_any_change)
        self._key_edit.textChanged.connect(self._on_key_changed)
        self._generate_button.clicked.connect(self._on_generate_clicked)

        if app_settings is not None:
            self.load_state(app_settings)
        self._refresh_status()

    # ----- state round-trip (parent-dialog contract) --------------------- #

    def load_state(self, app_settings: AppSettings) -> None:
        """Populate widgets from ``app_settings`` without emitting changes."""
        self._app_settings = app_settings
        enabled = getattr(app_settings, "enable_http_api", False)
        port = getattr(app_settings, "http_api_port", 7778)
        key = getattr(app_settings, "http_api_key", "") or ""

        for widget in (self._enable_checkbox, self._port_spin, self._key_edit):
            widget.blockSignals(True)
        try:
            self._enable_checkbox.setChecked(bool(enabled))
            self._port_spin.setValue(int(port) if port else 7778)
            self._key_edit.setText(key)
        finally:
            for widget in (self._enable_checkbox, self._port_spin, self._key_edit):
                widget.blockSignals(False)
        self._refresh_status()

    def save_state(self, app_settings: AppSettings) -> None:
        """Write widget values into ``app_settings``."""
        app_settings.enable_http_api = self._enable_checkbox.isChecked()
        app_settings.http_api_port = int(self._port_spin.value())
        app_settings.http_api_key = self._key_edit.text().strip()

    # ----- slots --------------------------------------------------------- #

    def _on_enable_toggled(self, checked: bool) -> None:
        # Security default (F9/G12): first enable with no key auto-generates one
        # so the API never comes up open.
        if checked and not self._key_edit.text().strip():
            self._key_edit.setText(self._new_key())
            self._logger.info("Auto-generated API key on first enable")
        self._on_any_change()

    def _on_generate_clicked(self) -> None:
        self._key_edit.setText(self._new_key())
        self._on_any_change()

    def _on_key_changed(self, _text: str) -> None:
        self._on_any_change()

    def _on_any_change(self, *_args) -> None:
        self._refresh_status()
        self.api_settings_changed.emit()

    # ----- helpers ------------------------------------------------------- #

    @staticmethod
    def _new_key() -> str:
        from myvoice.services.api_server.security import generate_api_key

        return generate_api_key()

    def _refresh_status(self) -> None:
        enabled = self._enable_checkbox.isChecked()
        port = self._port_spin.value()
        if enabled:
            self._status_label.setText(f"Enabled — http://127.0.0.1:{port}/v1")
        else:
            self._status_label.setText("Disabled")

        if enabled and not self._key_edit.text().strip():
            self._warning_label.setText(
                "⚠ No API key set: any local process or web page could drive "
                "generation. Click Generate to add a key (recommended)."
            )
        else:
            self._warning_label.setText("")


__all__ = ["APIAccessSettingsPanel"]
