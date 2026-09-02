"""Standalone test client for the MyVoice Local TTS API.

A tiny PyQt6 GUI to exercise the OpenAI-compatible ``/v1/audio/speech`` endpoint
and play the audio back — useful for eyeballing the API on a machine without a
full OpenAI client installed.

Why a desktop app and not an HTML page: the server denies cross-origin (CORS)
and guards the Host header, and a browser ``<audio>`` element can't attach the
``Authorization: Bearer`` header. Making the request server-side (requests)
sidesteps all of that.

Run it with the bundled portable interpreter (no torch needed):

    python310\\python.exe tools\\api_test_client.py

or double-click ``09_API_Test_Client.bat``.

This is a dev/QA convenience tool — it is NOT bundled into the production exe.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import uuid
import wave

import requests
from PyQt6.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

DEFAULT_URL = "http://127.0.0.1:7778"
SAMPLE_RATE = 24000
EXT = {"mp3": "mp3", "wav": "wav", "pcm": "pcm"}


def pcm_to_wav(pcm_bytes: bytes) -> bytes:
    """Wrap raw 24 kHz mono int16 PCM in a WAV container (for playback)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm_bytes)
    return buf.getvalue()


class RequestWorker(QThread):
    """Runs one /v1/audio/speech request off the UI thread."""

    succeeded = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, base_url, api_key, body, stream):
        super().__init__()
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._body = body
        self._stream = stream

    def run(self):
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        url = f"{self._base_url}/v1/audio/speech"
        try:
            t0 = time.perf_counter()
            resp = requests.post(
                url, json=self._body, headers=headers, stream=self._stream, timeout=120
            )
            if resp.status_code != 200:
                detail = resp.text
                try:
                    detail = resp.json().get("detail", detail)
                except Exception:
                    pass
                self.failed.emit(f"HTTP {resp.status_code}: {detail}")
                return

            content_type = resp.headers.get("content-type", "?")
            first_byte_ms = None
            if self._stream:
                chunks = []
                for chunk in resp.iter_content(chunk_size=4096):
                    if chunk:
                        if first_byte_ms is None:
                            first_byte_ms = (time.perf_counter() - t0) * 1000.0
                        chunks.append(chunk)
                payload = b"".join(chunks)
            else:
                payload = resp.content
            total_ms = (time.perf_counter() - t0) * 1000.0

            self.succeeded.emit(
                {
                    "payload": payload,
                    "content_type": content_type,
                    "fmt": self._body["response_format"],
                    "first_byte_ms": first_byte_ms,
                    "total_ms": total_ms,
                }
            )
        except requests.exceptions.RequestException as exc:
            self.failed.emit(f"Connection error: {exc}")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Unexpected error: {exc}")


class ApiTestClient(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MyVoice — Local TTS API Test Client")
        self.resize(560, 620)

        self._worker = None
        self._last_payload = b""
        self._last_fmt = "mp3"
        self._tmp_path = None

        # --- media player ---
        self._player = QMediaPlayer()
        self._audio_out = QAudioOutput()
        self._player.setAudioOutput(self._audio_out)
        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.playbackStateChanged.connect(self._on_play_state)

        root = QVBoxLayout(self)

        # --- connection group ---
        conn = QGroupBox("Connection")
        conn_form = QFormLayout(conn)
        self._url_edit = QLineEdit(DEFAULT_URL)
        conn_form.addRow("Base URL:", self._url_edit)
        self._key_edit = QLineEdit()
        self._key_edit.setPlaceholderText("Bearer key (leave blank if keyless)")
        conn_form.addRow("API key:", self._key_edit)

        conn_buttons = QHBoxLayout()
        self._health_btn = QPushButton("Check /health")
        self._health_btn.clicked.connect(self._check_health)
        self._voices_btn = QPushButton("Refresh voices")
        self._voices_btn.clicked.connect(self._refresh_voices)
        conn_buttons.addWidget(self._health_btn)
        conn_buttons.addWidget(self._voices_btn)
        conn_form.addRow(conn_buttons)
        self._conn_status = QLabel("—")
        self._conn_status.setWordWrap(True)
        conn_form.addRow("Status:", self._conn_status)
        root.addWidget(conn)

        # --- request group ---
        req = QGroupBox("Request")
        req_form = QFormLayout(req)
        self._voice_combo = QComboBox()
        self._voice_combo.setEditable(True)
        self._voice_combo.addItem("Ryan")
        req_form.addRow("Voice:", self._voice_combo)

        self._text_edit = QPlainTextEdit(
            "Hello! This is a test of the MyVoice local API."
        )
        self._text_edit.setFixedHeight(90)
        req_form.addRow("Text:", self._text_edit)

        opts = QHBoxLayout()
        self._fmt_combo = QComboBox()
        self._fmt_combo.addItems(["mp3", "wav", "pcm"])
        opts.addWidget(QLabel("Format:"))
        opts.addWidget(self._fmt_combo)
        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(0.25, 4.0)
        self._speed_spin.setSingleStep(0.25)
        self._speed_spin.setValue(1.0)
        opts.addWidget(QLabel("Speed:"))
        opts.addWidget(self._speed_spin)
        self._stream_check = QCheckBox("Stream")
        opts.addWidget(self._stream_check)
        opts.addStretch()
        req_form.addRow(opts)

        self._generate_btn = QPushButton("Generate")
        self._generate_btn.clicked.connect(self._generate)
        req_form.addRow(self._generate_btn)
        self._req_status = QLabel("—")
        self._req_status.setWordWrap(True)
        req_form.addRow("Result:", self._req_status)
        root.addWidget(req)

        # --- playback group ---
        play = QGroupBox("Playback")
        play_layout = QVBoxLayout(play)
        controls = QHBoxLayout()
        self._play_btn = QPushButton("▶ Play")
        self._play_btn.clicked.connect(self._toggle_play)
        self._play_btn.setEnabled(False)
        self._stop_btn = QPushButton("■ Stop")
        self._stop_btn.clicked.connect(self._player.stop)
        self._stop_btn.setEnabled(False)
        self._save_btn = QPushButton("Save as…")
        self._save_btn.clicked.connect(self._save_as)
        self._save_btn.setEnabled(False)
        controls.addWidget(self._play_btn)
        controls.addWidget(self._stop_btn)
        controls.addWidget(self._save_btn)
        play_layout.addLayout(controls)
        self._pos_slider = QSlider(Qt.Orientation.Horizontal)
        self._pos_slider.setEnabled(False)
        self._pos_slider.sliderMoved.connect(self._player.setPosition)
        play_layout.addWidget(self._pos_slider)
        self._time_label = QLabel("0:00 / 0:00")
        play_layout.addWidget(self._time_label)
        root.addWidget(play)

        root.addStretch()

    # ----- connection helpers ----------------------------------------- #

    def _base(self):
        return self._url_edit.text().strip().rstrip("/")

    def _headers(self):
        key = self._key_edit.text().strip()
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _check_health(self):
        try:
            r = requests.get(f"{self._base()}/health", timeout=5)
            self._conn_status.setText(f"/health → {r.status_code} {r.text}")
        except requests.exceptions.RequestException as exc:
            self._conn_status.setText(f"/health failed: {exc}")

    def _refresh_voices(self):
        try:
            r = requests.get(
                f"{self._base()}/v1/voices", headers=self._headers(), timeout=10
            )
            if r.status_code != 200:
                self._conn_status.setText(f"/v1/voices → {r.status_code}: {r.text}")
                return
            voices = [v["name"] for v in r.json().get("voices", [])]
            current = self._voice_combo.currentText()
            self._voice_combo.clear()
            self._voice_combo.addItems(voices or ["(none)"])
            if current in voices:
                self._voice_combo.setCurrentText(current)
            self._conn_status.setText(f"Loaded {len(voices)} voice(s).")
        except requests.exceptions.RequestException as exc:
            self._conn_status.setText(f"/v1/voices failed: {exc}")

    # ----- request ---------------------------------------------------- #

    def _generate(self):
        text = self._text_edit.toPlainText().strip()
        voice = self._voice_combo.currentText().strip()
        if not text or not voice:
            QMessageBox.warning(self, "Missing input", "Enter both text and a voice.")
            return
        body = {
            "model": "myvoice-1",
            "input": text,
            "voice": voice,
            "response_format": self._fmt_combo.currentText(),
            "speed": self._speed_spin.value(),
            "stream": self._stream_check.isChecked(),
        }
        self._generate_btn.setEnabled(False)
        self._req_status.setText("Generating…")
        self._worker = RequestWorker(
            self._base(), self._key_edit.text().strip(), body, body["stream"]
        )
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_failure)
        self._worker.finished.connect(lambda: self._generate_btn.setEnabled(True))
        self._worker.start()

    def _on_success(self, result):
        payload = result["payload"]
        self._last_payload = payload
        self._last_fmt = result["fmt"]
        msg = (
            f"200 OK · {result['content_type']} · {len(payload):,} bytes · "
            f"{result['total_ms']:.0f} ms total"
        )
        if result["first_byte_ms"] is not None:
            msg += f" · first byte {result['first_byte_ms']:.0f} ms"
        self._req_status.setText(msg)
        self._save_btn.setEnabled(bool(payload))
        self._load_into_player(payload, result["fmt"])

    def _on_failure(self, message):
        self._req_status.setText(f"❌ {message}")

    # ----- playback --------------------------------------------------- #

    def _load_into_player(self, payload: bytes, fmt: str):
        # QMediaPlayer can't play raw L16 — wrap pcm in a WAV container.
        if fmt == "pcm":
            data, ext = pcm_to_wav(payload), "wav"
        else:
            data, ext = payload, EXT[fmt]

        # Release any previous source so Windows doesn't lock the old temp file,
        # and use a unique name per generation.
        self._player.stop()
        self._player.setSource(QUrl())
        self._cleanup_tmp()
        path = os.path.join(
            tempfile.gettempdir(), f"myvoice_api_{uuid.uuid4().hex}.{ext}"
        )
        with open(path, "wb") as fh:
            fh.write(data)
        self._tmp_path = path
        self._player.setSource(QUrl.fromLocalFile(path))
        self._play_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._pos_slider.setEnabled(True)
        self._player.play()

    def _toggle_play(self):
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_play_state(self, state):
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._play_btn.setText("⏸ Pause" if playing else "▶ Play")

    def _on_position(self, pos):
        if not self._pos_slider.isSliderDown():
            self._pos_slider.setValue(pos)
        self._update_time(pos, self._player.duration())

    def _on_duration(self, dur):
        self._pos_slider.setRange(0, dur)
        self._update_time(self._player.position(), dur)

    def _update_time(self, pos, dur):
        self._time_label.setText(f"{self._fmt_ms(pos)} / {self._fmt_ms(dur)}")

    @staticmethod
    def _fmt_ms(ms):
        s = ms // 1000
        return f"{s // 60}:{s % 60:02d}"

    # ----- save / cleanup --------------------------------------------- #

    def _save_as(self):
        if not self._last_payload:
            return
        ext = EXT[self._last_fmt]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save audio", f"myvoice_response.{ext}", f"Audio (*.{ext})"
        )
        if path:
            with open(path, "wb") as fh:
                fh.write(self._last_payload)
            self._req_status.setText(f"Saved {len(self._last_payload):,} bytes → {path}")

    def _cleanup_tmp(self):
        if self._tmp_path and os.path.exists(self._tmp_path):
            try:
                os.remove(self._tmp_path)
            except OSError:
                pass
        self._tmp_path = None

    def closeEvent(self, event):
        self._player.stop()
        self._player.setSource(QUrl())
        self._cleanup_tmp()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = ApiTestClient()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
