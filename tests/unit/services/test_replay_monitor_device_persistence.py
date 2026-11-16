"""Lucy's report (2026-11): "Cannot hear own voice (playback) after using
repeat or the clear text button."

Root cause: ``MonitorAudioService.play_monitor_audio`` wrote its per-call
``device`` argument back into ``current_monitor_device``. Normal generation
streams through ``start_streaming_session``, which opens on
``current_monitor_device`` (``None`` = follow the OS default output). Replay
(and Clear Comms) dispatch through ``_play_generated_audio``'s batch path,
which resolves a device with its own heuristics -- the first enumerated
output for "Default", or the first non-virtual output on a monitor/virtual
collision -- and passes it explicitly. One Replay therefore re-pointed every
later streamed generation at that device.

Reproduced on the dev host against real PyAudio (OS default = Voicemeeter
Input, index 20): streaming opened on 20 before Replay and on 19
("Microsoft Sound Mapper - Output") after it, in both the monitor-only and
the Voicemeeter-collision settings shapes.

Second half of the fix: ``AudioCoordinator.update_device_settings`` never
refreshed the monitor device -- the persistence side effect above was the
only path by which a Settings change reached streaming. It now re-resolves
explicitly via ``apply_monitor_device_from_settings``.

The services here are REAL, constructed without ``initialize()`` (no
PortAudio): ``_pyaudio`` is a fake and the playback thread is patched out.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from myvoice.models.app_settings import AppSettings
from myvoice.models.audio_device import AudioDevice, DeviceType
from myvoice.services.audio_coordinator import AudioCoordinator
from myvoice.services.monitor_audio_service import MonitorAudioService


OS_DEFAULT_INDEX = 20
FIRST_ENUMERATED_INDEX = 19  # what Replay's "Default" fallback picks


def _device(index: int, name: str) -> AudioDevice:
    return AudioDevice(
        device_id=f"pyaudio_{index}", name=name, device_type=DeviceType.OUTPUT
    )


class _FakePyAudio:
    """Records the output index every stream is opened on."""

    def __init__(self):
        self.opened_indices = []

    def get_default_output_device_info(self):
        return {"index": OS_DEFAULT_INDEX, "name": "Voicemeeter Input"}

    def open(self, **kwargs):
        self.opened_indices.append(kwargs["output_device_index"])
        return MagicMock()


@pytest.fixture
def monitor() -> MonitorAudioService:
    svc = MonitorAudioService(AppSettings())
    svc._pyaudio = _FakePyAudio()
    svc._is_initialized = True
    return svc


async def _streaming_index(svc: MonitorAudioService) -> int:
    await svc.start_streaming_session(sample_rate=24000, channels=1, sample_width=2)
    await svc.stop_streaming_session()
    return svc._pyaudio.opened_indices[-1]


class TestReplayDoesNotRepointStreaming:
    def test_default_monitor_streaming_survives_replay(self, monitor):
        """The exact shipped sequence: monitor on "Default", generate
        (stream), Replay with the batch path's resolved device, generate
        again -- the second stream must still follow the OS default."""

        async def run():
            before = await _streaming_index(monitor)
            with patch.object(monitor, "_start_monitor_playback_thread"):
                task = await monitor.play_monitor_audio(
                    b"RIFF", _device(FIRST_ENUMERATED_INDEX, "Microsoft Sound Mapper - Output")
                )
            after = await _streaming_index(monitor)
            return before, task, after

        before, task, after = asyncio.run(run())
        assert before == OS_DEFAULT_INDEX
        # The replay itself still plays on the device it was handed ...
        assert task.device.device_id == f"pyaudio_{FIRST_ENUMERATED_INDEX}"
        # ... but it no longer leaks into the configured monitor.
        assert monitor.current_monitor_device is None
        assert after == OS_DEFAULT_INDEX

    def test_configured_monitor_streaming_survives_replay(self, monitor):
        """Explicitly configured monitor + a collision-rerouted Replay device."""
        headset = _device(21, "Headset Earphone (A50 Voice)")
        monitor.current_monitor_device = headset

        async def run():
            with patch.object(monitor, "_start_monitor_playback_thread"):
                await monitor.play_monitor_audio(
                    b"RIFF", _device(FIRST_ENUMERATED_INDEX, "Microsoft Sound Mapper - Output")
                )
            return await _streaming_index(monitor)

        assert asyncio.run(run()) == 21
        assert monitor.current_monitor_device is headset


class TestSettingsChangeReachesStreaming:
    """Removing the side effect must not regress Settings changes: they now
    reach the monitor through ``update_device_settings`` explicitly."""

    @pytest.fixture
    def coordinator(self, monitor):
        coord = AudioCoordinator()
        coord.monitor_service = monitor
        coord.virtual_service = None
        return coord

    def test_selecting_a_monitor_device_applies_to_streaming(self, coordinator, monitor):
        headset = _device(21, "Headset Earphone (A50 Voice)")
        settings = AppSettings()
        settings.monitor_device_id = headset.device_id

        async def run():
            with patch.object(monitor, "enumerate_monitor_devices", return_value=[headset]):
                await coordinator.update_device_settings(settings)
            return await _streaming_index(monitor)

        assert asyncio.run(run()) == 21
        assert monitor.current_monitor_device == headset

    def test_switching_back_to_default_follows_os_default(self, coordinator, monitor):
        monitor.current_monitor_device = _device(21, "Headset Earphone (A50 Voice)")
        settings = AppSettings()
        settings.monitor_device_id = None

        async def run():
            await coordinator.update_device_settings(settings)
            return await _streaming_index(monitor)

        assert asyncio.run(run()) == OS_DEFAULT_INDEX
        assert monitor.current_monitor_device is None

    def test_missing_configured_device_keeps_previous(self, coordinator, monitor):
        headset = _device(21, "Headset Earphone (A50 Voice)")
        monitor.current_monitor_device = headset
        settings = AppSettings()
        settings.monitor_device_id = "pyaudio_99"

        async def run():
            with patch.object(monitor, "enumerate_monitor_devices", return_value=[headset]):
                await coordinator.update_device_settings(settings)

        asyncio.run(run())
        assert monitor.current_monitor_device is headset
