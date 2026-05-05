"""
Tests for DeviceResilienceManager

Story 2.5: Audio Device Resilience
- FR30: Handle audio device changes gracefully
- NFR8: Audio device disconnect handling
"""

import asyncio
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime

# Skip this module if PyQt6 is not available (required for myvoice imports)
pytest.importorskip("PyQt6", reason="PyQt6 required for myvoice imports")

from myvoice.services.device_resilience_manager import (
    DeviceResilienceManager,
    DeviceResilienceConfig,
    DeviceRole,
    DeviceState,
)
from myvoice.models.audio_device import AudioDevice, DeviceType
from myvoice.models.device_notification import (
    DeviceNotification,
    NotificationType,
    NotificationSeverity,
)
from myvoice.services.integrations.windows_audio_client import DeviceChangeEvent


@pytest.fixture
def mock_audio_client():
    """Create a mock WindowsAudioClient."""
    with patch('myvoice.services.device_resilience_manager.WindowsAudioClient') as mock:
        client = Mock()
        client.start_device_monitoring = Mock()
        client.stop_device_monitoring = Mock()
        client.close = Mock()
        client.add_device_change_callback = Mock()
        client.refresh_audio_devices = Mock(return_value=True)
        client.enumerate_audio_devices = Mock(return_value=[])
        client.find_device_by_metadata = Mock(return_value=None)
        client.get_default_output_device = Mock(return_value=None)
        mock.return_value = client
        yield client


@pytest.fixture
def config():
    """Create a test configuration."""
    return DeviceResilienceConfig(
        enable_monitoring=True,
        poll_interval_seconds=1.0,
        auto_recovery_enabled=True,
        show_disconnect_warning=True,
        show_reconnect_notification=True,
        show_fallback_notification=True,
        fallback_to_default_on_disconnect=True,
    )


@pytest.fixture
def manager(config, mock_audio_client):
    """Create a DeviceResilienceManager instance with mocked dependencies."""
    return DeviceResilienceManager(config)


@pytest.fixture
def monitor_device():
    """Create a test monitor device."""
    return AudioDevice(
        device_id="pyaudio_5",
        name="Speakers (High Definition Audio)",
        device_type=DeviceType.OUTPUT,
        is_default=True,
        is_available=True,
    )


@pytest.fixture
def virtual_device():
    """Create a test virtual microphone device."""
    return AudioDevice(
        device_id="pyaudio_10",
        name="CABLE Input (VB-Audio Virtual Cable)",
        device_type=DeviceType.VIRTUAL,
        is_default=False,
        is_available=True,
    )


@pytest.fixture
def fallback_device():
    """Create a test fallback device."""
    return AudioDevice(
        device_id="pyaudio_0",
        name="Default Output Device",
        device_type=DeviceType.OUTPUT,
        is_default=True,
        is_available=True,
    )


class TestDeviceResilienceManagerInit:
    """Tests for DeviceResilienceManager initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        manager = DeviceResilienceManager()
        assert manager.config is not None
        assert manager.config.enable_monitoring is True
        assert manager._is_initialized is False

    def test_init_with_custom_config(self, config):
        """Test initialization with custom configuration."""
        manager = DeviceResilienceManager(config)
        assert manager.config == config
        assert manager.config.poll_interval_seconds == 1.0

    def test_device_states_initialized(self):
        """Test that device states are initialized for both roles."""
        manager = DeviceResilienceManager()
        assert DeviceRole.MONITOR in manager._device_states
        assert DeviceRole.VIRTUAL_MIC in manager._device_states
        assert manager._device_states[DeviceRole.MONITOR].device is None
        assert manager._device_states[DeviceRole.VIRTUAL_MIC].device is None


class TestDeviceResilienceManagerLifecycle:
    """Tests for service lifecycle (start/stop)."""

    @pytest.mark.asyncio
    async def test_initialize_success(self, manager, mock_audio_client):
        """Test successful initialization."""
        result = await manager.initialize()

        assert result is True
        assert manager._is_initialized is True
        mock_audio_client.add_device_change_callback.assert_called_once()
        mock_audio_client.start_device_monitoring.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_success(self, manager, mock_audio_client):
        """Test successful shutdown."""
        await manager.initialize()
        result = await manager.shutdown()

        assert result is True
        assert manager._is_initialized is False
        mock_audio_client.stop_device_monitoring.assert_called_once()
        mock_audio_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_uninitialized(self, manager):
        """Test health check when not initialized."""
        is_healthy, error = await manager.health_check()

        assert is_healthy is False
        assert error is not None
        assert "not initialized" in error.user_message.lower()

    @pytest.mark.asyncio
    async def test_health_check_initialized(self, manager, mock_audio_client):
        """Test health check when initialized."""
        await manager.initialize()
        is_healthy, error = await manager.health_check()

        assert is_healthy is True
        assert error is None


class TestDeviceRegistration:
    """Tests for device registration and tracking."""

    @pytest.mark.asyncio
    async def test_register_monitor_device(self, manager, mock_audio_client, monitor_device):
        """Test registering a monitor device."""
        await manager.initialize()
        manager.register_device(DeviceRole.MONITOR, monitor_device)

        state = manager.get_device_state(DeviceRole.MONITOR)
        assert state.device == monitor_device
        assert state.is_connected is True
        assert state.last_seen is not None

    @pytest.mark.asyncio
    async def test_register_device_with_fallback(
        self, manager, mock_audio_client, monitor_device, fallback_device
    ):
        """Test registering a device with a fallback device."""
        await manager.initialize()
        manager.register_device(DeviceRole.MONITOR, monitor_device, fallback_device)

        state = manager.get_device_state(DeviceRole.MONITOR)
        assert state.device == monitor_device
        assert state.fallback_device == fallback_device

    @pytest.mark.asyncio
    async def test_unregister_device(self, manager, mock_audio_client, monitor_device):
        """Test unregistering a device."""
        await manager.initialize()
        manager.register_device(DeviceRole.MONITOR, monitor_device)
        manager.unregister_device(DeviceRole.MONITOR)

        state = manager.get_device_state(DeviceRole.MONITOR)
        assert state.device is None
        assert state.is_connected is False


class TestDeviceChangeHandling:
    """Tests for device change event handling."""

    @pytest.mark.asyncio
    async def test_device_removed_notification(
        self, manager, mock_audio_client, monitor_device
    ):
        """Test that device removal triggers a notification (Story 2.5 AC1)."""
        await manager.initialize()
        manager.register_device(DeviceRole.MONITOR, monitor_device)

        # Track notifications
        notifications = []
        manager.add_notification_callback(lambda n: notifications.append(n))

        # Simulate device removal
        event = DeviceChangeEvent(
            event_type="removed",
            device_index=5,
            device_name="Speakers (High Definition Audio)",
            timestamp=datetime.now(),
        )
        manager._handle_device_change_event(event)

        # Verify notification was sent
        assert len(notifications) == 1
        assert notifications[0].notification_type == NotificationType.DEVICE_REMOVED
        assert notifications[0].severity == NotificationSeverity.WARNING

    @pytest.mark.asyncio
    async def test_device_reconnect_auto_recovery(
        self, manager, mock_audio_client, monitor_device
    ):
        """Test that device reconnect triggers auto-recovery (Story 2.5 AC2)."""
        await manager.initialize()
        manager.register_device(DeviceRole.MONITOR, monitor_device)

        # Simulate disconnect first
        state = manager.get_device_state(DeviceRole.MONITOR)
        state.is_connected = False
        manager._pending_recoveries[monitor_device.device_id] = datetime.now()

        # Track recovery callbacks
        recovery_calls = []
        manager.add_recovery_callback(lambda role, device: recovery_calls.append((role, device)))

        # Track notifications
        notifications = []
        manager.add_notification_callback(lambda n: notifications.append(n))

        # Simulate device reconnect
        event = DeviceChangeEvent(
            event_type="added",
            device_index=5,
            device_name="Speakers (High Definition Audio)",
            timestamp=datetime.now(),
        )
        manager._handle_device_change_event(event)

        # Verify auto-recovery triggered
        assert state.is_connected is True
        assert len(recovery_calls) == 1
        assert recovery_calls[0][0] == DeviceRole.MONITOR
        assert recovery_calls[0][1] == monitor_device

        # Verify restoration notification
        restored_notifications = [
            n for n in notifications
            if n.notification_type == NotificationType.DEVICE_RESTORED
        ]
        assert len(restored_notifications) == 1

    @pytest.mark.asyncio
    async def test_virtual_mic_disconnect_independent(
        self, manager, mock_audio_client, monitor_device, virtual_device
    ):
        """Test that virtual mic disconnect doesn't affect monitor (Story 2.5 AC3)."""
        await manager.initialize()
        manager.register_device(DeviceRole.MONITOR, monitor_device)
        manager.register_device(DeviceRole.VIRTUAL_MIC, virtual_device)

        # Simulate virtual mic removal
        event = DeviceChangeEvent(
            event_type="removed",
            device_index=10,
            device_name="CABLE Input (VB-Audio Virtual Cable)",
            timestamp=datetime.now(),
        )
        manager._handle_device_change_event(event)

        # Verify monitor device is still connected
        monitor_state = manager.get_device_state(DeviceRole.MONITOR)
        assert monitor_state.is_connected is True

        # Verify virtual mic is disconnected
        virtual_state = manager.get_device_state(DeviceRole.VIRTUAL_MIC)
        assert virtual_state.is_connected is False

    @pytest.mark.asyncio
    async def test_fallback_on_disconnect(
        self, manager, mock_audio_client, monitor_device, fallback_device
    ):
        """Test fallback device is used when primary disconnects (Story 2.5 AC5)."""
        await manager.initialize()
        manager.register_device(DeviceRole.MONITOR, monitor_device, fallback_device)

        # Track recovery callbacks
        recovery_calls = []
        manager.add_recovery_callback(lambda role, device: recovery_calls.append((role, device)))

        # Track notifications
        notifications = []
        manager.add_notification_callback(lambda n: notifications.append(n))

        # Simulate device removal
        event = DeviceChangeEvent(
            event_type="removed",
            device_index=5,
            device_name="Speakers (High Definition Audio)",
            timestamp=datetime.now(),
        )
        manager._handle_device_change_event(event)

        # Verify fallback was triggered
        assert len(recovery_calls) == 1
        assert recovery_calls[0][1] == fallback_device

        # Verify fallback notification
        fallback_notifications = [
            n for n in notifications
            if n.notification_type == NotificationType.DEVICE_FALLBACK
        ]
        assert len(fallback_notifications) == 1


class TestDeviceRefresh:
    """Tests for device refresh functionality."""

    @pytest.mark.asyncio
    async def test_refresh_devices_success(self, manager, mock_audio_client):
        """Test successful device refresh (Story 2.5 AC4)."""
        await manager.initialize()
        mock_audio_client.refresh_audio_devices.return_value = True

        result = manager.refresh_devices()

        assert result is True
        mock_audio_client.refresh_audio_devices.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_devices_not_initialized(self, manager):
        """Test device refresh when not initialized."""
        result = manager.refresh_devices()
        assert result is False

    @pytest.mark.asyncio
    async def test_enumerate_devices(self, manager, mock_audio_client, monitor_device):
        """Test device enumeration."""
        await manager.initialize()
        mock_audio_client.enumerate_audio_devices.return_value = [monitor_device]

        devices = manager.enumerate_devices()

        assert len(devices) == 1
        assert devices[0] == monitor_device


class TestCallbackManagement:
    """Tests for callback registration and management."""

    @pytest.mark.asyncio
    async def test_add_notification_callback(self, manager, mock_audio_client):
        """Test adding a notification callback."""
        await manager.initialize()

        callback = Mock()
        manager.add_notification_callback(callback)

        assert callback in manager._notification_callbacks

    @pytest.mark.asyncio
    async def test_remove_notification_callback(self, manager, mock_audio_client):
        """Test removing a notification callback."""
        await manager.initialize()

        callback = Mock()
        manager.add_notification_callback(callback)
        manager.remove_notification_callback(callback)

        assert callback not in manager._notification_callbacks

    @pytest.mark.asyncio
    async def test_add_recovery_callback(self, manager, mock_audio_client):
        """Test adding a recovery callback."""
        await manager.initialize()

        callback = Mock()
        manager.add_recovery_callback(callback)

        assert callback in manager._recovery_callbacks


class TestHealthStatus:
    """Tests for health status reporting."""

    @pytest.mark.asyncio
    async def test_get_health_status(self, manager, mock_audio_client, monitor_device):
        """Test getting health status."""
        await manager.initialize()
        manager.register_device(DeviceRole.MONITOR, monitor_device)

        health = await manager.get_health()

        assert health["service_name"] == "DeviceResilienceManager"
        assert health["initialized"] is True
        assert health["monitoring_active"] is True
        assert health["monitor_device"]["name"] == monitor_device.name
        assert health["monitor_device"]["connected"] is True


class TestEventMatching:
    """Tests for device event matching logic."""

    @pytest.mark.asyncio
    async def test_event_matches_by_name(self, manager, mock_audio_client, monitor_device):
        """Test event matching by device name."""
        await manager.initialize()

        event = DeviceChangeEvent(
            event_type="removed",
            device_index=99,  # Different index
            device_name="Speakers (High Definition Audio)",
            timestamp=datetime.now(),
        )

        result = manager._event_matches_device(event, monitor_device)
        assert result is True

    @pytest.mark.asyncio
    async def test_event_matches_by_index(self, manager, mock_audio_client, monitor_device):
        """Test event matching by device index."""
        await manager.initialize()

        event = DeviceChangeEvent(
            event_type="removed",
            device_index=5,
            device_name="Different Name",
            timestamp=datetime.now(),
        )

        result = manager._event_matches_device(event, monitor_device)
        assert result is True

    @pytest.mark.asyncio
    async def test_event_no_match(self, manager, mock_audio_client, monitor_device):
        """Test event doesn't match unrelated device."""
        await manager.initialize()

        event = DeviceChangeEvent(
            event_type="removed",
            device_index=99,
            device_name="Completely Different Device",
            timestamp=datetime.now(),
        )

        result = manager._event_matches_device(event, monitor_device)
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
