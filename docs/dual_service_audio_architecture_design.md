# Dual-Service Audio Architecture Design

## Overview

This document outlines the design for the proper dual-service audio architecture to replace the current single AudioManager implementation. The design follows Stories 1.4 and 1.5 specifications, implementing separate services for monitor speaker routing (Audio Service 1) and virtual microphone routing (Audio Service 2).

## Architecture Principles

### 1. Service Separation
- **Audio Service 1 (MonitorAudioService):** Dedicated to monitor speaker routing only
- **Audio Service 2 (VirtualMicrophoneService):** Dedicated to virtual microphone routing only
- **No Dual Responsibility:** Each service manages only its designated audio output type

### 2. Resource Isolation
- **Dedicated PyAudio Instances:** Each service maintains its own PyAudio instance
- **Independent Streaming:** No shared audio streams between services
- **Separate Thread Pools:** Each service manages its own worker threads

### 3. Clean Interface Design
- **TTSService Integration:** Simple, clear interfaces for audio routing requests
- **Service Discovery:** Services register capabilities and availability
- **Event-Driven Communication:** Services communicate via events, not direct method calls

## Service Designs

### MonitorAudioService (Audio Service 1)

#### Responsibilities
- Monitor speaker device enumeration and selection
- Audio playback to selected monitor devices
- Volume control and quality preservation
- Device change handling for monitor outputs

#### Core Interface
```python
class MonitorAudioService(BaseService):
    """Audio Service 1: Monitor speaker routing only."""

    async def play_monitor_audio(self, audio_data: bytes,
                               device: Optional[AudioDevice] = None,
                               volume: float = 1.0) -> MonitorPlaybackTask

    async def enumerate_monitor_devices(self) -> List[AudioDevice]

    async def set_monitor_device(self, device: AudioDevice) -> bool

    async def set_monitor_volume(self, volume: float) -> bool

    def on_device_changed(self, callback: Callable[[DeviceChangeEvent], None])
```

#### Resource Management
- **Single PyAudio Instance:** `self._pyaudio = pyaudio.PyAudio()`
- **Monitor-Specific Configuration:**
  - Sample Rate: 44.1kHz (optimal for speakers)
  - Channels: 2 (stereo)
  - Chunk Size: 1024 frames
  - Buffer Size: 4096 frames (low latency)
- **Thread Pool:** Dedicated monitor playback workers
- **Stream Lifecycle:** Open → Play → Close (no persistent streams)

#### Key Features
- **Quality Preservation:** No resampling or compression
- **Device Fallback:** Automatic fallback to default device
- **Volume Integration:** Respects system volume settings
- **Error Recovery:** Graceful handling of device disconnection

### VirtualMicrophoneService (Audio Service 2)

#### Responsibilities
- Virtual microphone device detection and selection
- Audio routing to virtual microphone inputs
- Communication app compatibility optimization
- Virtual driver integration (VB-Audio Cable, Voicemeeter)

#### Core Interface
```python
class VirtualMicrophoneService(BaseService):
    """Audio Service 2: Virtual microphone routing only."""

    async def play_virtual_microphone(self, audio_data: bytes,
                                    device: Optional[AudioDevice] = None,
                                    volume: float = 1.0) -> VirtualPlaybackTask

    async def enumerate_virtual_devices(self) -> List[AudioDevice]

    async def set_virtual_device(self, device: AudioDevice) -> bool

    async def optimize_for_app(self, app: CommunicationApp) -> bool

    def on_virtual_device_changed(self, callback: Callable[[DeviceChangeEvent], None])
```

#### Resource Management
- **Single PyAudio Instance:** `self._pyaudio = pyaudio.PyAudio()`
- **Virtual-Specific Configuration:**
  - Sample Rate: 48kHz (optimal for communication apps)
  - Channels: 1 (mono for microphone simulation)
  - Chunk Size: 512 frames (lower latency)
  - Buffer Size: 2048 frames (minimal buffering)
- **Thread Pool:** Dedicated virtual microphone workers
- **Stream Lifecycle:** Open → Route → Close (no persistent streams)

#### Key Features
- **Level Management:** Automatic gain control for communication apps
- **Driver Detection:** VB-Audio Cable and Voicemeeter recognition
- **App Optimization:** Discord, Zoom, Teams-specific audio settings
- **Noise Gate:** Optional noise gate for clean virtual microphone output

## Service Coordination

### AudioCoordinator

A lightweight coordinator manages both services without performing audio operations:

```python
class AudioCoordinator:
    """Coordinates between MonitorAudioService and VirtualMicrophoneService."""

    def __init__(self):
        self.monitor_service = MonitorAudioService()
        self.virtual_service = VirtualMicrophoneService()
        self.sync_manager = StreamSynchronizationManager()

    async def play_dual_stream(self, audio_data: bytes,
                             monitor_device: Optional[AudioDevice] = None,
                             virtual_device: Optional[AudioDevice] = None,
                             volume: float = 1.0) -> DualStreamResult
```

### Synchronization Mechanism

#### Event-Based Synchronization
Instead of trying to synchronize PyAudio streams directly, use event-based coordination:

```python
class StreamSynchronizationManager:
    """Manages synchronized playback without resource competition."""

    def __init__(self):
        self.sync_events = {}
        self.playback_coordinator = PlaybackCoordinator()

    async def coordinate_dual_playback(self, audio_data: bytes,
                                     monitor_service: MonitorAudioService,
                                     virtual_service: VirtualMicrophoneService) -> DualStreamResult:
        """
        Coordinate dual playback using event-based synchronization.

        1. Create shared synchronization event
        2. Start both services with sync event
        3. Services signal readiness
        4. Coordinator triggers simultaneous playback start
        5. Services begin streaming independently
        """
        sync_event = asyncio.Event()

        # Start both services concurrently
        tasks = await asyncio.gather(
            monitor_service.play_monitor_audio(audio_data, sync_event=sync_event),
            virtual_service.play_virtual_microphone(audio_data, sync_event=sync_event),
            return_exceptions=True
        )

        return DualStreamResult(monitor_task=tasks[0], virtual_task=tasks[1])
```

## TTSService Integration

### Clean Interface Design

The TTSService will interact with services through the coordinator:

```python
class TTSService:
    def __init__(self):
        self.audio_coordinator = AudioCoordinator()

    async def execute_dual_routing(self, audio_data: bytes,
                                 monitor_device: Optional[AudioDevice] = None,
                                 virtual_device: Optional[AudioDevice] = None) -> TTSResponse:
        """
        Execute dual audio routing through separate services.

        No more direct calls to AudioManager.play_synchronized_dual_stream()
        """
        dual_result = await self.audio_coordinator.play_dual_stream(
            audio_data=audio_data,
            monitor_device=monitor_device,
            virtual_device=virtual_device
        )

        return TTSResponse(
            audio_data=audio_data,
            monitor_task=dual_result.monitor_task,
            virtual_task=dual_result.virtual_task
        )

    async def execute_monitor_only(self, audio_data: bytes) -> TTSResponse:
        """Execute monitor-only routing."""
        monitor_task = await self.audio_coordinator.monitor_service.play_monitor_audio(audio_data)
        return TTSResponse(audio_data=audio_data, monitor_task=monitor_task)

    async def execute_virtual_only(self, audio_data: bytes) -> TTSResponse:
        """Execute virtual-only routing."""
        virtual_task = await self.audio_coordinator.virtual_service.play_virtual_microphone(audio_data)
        return TTSResponse(audio_data=audio_data, virtual_task=virtual_task)
```

### Elimination of Double Playback

**Current Problem:**
- TTSService → play_synchronized_dual_stream() → play_monitor_audio() (double call)

**New Solution:**
- TTSService → AudioCoordinator.play_dual_stream() → Independent service calls
- No internal method calls between services
- Each service handles only its designated output

## Resource Competition Elimination

### Dedicated PyAudio Instances

**Current Problem:** Single AudioManager with multiple competing streams

**New Solution:**
```python
# MonitorAudioService
class MonitorAudioService:
    def __init__(self):
        self._pyaudio = pyaudio.PyAudio()  # Dedicated instance
        self._stream_config = MonitorStreamConfig()

# VirtualMicrophoneService
class VirtualMicrophoneService:
    def __init__(self):
        self._pyaudio = pyaudio.PyAudio()  # Separate dedicated instance
        self._stream_config = VirtualStreamConfig()
```

### Unified Chunking Strategy

**Current Problem:** Inconsistent chunk sizes (1024 vs 512 config, 2048 actual)

**New Solution:**
- **MonitorAudioService:** 1024-sample chunks (2048 bytes for 16-bit stereo)
- **VirtualMicrophoneService:** 512-sample chunks (1024 bytes for 16-bit mono)
- **Consistent Implementation:** Configuration matches actual usage

```python
# Monitor Configuration
MONITOR_CHUNK_SIZE = 1024  # samples
MONITOR_CHUNK_BYTES = MONITOR_CHUNK_SIZE * 2 * 2  # samples * channels * bytes_per_sample

# Virtual Configuration
VIRTUAL_CHUNK_SIZE = 512   # samples
VIRTUAL_CHUNK_BYTES = VIRTUAL_CHUNK_SIZE * 1 * 2  # samples * channels * bytes_per_sample
```

## Service Lifecycle Management

### Initialization Sequence

```python
async def initialize_audio_services():
    """
    Initialize both audio services with proper error handling.
    """
    try:
        # Initialize services sequentially to avoid PyAudio conflicts
        monitor_service = MonitorAudioService()
        await monitor_service.initialize()

        virtual_service = VirtualMicrophoneService()
        await virtual_service.initialize()

        # Create coordinator
        coordinator = AudioCoordinator(monitor_service, virtual_service)

        return coordinator

    except Exception as e:
        # Cleanup on failure
        if 'monitor_service' in locals():
            await monitor_service.shutdown()
        if 'virtual_service' in locals():
            await virtual_service.shutdown()
        raise
```

### Graceful Shutdown

```python
async def shutdown_audio_services(coordinator: AudioCoordinator):
    """
    Shutdown both services gracefully.
    """
    await asyncio.gather(
        coordinator.monitor_service.shutdown(),
        coordinator.virtual_service.shutdown(),
        return_exceptions=True
    )
```

### Health Monitoring

```python
class AudioHealthMonitor:
    """Monitors health of both audio services."""

    async def check_services_health(self, coordinator: AudioCoordinator) -> HealthReport:
        monitor_health = await coordinator.monitor_service.get_health()
        virtual_health = await coordinator.virtual_service.get_health()

        return HealthReport(
            monitor_service=monitor_health,
            virtual_service=virtual_health,
            overall_status=self._determine_overall_status(monitor_health, virtual_health)
        )
```

## File Structure Changes

### New Files to Create

```
src/myvoice/services/
├── monitor_audio_service.py          # MonitorAudioService (Audio Service 1)
├── virtual_microphone_service.py     # VirtualMicrophoneService (Audio Service 2)
├── audio_coordinator.py              # AudioCoordinator
└── stream_synchronization_manager.py # StreamSynchronizationManager

src/myvoice/models/
├── monitor_playback_task.py          # MonitorPlaybackTask
├── virtual_playback_task.py          # VirtualPlaybackTask
└── dual_stream_result.py             # DualStreamResult
```

### Files to Modify

```
src/myvoice/services/
├── audio_service.py                  # Refactor to AudioCoordinator or deprecate
└── tts_service.py                    # Update to use AudioCoordinator

src/myvoice/models/
└── audio_playback_task.py            # Split into specific task types
```

## Implementation Priority

### Phase 1: Service Separation (Critical)
1. Create MonitorAudioService with monitor-only functionality
2. Create VirtualMicrophoneService with virtual-only functionality
3. Update TTSService to use both services independently

### Phase 2: Coordination (High)
1. Implement AudioCoordinator
2. Implement StreamSynchronizationManager
3. Update TTSService to use coordinator

### Phase 3: Resource Optimization (Medium)
1. Implement unified chunking strategy
2. Add service health monitoring
3. Optimize PyAudio resource usage

## Testing Strategy

### Unit Tests
- **MonitorAudioService:** Mock PyAudio, test monitor-specific functionality
- **VirtualMicrophoneService:** Mock virtual devices, test virtual-specific functionality
- **AudioCoordinator:** Test service coordination without audio hardware

### Integration Tests
- **Dual Service Playback:** Test actual audio output to both services
- **Device Change Handling:** Test graceful handling of device changes
- **Resource Competition:** Verify no resource conflicts between services

### Performance Tests
- **Latency Measurement:** Compare latency of new architecture vs. current
- **Resource Usage:** Monitor PyAudio resource consumption
- **Concurrent Playback:** Test multiple simultaneous audio requests

## Success Metrics

### Functional Requirements
- ✅ Separate services for monitor and virtual microphone routing
- ✅ Elimination of double playback issues
- ✅ No resource competition between services
- ✅ Clean TTSService integration interfaces

### Performance Requirements
- ✅ Reduced audio latency (< 100ms end-to-end)
- ✅ Lower CPU usage (< 10% during dual playback)
- ✅ No audio distortion or pitch issues
- ✅ Stable concurrent playback for >30 minutes

### Quality Requirements
- ✅ Audio quality preservation (no resampling artifacts)
- ✅ Consistent virtual microphone levels for communication apps
- ✅ Graceful device change handling
- ✅ Comprehensive error handling and recovery

---

**Design Date:** 2025-09-28
**Designer:** Claude Sonnet 4 (Development Agent)
**Task Reference:** Archon Task 4b4a93a8-8f80-4674-baa0-0e693524cfab
**Implements:** Stories 1.4 and 1.5 dual-service architecture specifications