"""
Dual-Stream Synchronization Service

This service provides synchronized playback timing between monitor speakers and
virtual microphone output to prevent audio drift and ensure perfect coordination
for real-time dual audio streaming.
"""

import asyncio
import logging
import time
import threading
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from threading import Lock, Event, Condition
import struct

from myvoice.services.core.base_service import BaseService, ServiceStatus
from myvoice.models.error import MyVoiceError, ErrorSeverity


class SynchronizationMode(Enum):
    """Synchronization modes for dual-stream audio."""
    MASTER_SLAVE = "master_slave"          # Monitor is master, virtual mic follows
    SHARED_CLOCK = "shared_clock"          # Both streams follow shared timing
    BUFFER_SYNC = "buffer_sync"            # Synchronize via buffer management
    TIMESTAMP_SYNC = "timestamp_sync"      # Use timestamps for coordination


class StreamPriority(Enum):
    """Priority levels for audio streams."""
    MONITOR_PRIORITY = "monitor_priority"   # Prioritize monitor output
    VIRTUAL_MIC_PRIORITY = "virtual_mic_priority"  # Prioritize virtual mic
    BALANCED = "balanced"                   # Equal priority


@dataclass
class SynchronizationMetrics:
    """Metrics for dual-stream synchronization."""
    timestamp: float
    monitor_latency_ms: float
    virtual_mic_latency_ms: float
    drift_ms: float
    buffer_underruns: int
    buffer_overruns: int
    sync_corrections: int
    average_drift_ms: float
    max_drift_ms: float
    sync_accuracy_percent: float


@dataclass
class AudioBuffer:
    """Thread-safe audio buffer for stream synchronization."""
    data: deque = field(default_factory=deque)
    max_size: int = 10  # Maximum number of audio chunks
    lock: Lock = field(default_factory=Lock)
    not_empty: Condition = field(default=None)
    not_full: Condition = field(default=None)
    sample_rate: int = 48000
    channels: int = 2
    chunk_size: int = 1024
    bytes_per_sample: int = 2
    total_samples_written: int = 0
    total_samples_read: int = 0
    underrun_count: int = 0
    overrun_count: int = 0

    def __post_init__(self):
        # Share the same lock between conditions
        self.not_empty = Condition(self.lock)
        self.not_full = Condition(self.lock)

    def put(self, audio_chunk: bytes, timeout: Optional[float] = None) -> bool:
        """
        Add audio chunk to buffer with optional timeout.

        Args:
            audio_chunk: Audio data as bytes
            timeout: Maximum time to wait for space

        Returns:
            bool: True if chunk was added, False if timeout
        """
        with self.not_full:
            end_time = None if timeout is None else time.time() + timeout

            while len(self.data) >= self.max_size:
                if timeout is not None:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        self.overrun_count += 1
                        return False
                    self.not_full.wait(remaining)
                else:
                    self.not_full.wait()

            # Calculate samples in this chunk
            bytes_per_frame = self.channels * self.bytes_per_sample
            samples_in_chunk = len(audio_chunk) // bytes_per_frame

            self.data.append({
                'data': audio_chunk,
                'timestamp': time.time(),
                'sample_count': samples_in_chunk
            })
            self.total_samples_written += samples_in_chunk
            self.not_empty.notify()
            return True

    def get(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Get audio chunk from buffer with optional timeout.

        Args:
            timeout: Maximum time to wait for data

        Returns:
            Optional[Dict]: Audio chunk data or None if timeout
        """
        with self.not_empty:
            end_time = None if timeout is None else time.time() + timeout

            while not self.data:
                if timeout is not None:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        self.underrun_count += 1
                        return None
                    self.not_empty.wait(remaining)
                else:
                    self.not_empty.wait()

            chunk = self.data.popleft()
            self.total_samples_read += chunk['sample_count']
            self.not_full.notify()
            return chunk

    def peek(self) -> Optional[Dict[str, Any]]:
        """Peek at next chunk without removing it."""
        with self.lock:
            return self.data[0] if self.data else None

    def size(self) -> int:
        """Get current buffer size."""
        with self.lock:
            return len(self.data)

    def clear(self) -> None:
        """Clear all buffered data."""
        with self.lock:
            self.data.clear()
            self.not_full.notify_all()

    def get_buffer_level_percent(self) -> float:
        """Get buffer fill level as percentage."""
        with self.lock:
            return (len(self.data) / self.max_size) * 100.0

    def get_sample_drift(self) -> int:
        """Get sample drift (difference between written and read)."""
        with self.lock:
            return self.total_samples_written - self.total_samples_read


@dataclass
class StreamConfiguration:
    """Configuration for a synchronized audio stream."""
    stream_id: str
    sample_rate: int = 48000
    channels: int = 2
    chunk_size: int = 1024
    buffer_size: int = 8
    target_latency_ms: float = 50.0
    max_drift_ms: float = 10.0
    priority: StreamPriority = StreamPriority.BALANCED


class DualStreamSynchronizer(BaseService):
    """
    Dual-Stream Synchronization Service for coordinated audio playback.

    This service provides synchronized playback timing between monitor speakers
    and virtual microphone output to prevent audio drift and ensure perfect
    coordination for real-time dual audio streaming.

    Features:
    - Multiple synchronization modes (master/slave, shared clock, buffer sync)
    - Real-time drift detection and correction
    - Adaptive buffer management
    - Performance monitoring and metrics
    - Thread-safe operations
    - Configurable latency targets
    """

    def __init__(self, sync_mode: SynchronizationMode = SynchronizationMode.SHARED_CLOCK):
        """
        Initialize the Dual-Stream Synchronizer.

        Args:
            sync_mode: Synchronization mode to use
        """
        super().__init__("DualStreamSynchronizer")
        self.logger = logging.getLogger(self.__class__.__name__)

        # Synchronization configuration
        self.sync_mode = sync_mode
        self.stream_priority = StreamPriority.BALANCED

        # Stream management
        self.streams: Dict[str, StreamConfiguration] = {}
        self.buffers: Dict[str, AudioBuffer] = {}
        self.stream_threads: Dict[str, threading.Thread] = {}

        # Synchronization state
        self.master_clock_start_time: Optional[float] = None
        self.shared_clock_lock = Lock()
        self.lock = Lock()  # General purpose lock for thread-safe operations
        self.sync_active = False

        # Timing and drift management
        self.drift_correction_enabled = True
        self.max_allowed_drift_ms = 10.0
        self.sync_tolerance_ms = 2.0
        self.correction_gain = 0.1  # How aggressively to correct drift

        # Performance monitoring
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 measurements
        self.metrics_lock = Lock()

        # Callbacks
        self.sync_callbacks: List[Callable[[SynchronizationMetrics], None]] = []
        self.drift_callbacks: List[Callable[[str, float], None]] = []

        # Control events
        self._stop_event = Event()
        self._pause_event = Event()

        self.logger.info(f"Dual-Stream Synchronizer initialized (mode: {sync_mode.value})")

    def register_stream(self, config: StreamConfiguration) -> bool:
        """
        Register a new audio stream for synchronization.

        Args:
            config: Stream configuration

        Returns:
            bool: True if stream was registered successfully
        """
        try:
            if config.stream_id in self.streams:
                self.logger.warning(f"Stream {config.stream_id} already registered")
                return False

            # Create buffer for this stream
            buffer = AudioBuffer(
                max_size=config.buffer_size,
                sample_rate=config.sample_rate,
                channels=config.channels,
                chunk_size=config.chunk_size
            )

            self.streams[config.stream_id] = config
            self.buffers[config.stream_id] = buffer

            self.logger.info(f"Registered stream: {config.stream_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to register stream {config.stream_id}: {e}")
            return False

    def unregister_stream(self, stream_id: str) -> bool:
        """
        Unregister an audio stream.

        Args:
            stream_id: ID of stream to unregister

        Returns:
            bool: True if stream was unregistered successfully
        """
        try:
            if stream_id not in self.streams:
                self.logger.warning(f"Stream {stream_id} not found")
                return False

            # Stop stream thread if running
            if stream_id in self.stream_threads:
                thread = self.stream_threads[stream_id]
                if thread.is_alive():
                    # Signal thread to stop and wait
                    self._stop_event.set()
                    thread.join(timeout=2.0)
                del self.stream_threads[stream_id]

            # Clear buffer
            if stream_id in self.buffers:
                self.buffers[stream_id].clear()
                del self.buffers[stream_id]

            del self.streams[stream_id]

            self.logger.info(f"Unregistered stream: {stream_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to unregister stream {stream_id}: {e}")
            return False

    def write_audio_data(self, stream_id: str, audio_data: bytes) -> bool:
        """
        Write audio data to a stream buffer with synchronization.

        Args:
            stream_id: ID of target stream
            audio_data: Audio data to write

        Returns:
            bool: True if data was written successfully
        """
        try:
            if stream_id not in self.buffers:
                self.logger.error(f"Stream {stream_id} not registered")
                return False

            buffer = self.buffers[stream_id]

            # Apply timing synchronization based on mode
            if self.sync_mode == SynchronizationMode.SHARED_CLOCK:
                # Wait for shared clock timing
                target_time = self._calculate_shared_clock_time(stream_id, len(audio_data))
                current_time = time.time()
                if target_time > current_time:
                    time.sleep(target_time - current_time)

            # Write to buffer with timeout
            success = buffer.put(audio_data, timeout=0.1)
            if not success:
                self.logger.warning(f"Buffer overrun in stream {stream_id}")

            # Update synchronization metrics
            self._update_sync_metrics()

            return success

        except Exception as e:
            self.logger.error(f"Failed to write audio data to stream {stream_id}: {e}")
            return False

    def read_audio_data(self, stream_id: str, timeout: Optional[float] = None) -> Optional[bytes]:
        """
        Read audio data from a stream buffer with synchronization.

        Args:
            stream_id: ID of source stream
            timeout: Maximum time to wait for data

        Returns:
            Optional[bytes]: Audio data or None if timeout/error
        """
        try:
            if stream_id not in self.buffers:
                self.logger.error(f"Stream {stream_id} not registered")
                return None

            buffer = self.buffers[stream_id]

            # Apply synchronization delay if needed
            if self.sync_mode == SynchronizationMode.BUFFER_SYNC:
                self._apply_buffer_sync_delay(stream_id)

            chunk = buffer.get(timeout=timeout)
            if chunk is None:
                return None

            return chunk['data']

        except Exception as e:
            self.logger.error(f"Failed to read audio data from stream {stream_id}: {e}")
            return None

    def start_stream_playback(self, stream_id: str, playback_id: str) -> bool:
        """
        Start playback for a registered stream.

        Args:
            stream_id: ID of the stream to start
            playback_id: Playback task ID for tracking

        Returns:
            bool: True if stream playback started successfully
        """
        try:
            if stream_id not in self.streams:
                self.logger.error(f"Stream {stream_id} not registered for playback")
                return False

            stream_config = self.streams[stream_id]

            # Mark stream as active
            with self.lock:
                if not hasattr(self, 'active_streams'):
                    self.active_streams = {}

                self.active_streams[stream_id] = {
                    'playback_id': playback_id,
                    'start_time': time.time(),
                    'status': 'playing'
                }

            # Log successful start
            self.logger.info(f"Started stream playback: {stream_id} (playback_id: {playback_id})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start stream playback for {stream_id}: {e}")
            return False

    def stop_stream_playback(self, stream_id: str) -> bool:
        """
        Stop playback for a stream.

        Args:
            stream_id: ID of the stream to stop

        Returns:
            bool: True if stream playback stopped successfully
        """
        try:
            with self.lock:
                if hasattr(self, 'active_streams') and stream_id in self.active_streams:
                    self.active_streams[stream_id]['status'] = 'stopped'
                    self.active_streams[stream_id]['stop_time'] = time.time()

            self.logger.info(f"Stopped stream playback: {stream_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop stream playback for {stream_id}: {e}")
            return False

    def synchronize_streams(self, stream_ids: List[str]) -> bool:
        """
        Synchronize multiple streams to start at the same time.

        Args:
            stream_ids: List of stream IDs to synchronize

        Returns:
            bool: True if synchronization was successful
        """
        try:
            # Validate all streams exist
            for stream_id in stream_ids:
                if stream_id not in self.streams:
                    self.logger.error(f"Stream {stream_id} not found for synchronization")
                    return False

            # Clear all buffers to start fresh
            for stream_id in stream_ids:
                self.buffers[stream_id].clear()

            # Set shared clock start time
            with self.shared_clock_lock:
                self.master_clock_start_time = time.time() + 0.1  # Start in 100ms

            self.sync_active = True
            self.logger.info(f"Synchronized {len(stream_ids)} streams")
            return True

        except Exception as e:
            self.logger.error(f"Failed to synchronize streams: {e}")
            return False

    def _calculate_shared_clock_time(self, stream_id: str, data_size: int) -> float:
        """Calculate timing for shared clock synchronization."""
        if not self.master_clock_start_time:
            return time.time()

        config = self.streams[stream_id]
        buffer = self.buffers[stream_id]

        # Calculate how many samples this represents
        bytes_per_frame = config.channels * 2  # Assuming 16-bit samples
        samples = data_size // bytes_per_frame

        # Calculate target time based on sample position
        sample_time = buffer.total_samples_written / config.sample_rate
        target_time = self.master_clock_start_time + sample_time

        return target_time

    def _apply_buffer_sync_delay(self, stream_id: str) -> None:
        """Apply buffer-based synchronization delay."""
        try:
            # Get buffer levels for all streams
            buffer_levels = {}
            for sid, buffer in self.buffers.items():
                buffer_levels[sid] = buffer.get_buffer_level_percent()

            # Calculate delay based on buffer level differences
            current_level = buffer_levels.get(stream_id, 0)
            avg_level = sum(buffer_levels.values()) / len(buffer_levels)

            if current_level > avg_level + 10:  # Buffer too full
                delay_ms = (current_level - avg_level) * 0.1  # Small delay
                time.sleep(delay_ms / 1000.0)

        except Exception as e:
            self.logger.warning(f"Error applying buffer sync delay: {e}")

    def _update_sync_metrics(self) -> None:
        """Update synchronization metrics."""
        try:
            current_time = time.time()

            # Calculate latencies and drift
            monitor_latency = self._calculate_stream_latency("monitor")
            virtual_mic_latency = self._calculate_stream_latency("virtual_mic")

            drift_ms = abs(monitor_latency - virtual_mic_latency)

            # Get buffer statistics
            total_underruns = sum(buf.underrun_count for buf in self.buffers.values())
            total_overruns = sum(buf.overrun_count for buf in self.buffers.values())

            # Calculate average and max drift
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
            if recent_metrics:
                avg_drift = sum(m.drift_ms for m in recent_metrics) / len(recent_metrics)
                max_drift = max(m.drift_ms for m in recent_metrics)
            else:
                avg_drift = drift_ms
                max_drift = drift_ms

            # Calculate sync accuracy
            sync_accuracy = max(0, 100 - (drift_ms / self.max_allowed_drift_ms * 100))

            metrics = SynchronizationMetrics(
                timestamp=current_time,
                monitor_latency_ms=monitor_latency,
                virtual_mic_latency_ms=virtual_mic_latency,
                drift_ms=drift_ms,
                buffer_underruns=total_underruns,
                buffer_overruns=total_overruns,
                sync_corrections=0,  # TODO: Track corrections
                average_drift_ms=avg_drift,
                max_drift_ms=max_drift,
                sync_accuracy_percent=sync_accuracy
            )

            with self.metrics_lock:
                self.metrics_history.append(metrics)

            # Check for drift correction
            if self.drift_correction_enabled and drift_ms > self.max_allowed_drift_ms:
                self._apply_drift_correction(drift_ms)

            # Notify callbacks
            for callback in self.sync_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    self.logger.warning(f"Sync callback error: {e}")

        except Exception as e:
            self.logger.error(f"Error updating sync metrics: {e}")

    def _calculate_stream_latency(self, stream_id: str) -> float:
        """Calculate current latency for a stream in milliseconds."""
        try:
            if stream_id not in self.buffers:
                return 0.0

            buffer = self.buffers[stream_id]
            config = self.streams[stream_id]

            # Calculate latency based on buffer level
            buffer_samples = buffer.size() * config.chunk_size
            latency_ms = (buffer_samples / config.sample_rate) * 1000.0

            return latency_ms

        except Exception as e:
            self.logger.warning(f"Error calculating latency for {stream_id}: {e}")
            return 0.0

    def _apply_drift_correction(self, drift_ms: float) -> None:
        """Apply drift correction to maintain synchronization."""
        try:
            correction_amount = drift_ms * self.correction_gain

            # Apply correction based on priority
            if self.stream_priority == StreamPriority.MONITOR_PRIORITY:
                # Adjust virtual mic timing
                if "virtual_mic" in self.buffers:
                    # Add small delay to virtual mic
                    time.sleep(correction_amount / 1000.0)

            elif self.stream_priority == StreamPriority.VIRTUAL_MIC_PRIORITY:
                # Adjust monitor timing (more complex)
                pass  # Would need to coordinate with audio output

            else:  # BALANCED
                # Adjust both streams slightly
                adjustment = correction_amount / 2.0
                time.sleep(adjustment / 1000.0)

            self.logger.debug(f"Applied drift correction: {correction_amount:.2f}ms")

        except Exception as e:
            self.logger.error(f"Error applying drift correction: {e}")

    def get_synchronization_metrics(self) -> Optional[SynchronizationMetrics]:
        """
        Get the latest synchronization metrics.

        Returns:
            Optional[SynchronizationMetrics]: Latest metrics or None
        """
        with self.metrics_lock:
            return self.metrics_history[-1] if self.metrics_history else None

    def correct_drift(self) -> bool:
        """
        Apply drift correction to synchronized streams.

        Returns:
            bool: True if drift correction was applied successfully
        """
        try:
            if not self.drift_correction_enabled:
                return False

            # Get current metrics
            metrics = self.get_synchronization_metrics()
            if not metrics:
                return False

            # Check if correction is needed
            if abs(metrics.drift_ms) <= self.sync_tolerance_ms:
                return True  # No correction needed

            # Calculate correction amount
            correction_ms = metrics.drift_ms * self.correction_gain

            self.logger.debug(f"Applying drift correction: {correction_ms:.2f}ms")

            # Apply correction to all streams
            correction_applied = False
            for stream_id, buffer in self.buffers.items():
                try:
                    # Adjust buffer timing for correction
                    if correction_ms > 0:
                        # Add small delay to slow stream
                        buffer._last_read_time += correction_ms / 1000.0
                    else:
                        # Reduce delay for fast stream
                        buffer._last_read_time -= abs(correction_ms) / 1000.0

                    correction_applied = True

                except Exception as e:
                    self.logger.warning(f"Failed to apply correction to stream {stream_id}: {e}")

            # Update correction count in metrics
            if correction_applied:
                with self.metrics_lock:
                    if self.metrics_history:
                        latest_metrics = self.metrics_history[-1]
                        latest_metrics.sync_corrections += 1

            return correction_applied

        except Exception as e:
            self.logger.error(f"Drift correction failed: {e}")
            return False

    def get_stream_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all registered streams.

        Returns:
            Dict[str, Dict[str, Any]]: Stream status information
        """
        status = {}

        for stream_id, config in self.streams.items():
            buffer = self.buffers.get(stream_id)
            if buffer:
                status[stream_id] = {
                    'sample_rate': config.sample_rate,
                    'channels': config.channels,
                    'buffer_level_percent': buffer.get_buffer_level_percent(),
                    'sample_drift': buffer.get_sample_drift(),
                    'underruns': buffer.underrun_count,
                    'overruns': buffer.overrun_count,
                    'latency_ms': self._calculate_stream_latency(stream_id)
                }

        return status

    def add_sync_callback(self, callback: Callable[[SynchronizationMetrics], None]) -> None:
        """Add a callback for synchronization metrics updates."""
        if callback not in self.sync_callbacks:
            self.sync_callbacks.append(callback)

    def remove_sync_callback(self, callback: Callable[[SynchronizationMetrics], None]) -> None:
        """Remove a synchronization metrics callback."""
        if callback in self.sync_callbacks:
            self.sync_callbacks.remove(callback)

    def add_drift_callback(self, callback: Callable[[str, float], None]) -> None:
        """Add a callback for drift notifications."""
        if callback not in self.drift_callbacks:
            self.drift_callbacks.append(callback)

    def remove_drift_callback(self, callback: Callable[[str, float], None]) -> None:
        """Remove a drift notification callback."""
        if callback in self.drift_callbacks:
            self.drift_callbacks.remove(callback)

    async def start(self) -> bool:
        """Start the synchronization service."""
        try:
            await self._update_status(ServiceStatus.STARTING)

            # Reset state
            self.sync_active = False
            self.master_clock_start_time = None
            self._stop_event.clear()
            self._pause_event.clear()

            # Clear metrics
            with self.metrics_lock:
                self.metrics_history.clear()

            await self._update_status(ServiceStatus.RUNNING)
            self.logger.info("Dual-Stream Synchronizer started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start Dual-Stream Synchronizer: {e}")
            await self._update_status(ServiceStatus.ERROR)
            return False

    async def stop(self) -> bool:
        """Stop the synchronization service."""
        try:
            await self._update_status(ServiceStatus.STOPPING)

            # Signal all threads to stop
            self._stop_event.set()

            # Stop all stream threads
            for stream_id, thread in self.stream_threads.items():
                if thread.is_alive():
                    thread.join(timeout=2.0)

            # Clear all streams and buffers
            for stream_id in list(self.streams.keys()):
                self.unregister_stream(stream_id)

            # Clear callbacks
            self.sync_callbacks.clear()
            self.drift_callbacks.clear()

            await self._update_status(ServiceStatus.STOPPED)
            self.logger.info("Dual-Stream Synchronizer stopped")
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop Dual-Stream Synchronizer: {e}")
            await self._update_status(ServiceStatus.ERROR)
            return False

    def optimize_performance(self, cpu_priority: str = "high", memory_limit_mb: int = 100) -> bool:
        """
        Apply performance optimizations for real-time audio streaming.

        Args:
            cpu_priority: CPU priority level ("low", "normal", "high", "realtime")
            memory_limit_mb: Memory usage limit in megabytes

        Returns:
            bool: True if optimizations were applied successfully
        """
        try:
            import psutil
            import os

            # Get current process
            process = psutil.Process(os.getpid())

            # Set CPU priority
            priority_map = {
                "low": psutil.BELOW_NORMAL_PRIORITY_CLASS,
                "normal": psutil.NORMAL_PRIORITY_CLASS,
                "high": psutil.HIGH_PRIORITY_CLASS,
                "realtime": psutil.REALTIME_PRIORITY_CLASS
            }

            if cpu_priority in priority_map:
                try:
                    process.nice(priority_map[cpu_priority])
                    self.logger.info(f"Set CPU priority to {cpu_priority}")
                except Exception as e:
                    self.logger.warning(f"Failed to set CPU priority: {e}")

            # Configure thread priorities for audio threads
            for stream_id, thread in self.stream_threads.items():
                try:
                    # On Windows, set thread priority to time-critical for audio threads
                    if hasattr(thread, 'native_id') and thread.native_id:
                        import ctypes
                        ctypes.windll.kernel32.SetThreadPriority(thread.native_id, 15)  # THREAD_PRIORITY_TIME_CRITICAL
                        self.logger.debug(f"Set high priority for stream thread: {stream_id}")
                except Exception as e:
                    self.logger.debug(f"Could not set thread priority for {stream_id}: {e}")

            # Optimize buffer sizes for real-time performance
            self._optimize_buffer_configurations()

            # Enable drift correction optimizations
            self._enable_performance_monitoring()

            self.logger.info("Performance optimizations applied successfully")
            return True

        except ImportError:
            self.logger.warning("psutil not available - some performance optimizations skipped")
            return False
        except Exception as e:
            self.logger.error(f"Failed to apply performance optimizations: {e}")
            return False

    def _optimize_buffer_configurations(self) -> None:
        """Optimize buffer configurations for real-time performance."""
        try:
            for stream_id, buffer in self.buffers.items():
                stream_config = self.streams.get(stream_id)
                if not stream_config:
                    continue

                # Calculate optimal buffer size based on latency target
                target_latency_ms = 20.0  # 20ms target latency
                samples_per_ms = stream_config.sample_rate / 1000.0
                optimal_buffer_size = int(target_latency_ms * samples_per_ms * stream_config.channels)

                # Update buffer if needed
                if buffer.max_size != optimal_buffer_size:
                    buffer.max_size = optimal_buffer_size
                    self.logger.debug(f"Optimized buffer size for {stream_id}: {optimal_buffer_size} samples")

        except Exception as e:
            self.logger.error(f"Buffer optimization failed: {e}")

    def _enable_performance_monitoring(self) -> None:
        """Enable enhanced performance monitoring."""
        try:
            # Reduce metrics collection overhead while maintaining accuracy
            self.metrics_history = deque(maxlen=500)  # Reduce from 1000 to 500

            # Optimize drift correction parameters for lower latency
            self.sync_tolerance_ms = 1.0  # Tighter tolerance
            self.correction_gain = 0.15   # More aggressive correction

            # Enable adaptive buffer management
            self.adaptive_buffer_enabled = True

            self.logger.debug("Enhanced performance monitoring enabled")

        except Exception as e:
            self.logger.error(f"Performance monitoring setup failed: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get detailed performance metrics for monitoring and optimization.

        Returns:
            Dict[str, Any]: Performance metrics including CPU usage, latency, and throughput
        """
        try:
            import psutil
            import threading

            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            process = psutil.Process()

            # Calculate audio processing metrics
            active_streams = len([s for s in self.streams.values() if hasattr(s, 'active') and s.active])
            total_buffer_usage = sum(buffer.size() for buffer in self.buffers.values())
            max_buffer_capacity = sum(buffer.max_size for buffer in self.buffers.values())

            buffer_usage_percent = (total_buffer_usage / max_buffer_capacity * 100) if max_buffer_capacity > 0 else 0

            # Get synchronization quality metrics
            sync_metrics = self.get_synchronization_metrics()

            return {
                "system_performance": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_info.percent,
                    "process_memory_mb": process.memory_info().rss / 1024 / 1024,
                    "active_threads": threading.active_count()
                },
                "audio_performance": {
                    "active_streams": active_streams,
                    "buffer_usage_percent": buffer_usage_percent,
                    "total_buffer_samples": total_buffer_usage,
                    "sync_quality": sync_metrics.average_drift_ms if sync_metrics else 0,
                    "underruns": sync_metrics.buffer_underruns if sync_metrics else 0,
                    "overruns": sync_metrics.buffer_overruns if sync_metrics else 0
                },
                "real_time_metrics": {
                    "latency_target_ms": 20.0,
                    "current_latency_ms": sync_metrics.monitor_latency_ms if sync_metrics else 0,
                    "drift_corrections": sync_metrics.sync_corrections if sync_metrics else 0,
                    "max_drift_observed_ms": sync_metrics.max_drift_ms if sync_metrics else 0
                }
            }

        except ImportError:
            return {
                "system_performance": {"note": "psutil not available"},
                "audio_performance": {
                    "active_streams": len(self.streams),
                    "total_buffers": len(self.buffers)
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}

    def enable_adaptive_optimization(self, enable: bool = True) -> None:
        """
        Enable or disable adaptive performance optimization.

        When enabled, the synchronizer will automatically adjust parameters
        based on system performance and audio quality metrics.

        Args:
            enable: Whether to enable adaptive optimization
        """
        try:
            self.adaptive_optimization_enabled = enable

            if enable:
                # Start adaptive optimization monitoring
                self.logger.info("Adaptive performance optimization enabled")

                # Create optimization monitoring thread
                def adaptive_monitor():
                    while not self._stop_event.is_set() and self.adaptive_optimization_enabled:
                        try:
                            # Check performance metrics every 5 seconds
                            self._stop_event.wait(5.0)

                            if self._stop_event.is_set():
                                break

                            # Get current performance metrics
                            metrics = self.get_performance_metrics()

                            # Adaptive buffer size adjustment
                            cpu_usage = metrics.get("system_performance", {}).get("cpu_percent", 0)
                            if cpu_usage > 80:
                                # High CPU - reduce buffer update frequency
                                self._reduce_processing_load()
                            elif cpu_usage < 40:
                                # Low CPU - optimize for lower latency
                                self._optimize_for_latency()

                        except Exception as e:
                            self.logger.error(f"Adaptive optimization error: {e}")

                # Start monitoring thread
                adaptive_thread = threading.Thread(target=adaptive_monitor, daemon=True)
                adaptive_thread.start()

            else:
                self.logger.info("Adaptive performance optimization disabled")

        except Exception as e:
            self.logger.error(f"Failed to configure adaptive optimization: {e}")

    def _reduce_processing_load(self) -> None:
        """Reduce processing load when CPU usage is high."""
        try:
            # Increase sync tolerance to reduce correction frequency
            self.sync_tolerance_ms = min(self.sync_tolerance_ms * 1.2, 5.0)

            # Reduce metrics collection frequency
            if len(self.metrics_history) > 100:
                self.metrics_history = deque(list(self.metrics_history)[::2], maxlen=250)

            self.logger.debug("Reduced processing load due to high CPU usage")

        except Exception as e:
            self.logger.error(f"Failed to reduce processing load: {e}")

    def _optimize_for_latency(self) -> None:
        """Optimize for lower latency when CPU usage is low."""
        try:
            # Tighten sync tolerance for better accuracy
            self.sync_tolerance_ms = max(self.sync_tolerance_ms * 0.9, 0.5)

            # Increase correction responsiveness
            self.correction_gain = min(self.correction_gain * 1.1, 0.3)

            self.logger.debug("Optimized for lower latency due to low CPU usage")

        except Exception as e:
            self.logger.error(f"Failed to optimize for latency: {e}")

    async def health_check(self) -> Tuple[bool, Optional[MyVoiceError]]:
        """Check service health."""
        try:
            # Check if synchronization is working within tolerance
            metrics = self.get_synchronization_metrics()
            if metrics and metrics.drift_ms > self.max_allowed_drift_ms * 2:
                return False, MyVoiceError(
                    severity=ErrorSeverity.WARNING,
                    code="SYNC_DRIFT_HIGH",
                    user_message="Audio synchronization drift is high",
                    technical_details=f"Drift: {metrics.drift_ms:.2f}ms",
                    suggested_action="Check audio device performance and system load"
                )

            return True, None

        except Exception as e:
            return False, MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="DUAL_STREAM_HEALTH_FAILED",
                user_message="Dual-stream synchronizer health check failed",
                technical_details=str(e),
                suggested_action="Restart the synchronization service"
            )