"""
Transcription Queue Service

This module implements a background queue system for processing voice file transcriptions
using asyncio and the WhisperService. It provides automatic transcription processing
with priority queuing, concurrent processing limits, and error handling.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path

from myvoice.services.core.base_service import BaseService
from myvoice.models.service_enums import ServiceStatus
from myvoice.models.voice_profile import VoiceProfile, TranscriptionStatus
from myvoice.models.error import MyVoiceError, ErrorSeverity


logger = logging.getLogger(__name__)


class QueuePriority(Enum):
    """Priority levels for transcription queue items."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class TranscriptionQueueItem:
    """Item in the transcription queue."""
    voice_profile: VoiceProfile
    priority: QueuePriority = QueuePriority.NORMAL
    added_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3

    def __lt__(self, other):
        """Enable priority queue ordering (higher priority first)."""
        if not isinstance(other, TranscriptionQueueItem):
            return NotImplemented
        # Higher priority values come first, then by older added_at
        return (other.priority.value, self.added_at) < (self.priority.value, other.added_at)


class TranscriptionQueueService(BaseService):
    """
    Background service for processing voice file transcriptions in a queue.

    This service manages automatic transcription of voice files using WhisperService
    with features including:
    - Priority-based queue processing
    - Concurrent processing with configurable limits
    - Automatic retry with exponential backoff
    - Progress tracking and status updates
    - Integration with VoiceProfile status tracking
    """

    def __init__(
        self,
        whisper_service: Optional[Any] = None,  # WhisperService instance
        max_concurrent_transcriptions: int = 2,
        max_queue_size: int = 100,
        retry_delay_base: float = 5.0,  # Base delay in seconds for retries
        enable_auto_processing: bool = True
    ):
        """
        Initialize the transcription queue service.

        Args:
            whisper_service: WhisperService instance for processing transcriptions
            max_concurrent_transcriptions: Maximum number of concurrent transcription tasks
            max_queue_size: Maximum number of items in the queue
            retry_delay_base: Base delay for exponential backoff retries
            enable_auto_processing: Whether to automatically process queue items
        """
        super().__init__("TranscriptionQueueService")

        self.whisper_service = whisper_service
        self.max_concurrent_transcriptions = max_concurrent_transcriptions
        self.max_queue_size = max_queue_size
        self.retry_delay_base = retry_delay_base
        self.enable_auto_processing = enable_auto_processing

        # Queue and processing state
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._processing_tasks: Dict[str, asyncio.Task] = {}
        self._processing_semaphore: Optional[asyncio.Semaphore] = None
        self._queue_processor_task: Optional[asyncio.Task] = None

        # Statistics and callbacks
        self._stats = {
            'total_queued': 0,
            'total_processed': 0,
            'total_failed': 0,
            'total_skipped': 0,
            'current_queue_size': 0,
            'active_transcriptions': 0
        }

        # Callbacks for external integration
        self._progress_callback: Optional[Callable[[VoiceProfile, str], None]] = None
        self._completion_callback: Optional[Callable[[VoiceProfile, bool, str], None]] = None

    async def start_service(self) -> None:
        """Start the transcription queue service."""
        try:
            self.status = ServiceStatus.STARTING
            self.logger.info("Starting transcription queue service")

            # Validate dependencies
            if not self.whisper_service:
                raise MyVoiceError(
                    severity=ErrorSeverity.CRITICAL,
                    code="WHISPER_SERVICE_REQUIRED",
                    user_message="WhisperService is required for transcription queue",
                    technical_details="WhisperService instance must be provided",
                    suggested_action="Initialize with a valid WhisperService instance"
                )

            # Initialize processing semaphore
            self._processing_semaphore = asyncio.Semaphore(self.max_concurrent_transcriptions)

            # Start the queue processor if auto-processing is enabled
            if self.enable_auto_processing:
                self._queue_processor_task = asyncio.create_task(self._process_queue())

            self.status = ServiceStatus.RUNNING
            self.logger.info("Transcription queue service started successfully")

        except MyVoiceError:
            # Re-raise MyVoiceError as-is (don't wrap it)
            self.status = ServiceStatus.ERROR
            raise
        except Exception as e:
            self.status = ServiceStatus.ERROR
            error = MyVoiceError(
                severity=ErrorSeverity.CRITICAL,
                code="TRANSCRIPTION_QUEUE_START_FAILED",
                user_message="Failed to start transcription queue service",
                technical_details=str(e),
                suggested_action="Check WhisperService availability and try again"
            )
            self.logger.error(f"Failed to start transcription queue service: {e}")
            raise error

    async def stop_service(self) -> None:
        """Stop the transcription queue service."""
        try:
            self.status = ServiceStatus.STOPPING
            self.logger.info("Stopping transcription queue service")

            # Stop queue processor
            if self._queue_processor_task and not self._queue_processor_task.done():
                self._queue_processor_task.cancel()
                try:
                    await self._queue_processor_task
                except asyncio.CancelledError:
                    pass

            # Cancel all active transcription tasks
            if self._processing_tasks:
                for task_id, task in self._processing_tasks.items():
                    if not task.done():
                        task.cancel()
                        self.logger.debug(f"Cancelled transcription task: {task_id}")

                # Wait for tasks to complete or timeout
                if self._processing_tasks:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*self._processing_tasks.values(), return_exceptions=True),
                            timeout=10.0
                        )
                    except asyncio.TimeoutError:
                        self.logger.warning("Some transcription tasks did not complete within timeout")

                self._processing_tasks.clear()

            # Clear queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except asyncio.QueueEmpty:
                    break

            self.status = ServiceStatus.STOPPED
            self.logger.info("Transcription queue service stopped successfully")

        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.logger.error(f"Error stopping transcription queue service: {e}")
            raise

    async def queue_transcription(
        self,
        voice_profile: VoiceProfile,
        priority: QueuePriority = QueuePriority.NORMAL
    ) -> bool:
        """
        Queue a voice profile for transcription.

        Args:
            voice_profile: VoiceProfile to transcribe
            priority: Priority level for the transcription

        Returns:
            bool: True if successfully queued, False otherwise

        Raises:
            MyVoiceError: If queueing fails
        """
        try:
            # Validate voice profile can be queued
            if not voice_profile.can_queue_transcription():
                self.logger.debug(f"Voice profile {voice_profile.name} cannot be queued: {voice_profile.transcription_status}")
                return False

            # Check queue capacity
            if self._queue.qsize() >= self.max_queue_size:
                raise MyVoiceError(
                    severity=ErrorSeverity.WARNING,
                    code="TRANSCRIPTION_QUEUE_FULL",
                    user_message="Transcription queue is full",
                    technical_details=f"Queue has reached maximum size of {self.max_queue_size}",
                    suggested_action="Wait for current transcriptions to complete"
                )

            # Create queue item
            queue_item = TranscriptionQueueItem(
                voice_profile=voice_profile,
                priority=priority
            )

            # Update voice profile status
            voice_profile.update_transcription_status(TranscriptionStatus.QUEUED)

            # Add to queue
            await self._queue.put(queue_item)

            # Update statistics
            self._stats['total_queued'] += 1
            self._stats['current_queue_size'] = self._queue.qsize()

            self.logger.info(f"Queued transcription for {voice_profile.name} with priority {priority.name}")

            # Notify progress callback
            if self._progress_callback:
                try:
                    self._progress_callback(voice_profile, "Queued for transcription")
                except Exception as e:
                    self.logger.error(f"Error in progress callback: {e}")

            return True

        except Exception as e:
            if isinstance(e, MyVoiceError):
                raise

            error = MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="TRANSCRIPTION_QUEUE_FAILED",
                user_message="Failed to queue transcription",
                technical_details=str(e),
                suggested_action="Check queue status and try again"
            )
            self.logger.error(f"Failed to queue transcription for {voice_profile.name}: {e}")
            raise error

    async def _process_queue(self) -> None:
        """Background task to process the transcription queue."""
        self.logger.info("Starting queue processor")

        try:
            while True:
                try:
                    # Get item from queue (this will block until item available)
                    queue_item = await self._queue.get()

                    # Update statistics
                    self._stats['current_queue_size'] = self._queue.qsize()

                    # Process the item in background
                    task_id = f"transcription_{queue_item.voice_profile.name}_{datetime.now().timestamp()}"
                    task = asyncio.create_task(
                        self._process_transcription_item(queue_item)
                    )
                    self._processing_tasks[task_id] = task

                    # Clean up completed tasks
                    await self._cleanup_completed_tasks()

                    # Mark queue item as done
                    self._queue.task_done()

                except asyncio.CancelledError:
                    self.logger.info("Queue processor cancelled")
                    break
                except Exception as e:
                    self.logger.error(f"Error in queue processor: {e}")
                    # Continue processing to avoid stopping the queue

        except Exception as e:
            self.logger.error(f"Fatal error in queue processor: {e}")
            raise

    async def _process_transcription_item(self, queue_item: TranscriptionQueueItem) -> None:
        """
        Process a single transcription queue item with enhanced error handling and recovery.

        Args:
            queue_item: Item to process
        """
        voice_profile = queue_item.voice_profile
        start_time = datetime.now()

        # Acquire processing semaphore to limit concurrent transcriptions
        async with self._processing_semaphore:
            try:
                self._stats['active_transcriptions'] += 1

                # Pre-processing validation
                if not voice_profile.file_path.exists():
                    raise FileNotFoundError(f"Voice file not found: {voice_profile.file_path}")

                # Update voice profile status
                voice_profile.update_transcription_status(TranscriptionStatus.PROCESSING)

                # Notify progress callback
                if self._progress_callback:
                    try:
                        self._progress_callback(voice_profile, "Starting transcription...")
                    except Exception as e:
                        self.logger.error(f"Error in progress callback: {e}")

                self.logger.info(f"Starting transcription for {voice_profile.name}")

                # Calculate timeout based on file size and queue position
                file_size_mb = voice_profile.file_path.stat().st_size / (1024 * 1024)
                # Base timeout: 60s + 10s per MB, max 600s (10 minutes)
                timeout_seconds = min(600.0, 60.0 + (file_size_mb * 10.0))

                # Perform transcription using WhisperService with timeout
                transcription_result = await self.whisper_service.transcribe_file(
                    file_path=voice_profile.file_path,
                    word_timestamps=True,
                    timeout_seconds=timeout_seconds
                )

                # Validate transcription result
                if not transcription_result or not transcription_result.text.strip():
                    raise RuntimeError("Transcription returned empty result")

                # Update voice profile with results
                voice_profile.set_transcription_result(
                    transcription_text=transcription_result.text,
                    confidence=transcription_result.confidence,
                    model_name=getattr(transcription_result, 'model_name', 'whisper')
                )

                # Update statistics
                self._stats['total_processed'] += 1
                processing_time = (datetime.now() - start_time).total_seconds()

                self.logger.info(f"Successfully transcribed {voice_profile.name} in {processing_time:.1f}s")

                # Notify completion callback
                if self._completion_callback:
                    try:
                        self._completion_callback(voice_profile, True, "Transcription completed successfully")
                    except Exception as e:
                        self.logger.error(f"Error in completion callback: {e}")

            except Exception as e:
                # Enhanced error handling with categorization
                error_message = str(e)
                error_category = self._categorize_error(e)
                processing_time = (datetime.now() - start_time).total_seconds()

                self.logger.error(f"Transcription failed for {voice_profile.name} after {processing_time:.1f}s: {error_message}")

                # Determine if error is retryable based on category
                is_retryable = self._is_error_retryable(error_category, e)

                # Check if we should retry
                if is_retryable and queue_item.retry_count < queue_item.max_retries:
                    queue_item.retry_count += 1

                    # Calculate retry delay with exponential backoff, adjusted by error type
                    base_delay = self.retry_delay_base
                    if error_category in ['network', 'temporary']:
                        # Shorter delay for network/temporary issues
                        base_delay = max(2.0, self.retry_delay_base / 2)
                    elif error_category == 'resource':
                        # Longer delay for resource issues (memory, GPU)
                        base_delay = self.retry_delay_base * 2

                    delay = base_delay * (2 ** (queue_item.retry_count - 1))
                    # Cap maximum delay at 5 minutes
                    delay = min(delay, 300.0)

                    self.logger.warning(
                        f"Transcription failed for {voice_profile.name} (attempt {queue_item.retry_count}, category: {error_category}), "
                        f"retrying in {delay:.1f}s: {error_message}"
                    )

                    # Update voice profile status for retry
                    voice_profile.update_transcription_status(TranscriptionStatus.QUEUED)

                    # Re-queue with delay
                    asyncio.create_task(self._retry_transcription(queue_item, delay))

                else:
                    # Max retries exceeded or non-retryable error, mark as failed
                    if not is_retryable:
                        failure_reason = f"Non-retryable error ({error_category}): {error_message}"
                    else:
                        failure_reason = f"Max retries exceeded: {error_message}"

                    voice_profile.mark_transcription_failed(failure_reason)
                    self._stats['total_failed'] += 1

                    self.logger.error(f"Transcription failed permanently for {voice_profile.name}: {failure_reason}")

                    # Notify completion callback
                    if self._completion_callback:
                        try:
                            self._completion_callback(voice_profile, False, f"Transcription failed: {failure_reason}")
                        except Exception as e:
                            self.logger.error(f"Error in completion callback: {e}")

            finally:
                self._stats['active_transcriptions'] -= 1

    def _categorize_error(self, error: Exception) -> str:
        """
        Categorize an error for better retry logic.

        Args:
            error: Exception to categorize

        Returns:
            str: Error category ('permanent', 'temporary', 'resource', 'network', 'file')
        """
        error_str = str(error).lower()

        # Permanent errors - don't retry
        if isinstance(error, (FileNotFoundError, PermissionError)):
            return 'permanent'
        if any(keyword in error_str for keyword in ['corrupt', 'invalid format', 'unsupported', 'not found']):
            return 'permanent'

        # File access errors
        if any(keyword in error_str for keyword in ['permission', 'access denied', 'locked']):
            return 'file'

        # Resource errors - retry with longer delay
        if any(keyword in error_str for keyword in ['memory', 'out of memory', 'cuda', 'gpu']):
            return 'resource'

        # Network/temporary errors - retry quickly
        if any(keyword in error_str for keyword in ['network', 'connection', 'timeout', 'download']):
            return 'network'

        # Timeout errors
        if 'timeout' in error_str or isinstance(error, asyncio.TimeoutError):
            return 'temporary'

        # Default: treat as temporary (retryable)
        return 'temporary'

    def _is_error_retryable(self, error_category: str, error: Exception) -> bool:
        """
        Determine if an error should be retried.

        Args:
            error_category: Error category from _categorize_error
            error: Original exception

        Returns:
            bool: True if error should be retried
        """
        # Never retry permanent errors
        if error_category == 'permanent':
            return False

        # Retry most other error types
        if error_category in ['temporary', 'network', 'resource']:
            return True

        # File errors: only retry if it might be a temporary lock
        if error_category == 'file':
            error_str = str(error).lower()
            return 'locked' in error_str or 'busy' in error_str

        # Default: retry unknown errors
        return True

    async def _retry_transcription(self, queue_item: TranscriptionQueueItem, delay: float) -> None:
        """
        Retry a failed transcription after delay.

        Args:
            queue_item: Queue item to retry
            delay: Delay in seconds before retrying
        """
        try:
            await asyncio.sleep(delay)
            await self._queue.put(queue_item)
            self.logger.debug(f"Re-queued transcription for {queue_item.voice_profile.name}")
        except Exception as e:
            self.logger.error(f"Failed to re-queue transcription: {e}")

    async def _cleanup_completed_tasks(self) -> None:
        """Clean up completed processing tasks."""
        completed_tasks = [
            task_id for task_id, task in self._processing_tasks.items()
            if task.done()
        ]

        for task_id in completed_tasks:
            task = self._processing_tasks.pop(task_id)
            if task.exception():
                self.logger.error(f"Task {task_id} completed with exception: {task.exception()}")

    def set_progress_callback(self, callback: Callable[[VoiceProfile, str], None]) -> None:
        """
        Set callback for transcription progress updates.

        Args:
            callback: Function to call with (voice_profile, progress_message)
        """
        self._progress_callback = callback

    def set_completion_callback(self, callback: Callable[[VoiceProfile, bool, str], None]) -> None:
        """
        Set callback for transcription completion.

        Args:
            callback: Function to call with (voice_profile, success, message)
        """
        self._completion_callback = callback

    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get current queue statistics.

        Returns:
            dict: Queue statistics including counts and active tasks
        """
        # Calculate active transcriptions based on semaphore usage
        active_transcriptions = 0
        if self._processing_semaphore:
            # Number of active transcriptions is max_concurrent - available semaphore slots
            active_transcriptions = self.max_concurrent_transcriptions - self._processing_semaphore._value

        return {
            **self._stats,
            'current_queue_size': self._queue.qsize(),
            'active_transcriptions': active_transcriptions,
            'service_status': self.status.value
        }

    async def clear_queue(self) -> int:
        """
        Clear all pending items from the queue.

        Returns:
            int: Number of items removed from queue
        """
        cleared_count = 0

        while not self._queue.empty():
            try:
                queue_item = self._queue.get_nowait()
                queue_item.voice_profile.update_transcription_status(TranscriptionStatus.NOT_STARTED)
                self._queue.task_done()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break

        self._stats['current_queue_size'] = 0
        self.logger.info(f"Cleared {cleared_count} items from transcription queue")

        return cleared_count

    async def get_queue_items(self) -> List[VoiceProfile]:
        """
        Get list of voice profiles currently in queue.

        Returns:
            List[VoiceProfile]: Voice profiles waiting for transcription
        """
        # Note: This is a read-only operation that doesn't modify the queue
        # In practice, this would need more sophisticated implementation
        # to avoid blocking the queue or use a separate tracking mechanism
        items = []
        temp_items = []

        # Temporarily drain queue to inspect items
        try:
            while not self._queue.empty():
                item = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                temp_items.append(item)
                items.append(item.voice_profile)
        except asyncio.TimeoutError:
            pass

        # Put items back in queue
        for item in temp_items:
            await self._queue.put(item)

        return items

    # BaseService abstract method implementations
    async def start(self) -> bool:
        """Start the service (delegates to start_service)."""
        await self.start_service()
        return self.status == ServiceStatus.RUNNING

    async def stop(self) -> None:
        """Stop the service (delegates to stop_service)."""
        await self.stop_service()

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the transcription queue service.

        Returns:
            dict: Health status information
        """
        try:
            health_info = {
                "status": self.status.value,
                "whisper_service_available": self.whisper_service is not None,
                "queue_size": self._queue.qsize() if self._queue else 0,
                "active_transcriptions": self.max_concurrent_transcriptions - self._processing_semaphore._value if self._processing_semaphore else 0,
                "auto_processing_enabled": self.enable_auto_processing,
                "max_concurrent_transcriptions": self.max_concurrent_transcriptions,
                "max_queue_size": self.max_queue_size
            }

            if self.whisper_service:
                try:
                    # Test whisper service health if available
                    whisper_health = await self.whisper_service.health_check()
                    health_info["whisper_service_health"] = whisper_health
                except Exception as e:
                    health_info["whisper_service_health"] = {"error": str(e)}

            return health_info

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }