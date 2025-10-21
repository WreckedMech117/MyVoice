"""
Background Transcription Manager

This module provides a high-level manager for coordinating background transcription
processing with caching, user notifications, and UI integration. It acts as a
facade over the transcription queue and cache services.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from pathlib import Path

from myvoice.services.core.base_service import BaseService
from myvoice.models.service_enums import ServiceStatus
from myvoice.models.voice_profile import VoiceProfile, TranscriptionStatus
from myvoice.models.error import MyVoiceError, ErrorSeverity
from myvoice.services.transcription_queue_service import TranscriptionQueueService, QueuePriority
from myvoice.services.transcription_cache_service import TranscriptionCacheService


class NotificationType(Enum):
    """Types of transcription notifications."""
    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BATCH_COMPLETED = "batch_completed"


@dataclass
class TranscriptionNotification:
    """Notification about transcription events."""
    type: NotificationType
    voice_profile: VoiceProfile
    message: str
    timestamp: datetime
    progress_percent: Optional[float] = None
    batch_info: Optional[Dict[str, Any]] = None


class BackgroundTranscriptionManager(BaseService):
    """
    High-level manager for background transcription processing.

    This service coordinates transcription queue operations with caching,
    provides user notifications, and integrates with UI components for
    seamless background processing.

    Features:
    - Automatic cache checking before queuing transcriptions
    - Batch transcription operations with progress tracking
    - User notification system for transcription events
    - Smart retry logic with exponential backoff
    - Integration hooks for UI components
    """

    def __init__(
        self,
        whisper_service: Optional[Any] = None,
        max_concurrent_transcriptions: int = 2,
        enable_notifications: bool = True,
        auto_save_cache: bool = True
    ):
        """
        Initialize the background transcription manager.

        Args:
            whisper_service: WhisperService instance
            max_concurrent_transcriptions: Maximum concurrent transcription tasks
            enable_notifications: Whether to emit user notifications
            auto_save_cache: Whether to automatically save cache
        """
        super().__init__("BackgroundTranscriptionManager")

        # Initialize sub-services
        self.cache_service = TranscriptionCacheService()
        self.queue_service = TranscriptionQueueService(
            whisper_service=whisper_service,
            max_concurrent_transcriptions=max_concurrent_transcriptions
        )

        self.enable_notifications = enable_notifications
        self.auto_save_cache = auto_save_cache

        # Notification callbacks
        self._notification_callbacks: List[Callable[[TranscriptionNotification], None]] = []

        # Batch tracking
        self._active_batches: Dict[str, Dict[str, Any]] = {}
        self._batch_counter = 0

        # Statistics
        self._stats = {
            'total_requested': 0,
            'cache_hits': 0,
            'queue_submissions': 0,
            'completions': 0,
            'failures': 0,
            'notifications_sent': 0
        }

    async def start_service(self) -> None:
        """Start the background transcription manager."""
        try:
            self.status = ServiceStatus.STARTING
            self.logger.info("Starting background transcription manager")

            # Start sub-services
            await self.cache_service.start_service()
            await self.queue_service.start_service()

            # Set up queue callbacks
            self.queue_service.set_progress_callback(self._on_transcription_progress)
            self.queue_service.set_completion_callback(self._on_transcription_completion)

            self.status = ServiceStatus.RUNNING
            self.logger.info("Background transcription manager started successfully")

        except Exception as e:
            self.status = ServiceStatus.ERROR
            error = MyVoiceError(
                severity=ErrorSeverity.CRITICAL,
                code="BACKGROUND_TRANSCRIPTION_START_FAILED",
                user_message="Failed to start background transcription manager",
                technical_details=str(e),
                suggested_action="Check sub-service dependencies and try again"
            )
            self.logger.error(f"Failed to start background transcription manager: {e}")
            raise error

    async def stop_service(self) -> None:
        """Stop the background transcription manager."""
        try:
            self.status = ServiceStatus.STOPPING
            self.logger.info("Stopping background transcription manager")

            # Stop sub-services
            await self.queue_service.stop_service()
            await self.cache_service.stop_service()

            self.status = ServiceStatus.STOPPED
            self.logger.info("Background transcription manager stopped successfully")

        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.logger.error(f"Error stopping background transcription manager: {e}")
            raise

    async def request_transcription(
        self,
        voice_profile: VoiceProfile,
        priority: QueuePriority = QueuePriority.NORMAL,
        force_regenerate: bool = False
    ) -> bool:
        """
        Request transcription for a voice profile with intelligent caching.

        Args:
            voice_profile: VoiceProfile to transcribe
            priority: Priority level for the transcription
            force_regenerate: Whether to bypass cache and regenerate

        Returns:
            bool: True if transcription was initiated or cached result used
        """
        try:
            self._stats['total_requested'] += 1

            # Check cache first (unless forcing regeneration)
            if not force_regenerate:
                cached_result = await self.cache_service.get_cached_transcription(voice_profile.file_path)
                if cached_result:
                    # Use cached result
                    voice_profile.set_transcription_result(
                        transcription_text=cached_result.transcription_text,
                        confidence=cached_result.confidence,
                        model_name=cached_result.model_name
                    )

                    self._stats['cache_hits'] += 1
                    self.logger.info(f"Used cached transcription for {voice_profile.name}")

                    # Send completion notification
                    if self.enable_notifications:
                        await self._emit_notification(TranscriptionNotification(
                            type=NotificationType.COMPLETED,
                            voice_profile=voice_profile,
                            message=f"Transcription loaded from cache",
                            timestamp=datetime.now()
                        ))

                    return True

            # Queue for transcription
            success = await self.queue_service.queue_transcription(voice_profile, priority)
            if success:
                self._stats['queue_submissions'] += 1

                # Send started notification
                if self.enable_notifications:
                    await self._emit_notification(TranscriptionNotification(
                        type=NotificationType.STARTED,
                        voice_profile=voice_profile,
                        message=f"Transcription started",
                        timestamp=datetime.now()
                    ))

            return success

        except Exception as e:
            self.logger.error(f"Error requesting transcription for {voice_profile.name}: {e}")
            return False

    async def request_batch_transcription(
        self,
        voice_profiles: List[VoiceProfile],
        priority: QueuePriority = QueuePriority.NORMAL,
        force_regenerate: bool = False
    ) -> str:
        """
        Request batch transcription for multiple voice profiles.

        Args:
            voice_profiles: List of VoiceProfiles to transcribe
            priority: Priority level for the transcriptions
            force_regenerate: Whether to bypass cache and regenerate

        Returns:
            str: Batch ID for tracking progress
        """
        try:
            batch_id = f"batch_{self._batch_counter}_{datetime.now().timestamp()}"
            self._batch_counter += 1

            batch_info = {
                'id': batch_id,
                'total_profiles': len(voice_profiles),
                'completed': 0,
                'failed': 0,
                'cache_hits': 0,
                'started_at': datetime.now(),
                'profile_names': [p.name for p in voice_profiles]
            }

            self._active_batches[batch_id] = batch_info

            self.logger.info(f"Starting batch transcription {batch_id} with {len(voice_profiles)} profiles")

            # Process each profile
            for voice_profile in voice_profiles:
                # Check cache first
                if not force_regenerate:
                    cached_result = await self.cache_service.get_cached_transcription(voice_profile.file_path)
                    if cached_result:
                        voice_profile.set_transcription_result(
                            transcription_text=cached_result.transcription_text,
                            confidence=cached_result.confidence,
                            model_name=cached_result.model_name
                        )
                        batch_info['completed'] += 1
                        batch_info['cache_hits'] += 1
                        continue

                # Queue for transcription
                await self.queue_service.queue_transcription(voice_profile, priority)

            # Send batch started notification
            if self.enable_notifications:
                await self._emit_notification(TranscriptionNotification(
                    type=NotificationType.STARTED,
                    voice_profile=voice_profiles[0],  # Use first profile as representative
                    message=f"Batch transcription started ({len(voice_profiles)} files)",
                    timestamp=datetime.now(),
                    batch_info=batch_info
                ))

            return batch_id

        except Exception as e:
            self.logger.error(f"Error in batch transcription: {e}")
            raise

    async def cancel_transcription(self, voice_profile: VoiceProfile) -> bool:
        """
        Cancel a pending transcription.

        Args:
            voice_profile: VoiceProfile to cancel

        Returns:
            bool: True if cancellation was successful
        """
        try:
            # For now, we can only update the profile status
            # The queue service doesn't currently support individual cancellation
            voice_profile.update_transcription_status(TranscriptionStatus.NOT_STARTED)
            self.logger.info(f"Cancelled transcription for {voice_profile.name}")
            return True

        except Exception as e:
            self.logger.error(f"Error cancelling transcription: {e}")
            return False

    async def get_transcription_progress(self) -> Dict[str, Any]:
        """
        Get current transcription progress information.

        Returns:
            dict: Progress information including queue stats and active batches
        """
        try:
            queue_stats = self.queue_service.get_queue_stats()
            cache_stats = self.cache_service.get_cache_stats()

            return {
                'queue_stats': queue_stats,
                'cache_stats': cache_stats,
                'active_batches': len(self._active_batches),
                'batch_details': list(self._active_batches.values()),
                'manager_stats': self._stats
            }

        except Exception as e:
            self.logger.error(f"Error getting transcription progress: {e}")
            return {}

    def add_notification_callback(self, callback: Callable[[TranscriptionNotification], None]) -> None:
        """
        Add a callback for transcription notifications.

        Args:
            callback: Function to call with TranscriptionNotification
        """
        self._notification_callbacks.append(callback)

    def remove_notification_callback(self, callback: Callable[[TranscriptionNotification], None]) -> None:
        """
        Remove a notification callback.

        Args:
            callback: Callback function to remove
        """
        if callback in self._notification_callbacks:
            self._notification_callbacks.remove(callback)

    async def clear_cache(self, file_path: Optional[Path] = None) -> int:
        """
        Clear transcription cache.

        Args:
            file_path: Specific file to clear, or None for all

        Returns:
            int: Number of entries cleared
        """
        return await self.cache_service.invalidate_cache(file_path)

    async def cleanup_expired_cache(self) -> int:
        """
        Clean up expired cache entries.

        Returns:
            int: Number of entries removed
        """
        return await self.cache_service.cleanup_expired_entries()

    async def _emit_notification(self, notification: TranscriptionNotification) -> None:
        """
        Emit a transcription notification to all registered callbacks.

        Args:
            notification: Notification to emit
        """
        try:
            for callback in self._notification_callbacks:
                try:
                    callback(notification)
                except Exception as e:
                    self.logger.error(f"Error in notification callback: {e}")

            self._stats['notifications_sent'] += 1

        except Exception as e:
            self.logger.error(f"Error emitting notification: {e}")

    def _on_transcription_progress(self, voice_profile: VoiceProfile, message: str) -> None:
        """
        Handle transcription progress updates from the queue service.

        Args:
            voice_profile: VoiceProfile being processed
            message: Progress message
        """
        try:
            if self.enable_notifications:
                asyncio.create_task(self._emit_notification(TranscriptionNotification(
                    type=NotificationType.PROGRESS,
                    voice_profile=voice_profile,
                    message=message,
                    timestamp=datetime.now()
                )))

        except Exception as e:
            self.logger.error(f"Error handling transcription progress: {e}")

    def _on_transcription_completion(self, voice_profile: VoiceProfile, success: bool, message: str) -> None:
        """
        Handle transcription completion from the queue service.

        Args:
            voice_profile: VoiceProfile that was processed
            success: Whether transcription was successful
            message: Completion message
        """
        try:
            if success:
                self._stats['completions'] += 1

                # Cache the result if auto-save is enabled
                if self.auto_save_cache and voice_profile.transcription:
                    asyncio.create_task(self.cache_service.cache_transcription(
                        file_path=voice_profile.file_path,
                        transcription_text=voice_profile.transcription,
                        confidence=voice_profile.transcription_confidence,
                        model_name=getattr(voice_profile, 'transcription_model', 'whisper'),
                        language='en'  # Default to English for now
                    ))
            else:
                self._stats['failures'] += 1

            # Update batch progress if applicable
            self._update_batch_progress(voice_profile, success)

            # Send completion notification
            if self.enable_notifications:
                notification_type = NotificationType.COMPLETED if success else NotificationType.FAILED
                asyncio.create_task(self._emit_notification(TranscriptionNotification(
                    type=notification_type,
                    voice_profile=voice_profile,
                    message=message,
                    timestamp=datetime.now()
                )))

        except Exception as e:
            self.logger.error(f"Error handling transcription completion: {e}")

    def _update_batch_progress(self, voice_profile: VoiceProfile, success: bool) -> None:
        """
        Update progress for any active batches containing this profile.

        Args:
            voice_profile: VoiceProfile that was completed
            success: Whether transcription was successful
        """
        try:
            profile_name = voice_profile.name

            for batch_id, batch_info in self._active_batches.items():
                if profile_name in batch_info['profile_names']:
                    if success:
                        batch_info['completed'] += 1
                    else:
                        batch_info['failed'] += 1

                    # Check if batch is complete
                    total_processed = batch_info['completed'] + batch_info['failed']
                    if total_processed >= batch_info['total_profiles']:
                        # Batch is complete
                        batch_info['completed_at'] = datetime.now()

                        if self.enable_notifications:
                            asyncio.create_task(self._emit_notification(TranscriptionNotification(
                                type=NotificationType.BATCH_COMPLETED,
                                voice_profile=voice_profile,
                                message=f"Batch completed: {batch_info['completed']} successful, {batch_info['failed']} failed",
                                timestamp=datetime.now(),
                                batch_info=batch_info
                            )))

                        # Clean up completed batch
                        del self._active_batches[batch_id]
                        break

        except Exception as e:
            self.logger.error(f"Error updating batch progress: {e}")

    # BaseService abstract method implementations
    async def start(self) -> bool:
        """Start the service."""
        await self.start_service()
        return self.status == ServiceStatus.RUNNING

    async def stop(self) -> None:
        """Stop the service."""
        await self.stop_service()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            cache_health = await self.cache_service.health_check()
            queue_health = await self.queue_service.health_check()

            return {
                "status": self.status.value,
                "cache_service": cache_health,
                "queue_service": queue_health,
                "active_batches": len(self._active_batches),
                "notification_callbacks": len(self._notification_callbacks),
                "stats": self._stats
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }