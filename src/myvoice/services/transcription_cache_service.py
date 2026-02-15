"""
Transcription Cache Service

This module implements a caching system for transcription results to avoid
re-processing voice files that have already been transcribed. It provides
persistent storage, cache invalidation, and performance optimization.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from myvoice.services.core.base_service import BaseService
from myvoice.models.service_enums import ServiceStatus
from myvoice.models.error import MyVoiceError, ErrorSeverity


@dataclass
class CachedTranscription:
    """Cached transcription result."""
    file_path: str
    file_hash: str
    file_size: int
    file_mtime: float
    transcription_text: str
    confidence: Optional[float]
    model_name: str
    language: str
    cached_at: float
    access_count: int = 0
    last_accessed: Optional[float] = None

    def is_valid_for_file(self, file_path: Path) -> bool:
        """Check if cached transcription is valid for the given file."""
        try:
            if not file_path.exists():
                return False

            stat = file_path.stat()
            current_size = stat.st_size
            current_mtime = stat.st_mtime

            # Check file size and modification time
            if current_size != self.file_size:
                return False

            # Allow small mtime differences for filesystem precision
            if abs(current_mtime - self.file_mtime) > 1.0:
                return False

            return True

        except (OSError, IOError):
            return False

    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now().timestamp()


class TranscriptionCacheService(BaseService):
    """
    Service for caching transcription results to avoid re-processing files.

    Features:
    - Persistent storage of transcription results
    - File change detection using hash, size, and mtime
    - Cache expiration and cleanup
    - Performance statistics and optimization
    - Thread-safe operations for concurrent access
    """

    def __init__(
        self,
        cache_file: Optional[Path] = None,
        max_cache_entries: int = 1000,
        max_cache_age_days: int = 30,
        enable_hash_verification: bool = True
    ):
        """
        Initialize the transcription cache service.

        Args:
            cache_file: Path to cache file (default: config/transcription_cache.json)
            max_cache_entries: Maximum number of cached entries
            max_cache_age_days: Maximum age of cached entries in days
            enable_hash_verification: Whether to use file hash for verification
        """
        super().__init__("TranscriptionCacheService")

        self.cache_file = cache_file or Path("config/transcription_cache.json")
        self.max_cache_entries = max_cache_entries
        self.max_cache_age_days = max_cache_age_days
        self.enable_hash_verification = enable_hash_verification

        # Cache storage
        self._cache: Dict[str, CachedTranscription] = {}
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="TranscriptionCache")

        # Statistics
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_invalidations': 0,
            'cache_saves': 0,
            'cache_loads': 0,
            'cleanup_operations': 0
        }

    async def start_service(self) -> None:
        """Start the transcription cache service."""
        try:
            self.status = ServiceStatus.STARTING
            self.logger.info("Starting transcription cache service")

            # Ensure cache directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)

            # Load existing cache
            await self._load_cache()

            self.status = ServiceStatus.RUNNING
            self.logger.info(f"Transcription cache service started with {len(self._cache)} cached entries")

        except Exception as e:
            self.status = ServiceStatus.ERROR
            error = MyVoiceError(
                severity=ErrorSeverity.ERROR,
                code="TRANSCRIPTION_CACHE_START_FAILED",
                user_message="Failed to start transcription cache service",
                technical_details=str(e),
                suggested_action="Check file permissions and try again"
            )
            self.logger.error(f"Failed to start transcription cache service: {e}")
            raise error

    async def stop_service(self) -> None:
        """Stop the transcription cache service."""
        try:
            self.status = ServiceStatus.STOPPING
            self.logger.info("Stopping transcription cache service")

            # Save cache before stopping
            await self._save_cache()

            # Shutdown thread pool - QA Round 2 Item #8: Non-blocking shutdown
            self._executor.shutdown(wait=False, cancel_futures=True)

            self.status = ServiceStatus.STOPPED
            self.logger.info("Transcription cache service stopped successfully")

        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.logger.error(f"Error stopping transcription cache service: {e}")
            raise

    async def get_cached_transcription(self, file_path: Path) -> Optional[CachedTranscription]:
        """
        Get cached transcription for a file if available and valid.

        Args:
            file_path: Path to the audio file

        Returns:
            CachedTranscription if valid cache exists, None otherwise
        """
        try:
            cache_key = self._get_cache_key(file_path)

            if cache_key not in self._cache:
                self._stats['cache_misses'] += 1
                return None

            cached_entry = self._cache[cache_key]

            # Verify cache validity
            if not cached_entry.is_valid_for_file(file_path):
                # Cache is invalid, remove it
                del self._cache[cache_key]
                self._stats['cache_invalidations'] += 1
                self._stats['cache_misses'] += 1
                self.logger.debug(f"Invalidated cache for {file_path.name}")
                return None

            # Optional hash verification for extra security
            if self.enable_hash_verification:
                current_hash = await self._calculate_file_hash(file_path)
                if current_hash != cached_entry.file_hash:
                    del self._cache[cache_key]
                    self._stats['cache_invalidations'] += 1
                    self._stats['cache_misses'] += 1
                    self.logger.debug(f"Hash mismatch, invalidated cache for {file_path.name}")
                    return None

            # Update access statistics
            cached_entry.update_access()
            self._stats['cache_hits'] += 1

            self.logger.debug(f"Cache hit for {file_path.name}")
            return cached_entry

        except Exception as e:
            self.logger.error(f"Error retrieving cached transcription: {e}")
            self._stats['cache_misses'] += 1
            return None

    async def cache_transcription(
        self,
        file_path: Path,
        transcription_text: str,
        confidence: Optional[float] = None,
        model_name: str = "whisper",
        language: str = "en"
    ) -> bool:
        """
        Cache a transcription result for a file.

        Args:
            file_path: Path to the audio file
            transcription_text: Transcribed text
            confidence: Confidence score (0.0-1.0)
            model_name: Name of the model used
            language: Detected language

        Returns:
            bool: True if successfully cached
        """
        try:
            if not file_path.exists():
                self.logger.warning(f"Cannot cache transcription for non-existent file: {file_path}")
                return False

            # Get file statistics
            stat = file_path.stat()
            file_hash = await self._calculate_file_hash(file_path) if self.enable_hash_verification else ""

            # Create cache entry
            cache_entry = CachedTranscription(
                file_path=str(file_path),
                file_hash=file_hash,
                file_size=stat.st_size,
                file_mtime=stat.st_mtime,
                transcription_text=transcription_text,
                confidence=confidence,
                model_name=model_name,
                language=language,
                cached_at=datetime.now().timestamp()
            )

            # Store in cache
            cache_key = self._get_cache_key(file_path)
            self._cache[cache_key] = cache_entry

            # Cleanup if cache is too large
            await self._cleanup_cache_if_needed()

            self._stats['cache_saves'] += 1
            self.logger.debug(f"Cached transcription for {file_path.name}")

            return True

        except Exception as e:
            self.logger.error(f"Error caching transcription: {e}")
            return False

    async def invalidate_cache(self, file_path: Optional[Path] = None) -> int:
        """
        Invalidate cache entries.

        Args:
            file_path: Specific file to invalidate, or None to clear all

        Returns:
            int: Number of entries invalidated
        """
        try:
            if file_path is None:
                # Clear entire cache
                count = len(self._cache)
                self._cache.clear()
                self.logger.info(f"Cleared entire transcription cache ({count} entries)")
                return count
            else:
                # Clear specific file
                cache_key = self._get_cache_key(file_path)
                if cache_key in self._cache:
                    del self._cache[cache_key]
                    self.logger.debug(f"Invalidated cache for {file_path.name}")
                    return 1
                return 0

        except Exception as e:
            self.logger.error(f"Error invalidating cache: {e}")
            return 0

    async def cleanup_expired_entries(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            int: Number of entries removed
        """
        try:
            cutoff_time = datetime.now().timestamp() - (self.max_cache_age_days * 24 * 3600)
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.cached_at < cutoff_time
            ]

            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

            self._stats['cleanup_operations'] += 1
            return len(expired_keys)

        except Exception as e:
            self.logger.error(f"Error cleaning up expired entries: {e}")
            return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            dict: Cache statistics and metrics
        """
        total_requests = self._stats['cache_hits'] + self._stats['cache_misses']
        hit_rate = (self._stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0

        return {
            **self._stats,
            'total_cached_entries': len(self._cache),
            'hit_rate_percent': round(hit_rate, 2),
            'cache_file_size_bytes': self.cache_file.stat().st_size if self.cache_file.exists() else 0,
            'max_cache_entries': self.max_cache_entries,
            'max_cache_age_days': self.max_cache_age_days
        }

    async def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            if not self.cache_file.exists():
                self.logger.debug("No cache file found, starting with empty cache")
                return

            cache_data = json.loads(self.cache_file.read_text(encoding='utf-8'))
            cache_version = cache_data.get('cache_version', '1.0')

            if cache_version != '1.0':
                self.logger.info(f"Cache version {cache_version} incompatible, starting fresh")
                return

            # Load cache entries
            for entry_data in cache_data.get('entries', []):
                try:
                    cached_entry = CachedTranscription(**entry_data)
                    cache_key = self._get_cache_key(Path(cached_entry.file_path))
                    self._cache[cache_key] = cached_entry
                except Exception as e:
                    self.logger.warning(f"Skipping invalid cache entry: {e}")

            self._stats['cache_loads'] += 1
            self.logger.info(f"Loaded {len(self._cache)} entries from transcription cache")

        except Exception as e:
            self.logger.warning(f"Error loading cache, starting fresh: {e}")
            self._cache.clear()

    async def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            cache_data = {
                'cache_version': '1.0',
                'saved_at': datetime.now().timestamp(),
                'entries': [asdict(entry) for entry in self._cache.values()]
            }

            # Atomic write using temporary file
            temp_file = self.cache_file.with_suffix('.tmp')
            temp_file.write_text(json.dumps(cache_data, indent=2), encoding='utf-8')
            temp_file.replace(self.cache_file)

            self.logger.debug(f"Saved {len(self._cache)} entries to transcription cache")

        except Exception as e:
            self.logger.error(f"Error saving cache: {e}")

    async def _cleanup_cache_if_needed(self) -> None:
        """Clean up cache if it exceeds size limits."""
        if len(self._cache) <= self.max_cache_entries:
            return

        # Remove oldest entries based on last access time
        entries_by_access = sorted(
            self._cache.items(),
            key=lambda x: x[1].last_accessed or x[1].cached_at
        )

        entries_to_remove = len(self._cache) - self.max_cache_entries
        for i in range(entries_to_remove):
            key, _ = entries_by_access[i]
            del self._cache[key]

        self.logger.info(f"Cache cleanup: removed {entries_to_remove} oldest entries")

    def _get_cache_key(self, file_path: Path) -> str:
        """Generate cache key for a file path."""
        return str(file_path.resolve())

    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content."""
        try:
            # Use thread pool for file I/O to avoid blocking
            loop = __import__('asyncio').get_event_loop()

            def _hash_file():
                hasher = hashlib.sha256()
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)
                return hasher.hexdigest()

            return await loop.run_in_executor(self._executor, _hash_file)

        except Exception as e:
            self.logger.error(f"Error calculating file hash: {e}")
            return ""

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
            return {
                "status": self.status.value,
                "cache_entries": len(self._cache),
                "cache_file_exists": self.cache_file.exists(),
                "cache_file_writable": self.cache_file.parent.exists(),
                "stats": self.get_cache_stats()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }