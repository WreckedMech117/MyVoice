"""
Base Service Interface

This module provides the base service class and interfaces for the MyVoice
service-oriented architecture.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from enum import Enum

from myvoice.models.error import MyVoiceError, ErrorSeverity
from myvoice.models.validation import UserFriendlyMessage
from myvoice.models.service_enums import ServiceStatus


class BaseService(ABC):
    """
    Base class for all MyVoice services.

    This provides common functionality for service management, health checking,
    error handling, and logging.

    Attributes:
        service_name: Name of the service
        status: Current service status
        logger: Service logger instance
    """

    def __init__(self, service_name: str):
        """
        Initialize the base service.

        Args:
            service_name: Name of the service for logging and identification
        """
        self.service_name = service_name
        self.status = ServiceStatus.STOPPED
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._startup_time: Optional[float] = None
        self._shutdown_time: Optional[float] = None
        self._health_check_interval: float = 30.0  # seconds

        self.logger.debug(f"Service '{service_name}' initialized")

    @abstractmethod
    async def start(self) -> bool:
        """
        Start the service.

        Returns:
            bool: True if service started successfully, False otherwise
        """
        pass

    @abstractmethod
    async def stop(self) -> bool:
        """
        Stop the service.

        Returns:
            bool: True if service stopped successfully, False otherwise
        """
        pass

    @abstractmethod
    async def health_check(self) -> tuple[bool, Optional[MyVoiceError]]:
        """
        Check service health.

        Returns:
            tuple[bool, Optional[MyVoiceError]]: (is_healthy, error_if_any)
        """
        pass

    async def restart(self) -> bool:
        """
        Restart the service.

        Returns:
            bool: True if restart successful, False otherwise
        """
        self.logger.info(f"Restarting service '{self.service_name}'")

        try:
            # Stop the service
            if not await self.stop():
                self.logger.error("Failed to stop service during restart")
                return False

            # Wait a moment for cleanup
            await asyncio.sleep(0.5)

            # Start the service
            if not await self.start():
                self.logger.error("Failed to start service during restart")
                return False

            self.logger.info(f"Service '{self.service_name}' restarted successfully")
            return True

        except Exception as e:
            self.logger.exception(f"Error during service restart: {e}")
            self.status = ServiceStatus.ERROR
            return False

    def is_running(self) -> bool:
        """Check if the service is currently running."""
        return self.status == ServiceStatus.RUNNING

    def is_healthy(self) -> bool:
        """Check if the service is in a healthy state."""
        return self.status in [ServiceStatus.RUNNING, ServiceStatus.STARTING]

    def get_status_info(self) -> Dict[str, Any]:
        """
        Get comprehensive status information.

        Returns:
            Dict[str, Any]: Status information dictionary
        """
        return {
            "service_name": self.service_name,
            "status": self.status.value,
            "is_running": self.is_running(),
            "is_healthy": self.is_healthy(),
            "startup_time": self._startup_time,
            "shutdown_time": self._shutdown_time,
        }

    def handle_service_error(self, error: Exception, context: str = "") -> UserFriendlyMessage:
        """
        Convert service errors to user-friendly messages.

        Args:
            error: The exception that occurred
            context: Additional context about when/where the error occurred

        Returns:
            UserFriendlyMessage: User-friendly error message
        """
        self.logger.error(f"Service error in {context}: {error}")

        # Default error handling - can be overridden by specific services
        if "connection" in str(error).lower():
            return UserFriendlyMessage(
                title="Connection Error",
                message="Unable to connect to the required service",
                details=str(error) if context else None,
                action_suggestions=[
                    "Check your internet connection",
                    "Verify service is running",
                    "Try again in a moment"
                ],
                severity="error",
                is_recoverable=True
            )
        elif "timeout" in str(error).lower():
            return UserFriendlyMessage(
                title="Service Timeout",
                message="The operation took too long to complete",
                details=str(error) if context else None,
                action_suggestions=[
                    "Try again with a smaller request",
                    "Check your internet connection",
                    "Contact support if problem persists"
                ],
                severity="warning",
                is_recoverable=True
            )
        elif "permission" in str(error).lower() or "access" in str(error).lower():
            return UserFriendlyMessage(
                title="Access Error",
                message="Permission denied or insufficient access rights",
                details=str(error) if context else None,
                action_suggestions=[
                    "Check file permissions",
                    "Run as administrator if needed",
                    "Verify API credentials"
                ],
                severity="error",
                is_recoverable=False
            )
        else:
            return UserFriendlyMessage(
                title="Service Error",
                message="An unexpected error occurred",
                details=str(error) if context else None,
                action_suggestions=[
                    "Try the operation again",
                    "Restart the application if problem persists",
                    "Contact support with error details"
                ],
                severity="error",
                is_recoverable=True
            )

    async def _update_status(self, new_status: ServiceStatus):
        """
        Update service status with logging.

        Args:
            new_status: New status to set
        """
        old_status = self.status
        self.status = new_status

        if old_status != new_status:
            self.logger.info(f"Service '{self.service_name}' status changed: {old_status.value} -> {new_status.value}")

            # Track timing
            import time
            if new_status == ServiceStatus.RUNNING:
                self._startup_time = time.time()
            elif new_status == ServiceStatus.STOPPED:
                self._shutdown_time = time.time()

    def __repr__(self) -> str:
        """String representation of the service."""
        return f"{self.__class__.__name__}(name={self.service_name}, status={self.status.value})"