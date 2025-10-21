"""
Service Status Enumerations

This module contains service status enums to avoid circular imports.
"""

from enum import Enum


class ServiceStatus(Enum):
    """Service status enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"