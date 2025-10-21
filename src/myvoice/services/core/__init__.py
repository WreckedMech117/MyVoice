"""
MyVoice Service Core Package

This package contains core service infrastructure and base classes.
"""

from .base_service import BaseService, ServiceStatus

__all__ = ["BaseService", "ServiceStatus"]