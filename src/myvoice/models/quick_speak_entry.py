"""
Quick Speak Entry Model

This module contains the QuickSpeakEntry data model for managing
preset text responses in the Quick Speak feature.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional
from uuid import uuid4


logger = logging.getLogger(__name__)


@dataclass
class QuickSpeakEntry:
    """
    Quick Speak entry model for preset text responses.

    Attributes:
        text: The text content to be spoken
        label: Optional label/name for the entry
        id: Unique identifier for the entry
    """

    text: str
    label: Optional[str] = None
    id: str = None

    def __post_init__(self):
        """Initialize entry with unique ID if not provided."""
        if self.id is None:
            self.id = str(uuid4())

        # Validate entry
        if not self.text or not self.text.strip():
            raise ValueError("Quick Speak entry text cannot be empty")

    def get_display_text(self) -> str:
        """
        Get display text for the entry.

        Returns:
            Label if available, otherwise truncated text
        """
        if self.label and self.label.strip():
            return self.label.strip()

        # Return truncated text as fallback
        max_length = 50
        text = self.text.strip()
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text

    def to_dict(self) -> dict:
        """
        Convert entry to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            'id': self.id,
            'text': self.text,
            'label': self.label
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'QuickSpeakEntry':
        """
        Create entry from dictionary.

        Args:
            data: Dictionary with entry data

        Returns:
            QuickSpeakEntry instance
        """
        return cls(
            id=data.get('id'),
            text=data.get('text', ''),
            label=data.get('label')
        )

    def __str__(self) -> str:
        """String representation of the entry."""
        return f"QuickSpeakEntry(label='{self.label}', text='{self.text[:30]}...')"