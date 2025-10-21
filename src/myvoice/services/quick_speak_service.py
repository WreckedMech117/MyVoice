"""
Quick Speak Service

This module manages Quick Speak entries including CSV persistence,
loading, and saving of preset text responses.
"""

import logging
import csv
from pathlib import Path
from typing import List, Optional

from myvoice.models.quick_speak_entry import QuickSpeakEntry


logger = logging.getLogger(__name__)


class QuickSpeakService:
    """
    Service for managing Quick Speak entries with CSV persistence.

    Handles loading, saving, and managing preset text entries that users
    can quickly select for TTS generation.
    """

    def __init__(self, config_directory: str = "config"):
        """
        Initialize Quick Speak service.

        Args:
            config_directory: Directory path for configuration files
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_directory = Path(config_directory)
        self.profiles_directory = self.config_directory / "quickspeak_profiles"
        self.current_profile = "default"
        self.csv_file_path = self.profiles_directory / f"{self.current_profile}.csv"

        # Ensure directories exist
        self.config_directory.mkdir(parents=True, exist_ok=True)
        self.profiles_directory.mkdir(parents=True, exist_ok=True)

        # Create default profile if it doesn't exist
        if not self.csv_file_path.exists():
            self._create_default_profile()

        # In-memory cache of entries
        self._entries: List[QuickSpeakEntry] = []

        self.logger.info(f"QuickSpeakService initialized with profile: {self.current_profile}")

    def load_entries(self) -> List[QuickSpeakEntry]:
        """
        Load Quick Speak entries from CSV file.

        Returns:
            List of QuickSpeakEntry objects
        """
        self._entries = []

        if not self.csv_file_path.exists():
            self.logger.info("Quick Speak CSV file does not exist, starting with empty list")
            return self._entries

        try:
            with open(self.csv_file_path, 'r', encoding='utf-8', newline='') as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:
                    try:
                        # Create entry from CSV row
                        entry = QuickSpeakEntry(
                            id=row.get('id'),
                            text=row.get('text', ''),
                            label=row.get('label')
                        )
                        self._entries.append(entry)
                    except ValueError as e:
                        self.logger.warning(f"Skipping invalid entry: {e}")
                        continue

            self.logger.info(f"Loaded {len(self._entries)} Quick Speak entries from CSV")

        except Exception as e:
            self.logger.error(f"Error loading Quick Speak entries: {e}")

        return self._entries

    def save_entries(self, entries: List[QuickSpeakEntry]) -> bool:
        """
        Save Quick Speak entries to CSV file.

        Args:
            entries: List of entries to save

        Returns:
            True if save successful, False otherwise
        """
        try:
            # Ensure config directory exists
            self.config_directory.mkdir(parents=True, exist_ok=True)

            # Write to CSV file
            with open(self.csv_file_path, 'w', encoding='utf-8', newline='') as csvfile:
                fieldnames = ['id', 'label', 'text']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for entry in entries:
                    writer.writerow({
                        'id': entry.id,
                        'label': entry.label or '',
                        'text': entry.text
                    })

            # Update cache
            self._entries = entries.copy()

            self.logger.info(f"Saved {len(entries)} Quick Speak entries to CSV")
            return True

        except Exception as e:
            self.logger.error(f"Error saving Quick Speak entries: {e}")
            return False

    def get_entries(self) -> List[QuickSpeakEntry]:
        """
        Get current list of Quick Speak entries.

        Returns:
            List of QuickSpeakEntry objects
        """
        return self._entries.copy()

    def add_entry(self, text: str, label: Optional[str] = None) -> QuickSpeakEntry:
        """
        Add a new Quick Speak entry.

        Args:
            text: Text content for the entry
            label: Optional label for the entry

        Returns:
            The created QuickSpeakEntry
        """
        entry = QuickSpeakEntry(text=text, label=label)
        self._entries.append(entry)
        self.save_entries(self._entries)

        self.logger.info(f"Added Quick Speak entry: {entry.get_display_text()}")
        return entry

    def update_entry(self, entry_id: str, text: str, label: Optional[str] = None) -> bool:
        """
        Update an existing Quick Speak entry.

        Args:
            entry_id: ID of entry to update
            text: New text content
            label: New label

        Returns:
            True if update successful, False if entry not found
        """
        for entry in self._entries:
            if entry.id == entry_id:
                entry.text = text
                entry.label = label
                self.save_entries(self._entries)
                self.logger.info(f"Updated Quick Speak entry: {entry_id}")
                return True

        self.logger.warning(f"Quick Speak entry not found for update: {entry_id}")
        return False

    def delete_entry(self, entry_id: str) -> bool:
        """
        Delete a Quick Speak entry.

        Args:
            entry_id: ID of entry to delete

        Returns:
            True if deletion successful, False if entry not found
        """
        for i, entry in enumerate(self._entries):
            if entry.id == entry_id:
                del self._entries[i]
                self.save_entries(self._entries)
                self.logger.info(f"Deleted Quick Speak entry: {entry_id}")
                return True

        self.logger.warning(f"Quick Speak entry not found for deletion: {entry_id}")
        return False

    def get_entry_by_id(self, entry_id: str) -> Optional[QuickSpeakEntry]:
        """
        Get a specific entry by ID.

        Args:
            entry_id: ID of the entry

        Returns:
            QuickSpeakEntry if found, None otherwise
        """
        for entry in self._entries:
            if entry.id == entry_id:
                return entry
        return None

    # Profile Management Methods

    def _create_default_profile(self) -> None:
        """Create default profile with sample entries."""
        try:
            # Ensure profile directory exists
            self.profiles_directory.mkdir(parents=True, exist_ok=True)

            default_entries = [
                {"id": "1", "label": "Greeting", "text": "Hello! How can I help you today?"},
                {"id": "2", "label": "Thanks", "text": "Thank you very much!"},
                {"id": "3", "label": "Goodbye", "text": "It was nice talking to you. Have a great day!"},
                {"id": "4", "label": "Thinking", "text": "Let me think about that for a moment..."},
                {"id": "5", "label": "Affirmative", "text": "Yes absolutely!"},
                {"id": "6", "label": "Negative", "text": "I'm sorry"},
                {"id": "7", "label": "BRB", "text": "Be right back"},
                {"id": "8", "label": "OneM\noment", "text": "One moment please"},
                {"id": "9", "label": "Laugh", "text": "Haha that's funny!"},
                {"id": "10", "label": "Agree", "text": "I agree with you"},
                {"id": "11", "label": "Disagree", "text": "I see it differently"},
                {"id": "12", "label": "GoodMorning", "text": "Good morning!"},
                {"id": "13", "label": "GoodAfternoon", "text": "Good afternoon!"},
                {"id": "14", "label": "GoodEvening", "text": "Good evening!"},
                {"id": "15", "label": "HowAreYou", "text": "How are you doing today?"},
            ]

            default_profile_csv = self.profiles_directory / "default.csv"
            with open(default_profile_csv, 'w', encoding='utf-8', newline='') as csvfile:
                fieldnames = ['id', 'label', 'text']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(default_entries)

            self.logger.info("Created default profile with sample entries")

        except Exception as e:
            self.logger.error(f"Error creating default profile: {e}")

    def get_profiles(self) -> List[str]:
        """
        Get list of available Quick Speak profiles.

        Returns:
            List of profile names (without .csv extension)
        """
        try:
            profiles = []
            for file in self.profiles_directory.glob("*.csv"):
                profiles.append(file.stem)
            return sorted(profiles) if profiles else ["default"]
        except Exception as e:
            self.logger.error(f"Error listing profiles: {e}")
            return ["default"]

    def get_current_profile(self) -> str:
        """Get the name of the current active profile."""
        return self.current_profile

    def switch_profile(self, profile_name: str) -> bool:
        """
        Switch to a different Quick Speak profile.

        Args:
            profile_name: Name of the profile to switch to

        Returns:
            True if switch successful, False otherwise
        """
        try:
            new_csv_path = self.profiles_directory / f"{profile_name}.csv"

            # Create empty profile with proper CSV headers if it doesn't exist
            if not new_csv_path.exists():
                with open(new_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
                    fieldnames = ['id', 'label', 'text']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                self.logger.info(f"Created new profile with headers: {profile_name}")

            self.current_profile = profile_name
            self.csv_file_path = new_csv_path
            self.load_entries()

            self.logger.info(f"Switched to profile: {profile_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error switching profile: {e}")
            return False

    def create_profile(self, profile_name: str) -> bool:
        """
        Create a new Quick Speak profile.

        Args:
            profile_name: Name for the new profile

        Returns:
            True if creation successful, False otherwise
        """
        try:
            # Sanitize profile name (remove invalid characters)
            import re
            sanitized_name = re.sub(r'[<>:"/\\|?*]', '', profile_name).strip()

            if not sanitized_name:
                self.logger.warning("Invalid profile name")
                return False

            new_csv_path = self.profiles_directory / f"{sanitized_name}.csv"

            if new_csv_path.exists():
                self.logger.warning(f"Profile already exists: {sanitized_name}")
                return False

            # Create empty CSV file with headers
            with open(new_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
                fieldnames = ['id', 'label', 'text']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

            self.logger.info(f"Created new profile: {sanitized_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating profile: {e}")
            return False

    def delete_profile(self, profile_name: str) -> bool:
        """
        Delete a Quick Speak profile.

        Args:
            profile_name: Name of the profile to delete

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            # Don't allow deleting the current profile or default
            if profile_name == self.current_profile:
                self.logger.warning("Cannot delete current active profile")
                return False

            if profile_name == "default":
                self.logger.warning("Cannot delete default profile")
                return False

            csv_path = self.profiles_directory / f"{profile_name}.csv"

            if not csv_path.exists():
                self.logger.warning(f"Profile does not exist: {profile_name}")
                return False

            csv_path.unlink()
            self.logger.info(f"Deleted profile: {profile_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting profile: {e}")
            return False