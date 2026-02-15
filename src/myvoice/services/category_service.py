"""
Category Service for Voice Template Categories

Manages custom user-defined categories for voice templates.
Categories are persisted to config/templates/categories.json.

Story 3.1: Create Custom Category
- Load/save custom categories from JSON file
- Merge built-in + custom categories
- Validate duplicate category names
- Persist across sessions
"""

import json
import logging
from pathlib import Path
from typing import Optional

from myvoice.utils.portable_paths import get_categories_file_path


class CategoryService:
    """
    Service for managing custom voice template categories.

    Custom categories are stored in categories.json and merged with
    built-in categories from VOICE_TEMPLATES.
    """

    def __init__(self):
        """Initialize the CategoryService."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._categories_file = get_categories_file_path()
        self._custom_categories: list[str] = []
        self._load_categories()

    def _load_categories(self) -> None:
        """Load custom categories from JSON file."""
        if not self._categories_file.exists():
            self.logger.debug("No categories.json found, starting with empty list")
            self._custom_categories = []
            return

        try:
            with open(self._categories_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._custom_categories = data.get('categories', [])
                self.logger.info(f"Loaded {len(self._custom_categories)} custom categories")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse categories.json: {e}")
            self._custom_categories = []
        except Exception as e:
            self.logger.error(f"Failed to load categories: {e}")
            self._custom_categories = []

    def _save_categories(self) -> bool:
        """
        Save custom categories to JSON file.

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure parent directory exists
            self._categories_file.parent.mkdir(parents=True, exist_ok=True)

            data = {'categories': self._custom_categories}
            with open(self._categories_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved {len(self._custom_categories)} custom categories")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save categories: {e}")
            return False

    def get_custom_categories(self) -> list[str]:
        """
        Get list of custom categories.

        Returns:
            List of custom category names
        """
        return self._custom_categories.copy()

    def add_category(self, name: str, builtin_categories: list[str]) -> tuple[bool, str]:
        """
        Add a new custom category.

        Args:
            name: Category name to add
            builtin_categories: List of built-in category names to check for duplicates

        Returns:
            Tuple of (success, error_message)
            - (True, "") on success
            - (False, "error description") on failure
        """
        # Validate name
        name = name.strip()
        if not name:
            return False, "Category name cannot be empty"

        # Check for duplicate in built-in categories (case-insensitive)
        for builtin in builtin_categories:
            if name.lower() == builtin.lower():
                return False, f"Category '{name}' already exists as a built-in category"

        # Check for duplicate in custom categories (case-insensitive)
        for custom in self._custom_categories:
            if name.lower() == custom.lower():
                return False, f"Category '{name}' already exists"

        # Add category
        self._custom_categories.append(name)

        # Save to file
        if not self._save_categories():
            # Rollback on save failure
            self._custom_categories.remove(name)
            return False, "Failed to save category"

        self.logger.info(f"Added custom category: {name}")
        return True, ""

    def remove_category(self, name: str) -> tuple[bool, str]:
        """
        Remove a custom category.

        Args:
            name: Category name to remove

        Returns:
            Tuple of (success, error_message)
        """
        if name not in self._custom_categories:
            return False, f"Category '{name}' not found"

        self._custom_categories.remove(name)

        if not self._save_categories():
            # Rollback on save failure
            self._custom_categories.append(name)
            return False, "Failed to save changes"

        self.logger.info(f"Removed custom category: {name}")
        return True, ""

    def category_exists(self, name: str, builtin_categories: list[str]) -> bool:
        """
        Check if a category name exists (built-in or custom).

        Args:
            name: Category name to check
            builtin_categories: List of built-in category names

        Returns:
            True if category exists, False otherwise
        """
        name_lower = name.lower().strip()

        # Check built-in
        for builtin in builtin_categories:
            if name_lower == builtin.lower():
                return True

        # Check custom
        for custom in self._custom_categories:
            if name_lower == custom.lower():
                return True

        return False

    def reload(self) -> None:
        """Reload categories from file."""
        self._load_categories()
