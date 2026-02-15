"""
Template Service for Voice Templates

Manages custom user-defined voice templates.
Templates are persisted to config/templates/{category}/{name}.json.

Story 3.2: Create Custom Template
- Load/save custom templates from JSON files
- Organize templates by category
- Merge with built-in templates
- Persist across sessions
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from myvoice.utils.portable_paths import get_templates_path


class TemplateService:
    """
    Service for managing custom voice templates.

    Custom templates are stored as individual JSON files organized by category:
    config/templates/{category}/{template_name}.json

    Each template file contains:
    {
        "name": "Template Name",
        "description": "Voice description text...",
        "category": "Category Name"
    }
    """

    def __init__(self):
        """Initialize the TemplateService."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._templates_dir = get_templates_path()
        self._custom_templates: dict[str, dict[str, str]] = {}
        self._load_all_templates()

    def _sanitize_filename(self, name: str) -> str:
        """
        Sanitize a name for use as a filename.

        Args:
            name: Original name

        Returns:
            Safe filename (no special characters)
        """
        # Replace spaces with underscores, remove special characters
        safe = re.sub(r'[^\w\s-]', '', name)
        safe = re.sub(r'[\s]+', '_', safe)
        return safe.strip('_')

    def _get_category_dir(self, category: str) -> Path:
        """
        Get the directory path for a category.

        Args:
            category: Category name

        Returns:
            Path to category directory
        """
        safe_category = self._sanitize_filename(category)
        return self._templates_dir / safe_category

    def _get_template_file(self, category: str, name: str) -> Path:
        """
        Get the file path for a template.

        Args:
            category: Category name
            name: Template name

        Returns:
            Path to template JSON file
        """
        safe_name = self._sanitize_filename(name)
        return self._get_category_dir(category) / f"{safe_name}.json"

    def _load_all_templates(self) -> None:
        """Load all custom templates from disk."""
        self._custom_templates = {}

        if not self._templates_dir.exists():
            self.logger.debug("Templates directory does not exist yet")
            return

        # Iterate through category directories
        for category_dir in self._templates_dir.iterdir():
            if not category_dir.is_dir():
                continue

            # Skip categories.json file
            if category_dir.name == 'categories.json':
                continue

            category_templates = {}

            # Load each template file in the category
            for template_file in category_dir.glob('*.json'):
                try:
                    with open(template_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        name = data.get('name', template_file.stem)
                        description = data.get('description', '')
                        category = data.get('category', category_dir.name)

                        if name and description:
                            category_templates[name] = description

                except (json.JSONDecodeError, IOError) as e:
                    self.logger.warning(f"Failed to load template {template_file}: {e}")

            if category_templates:
                # Use the original category name from the template if available
                # Otherwise use the directory name
                category_name = category_dir.name.replace('_', ' ')
                self._custom_templates[category_name] = category_templates

        total = sum(len(t) for t in self._custom_templates.values())
        self.logger.info(f"Loaded {total} custom templates from {len(self._custom_templates)} categories")

    def get_custom_templates(self) -> dict[str, dict[str, str]]:
        """
        Get all custom templates organized by category.

        Returns:
            Dict of {category: {template_name: description}}
        """
        return {cat: dict(templates) for cat, templates in self._custom_templates.items()}

    def get_templates_for_category(self, category: str) -> dict[str, str]:
        """
        Get custom templates for a specific category.

        Args:
            category: Category name

        Returns:
            Dict of {template_name: description}
        """
        return dict(self._custom_templates.get(category, {}))

    def add_template(
        self,
        category: str,
        name: str,
        description: str,
        builtin_templates: dict[str, dict[str, str]] = None
    ) -> tuple[bool, str]:
        """
        Add a new custom template.

        Args:
            category: Category to add template to
            name: Template name
            description: Voice description text
            builtin_templates: Built-in templates dict to check for duplicates

        Returns:
            Tuple of (success, error_message)
        """
        builtin_templates = builtin_templates or {}

        # Validate inputs
        name = name.strip()
        description = description.strip()
        category = category.strip()

        if not name:
            return False, "Template name cannot be empty"

        if not description:
            return False, "Template description cannot be empty"

        if not category:
            return False, "Category cannot be empty"

        # Check for duplicate in built-in templates (case-insensitive)
        if category in builtin_templates:
            for builtin_name in builtin_templates[category].keys():
                if name.lower() == builtin_name.lower():
                    return False, f"Template '{name}' already exists as a built-in template"

        # Check for duplicate in custom templates (case-insensitive)
        if category in self._custom_templates:
            for custom_name in self._custom_templates[category].keys():
                if name.lower() == custom_name.lower():
                    return False, f"Template '{name}' already exists in this category"

        # Create category directory if needed
        category_dir = self._get_category_dir(category)
        try:
            category_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return False, f"Failed to create category directory: {e}"

        # Save template file
        template_file = self._get_template_file(category, name)
        template_data = {
            "name": name,
            "description": description,
            "category": category
        }

        try:
            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            return False, f"Failed to save template: {e}"

        # Update in-memory cache
        if category not in self._custom_templates:
            self._custom_templates[category] = {}
        self._custom_templates[category][name] = description

        self.logger.info(f"Added custom template: {category}/{name}")
        return True, ""

    def remove_template(self, category: str, name: str) -> tuple[bool, str]:
        """
        Remove a custom template.

        Args:
            category: Category name
            name: Template name

        Returns:
            Tuple of (success, error_message)
        """
        if category not in self._custom_templates:
            return False, f"Category '{category}' not found"

        if name not in self._custom_templates[category]:
            return False, f"Template '{name}' not found in category '{category}'"

        # Delete the file
        template_file = self._get_template_file(category, name)
        try:
            if template_file.exists():
                template_file.unlink()
        except OSError as e:
            return False, f"Failed to delete template file: {e}"

        # Update in-memory cache
        del self._custom_templates[category][name]

        # Remove empty category
        if not self._custom_templates[category]:
            del self._custom_templates[category]
            # Try to remove empty directory
            try:
                category_dir = self._get_category_dir(category)
                if category_dir.exists() and not any(category_dir.iterdir()):
                    category_dir.rmdir()
            except OSError:
                pass  # Ignore directory removal errors

        self.logger.info(f"Removed custom template: {category}/{name}")
        return True, ""

    def update_template(
        self,
        category: str,
        old_name: str,
        new_name: str,
        new_description: str,
        builtin_templates: dict[str, dict[str, str]] = None
    ) -> tuple[bool, str]:
        """
        Update an existing custom template.

        Args:
            category: Category name
            old_name: Current template name
            new_name: New template name (can be same as old_name)
            new_description: New voice description
            builtin_templates: Built-in templates to check for name conflicts

        Returns:
            Tuple of (success, error_message)
        """
        builtin_templates = builtin_templates or {}

        # Validate inputs
        new_name = new_name.strip()
        new_description = new_description.strip()

        if not new_name:
            return False, "Template name cannot be empty"

        if not new_description:
            return False, "Template description cannot be empty"

        # Check template exists
        if category not in self._custom_templates:
            return False, f"Category '{category}' not found"

        if old_name not in self._custom_templates[category]:
            return False, f"Template '{old_name}' not found"

        # If name changed, check for conflicts
        if new_name.lower() != old_name.lower():
            # Check built-in
            if category in builtin_templates:
                for builtin_name in builtin_templates[category].keys():
                    if new_name.lower() == builtin_name.lower():
                        return False, f"Template '{new_name}' already exists as a built-in template"

            # Check custom (excluding current template)
            for custom_name in self._custom_templates[category].keys():
                if custom_name != old_name and new_name.lower() == custom_name.lower():
                    return False, f"Template '{new_name}' already exists in this category"

        # Delete old file if name changed
        old_file = self._get_template_file(category, old_name)
        new_file = self._get_template_file(category, new_name)

        if old_file != new_file:
            try:
                if old_file.exists():
                    old_file.unlink()
            except OSError as e:
                return False, f"Failed to remove old template file: {e}"

        # Save new/updated template file
        template_data = {
            "name": new_name,
            "description": new_description,
            "category": category
        }

        try:
            with open(new_file, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            return False, f"Failed to save template: {e}"

        # Update in-memory cache
        if old_name != new_name:
            del self._custom_templates[category][old_name]
        self._custom_templates[category][new_name] = new_description

        self.logger.info(f"Updated custom template: {category}/{old_name} -> {new_name}")
        return True, ""

    def is_custom_template(
        self,
        category: str,
        name: str,
        builtin_templates: dict[str, dict[str, str]] = None
    ) -> bool:
        """
        Check if a template is a custom template (not built-in).

        Args:
            category: Category name
            name: Template name
            builtin_templates: Built-in templates to check against

        Returns:
            True if template is custom, False if built-in or not found
        """
        builtin_templates = builtin_templates or {}

        # Check if it's a built-in template
        if category in builtin_templates:
            if name in builtin_templates[category]:
                return False

        # Check if it exists in custom templates
        if category in self._custom_templates:
            if name in self._custom_templates[category]:
                return True

        return False

    def template_exists(
        self,
        category: str,
        name: str,
        builtin_templates: dict[str, dict[str, str]] = None
    ) -> bool:
        """
        Check if a template name exists in a category.

        Args:
            category: Category name
            name: Template name
            builtin_templates: Built-in templates to check

        Returns:
            True if template exists
        """
        builtin_templates = builtin_templates or {}
        name_lower = name.lower().strip()

        # Check built-in
        if category in builtin_templates:
            for builtin_name in builtin_templates[category].keys():
                if name_lower == builtin_name.lower():
                    return True

        # Check custom
        if category in self._custom_templates:
            for custom_name in self._custom_templates[category].keys():
                if name_lower == custom_name.lower():
                    return True

        return False

    def reload(self) -> None:
        """Reload all templates from disk."""
        self._load_all_templates()
