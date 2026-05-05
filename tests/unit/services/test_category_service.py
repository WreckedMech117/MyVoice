"""
Tests for CategoryService

Story 3.1: Create Custom Category
- Load/save custom categories from JSON file
- Merge built-in + custom categories
- Validate duplicate category names
- Persist across sessions
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from myvoice.services.category_service import CategoryService


@pytest.fixture
def temp_categories_file(tmp_path):
    """Create a temporary categories file path."""
    templates_dir = tmp_path / "config" / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    return templates_dir / "categories.json"


@pytest.fixture
def category_service(temp_categories_file):
    """Create a CategoryService with mocked file path."""
    with patch('myvoice.services.category_service.get_categories_file_path') as mock_path:
        mock_path.return_value = temp_categories_file
        service = CategoryService()
        yield service


@pytest.fixture
def builtin_categories():
    """List of built-in category names for testing."""
    return ["Narrators", "Characters", "Regional"]


class TestCategoryServiceInit:
    """Tests for CategoryService initialization."""

    def test_service_creation(self, category_service):
        """Test service creates successfully."""
        assert category_service is not None

    def test_custom_categories_starts_empty(self, category_service):
        """Test custom categories list starts empty when no file exists."""
        categories = category_service.get_custom_categories()
        assert categories == []

    def test_loads_existing_categories(self, temp_categories_file):
        """Test service loads existing categories from file."""
        # Create categories file
        temp_categories_file.parent.mkdir(parents=True, exist_ok=True)
        data = {"categories": ["My Category", "Another Category"]}
        temp_categories_file.write_text(json.dumps(data))

        with patch('myvoice.services.category_service.get_categories_file_path') as mock_path:
            mock_path.return_value = temp_categories_file
            service = CategoryService()

        categories = service.get_custom_categories()
        assert "My Category" in categories
        assert "Another Category" in categories

    def test_handles_invalid_json(self, temp_categories_file):
        """Test service handles invalid JSON gracefully."""
        temp_categories_file.parent.mkdir(parents=True, exist_ok=True)
        temp_categories_file.write_text("not valid json {{{")

        with patch('myvoice.services.category_service.get_categories_file_path') as mock_path:
            mock_path.return_value = temp_categories_file
            service = CategoryService()

        # Should not raise, returns empty list
        categories = service.get_custom_categories()
        assert categories == []


class TestAddCategory:
    """Tests for adding custom categories."""

    def test_add_category_success(self, category_service, builtin_categories):
        """Test adding a new category succeeds."""
        success, error = category_service.add_category("My Custom", builtin_categories)
        assert success is True
        assert error == ""
        assert "My Custom" in category_service.get_custom_categories()

    def test_add_category_empty_name_fails(self, category_service, builtin_categories):
        """Test adding empty category name fails."""
        success, error = category_service.add_category("", builtin_categories)
        assert success is False
        assert "empty" in error.lower()

    def test_add_category_whitespace_name_fails(self, category_service, builtin_categories):
        """Test adding whitespace-only category name fails."""
        success, error = category_service.add_category("   ", builtin_categories)
        assert success is False
        assert "empty" in error.lower()

    def test_add_duplicate_builtin_fails(self, category_service, builtin_categories):
        """Test adding category with same name as built-in fails."""
        success, error = category_service.add_category("Narrators", builtin_categories)
        assert success is False
        assert "already exists" in error.lower()

    def test_add_duplicate_builtin_case_insensitive(self, category_service, builtin_categories):
        """Test duplicate check is case-insensitive for built-in."""
        success, error = category_service.add_category("narrators", builtin_categories)
        assert success is False
        assert "already exists" in error.lower()

    def test_add_duplicate_custom_fails(self, category_service, builtin_categories):
        """Test adding duplicate custom category fails."""
        category_service.add_category("My Category", builtin_categories)
        success, error = category_service.add_category("My Category", builtin_categories)
        assert success is False
        assert "already exists" in error.lower()

    def test_add_duplicate_custom_case_insensitive(self, category_service, builtin_categories):
        """Test duplicate check is case-insensitive for custom."""
        category_service.add_category("My Category", builtin_categories)
        success, error = category_service.add_category("my category", builtin_categories)
        assert success is False
        assert "already exists" in error.lower()

    def test_add_category_persists_to_file(self, category_service, temp_categories_file, builtin_categories):
        """Test added category is saved to file."""
        category_service.add_category("Persistent Category", builtin_categories)

        # Read file directly
        data = json.loads(temp_categories_file.read_text())
        assert "Persistent Category" in data["categories"]

    def test_add_multiple_categories(self, category_service, builtin_categories):
        """Test adding multiple categories."""
        category_service.add_category("Category A", builtin_categories)
        category_service.add_category("Category B", builtin_categories)
        category_service.add_category("Category C", builtin_categories)

        categories = category_service.get_custom_categories()
        assert len(categories) == 3
        assert "Category A" in categories
        assert "Category B" in categories
        assert "Category C" in categories


class TestRemoveCategory:
    """Tests for removing custom categories."""

    def test_remove_category_success(self, category_service, builtin_categories):
        """Test removing a category succeeds."""
        category_service.add_category("To Remove", builtin_categories)
        success, error = category_service.remove_category("To Remove")
        assert success is True
        assert error == ""
        assert "To Remove" not in category_service.get_custom_categories()

    def test_remove_nonexistent_fails(self, category_service):
        """Test removing nonexistent category fails."""
        success, error = category_service.remove_category("Does Not Exist")
        assert success is False
        assert "not found" in error.lower()

    def test_remove_persists_to_file(self, category_service, temp_categories_file, builtin_categories):
        """Test removal is saved to file."""
        category_service.add_category("Will Remove", builtin_categories)
        category_service.remove_category("Will Remove")

        data = json.loads(temp_categories_file.read_text())
        assert "Will Remove" not in data["categories"]


class TestCategoryExists:
    """Tests for category_exists method."""

    def test_builtin_exists(self, category_service, builtin_categories):
        """Test builtin category exists returns True."""
        assert category_service.category_exists("Narrators", builtin_categories) is True

    def test_builtin_exists_case_insensitive(self, category_service, builtin_categories):
        """Test builtin exists check is case-insensitive."""
        assert category_service.category_exists("narrators", builtin_categories) is True
        assert category_service.category_exists("NARRATORS", builtin_categories) is True

    def test_custom_exists(self, category_service, builtin_categories):
        """Test custom category exists returns True."""
        category_service.add_category("My Custom", builtin_categories)
        assert category_service.category_exists("My Custom", builtin_categories) is True

    def test_custom_exists_case_insensitive(self, category_service, builtin_categories):
        """Test custom exists check is case-insensitive."""
        category_service.add_category("My Custom", builtin_categories)
        assert category_service.category_exists("my custom", builtin_categories) is True

    def test_nonexistent_returns_false(self, category_service, builtin_categories):
        """Test nonexistent category returns False."""
        assert category_service.category_exists("Does Not Exist", builtin_categories) is False


class TestReload:
    """Tests for reload functionality."""

    def test_reload_picks_up_external_changes(self, category_service, temp_categories_file, builtin_categories):
        """Test reload picks up changes made externally to file."""
        # Add a category
        category_service.add_category("Original", builtin_categories)

        # Simulate external modification
        data = {"categories": ["Original", "External Addition"]}
        temp_categories_file.write_text(json.dumps(data))

        # Reload
        category_service.reload()

        categories = category_service.get_custom_categories()
        assert "Original" in categories
        assert "External Addition" in categories


class TestGetCustomCategories:
    """Tests for get_custom_categories method."""

    def test_returns_copy(self, category_service, builtin_categories):
        """Test returns a copy, not the internal list."""
        category_service.add_category("Test", builtin_categories)
        categories = category_service.get_custom_categories()
        categories.append("Injected")

        # Internal list should not be modified
        assert "Injected" not in category_service.get_custom_categories()
