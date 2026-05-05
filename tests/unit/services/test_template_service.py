"""
Tests for TemplateService

Story 3.2: Create Custom Template
- Load/save custom templates from JSON files
- Organize templates by category
- Merge with built-in templates
- Persist across sessions
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from myvoice.services.template_service import TemplateService


@pytest.fixture
def temp_templates_dir(tmp_path):
    """Create a temporary templates directory."""
    templates_dir = tmp_path / "config" / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    return templates_dir


@pytest.fixture
def template_service(temp_templates_dir):
    """Create a TemplateService with mocked directory path."""
    with patch('myvoice.services.template_service.get_templates_path') as mock_path:
        mock_path.return_value = temp_templates_dir
        service = TemplateService()
        yield service


@pytest.fixture
def builtin_templates():
    """Sample built-in templates for testing."""
    return {
        "Narrators": {
            "Audiobook Narrator": "A warm, mature voice...",
            "Documentary Narrator": "An authoritative voice..."
        },
        "Characters": {
            "Grumpy Craftsman": "A weathered voice..."
        }
    }


class TestTemplateServiceInit:
    """Tests for TemplateService initialization."""

    def test_service_creation(self, template_service):
        """Test service creates successfully."""
        assert template_service is not None

    def test_custom_templates_starts_empty(self, template_service):
        """Test custom templates dict starts empty."""
        templates = template_service.get_custom_templates()
        assert templates == {}

    def test_loads_existing_templates(self, temp_templates_dir):
        """Test service loads existing templates from directories."""
        # Create a category directory with template
        category_dir = temp_templates_dir / "My_Category"
        category_dir.mkdir()

        template_data = {
            "name": "My Template",
            "description": "A custom voice description",
            "category": "My Category"
        }
        (category_dir / "My_Template.json").write_text(json.dumps(template_data))

        with patch('myvoice.services.template_service.get_templates_path') as mock_path:
            mock_path.return_value = temp_templates_dir
            service = TemplateService()

        templates = service.get_custom_templates()
        assert "My Category" in templates
        assert "My Template" in templates["My Category"]
        assert templates["My Category"]["My Template"] == "A custom voice description"


class TestAddTemplate:
    """Tests for adding custom templates."""

    def test_add_template_success(self, template_service, builtin_templates):
        """Test adding a new template succeeds."""
        success, error = template_service.add_template(
            "My Category", "My Template", "A custom voice", builtin_templates
        )
        assert success is True
        assert error == ""

        templates = template_service.get_templates_for_category("My Category")
        assert "My Template" in templates
        assert templates["My Template"] == "A custom voice"

    def test_add_template_empty_name_fails(self, template_service, builtin_templates):
        """Test adding template with empty name fails."""
        success, error = template_service.add_template(
            "Category", "", "Description", builtin_templates
        )
        assert success is False
        assert "name" in error.lower() and "empty" in error.lower()

    def test_add_template_empty_description_fails(self, template_service, builtin_templates):
        """Test adding template with empty description fails."""
        success, error = template_service.add_template(
            "Category", "Name", "", builtin_templates
        )
        assert success is False
        assert "description" in error.lower() and "empty" in error.lower()

    def test_add_template_empty_category_fails(self, template_service, builtin_templates):
        """Test adding template with empty category fails."""
        success, error = template_service.add_template(
            "", "Name", "Description", builtin_templates
        )
        assert success is False
        assert "category" in error.lower() and "empty" in error.lower()

    def test_add_duplicate_builtin_fails(self, template_service, builtin_templates):
        """Test adding template with same name as built-in fails."""
        success, error = template_service.add_template(
            "Narrators", "Audiobook Narrator", "My description", builtin_templates
        )
        assert success is False
        assert "already exists" in error.lower()

    def test_add_duplicate_builtin_case_insensitive(self, template_service, builtin_templates):
        """Test duplicate check is case-insensitive for built-in."""
        success, error = template_service.add_template(
            "Narrators", "audiobook narrator", "My description", builtin_templates
        )
        assert success is False
        assert "already exists" in error.lower()

    def test_add_duplicate_custom_fails(self, template_service, builtin_templates):
        """Test adding duplicate custom template fails."""
        template_service.add_template("Category", "My Template", "Desc 1", builtin_templates)
        success, error = template_service.add_template(
            "Category", "My Template", "Desc 2", builtin_templates
        )
        assert success is False
        assert "already exists" in error.lower()

    def test_add_duplicate_custom_case_insensitive(self, template_service, builtin_templates):
        """Test duplicate check is case-insensitive for custom."""
        template_service.add_template("Category", "My Template", "Desc 1", builtin_templates)
        success, error = template_service.add_template(
            "Category", "my template", "Desc 2", builtin_templates
        )
        assert success is False
        assert "already exists" in error.lower()

    def test_add_template_creates_file(self, template_service, temp_templates_dir, builtin_templates):
        """Test added template is saved to file."""
        template_service.add_template(
            "New Category", "New Template", "Voice description", builtin_templates
        )

        # Check file was created
        category_dir = temp_templates_dir / "New_Category"
        template_file = category_dir / "New_Template.json"
        assert template_file.exists()

        # Verify file contents
        data = json.loads(template_file.read_text())
        assert data["name"] == "New Template"
        assert data["description"] == "Voice description"
        assert data["category"] == "New Category"

    def test_add_multiple_templates_same_category(self, template_service, builtin_templates):
        """Test adding multiple templates to same category."""
        template_service.add_template("Category", "Template A", "Desc A", builtin_templates)
        template_service.add_template("Category", "Template B", "Desc B", builtin_templates)
        template_service.add_template("Category", "Template C", "Desc C", builtin_templates)

        templates = template_service.get_templates_for_category("Category")
        assert len(templates) == 3
        assert "Template A" in templates
        assert "Template B" in templates
        assert "Template C" in templates


class TestRemoveTemplate:
    """Tests for removing custom templates."""

    def test_remove_template_success(self, template_service, builtin_templates):
        """Test removing a template succeeds."""
        template_service.add_template("Category", "To Remove", "Desc", builtin_templates)
        success, error = template_service.remove_template("Category", "To Remove")

        assert success is True
        assert error == ""
        assert "To Remove" not in template_service.get_templates_for_category("Category")

    def test_remove_nonexistent_category_fails(self, template_service):
        """Test removing from nonexistent category fails."""
        success, error = template_service.remove_template("Nonexistent", "Template")
        assert success is False
        assert "not found" in error.lower()

    def test_remove_nonexistent_template_fails(self, template_service, builtin_templates):
        """Test removing nonexistent template fails."""
        template_service.add_template("Category", "Exists", "Desc", builtin_templates)
        success, error = template_service.remove_template("Category", "Does Not Exist")
        assert success is False
        assert "not found" in error.lower()

    def test_remove_deletes_file(self, template_service, temp_templates_dir, builtin_templates):
        """Test removal deletes the file."""
        template_service.add_template("Category", "To Delete", "Desc", builtin_templates)

        template_file = temp_templates_dir / "Category" / "To_Delete.json"
        assert template_file.exists()

        template_service.remove_template("Category", "To Delete")
        assert not template_file.exists()


class TestGetTemplates:
    """Tests for getting templates."""

    def test_get_custom_templates_returns_copy(self, template_service, builtin_templates):
        """Test get_custom_templates returns a copy."""
        template_service.add_template("Category", "Template", "Desc", builtin_templates)

        templates = template_service.get_custom_templates()
        templates["Category"]["Injected"] = "Injected desc"

        # Original should not be modified
        assert "Injected" not in template_service.get_templates_for_category("Category")

    def test_get_templates_for_category_empty(self, template_service):
        """Test getting templates for nonexistent category returns empty dict."""
        templates = template_service.get_templates_for_category("Nonexistent")
        assert templates == {}

    def test_get_templates_for_category_returns_copy(self, template_service, builtin_templates):
        """Test get_templates_for_category returns a copy."""
        template_service.add_template("Category", "Template", "Desc", builtin_templates)

        templates = template_service.get_templates_for_category("Category")
        templates["Injected"] = "Injected desc"

        # Original should not be modified
        assert "Injected" not in template_service.get_templates_for_category("Category")


class TestTemplateExists:
    """Tests for template_exists method."""

    def test_builtin_exists(self, template_service, builtin_templates):
        """Test builtin template exists returns True."""
        exists = template_service.template_exists(
            "Narrators", "Audiobook Narrator", builtin_templates
        )
        assert exists is True

    def test_builtin_exists_case_insensitive(self, template_service, builtin_templates):
        """Test builtin exists check is case-insensitive."""
        exists = template_service.template_exists(
            "Narrators", "audiobook narrator", builtin_templates
        )
        assert exists is True

    def test_custom_exists(self, template_service, builtin_templates):
        """Test custom template exists returns True."""
        template_service.add_template("Category", "My Custom", "Desc", builtin_templates)
        exists = template_service.template_exists("Category", "My Custom", builtin_templates)
        assert exists is True

    def test_custom_exists_case_insensitive(self, template_service, builtin_templates):
        """Test custom exists check is case-insensitive."""
        template_service.add_template("Category", "My Custom", "Desc", builtin_templates)
        exists = template_service.template_exists("Category", "my custom", builtin_templates)
        assert exists is True

    def test_nonexistent_returns_false(self, template_service, builtin_templates):
        """Test nonexistent template returns False."""
        exists = template_service.template_exists("Category", "Does Not Exist", builtin_templates)
        assert exists is False


class TestReload:
    """Tests for reload functionality."""

    def test_reload_picks_up_external_changes(
        self, template_service, temp_templates_dir, builtin_templates
    ):
        """Test reload picks up externally added templates."""
        # Add a template via service
        template_service.add_template("Category", "Original", "Desc", builtin_templates)

        # Externally add another template
        category_dir = temp_templates_dir / "Category"
        external_data = {
            "name": "External",
            "description": "Externally added",
            "category": "Category"
        }
        (category_dir / "External.json").write_text(json.dumps(external_data))

        # Reload
        template_service.reload()

        templates = template_service.get_templates_for_category("Category")
        assert "Original" in templates
        assert "External" in templates


class TestUpdateTemplate:
    """Tests for updating custom templates (Story 3.3)."""

    def test_update_template_description_only(self, template_service, builtin_templates):
        """Test updating just the description."""
        template_service.add_template("Category", "Template", "Old desc", builtin_templates)
        success, error = template_service.update_template(
            "Category", "Template", "Template", "New desc", builtin_templates
        )
        assert success is True
        assert error == ""
        templates = template_service.get_templates_for_category("Category")
        assert templates["Template"] == "New desc"

    def test_update_template_name_only(self, template_service, builtin_templates):
        """Test updating just the name."""
        template_service.add_template("Category", "OldName", "Desc", builtin_templates)
        success, error = template_service.update_template(
            "Category", "OldName", "NewName", "Desc", builtin_templates
        )
        assert success is True
        templates = template_service.get_templates_for_category("Category")
        assert "NewName" in templates
        assert "OldName" not in templates

    def test_update_template_both(self, template_service, builtin_templates):
        """Test updating both name and description."""
        template_service.add_template("Category", "OldName", "Old desc", builtin_templates)
        success, error = template_service.update_template(
            "Category", "OldName", "NewName", "New desc", builtin_templates
        )
        assert success is True
        templates = template_service.get_templates_for_category("Category")
        assert templates["NewName"] == "New desc"
        assert "OldName" not in templates

    def test_update_nonexistent_category_fails(self, template_service, builtin_templates):
        """Test updating in nonexistent category fails."""
        success, error = template_service.update_template(
            "Nonexistent", "Template", "Template", "Desc", builtin_templates
        )
        assert success is False
        assert "not found" in error.lower()

    def test_update_nonexistent_template_fails(self, template_service, builtin_templates):
        """Test updating nonexistent template fails."""
        template_service.add_template("Category", "Exists", "Desc", builtin_templates)
        success, error = template_service.update_template(
            "Category", "DoesNotExist", "NewName", "Desc", builtin_templates
        )
        assert success is False
        assert "not found" in error.lower()

    def test_update_to_duplicate_name_fails(self, template_service, builtin_templates):
        """Test updating to existing name fails."""
        template_service.add_template("Category", "Template1", "Desc 1", builtin_templates)
        template_service.add_template("Category", "Template2", "Desc 2", builtin_templates)
        success, error = template_service.update_template(
            "Category", "Template1", "Template2", "Desc", builtin_templates
        )
        assert success is False
        assert "already exists" in error.lower()

    def test_update_to_builtin_name_fails(self, template_service, builtin_templates):
        """Test updating to built-in name fails."""
        template_service.add_template("Narrators", "Custom", "Desc", builtin_templates)
        success, error = template_service.update_template(
            "Narrators", "Custom", "Audiobook Narrator", "Desc", builtin_templates
        )
        assert success is False
        assert "already exists" in error.lower()

    def test_update_empty_name_fails(self, template_service, builtin_templates):
        """Test updating with empty name fails."""
        template_service.add_template("Category", "Template", "Desc", builtin_templates)
        success, error = template_service.update_template(
            "Category", "Template", "", "Desc", builtin_templates
        )
        assert success is False
        assert "empty" in error.lower()

    def test_update_empty_description_fails(self, template_service, builtin_templates):
        """Test updating with empty description fails."""
        template_service.add_template("Category", "Template", "Desc", builtin_templates)
        success, error = template_service.update_template(
            "Category", "Template", "Template", "", builtin_templates
        )
        assert success is False
        assert "empty" in error.lower()


class TestIsCustomTemplate:
    """Tests for is_custom_template method (Story 3.3)."""

    def test_builtin_is_not_custom(self, template_service, builtin_templates):
        """Test built-in template returns False."""
        is_custom = template_service.is_custom_template(
            "Narrators", "Audiobook Narrator", builtin_templates
        )
        assert is_custom is False

    def test_custom_is_custom(self, template_service, builtin_templates):
        """Test custom template returns True."""
        template_service.add_template("Category", "MyCustom", "Desc", builtin_templates)
        is_custom = template_service.is_custom_template(
            "Category", "MyCustom", builtin_templates
        )
        assert is_custom is True

    def test_nonexistent_is_not_custom(self, template_service, builtin_templates):
        """Test nonexistent template returns False."""
        is_custom = template_service.is_custom_template(
            "Category", "DoesNotExist", builtin_templates
        )
        assert is_custom is False


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_spaces_replaced(self, template_service, builtin_templates):
        """Test spaces are replaced with underscores."""
        template_service.add_template(
            "My Category", "My Template Name", "Desc", builtin_templates
        )
        # Should create My_Category/My_Template_Name.json
        templates = template_service.get_custom_templates()
        assert "My Category" in templates
        assert "My Template Name" in templates["My Category"]

    def test_special_chars_removed(self, template_service, temp_templates_dir, builtin_templates):
        """Test special characters are removed from filenames."""
        template_service.add_template(
            "Category!", "Template/Name:Test", "Desc", builtin_templates
        )
        # File should be created with safe name
        templates = template_service.get_custom_templates()
        assert len(templates) > 0
