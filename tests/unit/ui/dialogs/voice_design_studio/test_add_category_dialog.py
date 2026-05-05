"""
Tests for AddCategoryDialog

Story 3.1: Create Custom Category
- Dialog for creating new category
- Name field with validation
- Create/Cancel buttons
- Error display for duplicates
"""

import pytest

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QDialog
from PyQt6.QtCore import Qt

from myvoice.ui.dialogs.voice_design_studio.add_category_dialog import AddCategoryDialog


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def existing_categories():
    """List of existing category names for validation."""
    return ["Narrators", "Characters", "Regional", "My Custom"]


@pytest.fixture
def dialog(qapp, existing_categories):
    """Create an AddCategoryDialog instance."""
    dialog = AddCategoryDialog(existing_categories=existing_categories)
    yield dialog
    dialog.deleteLater()


class TestAddCategoryDialogInit:
    """Tests for AddCategoryDialog initialization."""

    def test_dialog_creation(self, dialog):
        """Test dialog creates successfully."""
        assert dialog is not None
        assert isinstance(dialog, QDialog)

    def test_dialog_is_modal(self, dialog):
        """Test dialog is modal."""
        assert dialog.isModal()

    def test_has_name_edit(self, dialog):
        """Test dialog has name input field."""
        assert hasattr(dialog, 'name_edit')
        assert dialog.name_edit is not None

    def test_has_create_button(self, dialog):
        """Test dialog has Create button."""
        assert hasattr(dialog, 'create_button')
        assert dialog.create_button.text() == "Create"

    def test_has_cancel_button(self, dialog):
        """Test dialog has Cancel button."""
        assert hasattr(dialog, 'cancel_button')
        assert dialog.cancel_button.text() == "Cancel"

    def test_has_error_label(self, dialog):
        """Test dialog has error label."""
        assert hasattr(dialog, 'error_label')

    def test_error_label_initially_hidden(self, dialog):
        """Test error label is hidden initially."""
        assert not dialog.error_label.isVisible()


class TestNameField:
    """Tests for name input field."""

    def test_name_starts_empty(self, dialog):
        """Test name field starts empty."""
        assert dialog.name_edit.text() == ""

    def test_name_has_placeholder(self, dialog):
        """Test name field has placeholder text."""
        placeholder = dialog.name_edit.placeholderText()
        assert placeholder
        assert len(placeholder) > 0

    def test_name_has_max_length(self, dialog):
        """Test name field has max length set."""
        assert dialog.name_edit.maxLength() == 50

    def test_can_enter_name(self, dialog):
        """Test user can enter category name."""
        dialog.name_edit.setText("New Category")
        assert dialog.name_edit.text() == "New Category"


class TestCreateButtonState:
    """Tests for Create button enable/disable state."""

    def test_create_button_initially_disabled(self, dialog):
        """Test Create button is disabled when name is empty."""
        assert not dialog.create_button.isEnabled()

    def test_create_button_enables_with_name(self, dialog):
        """Test Create button enables when name is entered."""
        dialog.name_edit.setText("My Category")
        assert dialog.create_button.isEnabled()

    def test_create_button_disables_when_cleared(self, dialog):
        """Test Create button disables when name is cleared."""
        dialog.name_edit.setText("My Category")
        dialog.name_edit.clear()
        assert not dialog.create_button.isEnabled()

    def test_create_button_disabled_with_whitespace(self, dialog):
        """Test Create button disabled with whitespace-only name."""
        dialog.name_edit.setText("   ")
        # After stripping, it's empty, so should be disabled
        # But the textChanged handler checks stripped text
        # Actually our handler checks bool(name) where name = text.strip()
        # So whitespace should disable
        assert not dialog.create_button.isEnabled()


class TestValidation:
    """Tests for name validation."""

    def test_duplicate_shows_error(self, dialog, existing_categories):
        """Test entering duplicate name shows error."""
        dialog.name_edit.setText("Narrators")
        dialog.create_button.click()

        # Use isHidden() since parent is not shown (isVisible requires parent visible)
        assert not dialog.error_label.isHidden()
        assert "already exists" in dialog.error_label.text().lower()

    def test_duplicate_case_insensitive(self, dialog, existing_categories):
        """Test duplicate check is case-insensitive."""
        dialog.name_edit.setText("narrators")
        dialog.create_button.click()

        assert not dialog.error_label.isHidden()
        assert "already exists" in dialog.error_label.text().lower()

    def test_error_clears_on_text_change(self, dialog, existing_categories):
        """Test error clears when text changes."""
        dialog.name_edit.setText("Narrators")
        dialog.create_button.click()
        assert not dialog.error_label.isHidden()

        dialog.name_edit.setText("New Name")
        assert dialog.error_label.isHidden()

    def test_empty_name_shows_error(self, dialog):
        """Test empty name shows error on create attempt."""
        dialog.name_edit.setText("Test")  # Enable button
        dialog.name_edit.clear()
        # Button is disabled, but let's test the internal validation
        dialog.create_button.setEnabled(True)
        dialog.create_button.click()

        assert not dialog.error_label.isHidden()
        assert "empty" in dialog.error_label.text().lower()


class TestDialogResult:
    """Tests for dialog result."""

    def test_get_category_name_initially_none(self, dialog):
        """Test get_category_name returns None initially."""
        assert dialog.get_category_name() is None

    def test_cancel_returns_none(self, qapp, existing_categories):
        """Test cancelling returns None."""
        dialog = AddCategoryDialog(existing_categories=existing_categories)
        dialog.name_edit.setText("Test Category")
        dialog.reject()  # Cancel

        assert dialog.get_category_name() is None
        dialog.deleteLater()

    def test_create_valid_name_returns_name(self, qapp, existing_categories):
        """Test creating with valid name returns the name."""
        dialog = AddCategoryDialog(existing_categories=existing_categories)
        dialog.name_edit.setText("Brand New Category")
        dialog._on_create_clicked()  # Simulate click

        assert dialog.get_category_name() == "Brand New Category"
        dialog.deleteLater()

    def test_name_is_trimmed(self, qapp, existing_categories):
        """Test name is trimmed of whitespace."""
        dialog = AddCategoryDialog(existing_categories=existing_categories)
        dialog.name_edit.setText("  Trimmed Name  ")
        dialog._on_create_clicked()

        assert dialog.get_category_name() == "Trimmed Name"
        dialog.deleteLater()


class TestStaticMethod:
    """Tests for static convenience method."""

    def test_get_new_category_exists(self, dialog):
        """Test get_new_category static method exists."""
        assert hasattr(AddCategoryDialog, 'get_new_category')
        assert callable(AddCategoryDialog.get_new_category)


class TestAccessibility:
    """Tests for accessibility features."""

    def test_name_edit_has_accessible_name(self, dialog):
        """Test name edit has accessible name."""
        assert dialog.name_edit.accessibleName() == "Category Name"

    def test_name_edit_has_accessible_description(self, dialog):
        """Test name edit has accessible description."""
        desc = dialog.name_edit.accessibleDescription()
        assert desc
        assert len(desc) > 0


class TestUILayout:
    """Tests for UI layout."""

    def test_dialog_has_minimum_width(self, dialog):
        """Test dialog has minimum width set."""
        assert dialog.minimumWidth() >= 300

    def test_create_button_is_default(self, dialog):
        """Test Create button is the default button."""
        assert dialog.create_button.isDefault()
