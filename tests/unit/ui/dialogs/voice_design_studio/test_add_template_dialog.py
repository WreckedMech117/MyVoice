"""
Tests for AddTemplateDialog

Story 3.2: Create Custom Template
- Dialog for creating new template
- Name and description fields
- Category display
- Create/Cancel buttons
- Pre-populates with current description
"""

import pytest

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QDialog

from myvoice.ui.dialogs.voice_design_studio.add_template_dialog import AddTemplateDialog


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def existing_templates():
    """List of existing template names for validation."""
    return ["Audiobook Narrator", "Documentary Narrator", "My Custom Template"]


@pytest.fixture
def dialog(qapp, existing_templates):
    """Create an AddTemplateDialog instance."""
    dialog = AddTemplateDialog(
        category="Narrators",
        current_description="A warm, friendly voice...",
        existing_templates=existing_templates
    )
    yield dialog
    dialog.deleteLater()


class TestAddTemplateDialogInit:
    """Tests for AddTemplateDialog initialization."""

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

    def test_has_description_edit(self, dialog):
        """Test dialog has description input field."""
        assert hasattr(dialog, 'description_edit')
        assert dialog.description_edit is not None

    def test_has_category_display(self, dialog):
        """Test dialog has category display."""
        assert hasattr(dialog, 'category_display')
        assert dialog.category_display is not None

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
        assert dialog.error_label.isHidden()


class TestCategoryDisplay:
    """Tests for category display."""

    def test_category_displayed(self, dialog):
        """Test category is displayed."""
        assert dialog.category_display.text() == "Narrators"

    def test_get_category(self, dialog):
        """Test get_category returns the category."""
        assert dialog.get_category() == "Narrators"


class TestDescriptionPrePopulation:
    """Tests for description pre-population."""

    def test_description_prepopulated(self, dialog):
        """Test description is pre-populated with current description."""
        text = dialog.description_edit.toPlainText()
        assert text == "A warm, friendly voice..."

    def test_description_editable(self, dialog):
        """Test description is editable."""
        dialog.description_edit.setPlainText("Modified description")
        assert dialog.description_edit.toPlainText() == "Modified description"


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
        """Test user can enter template name."""
        dialog.name_edit.setText("New Template")
        assert dialog.name_edit.text() == "New Template"


class TestCreateButtonState:
    """Tests for Create button enable/disable state."""

    def test_create_button_initially_disabled(self, qapp):
        """Test Create button is disabled when name is empty."""
        # Create dialog without pre-populated description
        dialog = AddTemplateDialog(category="Category")
        assert not dialog.create_button.isEnabled()
        dialog.deleteLater()

    def test_create_button_enabled_with_both_fields(self, dialog):
        """Test Create button enables when name and description present."""
        dialog.name_edit.setText("My Template")
        # Description already pre-populated
        assert dialog.create_button.isEnabled()

    def test_create_button_disabled_without_name(self, dialog):
        """Test Create button disabled without name."""
        dialog.name_edit.clear()
        # Even with description, no name = disabled
        assert not dialog.create_button.isEnabled()

    def test_create_button_disabled_without_description(self, qapp, existing_templates):
        """Test Create button disabled without description."""
        dialog = AddTemplateDialog(
            category="Category",
            current_description="",  # Empty
            existing_templates=existing_templates
        )
        dialog.name_edit.setText("My Template")
        assert not dialog.create_button.isEnabled()
        dialog.deleteLater()


class TestValidation:
    """Tests for name validation."""

    def test_duplicate_shows_error(self, dialog):
        """Test entering duplicate name shows error."""
        dialog.name_edit.setText("Audiobook Narrator")
        dialog.create_button.click()

        assert not dialog.error_label.isHidden()
        assert "already exists" in dialog.error_label.text().lower()

    def test_duplicate_case_insensitive(self, dialog):
        """Test duplicate check is case-insensitive."""
        dialog.name_edit.setText("audiobook narrator")
        dialog.create_button.click()

        assert not dialog.error_label.isHidden()
        assert "already exists" in dialog.error_label.text().lower()

    def test_error_clears_on_text_change(self, dialog):
        """Test error clears when text changes."""
        dialog.name_edit.setText("Audiobook Narrator")
        dialog.create_button.click()
        assert not dialog.error_label.isHidden()

        dialog.name_edit.setText("New Name")
        assert dialog.error_label.isHidden()


class TestDialogResult:
    """Tests for dialog result."""

    def test_get_template_name_initially_none(self, dialog):
        """Test get_template_name returns None initially."""
        assert dialog.get_template_name() is None

    def test_get_template_description_initially_none(self, dialog):
        """Test get_template_description returns None initially."""
        assert dialog.get_template_description() is None

    def test_cancel_returns_none(self, qapp, existing_templates):
        """Test cancelling returns None."""
        dialog = AddTemplateDialog(
            category="Category",
            current_description="Desc",
            existing_templates=existing_templates
        )
        dialog.name_edit.setText("Test Template")
        dialog.reject()  # Cancel

        assert dialog.get_template_name() is None
        assert dialog.get_template_description() is None
        dialog.deleteLater()

    def test_create_valid_returns_values(self, qapp, existing_templates):
        """Test creating with valid input returns name and description."""
        dialog = AddTemplateDialog(
            category="Category",
            current_description="Original description",
            existing_templates=existing_templates
        )
        dialog.name_edit.setText("Brand New Template")
        dialog.description_edit.setPlainText("Modified description")
        dialog._on_create_clicked()

        assert dialog.get_template_name() == "Brand New Template"
        assert dialog.get_template_description() == "Modified description"
        dialog.deleteLater()

    def test_name_is_trimmed(self, qapp, existing_templates):
        """Test name is trimmed of whitespace."""
        dialog = AddTemplateDialog(
            category="Category",
            current_description="Description",
            existing_templates=existing_templates
        )
        dialog.name_edit.setText("  Trimmed Name  ")
        dialog._on_create_clicked()

        assert dialog.get_template_name() == "Trimmed Name"
        dialog.deleteLater()

    def test_description_is_trimmed(self, qapp, existing_templates):
        """Test description is trimmed of whitespace."""
        dialog = AddTemplateDialog(
            category="Category",
            current_description="Original",
            existing_templates=existing_templates
        )
        dialog.name_edit.setText("Name")
        dialog.description_edit.setPlainText("  Trimmed description  ")
        dialog._on_create_clicked()

        assert dialog.get_template_description() == "Trimmed description"
        dialog.deleteLater()


class TestStaticMethod:
    """Tests for static convenience method."""

    def test_get_new_template_exists(self, dialog):
        """Test get_new_template static method exists."""
        assert hasattr(AddTemplateDialog, 'get_new_template')
        assert callable(AddTemplateDialog.get_new_template)


class TestAccessibility:
    """Tests for accessibility features."""

    def test_name_edit_has_accessible_name(self, dialog):
        """Test name edit has accessible name."""
        assert dialog.name_edit.accessibleName() == "Template Name"

    def test_description_edit_has_accessible_name(self, dialog):
        """Test description edit has accessible name."""
        assert dialog.description_edit.accessibleName() == "Voice Description"


class TestUILayout:
    """Tests for UI layout."""

    def test_dialog_has_minimum_width(self, dialog):
        """Test dialog has minimum width set."""
        assert dialog.minimumWidth() >= 400

    def test_dialog_has_minimum_height(self, dialog):
        """Test dialog has minimum height set."""
        assert dialog.minimumHeight() >= 300

    def test_create_button_is_default(self, dialog):
        """Test Create button is the default button."""
        assert dialog.create_button.isDefault()
