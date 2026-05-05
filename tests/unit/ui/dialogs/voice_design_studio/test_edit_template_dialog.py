"""
Tests for EditTemplateDialog

Story 3.3: Manage Templates
- Dialog for editing existing custom template
- Name and description fields (pre-populated)
- Category display (read-only)
- Save/Cancel buttons
- Error display for name conflicts
"""

import pytest

# PyQt6 imports - skip tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QDialog

from myvoice.ui.dialogs.voice_design_studio.edit_template_dialog import EditTemplateDialog


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def existing_templates():
    """List of other template names for validation."""
    return ["Other Template 1", "Other Template 2"]


@pytest.fixture
def dialog(qapp, existing_templates):
    """Create an EditTemplateDialog instance."""
    dialog = EditTemplateDialog(
        category="Narrators",
        template_name="My Custom Voice",
        template_description="A warm, friendly voice with clear diction.",
        existing_templates=existing_templates
    )
    yield dialog
    dialog.deleteLater()


class TestEditTemplateDialogInit:
    """Tests for EditTemplateDialog initialization."""

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

    def test_has_save_button(self, dialog):
        """Test dialog has Save button."""
        assert hasattr(dialog, 'save_button')
        assert dialog.save_button.text() == "Save"

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


class TestPrePopulation:
    """Tests for pre-population of fields."""

    def test_name_prepopulated(self, dialog):
        """Test name field is pre-populated."""
        assert dialog.name_edit.text() == "My Custom Voice"

    def test_description_prepopulated(self, dialog):
        """Test description field is pre-populated."""
        assert dialog.description_edit.toPlainText() == "A warm, friendly voice with clear diction."

    def test_category_displayed(self, dialog):
        """Test category is displayed."""
        assert dialog.category_display.text() == "Narrators"


class TestGetters:
    """Tests for getter methods."""

    def test_get_original_name(self, dialog):
        """Test get_original_name returns original name."""
        assert dialog.get_original_name() == "My Custom Voice"

    def test_get_category(self, dialog):
        """Test get_category returns category."""
        assert dialog.get_category() == "Narrators"

    def test_get_new_name_initially_none(self, dialog):
        """Test get_new_name returns None before save."""
        assert dialog.get_new_name() is None

    def test_get_new_description_initially_none(self, dialog):
        """Test get_new_description returns None before save."""
        assert dialog.get_new_description() is None


class TestEditing:
    """Tests for editing functionality."""

    def test_can_edit_name(self, dialog):
        """Test name is editable."""
        dialog.name_edit.setText("Modified Name")
        assert dialog.name_edit.text() == "Modified Name"

    def test_can_edit_description(self, dialog):
        """Test description is editable."""
        dialog.description_edit.setPlainText("Modified description")
        assert dialog.description_edit.toPlainText() == "Modified description"


class TestValidation:
    """Tests for validation."""

    def test_empty_name_shows_error(self, dialog):
        """Test empty name shows error on save."""
        dialog.name_edit.clear()
        dialog.save_button.click()

        assert not dialog.error_label.isHidden()
        assert "empty" in dialog.error_label.text().lower()

    def test_empty_description_shows_error(self, dialog):
        """Test empty description shows error on save."""
        dialog.description_edit.clear()
        dialog.save_button.click()

        assert not dialog.error_label.isHidden()
        assert "empty" in dialog.error_label.text().lower()

    def test_duplicate_name_shows_error(self, dialog, existing_templates):
        """Test duplicate name shows error."""
        dialog.name_edit.setText("Other Template 1")
        dialog.save_button.click()

        assert not dialog.error_label.isHidden()
        assert "already exists" in dialog.error_label.text().lower()

    def test_duplicate_check_case_insensitive(self, dialog, existing_templates):
        """Test duplicate check is case-insensitive."""
        dialog.name_edit.setText("other template 1")
        dialog.save_button.click()

        assert not dialog.error_label.isHidden()
        assert "already exists" in dialog.error_label.text().lower()

    def test_error_clears_on_text_change(self, dialog, existing_templates):
        """Test error clears when text changes."""
        dialog.name_edit.setText("Other Template 1")
        dialog.save_button.click()
        assert not dialog.error_label.isHidden()

        dialog.name_edit.setText("New Unique Name")
        assert dialog.error_label.isHidden()

    def test_same_name_as_original_allowed(self, dialog):
        """Test keeping the same name is allowed."""
        # Name is already "My Custom Voice"
        dialog._on_save_clicked()

        # Should not show error, should accept
        assert dialog.get_new_name() == "My Custom Voice"


class TestDialogResult:
    """Tests for dialog result."""

    def test_save_returns_values(self, qapp, existing_templates):
        """Test saving returns new values."""
        dialog = EditTemplateDialog(
            category="Category",
            template_name="Original",
            template_description="Original desc",
            existing_templates=existing_templates
        )
        dialog.name_edit.setText("New Name")
        dialog.description_edit.setPlainText("New description")
        dialog._on_save_clicked()

        assert dialog.get_new_name() == "New Name"
        assert dialog.get_new_description() == "New description"
        dialog.deleteLater()

    def test_cancel_returns_none(self, qapp, existing_templates):
        """Test cancelling returns None."""
        dialog = EditTemplateDialog(
            category="Category",
            template_name="Template",
            template_description="Desc",
            existing_templates=existing_templates
        )
        dialog.reject()  # Cancel

        assert dialog.get_new_name() is None
        assert dialog.get_new_description() is None
        dialog.deleteLater()

    def test_name_is_trimmed(self, qapp, existing_templates):
        """Test name is trimmed."""
        dialog = EditTemplateDialog(
            category="Category",
            template_name="Template",
            template_description="Desc",
            existing_templates=existing_templates
        )
        dialog.name_edit.setText("  Trimmed  ")
        dialog._on_save_clicked()

        assert dialog.get_new_name() == "Trimmed"
        dialog.deleteLater()

    def test_description_is_trimmed(self, qapp, existing_templates):
        """Test description is trimmed."""
        dialog = EditTemplateDialog(
            category="Category",
            template_name="Template",
            template_description="Desc",
            existing_templates=existing_templates
        )
        dialog.description_edit.setPlainText("  Trimmed  ")
        dialog._on_save_clicked()

        assert dialog.get_new_description() == "Trimmed"
        dialog.deleteLater()


class TestHasChanges:
    """Tests for has_changes method."""

    def test_no_changes_before_save(self, dialog):
        """Test has_changes returns False before save."""
        assert dialog.has_changes() is False

    def test_has_changes_when_name_changed(self, qapp, existing_templates):
        """Test has_changes detects name change."""
        dialog = EditTemplateDialog(
            category="Category",
            template_name="Original",
            template_description="Original desc",
            existing_templates=existing_templates
        )
        dialog.name_edit.setText("Changed")
        dialog._on_save_clicked()

        assert dialog.has_changes() is True
        dialog.deleteLater()

    def test_has_changes_when_description_changed(self, qapp, existing_templates):
        """Test has_changes detects description change."""
        dialog = EditTemplateDialog(
            category="Category",
            template_name="Original",
            template_description="Original desc",
            existing_templates=existing_templates
        )
        dialog.description_edit.setPlainText("Changed desc")
        dialog._on_save_clicked()

        assert dialog.has_changes() is True
        dialog.deleteLater()

    def test_no_changes_when_same_values(self, qapp, existing_templates):
        """Test has_changes returns False when values unchanged."""
        dialog = EditTemplateDialog(
            category="Category",
            template_name="Original",
            template_description="Original desc",
            existing_templates=existing_templates
        )
        # Don't change anything
        dialog._on_save_clicked()

        assert dialog.has_changes() is False
        dialog.deleteLater()


class TestStaticMethod:
    """Tests for static convenience method."""

    def test_edit_template_exists(self, dialog):
        """Test edit_template static method exists."""
        assert hasattr(EditTemplateDialog, 'edit_template')
        assert callable(EditTemplateDialog.edit_template)


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

    def test_save_button_is_default(self, dialog):
        """Test Save button is the default button."""
        assert dialog.save_button.isDefault()


class TestExistingTemplatesFiltering:
    """Tests for filtering current template from existing list."""

    def test_current_template_excluded_from_duplicates(self, qapp):
        """Test current template name is excluded from duplicate check."""
        # If "My Template" is current, changing name back to "My Template" should work
        dialog = EditTemplateDialog(
            category="Category",
            template_name="My Template",
            template_description="Desc",
            existing_templates=["My Template", "Other Template"]
        )
        # Keep the same name
        dialog._on_save_clicked()

        # Should not show error - keeping same name is allowed
        assert dialog.get_new_name() == "My Template"
        dialog.deleteLater()
