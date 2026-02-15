"""
Edit Template Dialog

Dialog for editing an existing custom voice template.

Story 3.3: Manage Templates
- Name field (pre-populated, editable)
- Description field (pre-populated, editable)
- Shows category (read-only)
- Save and Cancel buttons
- Error display for name conflicts
"""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QTextEdit, QPushButton, QGroupBox
)
from PyQt6.QtCore import Qt


class EditTemplateDialog(QDialog):
    """
    Dialog for editing an existing custom template.

    Provides name input (pre-populated), description field (pre-populated),
    category display, and Save/Cancel buttons.
    """

    def __init__(
        self,
        parent=None,
        category: str = "",
        template_name: str = "",
        template_description: str = "",
        existing_templates: list[str] = None
    ):
        """
        Initialize the Edit Template Dialog.

        Args:
            parent: Parent widget
            category: Category name (read-only)
            template_name: Current template name
            template_description: Current template description
            existing_templates: List of other template names in this category (for validation)
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._category = category
        self._original_name = template_name
        self._original_description = template_description
        # Exclude current template name from existing list
        self._existing_templates = [
            t for t in (existing_templates or [])
            if t.lower() != template_name.lower()
        ]
        self._new_name: Optional[str] = None
        self._new_description: Optional[str] = None

        self._setup_ui()
        self._setup_connections()

    def _setup_ui(self):
        """Setup the dialog UI."""
        self.setWindowTitle("Edit Template")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Category display (read-only)
        category_layout = QHBoxLayout()
        category_label = QLabel("Category:")
        category_label.setStyleSheet("font-weight: bold;")
        category_layout.addWidget(category_label)

        self.category_display = QLabel(self._category)
        self.category_display.setObjectName("category_display")
        category_layout.addWidget(self.category_display)
        category_layout.addStretch()
        layout.addLayout(category_layout)

        # Name input
        name_group = QGroupBox("Template Name")
        name_layout = QVBoxLayout(name_group)

        self.name_edit = QLineEdit()
        self.name_edit.setObjectName("template_name_edit")
        self.name_edit.setPlaceholderText("e.g., My Custom Narrator")
        self.name_edit.setMaxLength(50)
        self.name_edit.setText(self._original_name)
        name_layout.addWidget(self.name_edit)

        layout.addWidget(name_group)

        # Description input
        desc_group = QGroupBox("Voice Description")
        desc_layout = QVBoxLayout(desc_group)

        desc_hint = QLabel("Edit the voice description for this template:")
        desc_hint.setStyleSheet("color: #666;")
        desc_layout.addWidget(desc_hint)

        self.description_edit = QTextEdit()
        self.description_edit.setObjectName("template_description_edit")
        self.description_edit.setPlaceholderText(
            "Describe the voice characteristics..."
        )
        self.description_edit.setMinimumHeight(100)
        self.description_edit.setPlainText(self._original_description)
        desc_layout.addWidget(self.description_edit)

        layout.addWidget(desc_group)

        # Error label (hidden by default)
        self.error_label = QLabel()
        self.error_label.setObjectName("error_label")
        self.error_label.setStyleSheet("color: red;")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)

        button_layout.addStretch()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("cancel_button")
        self.cancel_button.setMinimumWidth(80)
        button_layout.addWidget(self.cancel_button)

        self.save_button = QPushButton("Save")
        self.save_button.setObjectName("save_button")
        self.save_button.setDefault(True)
        self.save_button.setMinimumWidth(80)
        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)

        # Accessibility
        self.name_edit.setAccessibleName("Template Name")
        self.name_edit.setAccessibleDescription("Edit the template name")
        self.description_edit.setAccessibleName("Voice Description")
        self.description_edit.setAccessibleDescription(
            "Edit the voice description for this template"
        )

    def _setup_connections(self):
        """Setup signal connections."""
        self.name_edit.textChanged.connect(self._on_content_changed)
        self.description_edit.textChanged.connect(self._on_content_changed)
        self.cancel_button.clicked.connect(self.reject)
        self.save_button.clicked.connect(self._on_save_clicked)

    def _on_content_changed(self):
        """Handle content changes."""
        self.error_label.setVisible(False)

    def _on_save_clicked(self):
        """Handle Save button click."""
        name = self.name_edit.text().strip()
        description = self.description_edit.toPlainText().strip()

        if not name:
            self._show_error("Template name cannot be empty")
            return

        if not description:
            self._show_error("Voice description cannot be empty")
            return

        # Check for name conflicts (case-insensitive, excluding original name)
        for existing in self._existing_templates:
            if name.lower() == existing.lower():
                self._show_error(f"Template '{existing}' already exists in this category")
                return

        self._new_name = name
        self._new_description = description
        self.accept()

    def _show_error(self, message: str):
        """Show error message."""
        self.error_label.setText(message)
        self.error_label.setVisible(True)
        self.name_edit.setFocus()
        self.name_edit.selectAll()

    def get_original_name(self) -> str:
        """
        Get the original template name.

        Returns:
            Original template name
        """
        return self._original_name

    def get_new_name(self) -> Optional[str]:
        """
        Get the new template name.

        Returns:
            New template name if dialog was accepted, None otherwise
        """
        return self._new_name

    def get_new_description(self) -> Optional[str]:
        """
        Get the new template description.

        Returns:
            New template description if dialog was accepted, None otherwise
        """
        return self._new_description

    def get_category(self) -> str:
        """
        Get the category for this template.

        Returns:
            Category name
        """
        return self._category

    def has_changes(self) -> bool:
        """
        Check if any changes were made.

        Returns:
            True if name or description changed
        """
        if self._new_name is None:
            return False
        return (
            self._new_name != self._original_name or
            self._new_description != self._original_description
        )

    @staticmethod
    def edit_template(
        parent=None,
        category: str = "",
        template_name: str = "",
        template_description: str = "",
        existing_templates: list[str] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Static convenience method to show dialog and get result.

        Args:
            parent: Parent widget
            category: Category name
            template_name: Current template name
            template_description: Current template description
            existing_templates: List of other template names

        Returns:
            Tuple of (new_name, new_description) if saved, (None, None) if cancelled
        """
        dialog = EditTemplateDialog(
            parent, category, template_name, template_description, existing_templates
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.get_new_name(), dialog.get_new_description()
        return None, None
