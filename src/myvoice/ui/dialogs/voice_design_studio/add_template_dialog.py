"""
Add Template Dialog

Dialog for creating a new custom voice template.

Story 3.2: Create Custom Template
- Name field for template name
- Description field (pre-populated with current voice description)
- Shows selected category
- Create and Cancel buttons
- Error display for duplicates
"""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QTextEdit, QPushButton, QGroupBox
)
from PyQt6.QtCore import Qt


class AddTemplateDialog(QDialog):
    """
    Dialog for adding a new custom template.

    Provides name input, description field, category display,
    and Create/Cancel buttons.
    """

    def __init__(
        self,
        parent=None,
        category: str = "",
        current_description: str = "",
        existing_templates: list[str] = None
    ):
        """
        Initialize the Add Template Dialog.

        Args:
            parent: Parent widget
            category: Selected category name
            current_description: Current voice description to pre-populate
            existing_templates: List of existing template names in this category
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._category = category
        self._current_description = current_description
        self._existing_templates = existing_templates or []
        self._template_name: Optional[str] = None
        self._template_description: Optional[str] = None

        self._setup_ui()
        self._setup_connections()

    def _setup_ui(self):
        """Setup the dialog UI."""
        self.setWindowTitle("Add Template")
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
        # Pre-populate with current description
        if self._current_description:
            self.description_edit.setPlainText(self._current_description)
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

        self.create_button = QPushButton("Create")
        self.create_button.setObjectName("create_button")
        self.create_button.setDefault(True)
        self.create_button.setMinimumWidth(80)
        self.create_button.setEnabled(False)  # Disabled until name entered
        button_layout.addWidget(self.create_button)

        layout.addLayout(button_layout)

        # Accessibility
        self.name_edit.setAccessibleName("Template Name")
        self.name_edit.setAccessibleDescription("Enter a name for the new template")
        self.description_edit.setAccessibleName("Voice Description")
        self.description_edit.setAccessibleDescription(
            "Enter or edit the voice description for this template"
        )

    def _setup_connections(self):
        """Setup signal connections."""
        self.name_edit.textChanged.connect(self._on_content_changed)
        self.description_edit.textChanged.connect(self._on_content_changed)
        self.cancel_button.clicked.connect(self.reject)
        self.create_button.clicked.connect(self._on_create_clicked)

    def _on_content_changed(self):
        """Handle content changes."""
        name = self.name_edit.text().strip()
        description = self.description_edit.toPlainText().strip()
        self.create_button.setEnabled(bool(name) and bool(description))
        self.error_label.setVisible(False)

    def _on_create_clicked(self):
        """Handle Create button click."""
        name = self.name_edit.text().strip()
        description = self.description_edit.toPlainText().strip()

        if not name:
            self._show_error("Template name cannot be empty")
            return

        if not description:
            self._show_error("Voice description cannot be empty")
            return

        # Check for duplicates (case-insensitive)
        for existing in self._existing_templates:
            if name.lower() == existing.lower():
                self._show_error(f"Template '{existing}' already exists in this category")
                return

        self._template_name = name
        self._template_description = description
        self.accept()

    def _show_error(self, message: str):
        """Show error message."""
        self.error_label.setText(message)
        self.error_label.setVisible(True)
        self.name_edit.setFocus()
        self.name_edit.selectAll()

    def get_template_name(self) -> Optional[str]:
        """
        Get the entered template name.

        Returns:
            Template name if dialog was accepted, None otherwise
        """
        return self._template_name

    def get_template_description(self) -> Optional[str]:
        """
        Get the entered template description.

        Returns:
            Template description if dialog was accepted, None otherwise
        """
        return self._template_description

    def get_category(self) -> str:
        """
        Get the category for this template.

        Returns:
            Category name
        """
        return self._category

    @staticmethod
    def get_new_template(
        parent=None,
        category: str = "",
        current_description: str = "",
        existing_templates: list[str] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Static convenience method to show dialog and get result.

        Args:
            parent: Parent widget
            category: Selected category name
            current_description: Current voice description to pre-populate
            existing_templates: List of existing template names

        Returns:
            Tuple of (name, description) if created, (None, None) if cancelled
        """
        dialog = AddTemplateDialog(
            parent, category, current_description, existing_templates
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.get_template_name(), dialog.get_template_description()
        return None, None
