"""
Add Category Dialog

Dialog for creating a new custom voice template category.

Story 3.1: Create Custom Category
- Name field for category name
- Create and Cancel buttons
- Error display for duplicate names
"""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QMessageBox
)
from PyQt6.QtCore import Qt


class AddCategoryDialog(QDialog):
    """
    Dialog for adding a new custom category.

    Provides a name input field with Create/Cancel buttons.
    Validates input and returns the category name on accept.
    """

    def __init__(self, parent=None, existing_categories: list[str] = None):
        """
        Initialize the Add Category Dialog.

        Args:
            parent: Parent widget
            existing_categories: List of existing category names for validation
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._existing_categories = existing_categories or []
        self._category_name: Optional[str] = None

        self._setup_ui()
        self._setup_connections()

    def _setup_ui(self):
        """Setup the dialog UI."""
        self.setWindowTitle("Add Category")
        self.setModal(True)
        self.setMinimumWidth(300)
        self.setMaximumWidth(400)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Instruction label
        instruction = QLabel("Enter a name for the new category:")
        instruction.setWordWrap(True)
        layout.addWidget(instruction)

        # Name input
        self.name_edit = QLineEdit()
        self.name_edit.setObjectName("category_name_edit")
        self.name_edit.setPlaceholderText("e.g., My Custom Voices")
        self.name_edit.setMaxLength(50)
        layout.addWidget(self.name_edit)

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
        self.name_edit.setAccessibleName("Category Name")
        self.name_edit.setAccessibleDescription("Enter a name for the new category")

    def _setup_connections(self):
        """Setup signal connections."""
        self.name_edit.textChanged.connect(self._on_name_changed)
        self.cancel_button.clicked.connect(self.reject)
        self.create_button.clicked.connect(self._on_create_clicked)

    def _on_name_changed(self, text: str):
        """Handle name field changes."""
        name = text.strip()
        self.create_button.setEnabled(bool(name))
        self.error_label.setVisible(False)

    def _on_create_clicked(self):
        """Handle Create button click."""
        name = self.name_edit.text().strip()

        if not name:
            self._show_error("Category name cannot be empty")
            return

        # Check for duplicates (case-insensitive)
        for existing in self._existing_categories:
            if name.lower() == existing.lower():
                self._show_error(f"Category '{existing}' already exists")
                return

        self._category_name = name
        self.accept()

    def _show_error(self, message: str):
        """Show error message."""
        self.error_label.setText(message)
        self.error_label.setVisible(True)
        self.name_edit.setFocus()
        self.name_edit.selectAll()

    def get_category_name(self) -> Optional[str]:
        """
        Get the entered category name.

        Returns:
            Category name if dialog was accepted, None otherwise
        """
        return self._category_name

    @staticmethod
    def get_new_category(parent=None, existing_categories: list[str] = None) -> Optional[str]:
        """
        Static convenience method to show dialog and get result.

        Args:
            parent: Parent widget
            existing_categories: List of existing category names

        Returns:
            New category name if created, None if cancelled
        """
        dialog = AddCategoryDialog(parent, existing_categories)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.get_category_name()
        return None
