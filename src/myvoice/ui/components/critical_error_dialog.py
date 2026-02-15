"""
Critical Error Dialog

Story 7.6: Modal dialog for critical/unrecoverable errors with copy-to-clipboard support.
Implements NFR6 (stability) and NFR10 (user-friendly error messages).
"""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QWidget, QFrame, QApplication
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QIcon


class CriticalErrorDialog(QDialog):
    """
    Modal dialog for displaying critical errors with copy-to-clipboard support.

    Story 7.6: Critical errors show modal dialog with:
    - Clear error message (what happened + what to do)
    - Expandable technical details
    - Copy-to-clipboard button for support
    - Exit/Continue buttons
    """

    def __init__(
        self,
        title: str,
        message: str,
        details: Optional[str] = None,
        allow_continue: bool = True,
        parent: Optional[QWidget] = None
    ):
        """
        Initialize the critical error dialog.

        Args:
            title: Dialog title
            message: User-friendly error message
            details: Technical details for support (optional)
            allow_continue: Whether to show "Continue" button (False for fatal errors)
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)

        self._title = title
        self._message = message
        self._details = details
        self._allow_continue = allow_continue
        self._details_expanded = False

        self.setWindowTitle(title)
        self.setMinimumSize(450, 200)
        self.setModal(True)

        # Remove question mark from title bar
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint
        )

        self._create_ui()

        self.logger.debug(f"CriticalErrorDialog created: {title}")

    def _create_ui(self):
        """Create the dialog UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header with icon and title
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)

        # Error icon
        icon_label = QLabel()
        icon_label.setPixmap(
            self.style().standardIcon(
                self.style().StandardPixmap.SP_MessageBoxCritical
            ).pixmap(48, 48)
        )
        header_layout.addWidget(icon_label)

        # Title
        title_label = QLabel(self._title)
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setWordWrap(True)
        header_layout.addWidget(title_label, 1)

        layout.addLayout(header_layout)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        # Message
        message_label = QLabel(self._message)
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(message_label)

        # Details section (collapsible)
        if self._details:
            # Toggle button
            self._details_button = QPushButton("Show Details")
            self._details_button.setCheckable(True)
            self._details_button.setChecked(False)
            self._details_button.clicked.connect(self._toggle_details)
            self._details_button.setMaximumWidth(120)
            layout.addWidget(self._details_button)

            # Details text area (hidden initially)
            self._details_text = QTextEdit()
            self._details_text.setPlainText(self._details)
            self._details_text.setReadOnly(True)
            self._details_text.setMaximumHeight(150)
            self._details_text.setVisible(False)

            # Use monospace font for technical details
            details_font = QFont("Consolas", 8)
            self._details_text.setFont(details_font)

            layout.addWidget(self._details_text)

        # Spacer
        layout.addStretch()

        # Button row
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)

        # Copy button (if details available)
        if self._details:
            copy_button = QPushButton("Copy Details")
            copy_button.clicked.connect(self._copy_details)
            copy_button.setToolTip("Copy error details to clipboard for support")
            button_layout.addWidget(copy_button)

        button_layout.addStretch()

        # Continue button (if allowed)
        if self._allow_continue:
            continue_button = QPushButton("Continue")
            continue_button.clicked.connect(self.accept)
            continue_button.setToolTip("Continue using the application")
            button_layout.addWidget(continue_button)

        # Exit button
        exit_button = QPushButton("Exit")
        exit_button.clicked.connect(self.reject)
        exit_button.setDefault(not self._allow_continue)
        exit_button.setToolTip("Close the application")
        button_layout.addWidget(exit_button)

        layout.addLayout(button_layout)

        # Accessibility
        self.setAccessibleName(f"Error: {self._title}")
        self.setAccessibleDescription(self._message)

    def _toggle_details(self):
        """Toggle the visibility of the details section."""
        self._details_expanded = not self._details_expanded
        self._details_text.setVisible(self._details_expanded)

        if self._details_expanded:
            self._details_button.setText("Hide Details")
            # Expand dialog to fit details
            self.resize(self.width(), self.height() + 150)
        else:
            self._details_button.setText("Show Details")
            # Shrink dialog
            self.resize(self.width(), self.height() - 150)

    def _copy_details(self):
        """Copy error details to clipboard."""
        if self._details:
            clipboard = QApplication.clipboard()
            # Format details with context
            copy_text = f"""MyVoice Error Report
==================
Title: {self._title}
Message: {self._message}

Technical Details:
{self._details}

Time: {self._get_timestamp()}
"""
            clipboard.setText(copy_text)
            self.logger.info("Error details copied to clipboard")

            # Show brief feedback (change button text temporarily)
            sender = self.sender()
            if sender and hasattr(sender, 'setText'):
                original_text = sender.text()
                sender.setText("Copied!")
                # Reset after 2 seconds
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(2000, lambda: sender.setText(original_text))

    def _get_timestamp(self) -> str:
        """Get current timestamp for error report."""
        from datetime import datetime
        return datetime.now().isoformat()

    @staticmethod
    def show_error(
        title: str,
        message: str,
        details: Optional[str] = None,
        allow_continue: bool = True,
        parent: Optional[QWidget] = None
    ) -> bool:
        """
        Show a critical error dialog.

        Args:
            title: Dialog title
            message: User-friendly error message
            details: Technical details for support
            allow_continue: Whether to allow continuing
            parent: Parent widget

        Returns:
            True if user clicked "Continue", False if "Exit"
        """
        dialog = CriticalErrorDialog(
            title=title,
            message=message,
            details=details,
            allow_continue=allow_continue,
            parent=parent
        )
        result = dialog.exec()
        return result == QDialog.DialogCode.Accepted


class ErrorNotificationWidget(QWidget):
    """
    Non-modal error notification widget for recoverable errors.

    Story 7.6: Provides inline error display without blocking user interaction.
    """

    def __init__(self, message: str, parent: Optional[QWidget] = None):
        """
        Initialize the error notification widget.

        Args:
            message: Error message to display
            parent: Parent widget
        """
        super().__init__(parent)
        self._message = message
        self._create_ui()

    def _create_ui(self):
        """Create the notification UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Warning icon
        icon_label = QLabel()
        icon_label.setPixmap(
            self.style().standardIcon(
                self.style().StandardPixmap.SP_MessageBoxWarning
            ).pixmap(16, 16)
        )
        layout.addWidget(icon_label)

        # Message
        message_label = QLabel(self._message)
        message_label.setWordWrap(True)
        layout.addWidget(message_label, 1)

        # Dismiss button
        dismiss_button = QPushButton("x")
        dismiss_button.setFixedSize(20, 20)
        dismiss_button.setToolTip("Dismiss")
        dismiss_button.clicked.connect(self.hide)
        layout.addWidget(dismiss_button)

        # Style
        self.setStyleSheet("""
            ErrorNotificationWidget {
                background-color: #fef3c7;
                border: 1px solid #f59e0b;
                border-radius: 4px;
            }
            QLabel {
                color: #92400e;
            }
        """)
