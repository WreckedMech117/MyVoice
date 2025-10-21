"""
Resize Grip Component

This module implements edge resize grips for frameless windows,
allowing users to resize the window from all edges and corners.
"""

import logging
from typing import Optional

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPoint, QRect
from PyQt6.QtGui import QMouseEvent, QCursor


class SideGrip(QWidget):
    """
    Edge resize grip for frameless windows.

    Provides a thin, invisible resize area along window edges
    with appropriate cursors and mouse handling.
    """

    # Edge constants
    EDGE_LEFT = 'left'
    EDGE_TOP = 'top'
    EDGE_RIGHT = 'right'
    EDGE_BOTTOM = 'bottom'

    # Grip size in pixels (sleek design)
    GRIP_SIZE = 6

    def __init__(self, parent: Optional[QWidget] = None, edge: str = EDGE_LEFT):
        """
        Initialize the side grip.

        Args:
            parent: Parent widget (the main window)
            edge: Which edge this grip handles ('left', 'top', 'right', 'bottom')
        """
        super().__init__(parent)
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{edge}")

        self._parent_window = parent
        self._edge = edge
        self._is_resizing = False
        self._resize_start_pos = QPoint()
        self._resize_start_geometry = QRect()

        # Set fixed size based on edge orientation
        self._setup_geometry()

        # Set appropriate cursor for the edge
        self._setup_cursor()

        # Make grip semi-transparent (invisible but interactive)
        self.setStyleSheet("background: transparent;")

        self.logger.debug(f"SideGrip initialized for {edge} edge")

    def _setup_geometry(self):
        """Configure grip size based on edge orientation."""
        if self._edge in (self.EDGE_LEFT, self.EDGE_RIGHT):
            # Vertical grips (left/right)
            self.setFixedWidth(self.GRIP_SIZE)
        else:
            # Horizontal grips (top/bottom)
            self.setFixedHeight(self.GRIP_SIZE)

    def _setup_cursor(self):
        """Set the appropriate cursor shape for this edge."""
        cursor_map = {
            self.EDGE_LEFT: Qt.CursorShape.SizeHorCursor,
            self.EDGE_RIGHT: Qt.CursorShape.SizeHorCursor,
            self.EDGE_TOP: Qt.CursorShape.SizeVerCursor,
            self.EDGE_BOTTOM: Qt.CursorShape.SizeVerCursor,
        }
        self.setCursor(QCursor(cursor_map.get(self._edge, Qt.CursorShape.ArrowCursor)))

    def mousePressEvent(self, event: QMouseEvent):
        """
        Handle mouse press to initiate resizing.

        Args:
            event: Mouse event
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_resizing = True
            self._resize_start_pos = event.globalPosition().toPoint()
            self._resize_start_geometry = self._parent_window.geometry()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """
        Handle mouse move to resize the window.

        Args:
            event: Mouse event
        """
        if self._is_resizing and event.buttons() == Qt.MouseButton.LeftButton:
            delta = event.globalPosition().toPoint() - self._resize_start_pos
            self._resize_window(delta)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """
        Handle mouse release to stop resizing.

        Args:
            event: Mouse event
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_resizing = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def _resize_window(self, delta: QPoint):
        """
        Resize the window based on mouse delta and edge type.

        Args:
            delta: Mouse movement delta from start position
        """
        if not self._parent_window:
            return

        # Get current geometry
        rect = QRect(self._resize_start_geometry)

        # Apply resize based on edge
        if self._edge == self.EDGE_LEFT:
            # Resize from left edge
            new_width = rect.width() - delta.x()
            if new_width >= self._parent_window.minimumWidth():
                rect.setLeft(self._resize_start_geometry.left() + delta.x())
        elif self._edge == self.EDGE_RIGHT:
            # Resize from right edge
            new_width = rect.width() + delta.x()
            if new_width >= self._parent_window.minimumWidth():
                rect.setWidth(new_width)
        elif self._edge == self.EDGE_TOP:
            # Resize from top edge
            new_height = rect.height() - delta.y()
            if new_height >= self._parent_window.minimumHeight():
                rect.setTop(self._resize_start_geometry.top() + delta.y())
        elif self._edge == self.EDGE_BOTTOM:
            # Resize from bottom edge
            new_height = rect.height() + delta.y()
            if new_height >= self._parent_window.minimumHeight():
                rect.setHeight(new_height)

        # Apply the new geometry
        self._parent_window.setGeometry(rect)


class CornerGrip(QWidget):
    """
    Corner resize grip for frameless windows.

    Provides diagonal resizing from window corners with
    appropriate bi-directional cursors.
    """

    # Corner constants
    CORNER_TOP_LEFT = 'top_left'
    CORNER_TOP_RIGHT = 'top_right'
    CORNER_BOTTOM_LEFT = 'bottom_left'
    CORNER_BOTTOM_RIGHT = 'bottom_right'

    # Grip size in pixels
    GRIP_SIZE = 6

    def __init__(self, parent: Optional[QWidget] = None, corner: str = CORNER_TOP_LEFT):
        """
        Initialize the corner grip.

        Args:
            parent: Parent widget (the main window)
            corner: Which corner this grip handles
        """
        super().__init__(parent)
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{corner}")

        self._parent_window = parent
        self._corner = corner
        self._is_resizing = False
        self._resize_start_pos = QPoint()
        self._resize_start_geometry = QRect()

        # Set fixed size
        self.setFixedSize(self.GRIP_SIZE, self.GRIP_SIZE)

        # Set appropriate cursor for the corner
        self._setup_cursor()

        # Make grip semi-transparent
        self.setStyleSheet("background: transparent;")

        self.logger.debug(f"CornerGrip initialized for {corner} corner")

    def _setup_cursor(self):
        """Set the appropriate diagonal cursor shape for this corner."""
        cursor_map = {
            self.CORNER_TOP_LEFT: Qt.CursorShape.SizeFDiagCursor,
            self.CORNER_TOP_RIGHT: Qt.CursorShape.SizeBDiagCursor,
            self.CORNER_BOTTOM_LEFT: Qt.CursorShape.SizeBDiagCursor,
            self.CORNER_BOTTOM_RIGHT: Qt.CursorShape.SizeFDiagCursor,
        }
        self.setCursor(QCursor(cursor_map.get(self._corner, Qt.CursorShape.ArrowCursor)))

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press to initiate corner resizing."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_resizing = True
            self._resize_start_pos = event.globalPosition().toPoint()
            self._resize_start_geometry = self._parent_window.geometry()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move to resize from corner."""
        if self._is_resizing and event.buttons() == Qt.MouseButton.LeftButton:
            delta = event.globalPosition().toPoint() - self._resize_start_pos
            self._resize_window(delta)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release to stop resizing."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_resizing = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def _resize_window(self, delta: QPoint):
        """
        Resize the window diagonally from corner.

        Args:
            delta: Mouse movement delta from start position
        """
        if not self._parent_window:
            return

        rect = QRect(self._resize_start_geometry)
        min_width = self._parent_window.minimumWidth()
        min_height = self._parent_window.minimumHeight()

        if self._corner == self.CORNER_TOP_LEFT:
            # Resize from top-left corner
            new_width = rect.width() - delta.x()
            new_height = rect.height() - delta.y()
            if new_width >= min_width:
                rect.setLeft(self._resize_start_geometry.left() + delta.x())
            if new_height >= min_height:
                rect.setTop(self._resize_start_geometry.top() + delta.y())

        elif self._corner == self.CORNER_TOP_RIGHT:
            # Resize from top-right corner
            new_width = rect.width() + delta.x()
            new_height = rect.height() - delta.y()
            if new_width >= min_width:
                rect.setWidth(new_width)
            if new_height >= min_height:
                rect.setTop(self._resize_start_geometry.top() + delta.y())

        elif self._corner == self.CORNER_BOTTOM_LEFT:
            # Resize from bottom-left corner
            new_width = rect.width() - delta.x()
            new_height = rect.height() + delta.y()
            if new_width >= min_width:
                rect.setLeft(self._resize_start_geometry.left() + delta.x())
            if new_height >= min_height:
                rect.setHeight(new_height)

        elif self._corner == self.CORNER_BOTTOM_RIGHT:
            # Resize from bottom-right corner
            new_width = rect.width() + delta.x()
            new_height = rect.height() + delta.y()
            if new_width >= min_width:
                rect.setWidth(new_width)
            if new_height >= min_height:
                rect.setHeight(new_height)

        self._parent_window.setGeometry(rect)