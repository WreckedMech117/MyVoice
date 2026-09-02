"""Fixtures for the local TTS API tests."""

import pytest


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication (project convention; no pytest-qt)."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
