# tests/test_pyqt.py
import pytest
import sys
import os

# Skip PyQt tests if PyQt6 is not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for testing"""
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    yield app


def test_chat_widget_creation(qapp):
    """Test that ChatWidget can be created"""
    from frontend.pyqt_app.widgets.chat_widget import ChatWidget

    widget = ChatWidget()
    assert widget is not None
    assert widget.windowTitle() == ""  # Default title


def test_api_client():
    """Test API client basic functionality"""
    from frontend.pyqt_app.utils.api_client import APIClient

    client = APIClient("http://localhost:8000/api/v1")
    assert client.base_url == "http://localhost:8000/api/v1"


def test_main_window_creation(qapp):
    """Test that main window can be created"""
    from frontend.pyqt_app.main import MainWindow

    window = MainWindow()
    assert window is not None
    assert "CharaForge" in window.windowTitle()

    # Test that chat tab exists
    assert window.tab_widget.count() >= 1
    assert "Chat" in window.tab_widget.tabText(0)
