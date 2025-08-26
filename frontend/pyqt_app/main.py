# frontend/pyqt_app/main.py
"""
SagaForge T2I Lab - PyQt Desktop Application
Main entry point
"""
import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QStyleFactory, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPalette, QColor

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from frontend.shared.api_client import SagaForgeAPIClient
from frontend.pyqt_app.main_window import MainWindow


def setup_dark_theme(app):
    """Setup dark theme for the application"""
    app.setStyle(QStyleFactory.create("Fusion"))

    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)

    app.setPalette(palette)


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("SagaForge T2I Lab")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("SagaForge")

    # Set application icon
    icon_path = Path(__file__).parent / "resources" / "icons" / "app_icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # Setup dark theme
    setup_dark_theme(app)

    # Test API connection
    api_client = SagaForgeAPIClient()
    health = api_client.health_check()

    if health.get("status") == "error":
        QMessageBox.warning(
            None,
            "連線警告",
            f"無法連接到 API 服務器:\n{health.get('message', 'Unknown error')}\n\n"
            "請確保 API 服務器正在運行 (python -m api.main)",
        )

    # Create and show main window
    main_window = MainWindow(api_client)
    main_window.show()

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
