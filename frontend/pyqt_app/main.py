# frontend/pyqt_app/main.py
import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QFont
from widgets.chat_widget import ChatWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """Initialize main window UI"""
        self.setWindowTitle("CharaForge Multi-Modal Lab")
        self.setGeometry(100, 100, 800, 600)

        # Set application icon (if available)
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "assets", "icon.png")
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
        except:
            pass

        # Create central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Add chat tab
        self.chat_widget = ChatWidget()
        self.tab_widget.addTab(self.chat_widget, "💬 Chat")

        # TODO: Add more tabs for other features
        # self.tab_widget.addTab(CaptionWidget(), "📷 Caption")
        # self.tab_widget.addTab(VQAWidget(), "❓ VQA")

        # Set default tab
        self.tab_widget.setCurrentIndex(0)

        # Status bar
        self.statusBar().showMessage("Ready")

        # Set modern styling
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #ccc;
                background-color: white;
            }
            QTabWidget::tab-bar {
                alignment: left;
            }
            QTabBar::tab {
                background-color: #e1e1e1;
                border: 1px solid #ccc;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom-color: white;
            }
            QTabBar::tab:hover {
                background-color: #d1d1d1;
            }
        """
        )


def main():
    """Main application entry point"""
    # Set high DPI attributes
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("CharaForge Multi-Modal Lab")
    app.setApplicationVersion("0.3.0")
    app.setOrganizationName("CharaForge")

    # Set default font
    font = QFont("Arial", 10)
    app.setFont(font)

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
