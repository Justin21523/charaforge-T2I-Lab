# frontend/pyqt_app/main.py
import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QFont
from windows.chat_widget import ChatWidget
from windows.caption_widget import CaptionWidget
from windows.vqa_widget import VQAWidget
from windows.rag_widget import RAGWidget


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
        central_widget.setLayout(layout)

        # Add chat tab
        self.chat_widget = ChatWidget()
        self.tab_widget.addTab(self.chat_widget, "💬 Chat")

        # Create tab widget
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        tabs.setMovable(True)

        # Add tabs
        tabs.addTab(CaptionWidget(self.api_base_url), "📷 Caption")
        tabs.addTab(VQAWidget(self.api_base_url), "❓ VQA")
        tabs.addTab(ChatWidget(self.api_base_url), "💬 Chat")
        tabs.addTab(RAGWidget(self.api_base_url), "📚 RAG")  # NEW

        layout.addWidget(tabs)

        # Set window icon (if available)
        try:
            self.setWindowIcon(QIcon("icon.png"))
        except:
            pass

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

    def closeEvent(self, event):
        """Handle application close"""
        reply = self.confirm_exit()
        if reply:
            event.accept()
        else:
            event.ignore()

    def confirm_exit(self):
        """Confirm application exit"""
        from PyQt6.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self,
            "Exit Application",
            "Are you sure you want to exit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        return reply == QMessageBox.StandardButton.Yes


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
