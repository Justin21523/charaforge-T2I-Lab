# frontend/pyqt_app/main_window.py
"""
SagaForge T2I Lab - Main Window
"""
import os
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QMenuBar,
    QStatusBar,
    QAction,
    QMessageBox,
    QSplitter,
    QLabel,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont

from frontend.pyqt_app.widgets.generation_widget import GenerationWidget
from frontend.pyqt_app.widgets.lora_manager_widget import LoRAManagerWidget
from frontend.pyqt_app.widgets.batch_widget import BatchWidget
from frontend.pyqt_app.widgets.training_widget import TrainingWidget
from frontend.pyqt_app.widgets.gallery_widget import GalleryWidget
from frontend.pyqt_app.dialogs.settings_dialog import SettingsDialog
from frontend.pyqt_app.dialogs.about_dialog import AboutDialog


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client
        self.setup_ui()
        self.setup_menu()
        self.setup_statusbar()
        self.setup_connections()

        # Health check timer
        self.health_timer = QTimer()
        self.health_timer.timeout.connect(self.check_api_health)
        self.health_timer.start(30000)  # Check every 30 seconds

    def setup_ui(self):
        """Setup the main user interface"""
        self.setWindowTitle("SagaForge T2I Lab")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        # Left panel (controls)
        left_panel = QWidget()
        left_panel.setMinimumWidth(400)
        left_panel.setMaximumWidth(600)

        left_layout = QVBoxLayout(left_panel)

        # Tab widget for different functions
        self.tab_widget = QTabWidget()
        left_layout.addWidget(self.tab_widget)

        # Generation tab
        self.generation_widget = GenerationWidget(self.api_client)
        self.tab_widget.addTab(self.generation_widget, "生成")

        # LoRA Manager tab
        self.lora_widget = LoRAManagerWidget(self.api_client)
        self.tab_widget.addTab(self.lora_widget, "LoRA 管理")

        # Batch processing tab
        self.batch_widget = BatchWidget(self.api_client)
        self.tab_widget.addTab(self.batch_widget, "批次處理")

        # Training tab
        self.training_widget = TrainingWidget(self.api_client)
        self.tab_widget.addTab(self.training_widget, "訓練監控")

        # Right panel (gallery)
        self.gallery_widget = GalleryWidget()

        # Add to splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(self.gallery_widget)

        # Set splitter proportions
        main_splitter.setSizes([400, 800])

    def setup_menu(self):
        """Setup application menu"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("檔案")

        open_action = QAction("開啟圖片", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.gallery_widget.open_image)
        file_menu.addAction(open_action)

        save_action = QAction("儲存圖片", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.gallery_widget.save_current_image)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("編輯")

        settings_action = QAction("設定", self)
        settings_action.triggered.connect(self.show_settings)
        edit_menu.addAction(settings_action)

        # Help menu
        help_menu = menubar.addMenu("說明")

        about_action = QAction("關於", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_statusbar(self):
        """Setup status bar"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        # API status label
        self.api_status_label = QLabel("API: 檢查中...")
        self.statusbar.addPermanentWidget(self.api_status_label)

        # Progress info
        self.progress_label = QLabel("就緒")
        self.statusbar.addWidget(self.progress_label)

    def setup_connections(self):
        """Setup signal connections between widgets"""
        # Connect generation widget to gallery
        self.generation_widget.image_generated.connect(self.gallery_widget.add_image)

        # Connect batch widget to gallery
        self.batch_widget.images_generated.connect(self.gallery_widget.add_images)

        # Update status bar
        self.generation_widget.status_changed.connect(self.update_status)
        self.batch_widget.status_changed.connect(self.update_status)
        self.training_widget.status_changed.connect(self.update_status)

    def check_api_health(self):
        """Check API server health"""
        try:
            health = self.api_client.health_check()
            if health.get("status") == "ok":
                self.api_status_label.setText("API: ✅ 已連線")
                self.api_status_label.setStyleSheet("color: green;")
            else:
                self.api_status_label.setText("API: ❌ 錯誤")
                self.api_status_label.setStyleSheet("color: red;")
        except Exception as e:
            self.api_status_label.setText("API: ❌ 離線")
            self.api_status_label.setStyleSheet("color: red;")

    def update_status(self, message):
        """Update status bar message"""
        self.progress_label.setText(message)

    def show_settings(self):
        """Show settings dialog"""
        dialog = SettingsDialog(self)
        dialog.exec_()

    def show_about(self):
        """Show about dialog"""
        dialog = AboutDialog(self)
        dialog.exec_()

    def closeEvent(self, event):
        """Handle application close event"""
        reply = QMessageBox.question(
            self,
            "確認退出",
            "確定要退出 SagaForge T2I Lab？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # Stop timers
            if hasattr(self, "health_timer"):
                self.health_timer.stop()
            event.accept()
        else:
            event.ignore()
