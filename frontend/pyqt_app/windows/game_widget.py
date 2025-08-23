# frontend/pyqt_app/windows/game_widget.py
import sys
import json
import requests
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QLabel, QComboBox, QSplitter, QGroupBox,
    QListWidget, QMessageBox, QProgressBar, QTextBrowser,
    QScrollArea, QFrame
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QFont, QPixmap, QPalette

class GameWorker(QThread):
    """Worker thread for game API calls"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, endpoint, data=None):
        super().__init__()
        self.endpoint = endpoint
        self.data = data

    def run(self):
        try:
            if self.data:
                response = requests.post(
                    f"http://localhost:8000/api/v1/game/{self.endpoint}",
                    json=self.data,
                    timeout=30
                )
            else:
                response = requests.get(
                    f"http://localhost:8000/api/v1/game/{self.endpoint}",
                    timeout=30
                )

            response.raise_for_status()
            self.finished.emit(response.json())

        except Exception as e:
            self.error.emit(str(e))

class GameWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.session_id = None
        self.current_scene = None
        self.choices = []
        self.player_state = {}
        self.turn_number = 0
        self.worker = None

        self.init_ui()
        self.load_personas()
        self.load_saves()

    def init_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("🎮 文字冒險遊戲")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Game setup and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        # Game setup group
        setup_group = QGroupBox("🎯 遊戲設定")
        setup_layout = QVBoxLayout()

        setup_layout.addWidget(QLabel("角色導師:"))
        self.persona_combo = QComboBox()
        setup_layout.addWidget(self.persona_combo)

        # Persona description
        self.persona_desc = QTextBrowser()
        self.persona_desc.setMaximumHeight(80)
        setup_layout.addWidget(self.persona_desc)

        setup_layout.addWidget(QLabel("遊戲背景:"))
        self.setting_combo = QComboBox()
        self.setting_combo.addItems(["modern", "fantasy", "forest", "future"])
        setup_layout.addWidget(self.setting_combo)

        setup_layout.addWidget(QLabel("難度等級:"))
        self.difficulty_combo = QComboBox()
        self.difficulty_combo.addItems(["easy", "normal", "hard"])
        self.difficulty_combo.setCurrentText("normal")
        setup_layout.addWidget(self.difficulty_combo)

        setup_layout.addWidget(QLabel("角色名稱:"))
        self.player_name_input = QLineEdit()
        self.player_name_input.setPlaceholderText("冒險者")
        setup_layout.addWidget(self.player_name_input)

        self.start_btn = QPushButton("🚀 開始冒險")
        self.start_btn.clicked.connect(self.start_new_game)
        setup_layout.addWidget(self.start_btn)

        setup_group.setLayout(setup_layout)
        left_layout.addWidget(setup_group)

        # Save/Load group
        save_group = QGroupBox("💾 存檔管理")
        save_layout = QVBoxLayout()

        save_layout.addWidget(QLabel("存檔名稱:"))
        save_input_layout = QHBoxLayout()
        self.save_name_input = QLineEdit()
        self.save_name_input.setPlaceholderText("我的冒險")
        save_input_layout.addWidget(self.save_name_input)

        self.save_btn = QPushButton("💾")
        self.save_btn.clicked.connect(self.save_game)
        save_input_layout.addWidget(self.save_btn)
        save_layout.addLayout(save_input_layout)

        save_layout.addWidget(QLabel("載入存檔:"))
        load_layout = QHBoxLayout()
        self.saves_combo = QComboBox()
        load_layout.addWidget(self.saves_combo)

        self.refresh_saves_btn = QPushButton("🔄")
        self.refresh_saves_btn.clicked.connect(self.load_saves)
        load_layout.addWidget(self.refresh_saves_btn)
        save_layout.addLayout(load_layout)

        self.load_btn = QPushButton("📂 載入遊戲")
        self.load_btn.clicked.connect(self.load_game)
        save_layout.addWidget(self.load_btn)

        save_group.setLayout(save_layout)
        left_layout.addWidget(save_group)

        # Player status group
        self.status_group = QGroupBox("📊 玩家狀態")
        self.status_layout = QVBoxLayout()
        self.status_text = QTextBrowser()
        self.status_text.setMaximumHeight(120)
        self.status_layout.addWidget(self.status_text)
        self.status_group.setLayout(self.status_layout)
        left_layout.addWidget(self.status_group)

        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(300)
        main_splitter.addWidget(left_panel)

        # Right panel - Game content
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        # Scene display
        scene_group = QGroupBox("🌟 當前場景")
        scene_layout = QVBoxLayout()

        self.scene_display = QTextBrowser()
        self.scene_display.setMinimumHeight(200)
        scene_layout.addWidget(self.scene_display)

        scene_group.setLayout(scene_layout)
        right_layout.addWidget(scene_group)

        # Choices display
        choices_group = QGroupBox("⚡ 行動選擇")
        choices_layout = QVBoxLayout()

        # Quick choice buttons
        self.choice_buttons_layout = QVBoxLayout()
        choices_layout.addLayout(self.choice_buttons_layout)

        # Custom action input
        custom_layout = QVBoxLayout()
        custom_layout.addWidget(QLabel("自定義行動:"))

        self.custom_action_input = QLineEdit()
        self.custom_action_input.setPlaceholderText("描述你想要做的事情...")
        custom_layout.addWidget(self.custom_action_input)

        self.custom_message_input = QTextEdit()
        self.custom_message_input.setPlaceholderText("想要說的話... (可選)")
        self.custom_message_input.setMaximumHeight(60)
        custom_layout.addWidget(self.custom_message_input)

        self.custom_action_btn = QPushButton("⚡ 執行自定義行動")
        self.custom_action_btn.clicked.connect(self.take_custom_action)
        custom_layout.addWidget(self.custom_action_btn)

        choices_layout.addLayout(custom_layout)
        choices_group.setLayout(choices_layout)
        right_layout.addWidget(choices_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        # Status message
        self.status_message = QLabel("準備開始冒險！")
        self.status_message.setStyleSheet("QLabel { background-color: #e3f2fd; padding: 8px; border-radius: 4px; }")
        right_layout.addWidget(self.status_message)

        right_panel.setLayout(right_layout)
        main_splitter.addWidget(right_panel)

        main_splitter.setSizes([300, 700])
        layout.addWidget(main_splitter)

        self.setLayout(layout)

        # Connect persona selection to description update
        self.persona_combo.currentTextChanged.connect(self.update_persona_description)

    def load_personas(self):
        """Load available personas"""
        self.worker = GameWorker("personas")
        self.worker.finished.connect(self.on_personas_loaded)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_personas_loaded(self, data):
        """Handle personas loaded"""
        personas = data.get("personas", {})
        self.persona_combo.clear()

        for key, description in personas.items():
            self.persona_combo.addItem(key, description)

        if personas:
            self.update_persona_description()

    def update_persona_description(self):
        """Update persona description display"""
        current_data = self.persona_combo.currentData()
        if current_data:
            self.persona_desc.setPlainText(current_data)

    def load_saves(self):
        """Load available save files"""
        self.worker = GameWorker("saves")
        self.worker.finished.connect(self.on_saves_loaded)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_saves_loaded(self, data):
        """Handle saves loaded"""
        saves = data.get("saves", [])
        self.saves_combo.clear()
        self.saves_combo.addItems(saves)

    def start_new_game(self):
        """Start a new game"""
        persona = self.persona_combo.currentText()
        if not persona:
            QMessageBox.warning(self, "警告", "請選擇角色導師")
            return

        self.set_loading(True)

        data = {
            "persona": persona,
            "setting": self.setting_combo.currentText(),
            "difficulty": self.difficulty_combo.currentText(),
            "player_name": self.player_name_input.text() or "冒險者"
        }

        self.worker = GameWorker("new", data)
        self.worker.finished.connect(self.on_game_started)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_game_started(self, data):
        """Handle game started"""
        self.session_id = data["session_id"]
        self.update_game_display(data)
        self.status_message.setText(f"遊戲開始！會話ID：{self.session_id[:8]}...")
        self.set_loading(False)

    def take_action(self, action_id):
        """Take a predefined action"""
        if not self.session_id:
            return

        self.set_loading(True)

        data = {
            "session_id": self.session_id,
            "action": action_id
        }

        self.worker = GameWorker("step", data)
        self.worker.finished.connect(self.on_action_taken)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def take_custom_action(self):
        """Take a custom action"""
        if not self.session_id:
            return

        action = self.custom_action_input.text().strip()
        if not action:
            QMessageBox.warning(self, "警告", "請輸入行動描述")
            return

        self.set_loading(True)

        data = {
            "session_id": self.session_id,
            "action": action,
            "message": self.custom_message_input.toPlainText().strip() or None
        }

        self.worker = GameWorker("step", data)
        self.worker.finished.connect(self.on_action_taken)
        self.worker.error.connect(self.on_error)
        self.worker.start()

        # Clear inputs
        self.custom_action_input.clear()
        self.custom_message_input.clear()

    def on_action_taken(self, data):
        """Handle action taken"""
        self.update_game_display(data)
        self.status_message.setText("行動已執行！")
        self.set_loading(False)

    def save_game(self):
        """Save current game"""
        if not self.session_id:
            QMessageBox.warning(self, "警告", "沒有進行中的遊戲")
            return

        save_name = self.save_name_input.text().strip()
        if not save_name:
            QMessageBox.warning(self, "警告", "請輸入存檔名稱")
            return

        data = {
            "session_id": self.session_id,
            "save_name": save_name
        }

        self.worker = GameWorker("save", data)
        self.worker.finished.connect(lambda: self.on_game_saved(save_name))
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_game_saved(self, save_name):
        """Handle game saved"""
        self.status_message.setText(f"✅ 遊戲已保存：{save_name}")
        self.save_name_input.clear()
        self.load_saves()  # Refresh save list

    def load_game(self):
        """Load a saved game"""
        save_name = self.saves_combo.currentText()
        if not save_name:
            QMessageBox.warning(self, "警告", "請選擇存檔")
            return

        self.set_loading(True)

        data = {"save_name": save_name}

        self.worker = GameWorker("load", data)
        self.worker.finished.connect(lambda data: self.on_game_loaded(data, save_name))
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_game_loaded(self, data, save_name):
        """Handle game loaded"""
        self.session_id = data["session_id"]
        self.update_game_display(data)
        self.status_message.setText(f"✅ 遊戲已載入：{save_name}")
        self.set_loading(False)

    def update_game_display(self, data):
        """Update game display with new data"""
        self.current_scene = data["scene"]
        self.choices = data["choices"]
        self.player_state = data["player_state"]
        self.turn_number = data["turn_number