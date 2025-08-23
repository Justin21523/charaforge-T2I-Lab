# frontend/pyqt_app/widgets/chat_widget.py
import sys
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QLabel,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QPixmap
import json
import time
from typing import List, Dict
from frontend.pyqt_app.utils.api_client import APIClient


class ChatWorker(QThread):
    """Background worker for chat API calls"""

    response_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        api_client: APIClient,
        messages: List[Dict],
        session_id: str = None,  # type: ignore
        **kwargs,
    ):
        super().__init__()
        self.api_client = api_client
        self.messages = messages
        self.session_id = session_id
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.api_client.chat_completion(
                messages=self.messages, session_id=self.session_id, **self.kwargs
            )

            if "error" in result:
                self.error_occurred.emit(result["error"])
            else:
                self.response_ready.emit(result)

        except Exception as e:
            self.error_occurred.emit(str(e))


class ChatWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.api_client = APIClient()
        self.session_id = None
        self.messages = []
        self.worker = None

        self.init_ui()
        self.check_api_connection()

    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)

        # Header with title and status
        header_layout = QHBoxLayout()

        title_label = QLabel("CharaForge Chat")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        header_layout.addWidget(title_label)

        self.status_label = QLabel("🔴 Disconnected")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        header_layout.addWidget(self.status_label)

        layout.addLayout(header_layout)

        # Settings panel
        settings_group = QGroupBox("Settings")
        settings_layout = QHBoxLayout(settings_group)

        # Model selection
        settings_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["DialoGPT-medium", "Qwen-7B-Chat"])
        settings_layout.addWidget(self.model_combo)

        # Temperature control
        settings_layout.addWidget(QLabel("Temperature:"))
        self.temp_spinbox = QDoubleSpinBox()
        self.temp_spinbox.setRange(0.1, 2.0)
        self.temp_spinbox.setValue(0.7)
        self.temp_spinbox.setSingleStep(0.1)
        settings_layout.addWidget(self.temp_spinbox)

        # Max length control
        settings_layout.addWidget(QLabel("Max Length:"))
        self.length_spinbox = QSpinBox()
        self.length_spinbox.setRange(10, 1000)
        self.length_spinbox.setValue(200)
        settings_layout.addWidget(self.length_spinbox)

        # New session button
        self.new_session_btn = QPushButton("New Session")
        self.new_session_btn.clicked.connect(self.start_new_session)
        settings_layout.addWidget(self.new_session_btn)

        layout.addWidget(settings_group)

        # Chat display area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Arial", 10))
        self.chat_display.setMinimumHeight(400)
        self.chat_display.setStyleSheet(
            """
            QTextEdit {
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 10px;
                background-color: #f9f9f9;
            }
        """
        )
        layout.addWidget(self.chat_display)

        # Input area
        input_layout = QHBoxLayout()

        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type your message here...")
        self.message_input.setMinimumHeight(40)
        self.message_input.setStyleSheet(
            """
            QLineEdit {
                border: 2px solid #ddd;
                border-radius: 20px;
                padding: 10px 15px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #007acc;
            }
        """
        )
        self.message_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.message_input)

        self.send_button = QPushButton("Send")
        self.send_button.setMinimumHeight(40)
        self.send_button.setMinimumWidth(80)
        self.send_button.setStyleSheet(
            """
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                border-radius: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """
        )
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)

        layout.addLayout(input_layout)

        # Status bar
        self.info_label = QLabel("Ready to chat")
        self.info_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.info_label)

        self.setMinimumSize(600, 500)

    def check_api_connection(self):
        """Check API connection status"""
        try:
            health = self.api_client.health_check()
            if "error" not in health and health.get("status") == "healthy":
                self.status_label.setText("🟢 Connected")
                self.status_label.setStyleSheet("color: green; font-weight: bold;")
                self.info_label.setText(
                    f"API ready | GPU: {health.get('gpu_available', False)}"
                )

                # Load available models
                models_info = self.api_client.list_chat_models()
                if "available_models" in models_info:
                    self.model_combo.clear()
                    for model in models_info["available_models"]:
                        model_name = model.split("/")[-1]  # Get short name
                        self.model_combo.addItem(model_name, model)

            else:
                self.status_label.setText("🔴 API Error")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                self.info_label.setText("Failed to connect to API")

        except Exception as e:
            self.status_label.setText("🔴 Disconnected")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.info_label.setText(f"Connection error: {str(e)[:50]}...")

    def start_new_session(self):
        """Start a new chat session"""
        self.session_id = None
        self.messages.clear()
        self.chat_display.clear()
        self.add_system_message("New conversation started")
        self.info_label.setText("New session started")

    def add_system_message(self, message: str):
        """Add system message to chat display"""
        self.chat_display.append(
            f'<div style="color: #888; font-style: italic; margin: 5px 0;">🤖 {message}</div>'
        )

    def add_user_message(self, message: str):
        """Add user message to chat display"""
        timestamp = time.strftime("%H:%M")
        self.chat_display.append(
            f"""
        <div style="margin: 10px 0; text-align: right;">
            <div style="background: #007acc; color: white; padding: 10px; border-radius: 15px; display: inline-block; max-width: 70%; margin-left: 30%;">
                <strong>You</strong> <span style="opacity: 0.7; font-size: 10px;">{timestamp}</span><br>
                {message}
            </div>
        </div>
        """
        )
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    def add_assistant_message(self, message: str, model_info: str = ""):
        """Add assistant message to chat display"""
        timestamp = time.strftime("%H:%M")
        self.chat_display.append(
            f"""
        <div style="margin: 10px 0;">
            <div style="background: #e9e9e9; color: #333; padding: 10px; border-radius: 15px; display: inline-block; max-width: 70%;">
                <strong>Assistant</strong> <span style="opacity: 0.7; font-size: 10px;">{timestamp}</span><br>
                {message}
                {f'<br><small style="opacity: 0.5;">{model_info}</small>' if model_info else ''}
            </div>
        </div>
        """
        )
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    def send_message(self):
        """Send message to chat API"""
        message_text = self.message_input.text().strip()
        if not message_text:
            return

        # Disable input during processing
        self.message_input.setEnabled(False)
        self.send_button.setEnabled(False)
        self.info_label.setText("Thinking...")

        # Add user message to display
        self.add_user_message(message_text)

        # Add user message to conversation
        user_message = {
            "role": "user",
            "content": message_text,
            "timestamp": time.time(),
        }
        self.messages.append(user_message)

        # Clear input
        self.message_input.clear()

        # Start background API call
        self.worker = ChatWorker(
            api_client=self.api_client,
            messages=self.messages,
            session_id=self.session_id,
            max_length=self.length_spinbox.value(),
            temperature=self.temp_spinbox.value(),
        )

        self.worker.response_ready.connect(self.on_response_received)
        self.worker.error_occurred.connect(self.on_error_occurred)
        self.worker.start()

    def on_response_received(self, response: dict):
        """Handle successful API response"""
        try:
            # Extract response data
            assistant_message = response.get("message", {})
            content = assistant_message.get("content", "No response")

            # Update session ID
            if response.get("session_id"):
                self.session_id = response["session_id"]

            # Add assistant message to conversation
            self.messages.append(assistant_message)

            # Display response
            model_info = f"Model: {response.get('model_used', 'Unknown')} | Time: {response.get('elapsed_ms', 0)}ms"
            if response.get("safety_filtered"):
                model_info += " | ⚠️ Safety filtered"

            self.add_assistant_message(content, model_info)

            # Update status
            self.info_label.setText(
                f"Response received ({response.get('elapsed_ms', 0)}ms)"
            )

        except Exception as e:
            self.on_error_occurred(f"Failed to process response: {e}")

        finally:
            # Re-enable input
            self.message_input.setEnabled(True)
            self.send_button.setEnabled(True)
            self.message_input.setFocus()

    def on_error_occurred(self, error_message: str):
        """Handle API errors"""
        self.add_system_message(f"Error: {error_message}")
        self.info_label.setText(f"Error: {error_message[:50]}...")

        # Re-enable input
        self.message_input.setEnabled(True)
        self.send_button.setEnabled(True)
        self.message_input.setFocus()
