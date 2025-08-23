# frontend/pyqt_app/windows/agent_widget.py
import sys
import json
import requests
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QLabel,
    QSpinBox,
    QSplitter,
    QGroupBox,
    QListWidget,
    QMessageBox,
    QProgressBar,
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont


class AgentWorker(QThread):
    """Worker thread for agent API calls"""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, query, tools, max_iterations):
        super().__init__()
        self.query = query
        self.tools = tools
        self.max_iterations = max_iterations

    def run(self):
        try:
            tool_list = (
                [t.strip() for t in self.tools.split(",") if t.strip()]
                if self.tools
                else None
            )

            response = requests.post(
                "http://localhost:8000/api/v1/agent/act",
                json={
                    "query": self.query,
                    "tools": tool_list,
                    "max_iterations": self.max_iterations,
                    "temperature": 0.1,
                },
                timeout=30,
            )
            response.raise_for_status()

            self.finished.emit(response.json())

        except Exception as e:
            self.error.emit(str(e))


class AgentWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()
        self.load_available_tools()

    def init_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("🤖 AI Agent")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        # Input section
        input_group = QGroupBox("Query Input")
        input_layout = QVBoxLayout()

        # Query input
        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText("Ask your question here...")
        self.query_input.setMaximumHeight(100)
        input_layout.addWidget(QLabel("Your Question:"))
        input_layout.addWidget(self.query_input)

        # Tools and iterations
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Tools:"))
        self.tools_input = QLineEdit()
        self.tools_input.setPlaceholderText("calculator, web_search (empty for all)")
        controls_layout.addWidget(self.tools_input)

        controls_layout.addWidget(QLabel("Max Iterations:"))
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 5)
        self.max_iter_spin.setValue(3)
        controls_layout.addWidget(self.max_iter_spin)

        input_layout.addLayout(controls_layout)

        # Execute button
        self.execute_btn = QPushButton("🚀 Execute")
        self.execute_btn.clicked.connect(self.execute_agent)
        input_layout.addWidget(self.execute_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        input_layout.addWidget(self.progress_bar)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Results section
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Available tools
        tools_group = QGroupBox("Available Tools")
        tools_layout = QVBoxLayout()

        self.tools_list = QListWidget()
        tools_layout.addWidget(self.tools_list)

        reload_tools_btn = QPushButton("🔄 Reload Tools")
        reload_tools_btn.clicked.connect(self.load_available_tools)
        tools_layout.addWidget(reload_tools_btn)

        tools_group.setLayout(tools_layout)
        splitter.addWidget(tools_group)

        # Response section
        response_group = QGroupBox("Agent Response")
        response_layout = QVBoxLayout()

        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        response_layout.addWidget(self.response_text)

        response_group.setLayout(response_layout)
        splitter.addWidget(response_group)

        splitter.setSizes([300, 500])
        layout.addWidget(splitter)

        # Example queries
        examples_group = QGroupBox("Example Queries")
        examples_layout = QVBoxLayout()

        examples = [
            "Calculate the area of a circle with radius 5",
            "Search for information about machine learning",
            "What's 15% of 1250, then search for tax information",
            "List files in the current directory",
            "Calculate sin(pi/4) + cos(pi/3)",
        ]

        for example in examples:
            btn = QPushButton(example)
            btn.clicked.connect(
                lambda checked, text=example: self.query_input.setPlainText(text)
            )
            examples_layout.addWidget(btn)

        examples_group.setLayout(examples_layout)
        layout.addWidget(examples_group)

        self.setLayout(layout)

    def load_available_tools(self):
        """Load available tools from API"""
        try:
            response = requests.get(
                "http://localhost:8000/api/v1/agent/tools", timeout=5
            )
            response.raise_for_status()

            data = response.json()
            tools = data["tools"]

            self.tools_list.clear()
            for name, description in tools.items():
                self.tools_list.addItem(f"{name}: {description}")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load tools: {str(e)}")

    def execute_agent(self):
        """Execute agent query"""
        query = self.query_input.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "Warning", "Please enter a question.")
            return

        tools = self.tools_input.text().strip()
        max_iterations = self.max_iter_spin.value()

        # Disable UI and show progress
        self.execute_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

        # Start worker thread
        self.worker = AgentWorker(query, tools, max_iterations)
        self.worker.finished.connect(self.on_agent_finished)
        self.worker.error.connect(self.on_agent_error)
        self.worker.start()

    def on_agent_finished(self, response):
        """Handle successful agent response"""
        # Enable UI
        self.execute_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Format response
        output = f"Query: {response['query']}\n\n"
        output += f"Answer: {response['final_answer']}\n\n"
        output += f"Total Time: {response['total_time_ms']}ms\n\n"

        if response["reasoning_steps"]:
            output += "Reasoning Steps:\n"
            for step in response["reasoning_steps"]:
                output += f"• {step}\n"
            output += "\n"

        if response["tool_calls"]:
            output += "Tool Executions:\n"
            for tool_call in response["tool_calls"]:
                status = "✅" if tool_call["success"] else "❌"
                output += f"{status} {tool_call['tool_name']} ({tool_call['execution_time_ms']}ms)\n"
                if tool_call["success"]:
                    output += f"   Result: {tool_call['result']}\n"
                else:
                    output += f"   Error: {tool_call['error']}\n"
                output += "\n"

        self.response_text.setPlainText(output)

    def on_agent_error(self, error_msg):
        """Handle agent error"""
        # Enable UI
        self.execute_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        QMessageBox.critical(self, "Error", f"Agent execution failed: {error_msg}")
        self.response_text.setPlainText(f"Error: {error_msg}")
