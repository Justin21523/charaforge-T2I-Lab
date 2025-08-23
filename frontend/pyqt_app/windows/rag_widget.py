# frontend/pyqt_app/windows/rag_widget.py
import sys
import json
import requests
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QSlider,
    QSpinBox,
    QGroupBox,
    QGridLayout,
    QProgressBar,
    QTabWidget,
    QMessageBox,
    QSplitter,
    QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
import tempfile
import os


class RAGWorker(QThread):
    """Background worker for RAG operations"""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, operation, api_base_url, **kwargs):
        super().__init__()
        self.operation = operation
        self.api_base_url = api_base_url
        self.kwargs = kwargs

    def run(self):
        try:
            if self.operation == "upload":
                self.upload_document()
            elif self.operation == "query":
                self.query_rag()
            elif self.operation == "status":
                self.get_status()
        except Exception as e:
            self.error.emit(str(e))

    def upload_document(self):
        file_path = self.kwargs["file_path"]
        collection_name = self.kwargs["collection_name"]
        chunk_size = self.kwargs["chunk_size"]
        chunk_overlap = self.kwargs["chunk_overlap"]

        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {
                "collection_name": collection_name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }

            response = requests.post(
                f"{self.api_base_url}/rag/upload", files=files, data=data, timeout=60
            )

            if response.status_code == 200:
                self.finished.emit(response.json())
            else:
                self.error.emit(f"Upload failed: {response.text}")

    def query_rag(self):
        payload = {
            "question": self.kwargs["question"],
            "collection_name": self.kwargs["collection_name"],
            "top_k": self.kwargs["top_k"],
            "max_length": self.kwargs["max_length"],
            "temperature": self.kwargs["temperature"],
        }

        response = requests.post(
            f"{self.api_base_url}/rag/ask", json=payload, timeout=30
        )

        if response.status_code == 200:
            self.finished.emit(response.json())
        else:
            self.error.emit(f"Query failed: {response.text}")

    def get_status(self):
        response = requests.get(f"{self.api_base_url}/rag/status", timeout=10)

        if response.status_code == 200:
            self.finished.emit(response.json())
        else:
            self.error.emit(f"Status failed: {response.text}")


class RAGWidget(QWidget):
    """PyQt RAG interface widget"""

    def __init__(self, api_base_url="http://localhost:8000/api/v1"):
        super().__init__()
        self.api_base_url = api_base_url
        self.current_worker = None
        self.init_ui()
        self.load_status()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Title
        title = QLabel("📚 RAG Query System")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Upload and Status
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel: Query
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([400, 600])
        layout.addWidget(splitter)

        self.setLayout(layout)
        self.setWindowTitle("RAG Query System")
        self.resize(1000, 700)

    def create_left_panel(self):
        """Create left panel with upload and status"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Upload section
        upload_group = QGroupBox("📤 Upload Document")
        upload_layout = QVBoxLayout()

        # File selection
        file_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select a file...")
        self.file_browse_btn = QPushButton("📁 Browse")
        self.file_browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.file_browse_btn)
        upload_layout.addLayout(file_layout)

        # Collection name
        self.collection_edit = QLineEdit("default")
        upload_layout.addWidget(QLabel("📁 Collection Name:"))
        upload_layout.addWidget(self.collection_edit)

        # Parameters
        params_layout = QGridLayout()

        # Chunk size
        params_layout.addWidget(QLabel("📏 Chunk Size:"), 0, 0)
        self.chunk_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.chunk_size_slider.setRange(128, 1024)
        self.chunk_size_slider.setValue(512)
        self.chunk_size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.chunk_size_label = QLabel("512")
        self.chunk_size_slider.valueChanged.connect(
            lambda v: self.chunk_size_label.setText(str(v))
        )
        params_layout.addWidget(self.chunk_size_slider, 0, 1)
        params_layout.addWidget(self.chunk_size_label, 0, 2)

        # Chunk overlap
        params_layout.addWidget(QLabel("🔗 Overlap:"), 1, 0)
        self.chunk_overlap_slider = QSlider(Qt.Orientation.Horizontal)
        self.chunk_overlap_slider.setRange(0, 200)
        self.chunk_overlap_slider.setValue(50)
        self.chunk_overlap_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.chunk_overlap_label = QLabel("50")
        self.chunk_overlap_slider.valueChanged.connect(
            lambda v: self.chunk_overlap_label.setText(str(v))
        )
        params_layout.addWidget(self.chunk_overlap_slider, 1, 1)
        params_layout.addWidget(self.chunk_overlap_label, 1, 2)

        upload_layout.addLayout(params_layout)

        # Upload button
        self.upload_btn = QPushButton("📤 Upload Document")
        self.upload_btn.clicked.connect(self.upload_document)
        upload_layout.addWidget(self.upload_btn)

        # Upload status
        self.upload_status = QTextEdit()
        self.upload_status.setMaximumHeight(100)
        self.upload_status.setPlaceholderText("Upload status will appear here...")
        upload_layout.addWidget(self.upload_status)

        upload_group.setLayout(upload_layout)
        layout.addWidget(upload_group)

        # Status section
        status_group = QGroupBox("📊 System Status")
        status_layout = QVBoxLayout()

        # Refresh button
        refresh_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.load_status)
        refresh_layout.addWidget(self.refresh_btn)
        refresh_layout.addStretch()
        status_layout.addLayout(refresh_layout)

        # Status display
        self.status_display = QTextEdit()
        self.status_display.setMaximumHeight(150)
        self.status_display.setReadOnly(True)
        status_layout.addWidget(self.status_display)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_right_panel(self):
        """Create right panel with query interface"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Query section
        query_group = QGroupBox("❓ Ask Question")
        query_layout = QVBoxLayout()

        # Question input
        self.question_edit = QTextEdit()
        self.question_edit.setPlaceholderText(
            "What would you like to know about the uploaded documents?"
        )
        self.question_edit.setMaximumHeight(80)
        query_layout.addWidget(self.question_edit)

        # Query parameters
        params_layout = QGridLayout()

        # Collection for query
        params_layout.addWidget(QLabel("📁 Collection:"), 0, 0)
        self.query_collection_edit = QLineEdit("default")
        params_layout.addWidget(self.query_collection_edit, 0, 1)

        # Top K
        params_layout.addWidget(QLabel("🔍 Top K:"), 0, 2)
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 10)
        self.top_k_spin.setValue(3)
        params_layout.addWidget(self.top_k_spin, 0, 3)

        # Max length
        params_layout.addWidget(QLabel("📏 Max Length:"), 1, 0)
        self.max_length_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_length_slider.setRange(50, 500)
        self.max_length_slider.setValue(200)
        self.max_length_label = QLabel("200")
        self.max_length_slider.valueChanged.connect(
            lambda v: self.max_length_label.setText(str(v))
        )
        params_layout.addWidget(self.max_length_slider, 1, 1)
        params_layout.addWidget(self.max_length_label, 1, 2)

        # Temperature
        params_layout.addWidget(QLabel("🌡️ Temperature:"), 1, 3)
        self.temperature_slider = QSlider(Qt.Orientation.Horizontal)
        self.temperature_slider.setRange(1, 10)
        self.temperature_slider.setValue(7)
        self.temperature_label = QLabel("0.7")
        self.temperature_slider.valueChanged.connect(
            lambda v: self.temperature_label.setText(f"{v/10:.1f}")
        )
        params_layout.addWidget(self.temperature_slider, 2, 1)
        params_layout.addWidget(self.temperature_label, 2, 2)

        query_layout.addLayout(params_layout)

        # Query buttons
        button_layout = QHBoxLayout()
        self.query_btn = QPushButton("❓ Ask Question")
        self.query_btn.clicked.connect(self.query_rag)
        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.clicked.connect(self.clear_query)
        button_layout.addWidget(self.query_btn)
        button_layout.addWidget(self.clear_btn)
        query_layout.addLayout(button_layout)

        query_group.setLayout(query_layout)
        layout.addWidget(query_group)

        # Answer section
        answer_group = QGroupBox("💡 Answer")
        answer_layout = QVBoxLayout()

        # Answer display
        self.answer_display = QTextEdit()
        self.answer_display.setReadOnly(True)
        self.answer_display.setPlaceholderText("Answers will appear here...")
        answer_layout.addWidget(self.answer_display)

        answer_group.setLayout(answer_layout)
        layout.addWidget(answer_group)

        widget.setLayout(layout)
        return widget

    def browse_file(self):
        """Browse for file to upload"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Document",
            "",
            "Text Files (*.txt);;PDF Files (*.pdf);;All Files (*)",
        )
        if file_path:
            self.file_path_edit.setText(file_path)

    def upload_document(self):
        """Upload document to RAG system"""
        file_path = self.file_path_edit.text().strip()
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "Warning", "Please select a valid file")
            return

        self.upload_btn.setEnabled(False)
        self.upload_status.setText("⏳ Uploading...")

        self.current_worker = RAGWorker(
            "upload",
            self.api_base_url,
            file_path=file_path,
            collection_name=self.collection_edit.text(),
            chunk_size=self.chunk_size_slider.value(),
            chunk_overlap=self.chunk_overlap_slider.value(),
        )
        self.current_worker.finished.connect(self.on_upload_finished)
        self.current_worker.error.connect(self.on_upload_error)
        self.current_worker.start()

    def on_upload_finished(self, result):
        """Handle upload completion"""
        self.upload_btn.setEnabled(True)
        status_text = f"✅ Uploaded: {result['filename']}\n"
        status_text += f"📄 Chunks: {result['total_chunks']}\n"
        status_text += f"📁 Collection: {result['collection_name']}"
        self.upload_status.setText(status_text)

        # Refresh status
        QTimer.singleShot(1000, self.load_status)

    def on_upload_error(self, error_msg):
        """Handle upload error"""
        self.upload_btn.setEnabled(True)
        self.upload_status.setText(f"❌ Error: {error_msg}")

    def query_rag(self):
        """Query RAG system"""
        question = self.question_edit.toPlainText().strip()
        if not question:
            QMessageBox.warning(self, "Warning", "Please enter a question")
            return

        self.query_btn.setEnabled(False)
        self.answer_display.setText("⏳ Thinking...")

        self.current_worker = RAGWorker(
            "query",
            self.api_base_url,
            question=question,
            collection_name=self.query_collection_edit.text(),
            top_k=self.top_k_spin.value(),
            max_length=self.max_length_slider.value(),
            temperature=self.temperature_slider.value() / 10.0,
        )
        self.current_worker.finished.connect(self.on_query_finished)
        self.current_worker.error.connect(self.on_query_error)
        self.current_worker.start()

    def on_query_finished(self, result):
        """Handle query completion"""
        self.query_btn.setEnabled(True)

        answer_text = f"**Answer:** {result['answer']}\n\n"
        answer_text += f"**Confidence:** {result['confidence']:.1%}\n\n"
        answer_text += f"**Sources:** {', '.join(set(result['sources']))}\n\n"

        if result["relevant_chunks"]:
            answer_text += "**Relevant Context:**\n"
            for i, chunk in enumerate(result["relevant_chunks"], 1):
                answer_text += f"{i}. {chunk[:200]}...\n\n"

        self.answer_display.setText(answer_text)

    def on_query_error(self, error_msg):
        """Handle query error"""
        self.query_btn.setEnabled(True)
        self.answer_display.setText(f"❌ Error: {error_msg}")

    def clear_query(self):
        """Clear query form"""
        self.question_edit.clear()
        self.answer_display.clear()

    def load_status(self):
        """Load RAG system status"""
        self.refresh_btn.setEnabled(False)

        self.current_worker = RAGWorker("status", self.api_base_url)
        self.current_worker.finished.connect(self.on_status_finished)
        self.current_worker.error.connect(self.on_status_error)
        self.current_worker.start()

    def on_status_finished(self, result):
        """Handle status load completion"""
        self.refresh_btn.setEnabled(True)

        status_text = "📊 **RAG Status**\n\n"
        status_text += f"📁 Collections: {', '.join(result['collections']) if result['collections'] else 'None'}\n"
        status_text += f"📄 Total Documents: {result['total_documents']}\n"
        status_text += f"🔤 Total Chunks: {result['total_chunks']}"

        self.status_display.setText(status_text)

    def on_status_error(self, error_msg):
        """Handle status load error"""
        self.refresh_btn.setEnabled(True)
        self.status_display.setText(f"❌ Error: {error_msg}")


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = RAGWidget()
    widget.show()
    sys.exit(app.exec())
