# frontend/pyqt_app/widgets/training_widget.py
"""
Training Monitor Widget for PyQt
"""
import json
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QTabWidget,
    QTextEdit,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QLabel,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QLineEdit,
    QMessageBox,
    QSplitter,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont


class TrainingSubmissionThread(QThread):
    """Background thread for training submission"""

    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, api_client, config):
        super().__init__()
        self.api_client = api_client
        self.config = config

    def run(self):
        try:
            self.progress.emit("提交訓練任務...")
            result = self.api_client.submit_training_job(self.config)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class TrainingWidget(QWidget):
    """Training monitoring widget"""

    status_changed = pyqtSignal(str)  # Signal for status updates

    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client
        self.current_jobs = {}
        self.training_thread = None
        self.setup_ui()

        # Timer for job status updates
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_training_statuses)
        self.status_timer.start(10000)  # Update every 10 seconds

    def setup_ui(self):
        """Setup the training widget UI"""
        layout = QHBoxLayout(self)

        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Training submission
        left_panel = self.create_training_controls()
        splitter.addWidget(left_panel)

        # Right panel - Monitoring
        right_panel = self.create_monitoring_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([400, 600])

    def create_training_controls(self):
        """Create training controls panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Title
        title_label = QLabel("LoRA 訓練提交")
        title_font = title_label.font()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # Preset selection
        preset_group = QGroupBox("訓練預設")
        preset_layout = QVBoxLayout(preset_group)

        preset_buttons_layout = QHBoxLayout()
        self.character_preset_btn = QPushButton("角色訓練")
        self.style_preset_btn = QPushButton("風格訓練")
        self.custom_preset_btn = QPushButton("自定義")

        self.character_preset_btn.setCheckable(True)
        self.style_preset_btn.setCheckable(True)
        self.custom_preset_btn.setCheckable(True)
        self.character_preset_btn.setChecked(True)

        self.character_preset_btn.clicked.connect(lambda: self.load_preset("character"))
        self.style_preset_btn.clicked.connect(lambda: self.load_preset("style"))
        self.custom_preset_btn.clicked.connect(lambda: self.load_preset("custom"))

        preset_buttons_layout.addWidget(self.character_preset_btn)
        preset_buttons_layout.addWidget(self.style_preset_btn)
        preset_buttons_layout.addWidget(self.custom_preset_btn)
        preset_layout.addLayout(preset_buttons_layout)

        layout.addWidget(preset_group)

        # Training configuration
        config_group = QGroupBox("訓練配置")
        config_form = QFormLayout(config_group)

        # Basic settings
        self.run_id_edit = QLineEdit()
        self.run_id_edit.setPlaceholderText("例如: char_alice_v1")
        config_form.addRow("任務 ID:", self.run_id_edit)

        self.dataset_name_edit = QLineEdit()
        self.dataset_name_edit.setPlaceholderText("數據集目錄名稱")
        config_form.addRow("數據集名稱:", self.dataset_name_edit)

        # LoRA parameters
        self.rank_spin = QSpinBox()
        self.rank_spin.setRange(4, 128)
        self.rank_spin.setSingleStep(4)
        self.rank_spin.setValue(16)
        config_form.addRow("LoRA Rank:", self.rank_spin)

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.000001, 0.01)
        self.learning_rate_spin.setSingleStep(0.00001)
        self.learning_rate_spin.setValue(0.0001)
        self.learning_rate_spin.setDecimals(6)
        config_form.addRow("學習率:", self.learning_rate_spin)

        # Training parameters
        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(512, 1024)
        self.resolution_spin.setSingleStep(64)
        self.resolution_spin.setValue(768)
        config_form.addRow("解析度:", self.resolution_spin)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 8)
        self.batch_size_spin.setValue(1)
        config_form.addRow("批次大小:", self.batch_size_spin)

        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(500, 10000)
        self.max_steps_spin.setSingleStep(100)
        self.max_steps_spin.setValue(2000)
        config_form.addRow("最大步數:", self.max_steps_spin)

        self.save_every_spin = QSpinBox()
        self.save_every_spin.setRange(100, 1000)
        self.save_every_spin.setSingleStep(100)
        self.save_every_spin.setValue(500)
        config_form.addRow("保存間隔:", self.save_every_spin)

        layout.addWidget(config_group)

        # Submit button
        self.submit_btn = QPushButton("🚀 開始訓練")
        self.submit_btn.clicked.connect(self.submit_training)
        layout.addWidget(self.submit_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch()
        return widget

    def create_monitoring_panel(self):
        """Create monitoring panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Title
        title_label = QLabel("訓練監控")
        title_font = title_label.font()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # Refresh button
        refresh_btn = QPushButton("刷新任務列表")
        refresh_btn.clicked.connect(self.refresh_training_jobs)
        layout.addWidget(refresh_btn)

        # Jobs list
        self.jobs_list = QListWidget()
        self.jobs_list.itemClicked.connect(self.on_job_selected)
        layout.addWidget(self.jobs_list)

        # Job details
        self.job_details = QTextEdit()
        self.job_details.setReadOnly(True)
        layout.addWidget(self.job_details)

        return widget

    def load_preset(self, preset_type):
        """Load training preset"""
        # Update button states
        self.character_preset_btn.setChecked(preset_type == "character")
        self.style_preset_btn.setChecked(preset_type == "style")
        self.custom_preset_btn.setChecked(preset_type == "custom")

        # Load preset values
        presets = {
            "character": {
                "rank": 16,
                "learning_rate": 0.0001,
                "resolution": 768,
                "batch_size": 1,
                "max_steps": 2000,
                "save_every": 500,
            },
            "style": {
                "rank": 8,
                "learning_rate": 0.00008,
                "resolution": 768,
                "batch_size": 2,
                "max_steps": 1500,
                "save_every": 300,
            },
            "custom": {
                "rank": 32,
                "learning_rate": 0.00005,
                "resolution": 1024,
                "batch_size": 1,
                "max_steps": 3000,
                "save_every": 500,
            },
        }

        if preset_type in presets:
            preset = presets[preset_type]
            self.rank_spin.setValue(preset["rank"])
            self.learning_rate_spin.setValue(preset["learning_rate"])
            self.resolution_spin.setValue(preset["resolution"])
            self.batch_size_spin.setValue(preset["batch_size"])
            self.max_steps_spin.setValue(preset["max_steps"])
            self.save_every_spin.setValue(preset["save_every"])

    def submit_training(self):
        """Submit training job"""
        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.warning(self, "警告", "已有訓練任務正在提交中")
            return

        # Validate inputs
        run_id = self.run_id_edit.text().strip()
        dataset_name = self.dataset_name_edit.text().strip()

        if not run_id or not dataset_name:
            QMessageBox.warning(self, "錯誤", "請填寫任務 ID 和數據集名稱")
            return

        # Prepare configuration
        config = {
            "run_id": run_id,
            "dataset_name": dataset_name,
            "rank": self.rank_spin.value(),
            "learning_rate": self.learning_rate_spin.value(),
            "resolution": self.resolution_spin.value(),
            "batch_size": self.batch_size_spin.value(),
            "max_train_steps": self.max_steps_spin.value(),
            "save_every": self.save_every_spin.value(),
            "mixed_precision": "fp16",
            "gradient_checkpointing": True,
        }

        # Disable submit button and show progress
        self.submit_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate

        # Start training thread
        self.training_thread = TrainingSubmissionThread(self.api_client, config)
        self.training_thread.progress.connect(self.status_changed.emit)
        self.training_thread.finished.connect(self.on_training_submitted)
        self.training_thread.error.connect(self.on_training_error)
        self.training_thread.start()

    def on_training_submitted(self, result):
        """Handle successful training submission"""
        self.submit_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        if result.get("run_id") or result.get("job_id"):
            run_id = result.get("run_id") or result.get("job_id")
            self.status_changed.emit(f"訓練任務提交成功: {run_id}")
            self.refresh_training_jobs()

            QMessageBox.information(
                self, "成功", f"訓練任務提交成功！\n任務 ID: {run_id}"
            )

            # Generate new run_id for next job
            import time

            self.run_id_edit.setText(f"lora_train_{int(time.time())}")
        else:
            error_msg = result.get("message", "未知錯誤")
            QMessageBox.critical(self, "錯誤", f"訓練任務提交失敗: {error_msg}")

    def on_training_error(self, error_message):
        """Handle training submission error"""
        self.submit_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        self.status_changed.emit(f"訓練任務提交失敗: {error_message}")
        QMessageBox.critical(self, "錯誤", f"提交訓練任務時發生錯誤: {error_message}")

    def refresh_training_jobs(self):
        """Refresh training jobs list"""
        try:
            jobs = self.api_client.list_training_jobs()
            self.jobs_list.clear()
            self.current_jobs.clear()

            for job in jobs:
                run_id = job.get("run_id", "Unknown")
                status = job.get("status", "unknown")

                item_text = f"{run_id} - {status}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, job)
                self.jobs_list.addItem(item)

                self.current_jobs[run_id] = job

        except Exception as e:
            self.status_changed.emit(f"刷新訓練任務列表失敗: {str(e)}")

    def on_job_selected(self, item):
        """Handle job selection"""
        job_data = item.data(Qt.UserRole)
        if job_data:
            self.display_job_details(job_data)

    def display_job_details(self, job):
        """Display job details"""
        details = f"任務 ID: {job.get('run_id', 'N/A')}\n"
        details += f"狀態: {job.get('status', 'N/A')}\n"
        details += f"當前步數: {job.get('current_step', 0)}\n"
        details += f"最大步數: {job.get('max_steps', 'N/A')}\n"
        details += f"數據集: {job.get('dataset_name', 'N/A')}\n"

        if job.get("loss"):
            details += f"當前損失: {job.get('loss'):.4f}\n"

        if job.get("learning_rate"):
            details += f"學習率: {job.get('learning_rate'):.6f}\n"

        details += f"開始時間: {job.get('started_at', 'N/A')}\n"

        if job.get("completed_at"):
            details += f"完成時間: {job.get('completed_at')}\n"

        if job.get("error"):
            details += f"錯誤: {job.get('error')}\n"

        self.job_details.setPlainText(details)

    def update_job_statuses(self):
        """Update job statuses periodically"""
        for job_id in list(self.current_jobs.keys()):
            try:
                status = self.api_client.get_job_status(job_id)
                if status.get("status") != "error":
                    self.current_jobs[job_id] = status

                    # Update list item if it's still there
                    for i in range(self.jobs_list.count()):
                        item = self.jobs_list.item(i)
                        item_job = item.data(Qt.UserRole)
                        if item_job and item_job.get("job_id") == job_id:
                            # Update item text
                            job_name = status.get("job_name", job_id)
                            new_status = status.get("status", "unknown")
                            item.setText(f"{job_name} - {new_status}")
                            item.setData(Qt.UserRole, status)

                            # Update details if this job is selected
                            if self.jobs_list.currentItem() == item:
                                self.display_job_details(status)
                            break

            except Exception:
                # Silently ignore errors during status updates
                pass
