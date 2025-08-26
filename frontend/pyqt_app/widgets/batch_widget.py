# frontend/pyqt_app/widgets/batch_widget.py
"""
Batch Processing Widget for PyQt
"""
import os
import csv
import json
import tempfile
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QTabWidget,
    QTextEdit,
    QPushButton,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QLabel,
    QGroupBox,
    QMessageBox,
    QSpinBox,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont


class BatchProcessingThread(QThread):
    """Background thread for batch processing"""

    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, api_client, job_data):
        super().__init__()
        self.api_client = api_client
        self.job_data = job_data

    def run(self):
        try:
            self.progress.emit("提交批次任務...")
            result = self.api_client.submit_batch_job(self.job_data)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class BatchWidget(QWidget):
    """Batch processing widget"""

    images_generated = pyqtSignal(list)  # Signal for generated images
    status_changed = pyqtSignal(str)  # Signal for status updates

    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client
        self.current_jobs = {}
        self.batch_thread = None
        self.setup_ui()

        # Timer for job status updates
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_job_statuses)
        self.status_timer.start(5000)  # Update every 5 seconds

    def setup_ui(self):
        """Setup the batch widget UI"""
        layout = QVBoxLayout(self)

        # Tab widget for different input methods
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # CSV Upload Tab
        csv_tab = self.create_csv_tab()
        self.tab_widget.addTab(csv_tab, "CSV 上傳")

        # JSON Upload Tab
        json_tab = self.create_json_tab()
        self.tab_widget.addTab(json_tab, "JSON 上傳")

        # Manual Config Tab
        manual_tab = self.create_manual_tab()
        self.tab_widget.addTab(manual_tab, "手動設定")

        # Job Status Section
        status_group = QGroupBox("任務狀態")
        status_layout = QVBoxLayout(status_group)

        # Refresh button
        refresh_btn = QPushButton("刷新任務列表")
        refresh_btn.clicked.connect(self.refresh_jobs)
        status_layout.addWidget(refresh_btn)

        # Jobs list
        self.jobs_list = QListWidget()
        self.jobs_list.itemClicked.connect(self.on_job_selected)
        status_layout.addWidget(self.jobs_list)

        # Job details
        self.job_details = QTextEdit()
        self.job_details.setMaximumHeight(150)
        self.job_details.setReadOnly(True)
        status_layout.addWidget(self.job_details)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)

        layout.addWidget(status_group)

    def create_csv_tab(self):
        """Create CSV upload tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # File selection
        file_layout = QHBoxLayout()
        self.csv_file_path = QLineEdit()
        self.csv_file_path.setPlaceholderText("選擇 CSV 檔案...")
        self.csv_file_path.setReadOnly(True)
        file_layout.addWidget(self.csv_file_path)

        browse_btn = QPushButton("瀏覽")
        browse_btn.clicked.connect(self.browse_csv_file)
        file_layout.addWidget(browse_btn)

        layout.addLayout(file_layout)

        # CSV format help
        help_text = QTextEdit()
        help_text.setMaximumHeight(120)
        help_text.setReadOnly(True)
        help_text.setHtml(
            """
        <h4>CSV 格式說明：</h4>
        <pre>prompt,negative,width,height,steps,seed
"anime girl, blue hair",lowres,768,768,25,-1
"cat, cute, fluffy",blurry,512,512,20,12345</pre>
        <p><b>必填欄位：</b> prompt<br>
        <b>可選欄位：</b> negative, width, height, steps, cfg_scale, seed, sampler</p>
        """
        )
        layout.addWidget(help_text)

        # Submit button
        self.csv_submit_btn = QPushButton("提交 CSV 批次任務")
        self.csv_submit_btn.clicked.connect(self.submit_csv_batch)
        layout.addWidget(self.csv_submit_btn)

        return widget

    def create_json_tab(self):
        """Create JSON upload tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # File selection
        file_layout = QHBoxLayout()
        self.json_file_path = QLineEdit()
        self.json_file_path.setPlaceholderText("選擇 JSON 檔案...")
        self.json_file_path.setReadOnly(True)
        file_layout.addWidget(self.json_file_path)

        browse_btn = QPushButton("瀏覽")
        browse_btn.clicked.connect(self.browse_json_file)
        file_layout.addWidget(browse_btn)

        layout.addLayout(file_layout)

        # JSON format help
        help_text = QTextEdit()
        help_text.setMaximumHeight(150)
        help_text.setReadOnly(True)
        help_text.setHtml(
            """
        <h4>JSON 格式說明：</h4>
        <pre>{
  "job_name": "batch_generation_001",
  "tasks": [
    {
      "prompt": "anime girl, blue hair",
      "negative": "lowres, blurry",
      "width": 768,
      "height": 768,
      "steps": 25
    }
  ]
}</pre>
        """
        )
        layout.addWidget(help_text)

        # Submit button
        self.json_submit_btn = QPushButton("提交 JSON 批次任務")
        self.json_submit_btn.clicked.connect(self.submit_json_batch)
        layout.addWidget(self.json_submit_btn)

        return widget

    def create_manual_tab(self):
        """Create manual configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Prompts input
        prompts_group = QGroupBox("提示詞設定")
        prompts_layout = QVBoxLayout(prompts_group)

        prompts_layout.addWidget(QLabel("提示詞列表（每行一個）："))
        self.prompts_text = QTextEdit()
        self.prompts_text.setPlaceholderText(
            "anime girl, blue hair\ncat, cute, fluffy\nlandscape, mountains"
        )
        self.prompts_text.setMaximumHeight(100)
        prompts_layout.addWidget(self.prompts_text)

        prompts_layout.addWidget(QLabel("統一負面提示詞："))
        self.negative_text = QTextEdit()
        self.negative_text.setPlainText("lowres, blurry, bad anatomy")
        self.negative_text.setMaximumHeight(60)
        prompts_layout.addWidget(self.negative_text)

        layout.addWidget(prompts_group)

        # Parameters
        params_group = QGroupBox("生成參數")
        params_form = QFormLayout(params_group)

        # Dimensions
        dims_layout = QHBoxLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(256, 2048)
        self.width_spin.setSingleStep(64)
        self.width_spin.setValue(768)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(256, 2048)
        self.height_spin.setSingleStep(64)
        self.height_spin.setValue(768)

        dims_layout.addWidget(self.width_spin)
        dims_layout.addWidget(QLabel("×"))
        dims_layout.addWidget(self.height_spin)
        params_form.addRow("圖片尺寸:", dims_layout)

        # Other parameters
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 100)
        self.steps_spin.setValue(25)
        params_form.addRow("採樣步數:", self.steps_spin)

        self.cfg_spin = QSpinBox()
        self.cfg_spin.setRange(1, 30)
        self.cfg_spin.setValue(7)
        params_form.addRow("CFG 縮放:", self.cfg_spin)

        self.sampler_combo = QComboBox()
        self.sampler_combo.addItems(
            ["DPM++ 2M Karras", "DPM++ SDE Karras", "Euler a", "Euler", "DDIM"]
        )
        params_form.addRow("採樣器:", self.sampler_combo)

        layout.addWidget(params_group)

        # Submit button
        self.manual_submit_btn = QPushButton("提交手動批次任務")
        self.manual_submit_btn.clicked.connect(self.submit_manual_batch)
        layout.addWidget(self.manual_submit_btn)

        return widget

    def browse_csv_file(self):
        """Browse for CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇 CSV 檔案", "", "CSV Files (*.csv)"
        )
        if file_path:
            self.csv_file_path.setText(file_path)

    def browse_json_file(self):
        """Browse for JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇 JSON 檔案", "", "JSON Files (*.json)"
        )
        if file_path:
            self.json_file_path.setText(file_path)

    def submit_csv_batch(self):
        """Submit CSV batch job"""
        csv_file = self.csv_file_path.text()
        if not csv_file or not Path(csv_file).exists():
            QMessageBox.warning(self, "錯誤", "請選擇有效的 CSV 檔案")
            return

        try:
            tasks = []
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    task = {
                        "prompt": row.get("prompt", ""),
                        "negative": row.get("negative", "lowres, blurry"),
                        "width": int(row.get("width", 768)),
                        "height": int(row.get("height", 768)),
                        "steps": int(row.get("steps", 25)),
                        "cfg_scale": float(row.get("cfg_scale", 7.5)),
                        "seed": int(row.get("seed", -1)),
                        "sampler": row.get("sampler", "DPM++ 2M Karras"),
                    }
                    tasks.append(task)

            if not tasks:
                QMessageBox.warning(self, "錯誤", "CSV 檔案中沒有有效的任務")
                return

            job_data = {"job_name": f"csv_batch_{len(tasks)}_tasks", "tasks": tasks}

            self.submit_batch_job(job_data)

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"處理 CSV 檔案失敗: {str(e)}")

    def submit_json_batch(self):
        """Submit JSON batch job"""
        json_file = self.json_file_path.text()
        if not json_file or not Path(json_file).exists():
            QMessageBox.warning(self, "錯誤", "請選擇有效的 JSON 檔案")
            return

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                job_data = json.load(f)

            self.submit_batch_job(job_data)

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"處理 JSON 檔案失敗: {str(e)}")

    def submit_manual_batch(self):
        """Submit manual batch job"""
        prompts_text = self.prompts_text.toPlainText()
        if not prompts_text.strip():
            QMessageBox.warning(self, "錯誤", "請輸入至少一個提示詞")
            return

        prompts = [p.strip() for p in prompts_text.split("\n") if p.strip()]
        if not prompts:
            QMessageBox.warning(self, "錯誤", "沒有有效的提示詞")
            return

        negative = self.negative_text.toPlainText()

        tasks = []
        for i, prompt in enumerate(prompts):
            task = {
                "id": i + 1,
                "prompt": prompt,
                "negative": negative,
                "width": self.width_spin.value(),
                "height": self.height_spin.value(),
                "steps": self.steps_spin.value(),
                "cfg_scale": float(self.cfg_spin.value()),
                "sampler": self.sampler_combo.currentText(),
                "seed": -1,
            }
            tasks.append(task)

        job_data = {"job_name": f"manual_batch_{len(tasks)}_tasks", "tasks": tasks}

        self.submit_batch_job(job_data)

    def submit_batch_job(self, job_data):
        """Submit batch job to API"""
        if self.batch_thread and self.batch_thread.isRunning():
            QMessageBox.warning(self, "警告", "已有任務正在提交中")
            return

        # Disable submit buttons
        self.csv_submit_btn.setEnabled(False)
        self.json_submit_btn.setEnabled(False)
        self.manual_submit_btn.setEnabled(False)

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate

        # Start batch thread
        self.batch_thread = BatchProcessingThread(self.api_client, job_data)
        self.batch_thread.progress.connect(self.status_changed.emit)
        self.batch_thread.finished.connect(self.on_batch_submitted)
        self.batch_thread.error.connect(self.on_batch_error)
        self.batch_thread.start()

    def on_batch_submitted(self, result):
        """Handle successful batch submission"""
        # Re-enable submit buttons
        self.csv_submit_btn.setEnabled(True)
        self.json_submit_btn.setEnabled(True)
        self.manual_submit_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        if result.get("job_id"):
            job_id = result["job_id"]
            self.current_jobs[job_id] = {"status": "submitted", "details": result}
            self.status_changed.emit(f"批次任務提交成功: {job_id}")
            self.refresh_jobs()

            QMessageBox.information(
                self, "成功", f"批次任務提交成功！\n任務 ID: {job_id}"
            )
        else:
            error_msg = result.get("message", "未知錯誤")
            QMessageBox.critical(self, "錯誤", f"任務提交失敗: {error_msg}")

    def on_batch_error(self, error_message):
        """Handle batch submission error"""
        # Re-enable submit buttons
        self.csv_submit_btn.setEnabled(True)
        self.json_submit_btn.setEnabled(True)
        self.manual_submit_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        self.status_changed.emit(f"批次任務提交失敗: {error_message}")
        QMessageBox.critical(self, "錯誤", f"提交批次任務時發生錯誤: {error_message}")

    def refresh_jobs(self):
        """Refresh jobs list"""
        try:
            jobs = self.api_client.list_jobs()
            self.jobs_list.clear()
            self.current_jobs.clear()

            for job in jobs:
                job_id = job.get("job_id", "Unknown")
                status = job.get("status", "unknown")
                job_name = job.get("job_name", job_id)

                item_text = f"{job_name} - {status}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, job)
                self.jobs_list.addItem(item)

                self.current_jobs[job_id] = job

        except Exception as e:
            self.status_changed.emit(f"刷新任務列表失敗: {str(e)}")

    def on_job_selected(self, item):
        """Handle job selection"""
        job_data = item.data(Qt.UserRole)
        if job_data:
            self.display_job_details(job_data)

    def display_job_details(self, job):
        """Display job details"""
        details = f"任務 ID: {job.get('job_id', 'N/A')}\n"
        details += f"名稱: {job.get('job_name', 'N/A')}\n"
        details += f"狀態: {job.get('status', 'N/A')}\n"
        details += f"總任務數: {job.get('total', 'N/A')}\n"
        details += f"已完成: {job.get('completed', 0)}\n"
        details += f"失敗: {job.get('failed', 0)}\n"
        details += f"創建時間: {job.get('created_at', 'N/A')}\n"

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
