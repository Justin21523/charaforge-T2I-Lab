# frontend/pyqt_app/widgets/generation_widget.py
"""
Image Generation Control Panel
"""
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QPushButton,
    QTextEdit,
    QGroupBox,
    QCheckBox,
    QSlider,
    QLabel,
    QFileDialog,
    QProgressBar,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

from frontend.shared.constants import DEFAULT_GENERATION_PARAMS, CONTROLNET_TYPES


class GenerationThread(QThread):
    """Background thread for image generation"""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, api_client, params, control_params=None):
        super().__init__()
        self.api_client = api_client
        self.params = params
        self.control_params = control_params

    def run(self):
        try:
            self.progress.emit("正在生成圖片...")

            if self.control_params:
                result = self.api_client.controlnet_generate(
                    {**self.params, **self.control_params},
                    self.control_params.get("control_type", "pose"),
                )
            else:
                result = self.api_client.generate_image(self.params)

            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class GenerationWidget(QWidget):
    """Main image generation control panel"""

    image_generated = pyqtSignal(str, dict)  # image_path, metadata
    status_changed = pyqtSignal(str)

    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client
        self.setup_ui()
        self.generation_thread = None

    def setup_ui(self):
        """Setup the generation UI"""
        layout = QVBoxLayout(self)

        # Prompt group
        prompt_group = QGroupBox("提示詞設定")
        prompt_layout = QVBoxLayout(prompt_group)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("輸入正面提示詞...")
        self.prompt_edit.setMaximumHeight(80)
        prompt_layout.addWidget(QLabel("正面提示詞:"))
        prompt_layout.addWidget(self.prompt_edit)

        self.negative_edit = QTextEdit()
        self.negative_edit.setPlaceholderText("輸入負面提示詞...")
        self.negative_edit.setMaximumHeight(60)
        self.negative_edit.setPlainText(DEFAULT_GENERATION_PARAMS["negative"])
        prompt_layout.addWidget(QLabel("負面提示詞:"))
        prompt_layout.addWidget(self.negative_edit)

        layout.addWidget(prompt_group)

        # Parameters group
        params_group = QGroupBox("生成參數")
        params_form = QFormLayout(params_group)

        # Dimensions
        dims_layout = QHBoxLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(256, 2048)
        self.width_spin.setValue(DEFAULT_GENERATION_PARAMS["width"])
        self.width_spin.setSingleStep(64)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(256, 2048)
        self.height_spin.setValue(DEFAULT_GENERATION_PARAMS["height"])
        self.height_spin.setSingleStep(64)

        dims_layout.addWidget(self.width_spin)
        dims_layout.addWidget(QLabel("×"))
        dims_layout.addWidget(self.height_spin)
        params_form.addRow("圖片尺寸:", dims_layout)

        # Steps
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 100)
        self.steps_spin.setValue(DEFAULT_GENERATION_PARAMS["steps"])
        params_form.addRow("採樣步數:", self.steps_spin)

        # CFG Scale
        self.cfg_spin = QDoubleSpinBox()
        self.cfg_spin.setRange(1.0, 30.0)
        self.cfg_spin.setValue(DEFAULT_GENERATION_PARAMS["cfg_scale"])
        self.cfg_spin.setSingleStep(0.5)
        params_form.addRow("CFG 縮放:", self.cfg_spin)

        # Seed
        seed_layout = QHBoxLayout()
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 2147483647)
        self.seed_spin.setValue(DEFAULT_GENERATION_PARAMS["seed"])

        self.random_seed_btn = QPushButton("隨機")
        self.random_seed_btn.clicked.connect(self.randomize_seed)

        seed_layout.addWidget(self.seed_spin)
        seed_layout.addWidget(self.random_seed_btn)
        params_form.addRow("種子:", seed_layout)

        # Sampler
        self.sampler_combo = QComboBox()
        self.sampler_combo.addItems(
            [
                "DPM++ 2M Karras",
                "DPM++ SDE Karras",
                "Euler a",
                "Euler",
                "LMS",
                "Heun",
                "DDIM",
            ]
        )
        self.sampler_combo.setCurrentText(DEFAULT_GENERATION_PARAMS["sampler"])
        params_form.addRow("採樣器:", self.sampler_combo)

        # Batch size
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 8)
        self.batch_spin.setValue(DEFAULT_GENERATION_PARAMS["batch_size"])
        params_form.addRow("批次大小:", self.batch_spin)

        layout.addWidget(params_group)

        # ControlNet group
        controlnet_group = QGroupBox("ControlNet 控制")
        controlnet_layout = QVBoxLayout(controlnet_group)

        self.controlnet_enabled = QCheckBox("啟用 ControlNet")
        controlnet_layout.addWidget(self.controlnet_enabled)

        controlnet_form = QFormLayout()

        self.controlnet_type = QComboBox()
        self.controlnet_type.addItems(CONTROLNET_TYPES)
        self.controlnet_type.setEnabled(False)
        controlnet_form.addRow("控制類型:", self.controlnet_type)

        control_image_layout = QHBoxLayout()
        self.control_image_path = QLineEdit()
        self.control_image_path.setPlaceholderText("選擇控制圖片...")
        self.control_image_path.setEnabled(False)

        self.browse_control_btn = QPushButton("瀏覽")
        self.browse_control_btn.setEnabled(False)
        self.browse_control_btn.clicked.connect(self.browse_control_image)

        control_image_layout.addWidget(self.control_image_path)
        control_image_layout.addWidget(self.browse_control_btn)
        controlnet_form.addRow("控制圖片:", control_image_layout)

        self.controlnet_weight = QDoubleSpinBox()
        self.controlnet_weight.setRange(0.0, 2.0)
        self.controlnet_weight.setValue(1.0)
        self.controlnet_weight.setSingleStep(0.1)
        self.controlnet_weight.setEnabled(False)
        controlnet_form.addRow("控制強度:", self.controlnet_weight)

        controlnet_layout.addLayout(controlnet_form)

        # Connect ControlNet enable checkbox
        self.controlnet_enabled.toggled.connect(self.toggle_controlnet)

        layout.addWidget(controlnet_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Generate button
        self.generate_btn = QPushButton("生成圖片")
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.clicked.connect(self.generate_image)
        layout.addWidget(self.generate_btn)

        # Add stretch
        layout.addStretch()

    def toggle_controlnet(self, enabled):
        """Toggle ControlNet controls"""
        self.controlnet_type.setEnabled(enabled)
        self.control_image_path.setEnabled(enabled)
        self.browse_control_btn.setEnabled(enabled)
        self.controlnet_weight.setEnabled(enabled)

    def browse_control_image(self):
        """Browse for control image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇控制圖片", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if file_path:
            self.control_image_path.setText(file_path)

    def randomize_seed(self):
        """Generate random seed"""
        import random

        self.seed_spin.setValue(random.randint(0, 2147483647))

    def get_generation_params(self):
        """Get current generation parameters"""
        params = {
            "prompt": self.prompt_edit.toPlainText(),
            "negative": self.negative_edit.toPlainText(),
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "steps": self.steps_spin.value(),
            "cfg_scale": self.cfg_spin.value(),
            "seed": self.seed_spin.value(),
            "sampler": self.sampler_combo.currentText(),
            "batch_size": self.batch_spin.value(),
        }

        # Add ControlNet params if enabled
        control_params = None
        if self.controlnet_enabled.isChecked():
            control_image = self.control_image_path.text().strip()
            if control_image and os.path.exists(control_image):
                control_params = {
                    "control_type": self.controlnet_type.currentText(),
                    "control_image": control_image,
                    "control_weight": self.controlnet_weight.value(),
                }

        return params, control_params

    def generate_image(self):
        """Start image generation"""
        if self.generation_thread and self.generation_thread.isRunning():
            return

        params, control_params = self.get_generation_params()

        if not params["prompt"].strip():
            self.status_changed.emit("錯誤: 請輸入提示詞")
            return

        # Disable generate button and show progress
        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

        # Start generation thread
        self.generation_thread = GenerationThread(
            self.api_client, params, control_params
        )
        self.generation_thread.finished.connect(self.on_generation_finished)
        self.generation_thread.error.connect(self.on_generation_error)
        self.generation_thread.progress.connect(self.status_changed.emit)
        self.generation_thread.start()

    def on_generation_finished(self, result):
        """Handle successful generation"""
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        if "image_path" in result:
            metadata = {
                "prompt": self.prompt_edit.toPlainText(),
                "negative": self.negative_edit.toPlainText(),
                "seed": result.get("seed", self.seed_spin.value()),
                "width": self.width_spin.value(),
                "height": self.height_spin.value(),
                "steps": self.steps_spin.value(),
                "cfg_scale": self.cfg_spin.value(),
                "sampler": self.sampler_combo.currentText(),
            }

            self.image_generated.emit(result["image_path"], metadata)
            self.status_changed.emit("圖片生成完成")

            # Update seed to generated seed
            if "seed" in result:
                self.seed_spin.setValue(result["seed"])
        else:
            self.status_changed.emit("生成失敗: 未返回圖片路徑")

    def on_generation_error(self, error_message):
        """Handle generation error"""
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_changed.emit(f"生成失敗: {error_message}")
