# frontend/pyqt_app/dialogs/settings_dialog.py
"""
Settings Dialog for PyQt Application
"""
import os
import json
from pathlib import Path
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QTabWidget,
    QWidget,
    QLineEdit,
    QSpinBox,
    QCheckBox,
    QComboBox,
    QPushButton,
    QFileDialog,
    QGroupBox,
    QLabel,
    QMessageBox,
    QTextEdit,
)
from PyQt5.QtCore import Qt


class SettingsDialog(QDialog):
    """Application settings dialog"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("設定")
        self.setModal(True)
        self.setMinimumSize(500, 400)
        self.load_settings()
        self.setup_ui()

    def load_settings(self):
        """Load settings from config file"""
        self.config_path = Path.home() / ".sagaforge_t2i" / "settings.json"
        self.config_path.parent.mkdir(exist_ok=True)

        # Default settings
        self.settings = {
            "api": {
                "base_url": "http://localhost:8000",
                "timeout": 30,
                "retry_attempts": 3,
            },
            "generation": {
                "default_width": 768,
                "default_height": 768,
                "default_steps": 25,
                "default_cfg_scale": 7.5,
                "default_sampler": "DPM++ 2M Karras",
                "auto_save_images": True,
                "output_directory": str(Path.home() / "SagaForge_Images"),
            },
            "ui": {
                "theme": "dark",
                "auto_refresh_lora": True,
                "show_generation_info": True,
                "gallery_thumbnail_size": 150,
            },
            "advanced": {
                "enable_safety_filter": True,
                "log_level": "INFO",
                "max_concurrent_jobs": 3,
                "cache_directory": str(Path.home() / ".sagaforge_cache"),
            },
        }

        # Load existing settings
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    saved_settings = json.load(f)
                    self.settings.update(saved_settings)
            except Exception as e:
                print(f"Failed to load settings: {e}")

    def save_settings(self):
        """Save settings to config file"""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"無法保存設定: {str(e)}")
            return False

    def setup_ui(self):
        """Setup the settings UI"""
        layout = QVBoxLayout(self)

        # Tab widget
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        # API Settings Tab
        api_tab = self.create_api_tab()
        tab_widget.addTab(api_tab, "API 設定")

        # Generation Settings Tab
        generation_tab = self.create_generation_tab()
        tab_widget.addTab(generation_tab, "生成設定")

        # UI Settings Tab
        ui_tab = self.create_ui_tab()
        tab_widget.addTab(ui_tab, "介面設定")

        # Advanced Settings Tab
        advanced_tab = self.create_advanced_tab()
        tab_widget.addTab(advanced_tab, "進階設定")

        # Buttons
        button_layout = QHBoxLayout()

        self.reset_btn = QPushButton("重設為預設值")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(self.reset_btn)

        button_layout.addStretch()

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        self.ok_btn = QPushButton("確定")
        self.ok_btn.clicked.connect(self.accept_settings)
        self.ok_btn.setDefault(True)
        button_layout.addWidget(self.ok_btn)

        layout.addLayout(button_layout)

    def create_api_tab(self):
        """Create API settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # API Connection Group
        api_group = QGroupBox("API 連接設定")
        api_form = QFormLayout(api_group)

        self.api_url_edit = QLineEdit(self.settings["api"]["base_url"])
        api_form.addRow("API 位址:", self.api_url_edit)

        self.api_timeout_spin = QSpinBox()
        self.api_timeout_spin.setRange(5, 300)
        self.api_timeout_spin.setValue(self.settings["api"]["timeout"])
        self.api_timeout_spin.setSuffix(" 秒")
        api_form.addRow("連接超時:", self.api_timeout_spin)

        self.retry_spin = QSpinBox()
        self.retry_spin.setRange(1, 10)
        self.retry_spin.setValue(self.settings["api"]["retry_attempts"])
        api_form.addRow("重試次數:", self.retry_spin)

        layout.addWidget(api_group)
        layout.addStretch()

        return widget

    def create_generation_tab(self):
        """Create generation settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Default Parameters Group
        params_group = QGroupBox("預設生成參數")
        params_form = QFormLayout(params_group)

        self.default_width_spin = QSpinBox()
        self.default_width_spin.setRange(256, 2048)
        self.default_width_spin.setSingleStep(64)
        self.default_width_spin.setValue(self.settings["generation"]["default_width"])
        params_form.addRow("預設寬度:", self.default_width_spin)

        self.default_height_spin = QSpinBox()
        self.default_height_spin.setRange(256, 2048)
        self.default_height_spin.setSingleStep(64)
        self.default_height_spin.setValue(self.settings["generation"]["default_height"])
        params_form.addRow("預設高度:", self.default_height_spin)

        self.default_steps_spin = QSpinBox()
        self.default_steps_spin.setRange(1, 100)
        self.default_steps_spin.setValue(self.settings["generation"]["default_steps"])
        params_form.addRow("預設步數:", self.default_steps_spin)

        self.default_cfg_spin = QSpinBox()
        self.default_cfg_spin.setRange(1, 30)
        self.default_cfg_spin.setValue(
            int(self.settings["generation"]["default_cfg_scale"])
        )
        params_form.addRow("預設 CFG:", self.default_cfg_spin)

        self.default_sampler_combo = QComboBox()
        samplers = [
            "DPM++ 2M Karras",
            "DPM++ SDE Karras",
            "Euler a",
            "Euler",
            "LMS",
            "Heun",
            "DDIM",
        ]
        self.default_sampler_combo.addItems(samplers)
        self.default_sampler_combo.setCurrentText(
            self.settings["generation"]["default_sampler"]
        )
        params_form.addRow("預設採樣器:", self.default_sampler_combo)

        layout.addWidget(params_group)

        # Output Settings Group
        output_group = QGroupBox("輸出設定")
        output_form = QFormLayout(output_group)

        self.auto_save_check = QCheckBox("自動保存生成的圖片")
        self.auto_save_check.setChecked(self.settings["generation"]["auto_save_images"])
        output_form.addRow(self.auto_save_check)

        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit(
            self.settings["generation"]["output_directory"]
        )
        output_dir_layout.addWidget(self.output_dir_edit)

        browse_btn = QPushButton("瀏覽")
        browse_btn.clicked.connect(self.browse_output_directory)
        output_dir_layout.addWidget(browse_btn)

        output_form.addRow("輸出目錄:", output_dir_layout)

        layout.addWidget(output_group)
        layout.addStretch()

        return widget

    def create_ui_tab(self):
        """Create UI settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Appearance Group
        appearance_group = QGroupBox("外觀設定")
        appearance_form = QFormLayout(appearance_group)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light", "system"])
        self.theme_combo.setCurrentText(self.settings["ui"]["theme"])
        appearance_form.addRow("主題:", self.theme_combo)

        self.thumbnail_size_spin = QSpinBox()
        self.thumbnail_size_spin.setRange(100, 300)
        self.thumbnail_size_spin.setValue(self.settings["ui"]["gallery_thumbnail_size"])
        self.thumbnail_size_spin.setSuffix(" px")
        appearance_form.addRow("縮圖大小:", self.thumbnail_size_spin)

        layout.addWidget(appearance_group)

        # Behavior Group
        behavior_group = QGroupBox("行為設定")
        behavior_layout = QVBoxLayout(behavior_group)

        self.auto_refresh_check = QCheckBox("自動刷新 LoRA 列表")
        self.auto_refresh_check.setChecked(self.settings["ui"]["auto_refresh_lora"])
        behavior_layout.addWidget(self.auto_refresh_check)

        self.show_info_check = QCheckBox("顯示生成資訊")
        self.show_info_check.setChecked(self.settings["ui"]["show_generation_info"])
        behavior_layout.addWidget(self.show_info_check)

        layout.addWidget(behavior_group)
        layout.addStretch()

        return widget

    def create_advanced_tab(self):
        """Create advanced settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Safety Group
        safety_group = QGroupBox("安全設定")
        safety_layout = QVBoxLayout(safety_group)

        self.safety_filter_check = QCheckBox("啟用內容安全過濾器")
        self.safety_filter_check.setChecked(
            self.settings["advanced"]["enable_safety_filter"]
        )
        safety_layout.addWidget(self.safety_filter_check)

        layout.addWidget(safety_group)

        # Performance Group
        perf_group = QGroupBox("效能設定")
        perf_form = QFormLayout(perf_group)

        self.max_jobs_spin = QSpinBox()
        self.max_jobs_spin.setRange(1, 10)
        self.max_jobs_spin.setValue(self.settings["advanced"]["max_concurrent_jobs"])
        perf_form.addRow("最大同時任務數:", self.max_jobs_spin)

        layout.addWidget(perf_group)

        # Logging Group
        log_group = QGroupBox("日誌設定")
        log_form = QFormLayout(log_group)

        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level_combo.setCurrentText(self.settings["advanced"]["log_level"])
        log_form.addRow("日誌等級:", self.log_level_combo)

        layout.addWidget(log_group)

        # Cache Group
        cache_group = QGroupBox("快取設定")
        cache_form = QFormLayout(cache_group)

        cache_dir_layout = QHBoxLayout()
        self.cache_dir_edit = QLineEdit(self.settings["advanced"]["cache_directory"])
        cache_dir_layout.addWidget(self.cache_dir_edit)

        cache_browse_btn = QPushButton("瀏覽")
        cache_browse_btn.clicked.connect(self.browse_cache_directory)
        cache_dir_layout.addWidget(cache_browse_btn)

        cache_form.addRow("快取目錄:", cache_dir_layout)

        layout.addWidget(cache_group)
        layout.addStretch()

        return widget

    def browse_output_directory(self):
        """Browse for output directory"""
        directory = QFileDialog.getExistingDirectory(
            self, "選擇輸出目錄", self.output_dir_edit.text()
        )
        if directory:
            self.output_dir_edit.setText(directory)

    def browse_cache_directory(self):
        """Browse for cache directory"""
        directory = QFileDialog.getExistingDirectory(
            self, "選擇快取目錄", self.cache_dir_edit.text()
        )
        if directory:
            self.cache_dir_edit.setText(directory)

    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        reply = QMessageBox.question(
            self,
            "確認重設",
            "確定要重設所有設定為預設值嗎？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.load_settings()  # Reload defaults
            self.update_ui_from_settings()

    def update_ui_from_settings(self):
        """Update UI controls from settings"""
        # API settings
        self.api_url_edit.setText(self.settings["api"]["base_url"])
        self.api_timeout_spin.setValue(self.settings["api"]["timeout"])
        self.retry_spin.setValue(self.settings["api"]["retry_attempts"])

        # Generation settings
        self.default_width_spin.setValue(self.settings["generation"]["default_width"])
        self.default_height_spin.setValue(self.settings["generation"]["default_height"])
        self.default_steps_spin.setValue(self.settings["generation"]["default_steps"])
        self.default_cfg_spin.setValue(
            int(self.settings["generation"]["default_cfg_scale"])
        )
        self.default_sampler_combo.setCurrentText(
            self.settings["generation"]["default_sampler"]
        )
        self.auto_save_check.setChecked(self.settings["generation"]["auto_save_images"])
        self.output_dir_edit.setText(self.settings["generation"]["output_directory"])

        # UI settings
        self.theme_combo.setCurrentText(self.settings["ui"]["theme"])
        self.thumbnail_size_spin.setValue(self.settings["ui"]["gallery_thumbnail_size"])
        self.auto_refresh_check.setChecked(self.settings["ui"]["auto_refresh_lora"])
        self.show_info_check.setChecked(self.settings["ui"]["show_generation_info"])

        # Advanced settings
        self.safety_filter_check.setChecked(
            self.settings["advanced"]["enable_safety_filter"]
        )
        self.max_jobs_spin.setValue(self.settings["advanced"]["max_concurrent_jobs"])
        self.log_level_combo.setCurrentText(self.settings["advanced"]["log_level"])
        self.cache_dir_edit.setText(self.settings["advanced"]["cache_directory"])

    def accept_settings(self):
        """Accept and save settings"""
        # Update settings from UI
        self.settings["api"]["base_url"] = self.api_url_edit.text()
        self.settings["api"]["timeout"] = self.api_timeout_spin.value()
        self.settings["api"]["retry_attempts"] = self.retry_spin.value()

        self.settings["generation"]["default_width"] = self.default_width_spin.value()
        self.settings["generation"]["default_height"] = self.default_height_spin.value()
        self.settings["generation"]["default_steps"] = self.default_steps_spin.value()
        self.settings["generation"]["default_cfg_scale"] = float(
            self.default_cfg_spin.value()
        )
        self.settings["generation"][
            "default_sampler"
        ] = self.default_sampler_combo.currentText()
        self.settings["generation"][
            "auto_save_images"
        ] = self.auto_save_check.isChecked()
        self.settings["generation"]["output_directory"] = self.output_dir_edit.text()

        self.settings["ui"]["theme"] = self.theme_combo.currentText()
        self.settings["ui"]["gallery_thumbnail_size"] = self.thumbnail_size_spin.value()
        self.settings["ui"]["auto_refresh_lora"] = self.auto_refresh_check.isChecked()
        self.settings["ui"]["show_generation_info"] = self.show_info_check.isChecked()

        self.settings["advanced"][
            "enable_safety_filter"
        ] = self.safety_filter_check.isChecked()
        self.settings["advanced"]["max_concurrent_jobs"] = self.max_jobs_spin.value()
        self.settings["advanced"]["log_level"] = self.log_level_combo.currentText()
        self.settings["advanced"]["cache_directory"] = self.cache_dir_edit.text()

        if self.save_settings():
            self.accept()
