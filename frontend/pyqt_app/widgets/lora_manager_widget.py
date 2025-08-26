# frontend/pyqt_app/widgets/lora_manager_widget.py
"""
LoRA Model Manager Widget
"""
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QLabel,
    QSlider,
    QDoubleSpinBox,
    QGroupBox,
    QMessageBox,
    QProgressDialog,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont


class LoRAListThread(QThread):
    """Thread to fetch LoRA list"""

    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client

    def run(self):
        try:
            loras = self.api_client.list_loras()
            self.finished.emit(loras)
        except Exception as e:
            self.error.emit(str(e))


class LoRAManagerWidget(QWidget):
    """LoRA model management interface"""

    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client
        self.loaded_loras = {}  # {lora_id: weight}
        self.setup_ui()
        self.refresh_lora_list()

    def setup_ui(self):
        """Setup the LoRA manager UI"""
        layout = QVBoxLayout(self)

        # Available LoRAs group
        available_group = QGroupBox("可用 LoRA 模型")
        available_layout = QVBoxLayout(available_group)

        # Refresh button
        refresh_btn = QPushButton("重新整理列表")
        refresh_btn.clicked.connect(self.refresh_lora_list)
        available_layout.addWidget(refresh_btn)

        # LoRA list
        self.lora_list = QListWidget()
        self.lora_list.itemDoubleClicked.connect(self.load_selected_lora)
        available_layout.addWidget(self.lora_list)

        # Load controls
        load_layout = QHBoxLayout()

        weight_layout = QHBoxLayout()
        weight_layout.addWidget(QLabel("權重:"))
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(0.0, 2.0)
        self.weight_spin.setValue(1.0)
        self.weight_spin.setSingleStep(0.1)
        weight_layout.addWidget(self.weight_spin)

        self.load_btn = QPushButton("載入 LoRA")
        self.load_btn.clicked.connect(self.load_selected_lora)

        load_layout.addLayout(weight_layout)
        load_layout.addWidget(self.load_btn)
        available_layout.addLayout(load_layout)

        layout.addWidget(available_group)

        # Loaded LoRAs group
        loaded_group = QGroupBox("已載入 LoRA")
        loaded_layout = QVBoxLayout(loaded_group)

        self.loaded_list = QListWidget()
        loaded_layout.addWidget(self.loaded_list)

        # Unload button
        self.unload_btn = QPushButton("卸載選中的 LoRA")
        self.unload_btn.clicked.connect(self.unload_selected_lora)
        self.unload_btn.setEnabled(False)
        loaded_layout.addWidget(self.unload_btn)

        # Unload all button
        unload_all_btn = QPushButton("卸載所有 LoRA")
        unload_all_btn.clicked.connect(self.unload_all_loras)
        loaded_layout.addWidget(unload_all_btn)

        layout.addWidget(loaded_group)

        # Connect loaded list selection
        self.loaded_list.itemSelectionChanged.connect(self.on_loaded_selection_changed)

    def refresh_lora_list(self):
        """Refresh the available LoRA list"""
        self.lora_list.clear()

        # Show loading
        loading_item = QListWidgetItem("載入中...")
        loading_item.setFlags(Qt.NoItemFlags)
        font = loading_item.font()
        font.setItalic(True)
        loading_item.setFont(font)
        self.lora_list.addItem(loading_item)

        # Start loading thread
        self.list_thread = LoRAListThread(self.api_client)
        self.list_thread.finished.connect(self.on_lora_list_loaded)
        self.list_thread.error.connect(self.on_lora_list_error)
        self.list_thread.start()

    def on_lora_list_loaded(self, loras):
        """Handle loaded LoRA list"""
        self.lora_list.clear()

        if not loras:
            no_loras_item = QListWidgetItem("沒有找到 LoRA 模型")
            no_loras_item.setFlags(Qt.NoItemFlags)
            font = no_loras_item.font()
            font.setItalic(True)
            no_loras_item.setFont(font)
            self.lora_list.addItem(no_loras_item)
            return

        for lora in loras:
            item_text = f"{lora.get('name', 'Unknown')} ({lora.get('id', 'No ID')})"
            if "description" in lora:
                item_text += f"\n{lora['description']}"

            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, lora)
            self.lora_list.addItem(item)

    def on_lora_list_error(self, error):
        """Handle LoRA list loading error"""
        self.lora_list.clear()
        error_item = QListWidgetItem(f"載入失敗: {error}")
        error_item.setFlags(Qt.NoItemFlags)
        self.lora_list.addItem(error_item)

    def load_selected_lora(self):
        """Load the selected LoRA"""
        current_item = self.lora_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "警告", "請選擇要載入的 LoRA 模型")
            return

        lora_data = current_item.data(Qt.UserRole)
        if not lora_data:
            return

        lora_id = lora_data.get("id")
        if not lora_id:
            QMessageBox.warning(self, "錯誤", "LoRA 模型 ID 無效")
            return

        if lora_id in self.loaded_loras:
            QMessageBox.information(self, "提示", "該 LoRA 已經載入")
            return

        weight = self.weight_spin.value()

        # Show loading dialog
        progress = QProgressDialog("載入 LoRA 中...", "取消", 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        try:
            result = self.api_client.load_lora(lora_id, weight)
            progress.close()

            if result.get("status") == "success":
                self.loaded_loras[lora_id] = weight
                self.update_loaded_list()
                QMessageBox.information(
                    self, "成功", f"LoRA '{lora_data.get('name', lora_id)}' 載入成功"
                )
            else:
                QMessageBox.warning(
                    self, "失敗", f"載入失敗: {result.get('message', 'Unknown error')}"
                )

        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "錯誤", f"載入 LoRA 時發生錯誤: {str(e)}")

    def unload_selected_lora(self):
        """Unload the selected LoRA"""
        current_item = self.loaded_list.currentItem()
        if not current_item:
            return

        lora_id = current_item.data(Qt.UserRole)
        if not lora_id:
            return

        try:
            result = self.api_client.unload_lora(lora_id)

            if result.get("status") == "success":
                del self.loaded_loras[lora_id]
                self.update_loaded_list()
                QMessageBox.information(self, "成功", f"LoRA '{lora_id}' 卸載成功")
            else:
                QMessageBox.warning(
                    self, "失敗", f"卸載失敗: {result.get('message', 'Unknown error')}"
                )

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"卸載 LoRA 時發生錯誤: {str(e)}")

    def unload_all_loras(self):
        """Unload all loaded LoRAs"""
        if not self.loaded_loras:
            QMessageBox.information(self, "提示", "沒有已載入的 LoRA")
            return

        reply = QMessageBox.question(
            self,
            "確認",
            "確定要卸載所有 LoRA 嗎？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        failed_loras = []
        for lora_id in list(self.loaded_loras.keys()):
            try:
                result = self.api_client.unload_lora(lora_id)
                if result.get("status") == "success":
                    del self.loaded_loras[lora_id]
                else:
                    failed_loras.append(lora_id)
            except Exception:
                failed_loras.append(lora_id)

        self.update_loaded_list()

        if failed_loras:
            QMessageBox.warning(
                self, "部分失敗", f"以下 LoRA 卸載失敗: {', '.join(failed_loras)}"
            )
        else:
            QMessageBox.information(self, "成功", "所有 LoRA 已卸載")

    def update_loaded_list(self):
        """Update the loaded LoRAs list"""
        self.loaded_list.clear()

        for lora_id, weight in self.loaded_loras.items():
            item_text = f"{lora_id} (權重: {weight:.1f})"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, lora_id)
            self.loaded_list.addItem(item)

    def on_loaded_selection_changed(self):
        """Handle loaded LoRA selection change"""
        has_selection = bool(self.loaded_list.currentItem())
        self.unload_btn.setEnabled(has_selection)
