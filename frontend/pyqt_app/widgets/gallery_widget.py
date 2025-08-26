# frontend/pyqt_app/widgets/gallery_widget.py
"""
Image Gallery Widget
"""
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QPushButton,
    QGridLayout,
    QFrame,
    QFileDialog,
    QMessageBox,
    QMenu,
    QAction,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont


class ImageCard(QFrame):
    """Individual image card widget"""

    clicked = pyqtSignal(str, dict)  # image_path, metadata

    def __init__(self, image_path, metadata=None):
        super().__init__()
        self.image_path = image_path
        self.metadata = metadata or {}
        self.setup_ui()

    def setup_ui(self):
        """Setup image card UI"""
        self.setFrameStyle(QFrame.Box)
        self.setLineWidth(1)
        self.setMaximumSize(200, 280)
        self.setMinimumSize(180, 260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(180)
        self.image_label.setStyleSheet("border: 1px solid gray;")

        # Load and scale image
        pixmap = QPixmap(self.image_path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(
                180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setText("無法載入圖片")

        layout.addWidget(self.image_label)

        # Info labels
        filename = Path(self.image_path).name
        name_label = QLabel(filename)
        name_label.setWordWrap(True)
        name_label.setMaximumHeight(40)
        font = name_label.font()
        font.setPointSize(8)
        name_label.setFont(font)
        layout.addWidget(name_label)

        # Metadata info
        if self.metadata:
            seed = self.metadata.get("seed", "N/A")
            size = (
                f"{self.metadata.get('width', '?')}×{self.metadata.get('height', '?')}"
            )
            info_text = f"種子: {seed} | {size}"

            info_label = QLabel(info_text)
            info_label.setWordWrap(True)
            info_label.setMaximumHeight(30)
            font = info_label.font()
            font.setPointSize(7)
            info_label.setFont(font)
            layout.addWidget(info_label)

    def mousePressEvent(self, event):
        """Handle mouse click"""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.image_path, self.metadata)
        elif event.button() == Qt.RightButton:
            self.show_context_menu(event.pos())

    def show_context_menu(self, pos):
        """Show context menu"""
        menu = QMenu(self)

        open_action = QAction("開啟圖片", self)
        open_action.triggered.connect(
            lambda: self.clicked.emit(self.image_path, self.metadata)
        )
        menu.addAction(open_action)

        save_action = QAction("另存圖片", self)
        save_action.triggered.connect(self.save_image)
        menu.addAction(save_action)

        delete_action = QAction("刪除圖片", self)
        delete_action.triggered.connect(self.delete_image)
        menu.addAction(delete_action)

        menu.exec_(self.mapToGlobal(pos))

    def save_image(self):
        """Save image to new location"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "儲存圖片",
            Path(self.image_path).name,
            "PNG (*.png);;JPEG (*.jpg);;All Files (*.*)",
        )
        if file_path:
            try:
                pixmap = QPixmap(self.image_path)
                pixmap.save(file_path)
                QMessageBox.information(self, "成功", "圖片已儲存")
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"儲存失敗: {str(e)}")

    def delete_image(self):
        """Delete image file"""
        reply = QMessageBox.question(
            self,
            "確認刪除",
            "確定要刪除這張圖片嗎？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            try:
                os.remove(self.image_path)
                self.parent().remove_image_card(self)
                QMessageBox.information(self, "成功", "圖片已刪除")
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"刪除失敗: {str(e)}")


class GalleryWidget(QWidget):
    """Image gallery widget"""

    def __init__(self):
        super().__init__()
        self.current_image = None
        self.current_metadata = None
        self.image_cards = []
        self.setup_ui()

    def setup_ui(self):
        """Setup gallery UI"""
        layout = QVBoxLayout(self)

        # Header
        header_layout = QHBoxLayout()

        title_label = QLabel("圖片畫廊")
        title_font = title_label.font()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Buttons
        self.open_btn = QPushButton("開啟圖片")
        self.open_btn.clicked.connect(self.open_image)
        header_layout.addWidget(self.open_btn)

        self.save_btn = QPushButton("儲存當前")
        self.save_btn.clicked.connect(self.save_current_image)
        self.save_btn.setEnabled(False)
        header_layout.addWidget(self.save_btn)

        self.clear_btn = QPushButton("清空畫廊")
        self.clear_btn.clicked.connect(self.clear_gallery)
        header_layout.addWidget(self.clear_btn)

        layout.addLayout(header_layout)

        # Current image display
        self.current_frame = QFrame()
        self.current_frame.setFrameStyle(QFrame.Box)
        self.current_frame.setMinimumHeight(300)

        current_layout = QVBoxLayout(self.current_frame)

        self.current_image_label = QLabel("沒有選中的圖片")
        self.current_image_label.setAlignment(Qt.AlignCenter)
        self.current_image_label.setMinimumHeight(250)
        self.current_image_label.setStyleSheet("border: 1px dashed gray;")
        current_layout.addWidget(self.current_image_label)

        # Metadata display
        self.metadata_label = QLabel("")
        self.metadata_label.setWordWrap(True)
        self.metadata_label.setMaximumHeight(40)
        current_layout.addWidget(self.metadata_label)

        layout.addWidget(self.current_frame)

        # Gallery scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(200)

        # Gallery content widget
        self.gallery_widget = QWidget()
        self.gallery_layout = QGridLayout(self.gallery_widget)
        self.gallery_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.scroll_area.setWidget(self.gallery_widget)
        layout.addWidget(self.scroll_area)

    def add_image(self, image_path, metadata=None):
        """Add single image to gallery"""
        if not os.path.exists(image_path):
            return

        # Create image card
        card = ImageCard(image_path, metadata)
        card.clicked.connect(self.select_image)

        # Add to grid layout
        row = len(self.image_cards) // 4
        col = len(self.image_cards) % 4
        self.gallery_layout.addWidget(card, row, col)

        self.image_cards.append(card)

        # Auto-select first image
        if len(self.image_cards) == 1:
            self.select_image(image_path, metadata)

    def add_images(self, image_list):
        """Add multiple images to gallery"""
        for item in image_list:
            if isinstance(item, dict):
                self.add_image(item.get("path"), item.get("metadata"))
            else:
                self.add_image(item)

    def select_image(self, image_path, metadata):
        """Select and display image"""
        self.current_image = image_path
        self.current_metadata = metadata

        # Load and display image
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            # Scale to fit display area while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.current_image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.current_image_label.setPixmap(scaled_pixmap)
        else:
            self.current_image_label.setText("無法載入圖片")

        # Update metadata display
        if metadata:
            meta_text = f"提示詞: {metadata.get('prompt', 'N/A')[:50]}..."
            if len(metadata.get("prompt", "")) > 50:
                meta_text += f"\n種子: {metadata.get('seed', 'N/A')} | "
                meta_text += f"尺寸: {metadata.get('width', '?')}×{metadata.get('height', '?')} | "
                meta_text += f"步數: {metadata.get('steps', '?')} | CFG: {metadata.get('cfg_scale', '?')}"
            self.metadata_label.setText(meta_text)
        else:
            filename = Path(image_path).name
            self.metadata_label.setText(f"檔案: {filename}")

        self.save_btn.setEnabled(True)

    def open_image(self):
        """Open image from file system"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "開啟圖片",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*.*)",
        )
        if file_path:
            self.add_image(file_path)

    def save_current_image(self):
        """Save current selected image"""
        if not self.current_image:
            QMessageBox.warning(self, "警告", "沒有選中的圖片")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "儲存圖片",
            Path(self.current_image).name,
            "PNG (*.png);;JPEG (*.jpg);;All Files (*.*)",
        )
        if file_path:
            try:
                pixmap = QPixmap(self.current_image)
                pixmap.save(file_path)
                QMessageBox.information(self, "成功", "圖片已儲存")
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"儲存失敗: {str(e)}")

    def clear_gallery(self):
        """Clear all images from gallery"""
        if not self.image_cards:
            return

        reply = QMessageBox.question(
            self,
            "確認清空",
            "確定要清空所有圖片嗎？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # Remove all cards
            for card in self.image_cards:
                card.setParent(None)
                card.deleteLater()

            self.image_cards.clear()

            # Reset current image
            self.current_image = None
            self.current_metadata = None
            self.current_image_label.clear()
            self.current_image_label.setText("沒有選中的圖片")
            self.metadata_label.setText("")
            self.save_btn.setEnabled(False)

    def remove_image_card(self, card):
        """Remove specific image card"""
        if card in self.image_cards:
            self.image_cards.remove(card)
            card.setParent(None)
            card.deleteLater()

            # If this was the current image, clear it
            if self.current_image == card.image_path:
                self.current_image = None
                self.current_metadata = None
                self.current_image_label.clear()
                self.current_image_label.setText("沒有選中的圖片")
                self.metadata_label.setText("")
                self.save_btn.setEnabled(False)
