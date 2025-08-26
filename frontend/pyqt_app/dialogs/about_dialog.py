# frontend/pyqt_app/dialogs/about_dialog.py
"""
About Dialog for PyQt Application
"""
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont


class AboutDialog(QDialog):
    """About dialog showing application information"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("關於 SagaForge T2I Lab")
        self.setModal(True)
        self.setFixedSize(450, 350)
        self.setup_ui()

    def setup_ui(self):
        """Setup the about UI"""
        layout = QVBoxLayout(self)

        # Header section
        header_layout = QHBoxLayout()

        # Logo/Icon (if available)
        icon_label = QLabel()
        icon_label.setText("🎨")  # Fallback emoji if no icon
        icon_label.setStyleSheet("font-size: 48px;")
        icon_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(icon_label)

        # Title and version
        title_layout = QVBoxLayout()

        title_label = QLabel("SagaForge T2I Lab")
        title_font = title_label.font()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_layout.addWidget(title_label)

        version_label = QLabel("Version 1.0.0")
        version_label.setStyleSheet("color: #666666;")
        title_layout.addWidget(version_label)

        subtitle_label = QLabel("專業的動畫角色文生圖與 LoRA 訓練平台")
        subtitle_label.setWordWrap(True)
        title_layout.addWidget(subtitle_label)

        header_layout.addLayout(title_layout)
        layout.addLayout(header_layout)

        # Description
        description = QTextBrowser()
        description.setMaximumHeight(150)
        description.setOpenExternalLinks(True)
        description.setHtml(
            """
        <h3>功能特色</h3>
        <ul>
        <li><strong>文字轉圖片生成</strong> - 支援 SD1.5/SDXL 模型</li>
        <li><strong>LoRA 模型管理</strong> - 動態載入與權重調整</li>
        <li><strong>ControlNet 控制</strong> - 精確的姿勢與構圖控制</li>
        <li><strong>批次處理</strong> - 大量圖片生成與任務管理</li>
        <li><strong>LoRA 微調訓練</strong> - 自定義角色與風格模型</li>
        <li><strong>多平台介面</strong> - 桌面應用與網頁版本</li>
        </ul>

        <h3>技術棧</h3>
        <p>
        <strong>後端:</strong> FastAPI, Celery, Redis, diffusers, transformers<br>
        <strong>前端:</strong> PyQt5, Gradio, React.js<br>
        <strong>AI 模型:</strong> Stable Diffusion, ControlNet, CLIP, LoRA
        </p>
        """
        )
        layout.addWidget(description)

        # Copyright and links
        info_layout = QVBoxLayout()

        copyright_label = QLabel("© 2024 SagaForge Team. All rights reserved.")
        copyright_label.setAlignment(Qt.AlignCenter)
        copyright_label.setStyleSheet("color: #666666; font-size: 11px;")
        info_layout.addWidget(copyright_label)

        links_label = QLabel(
            '<a href="https://github.com/sagaforge/t2i-lab">GitHub</a> | '
            '<a href="https://sagaforge.ai/docs">文檔</a> | '
            '<a href="https://sagaforge.ai/support">技術支援</a>'
        )
        links_label.setAlignment(Qt.AlignCenter)
        links_label.setOpenExternalLinks(True)
        info_layout.addWidget(links_label)

        layout.addLayout(info_layout)

        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_btn = QPushButton("關閉")
        close_btn.clicked.connect(self.accept)
        close_btn.setDefault(True)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)
