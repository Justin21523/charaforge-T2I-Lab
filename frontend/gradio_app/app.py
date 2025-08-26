# frontend/gradio_app/app.py
"""
SagaForge T2I Lab - Gradio Web Interface
Main Gradio application
"""
import gradio as gr
import os
import sys
from pathlib import Path
import tempfile
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from frontend.shared.api_client import SagaForgeAPIClient
from frontend.shared.constants import DEFAULT_GENERATION_PARAMS, CONTROLNET_TYPES
from frontend.gradio_app.components import (
    generation,
    lora_management,
    batch_processing,
    training_monitor,
)

# Initialize API client
api_client = SagaForgeAPIClient()


def check_api_health():
    """Check API health and return status"""
    health = api_client.health_check()
    if health.get("status") == "ok":
        return "✅ API 連線正常", "green"
    else:
        return f"❌ API 連線失敗: {health.get('message', 'Unknown error')}", "red"


def create_main_interface():
    """Create the main Gradio interface"""

    # Custom CSS
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .status-ok { color: green; font-weight: bold; }
    .status-error { color: red; font-weight: bold; }
    .image-preview { max-height: 400px; }
    .parameter-group { border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; }
    .generate-btn {
        background: linear-gradient(45deg, #1e3a8a, #3b82f6) !important;
        color: white !important;
        font-size: 16px !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
    }
    """

    with gr.Blocks(css=css, title="SagaForge T2I Lab", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎨 SagaForge T2I Lab")
        gr.Markdown("專業的動畫角色文生圖與 LoRA 訓練平台")

        # API Status
        with gr.Row():
            api_status = gr.HTML()
            refresh_status_btn = gr.Button("刷新狀態", size="sm")

        def update_status():
            message, color = check_api_health()
            return f'<span style="color: {color}; font-weight: bold;">{message}</span>'

        refresh_status_btn.click(update_status, outputs=[api_status])
        app.load(update_status, outputs=[api_status])

        # Main tabs
        with gr.Tabs():
            # Image Generation Tab
            with gr.TabItem("🖼️ 圖片生成"):
                generation_interface = generation.create_generation_interface(
                    api_client
                )

            # LoRA Management Tab
            with gr.TabItem("🎭 LoRA 管理"):
                lora_interface = lora_management.create_lora_interface(api_client)

            # Batch Processing Tab
            with gr.TabItem("⚡ 批次處理"):
                batch_interface = batch_processing.create_batch_interface(api_client)

            # Training Monitor Tab
            with gr.TabItem("🔬 訓練監控"):
                training_interface = training_monitor.create_training_interface(
                    api_client
                )

            # Help Tab
            with gr.TabItem("❓ 說明"):
                with gr.Column():
                    gr.Markdown(
                        """
                    ## 使用說明

                    ### 🖼️ 圖片生成
                    - 輸入正面和負面提示詞
                    - 調整生成參數（尺寸、步數、CFG等）
                    - 可選擇啟用 ControlNet 進行精確控制

                    ### 🎭 LoRA 管理
                    - 瀏覽可用的 LoRA 模型
                    - 載入/卸載 LoRA 模型
                    - 調整 LoRA 權重

                    ### ⚡ 批次處理
                    - 上傳 CSV 或 JSON 格式的任務檔案
                    - 監控批次生成進度
                    - 下載生成結果

                    ### 🔬 訓練監控
                    - 提交 LoRA 訓練任務
                    - 實時監控訓練進度
                    - 查看訓練指標和損失曲線

                    ## 技術支援
                    - API 文檔: http://localhost:8000/docs
                    - 問題回報: GitHub Issues
                    """
                    )

        # Footer
        gr.Markdown("---")
        gr.Markdown("© 2024 SagaForge T2I Lab | Powered by Gradio")

    return app


if __name__ == "__main__":
    # Create and launch the interface
    app = create_main_interface()

    # Launch settings
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_tips=True,
        enable_queue=True,
        max_threads=10,
    )
