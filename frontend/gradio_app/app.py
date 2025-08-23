# frontend/gradio_app/app.py
import gradio as gr
import requests
import os
from PIL import Image
import json
from components.caption_tab import create_caption_tab
from components.vqa_tab import create_vqa_tab
from components.chat_tab import create_chat_tab
from components.rag_tab import create_rag_tab

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def create_app():
    """Create main Gradio application with all tabs"""

    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

    with gr.Blocks(
        title="CharaForge Multi-Modal Lab",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .tab-nav {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        """,
    ) as app:

        gr.Markdown(
            """
            # 🤖 CharaForge Multi-Modal Lab
            ### 整合圖像理解、對話、文件問答的 AI 工具平台
            """,
            elem_classes=["text-center"],
        )

        with gr.Tabs():
            # Existing tabs
            create_caption_tab(api_base_url)
            create_vqa_tab(api_base_url)
            create_chat_tab(api_base_url)

            # NEW: RAG tab
            create_rag_tab(api_base_url)

    return app


def generate_caption(image, max_length, num_beams, temperature):
    """Call caption API and return result"""
    try:
        if image is None:
            return "Please upload an image first."

        # Prepare request
        files = {"image": ("image.png", image, "image/png")}
        params = {
            "max_length": max_length,
            "num_beams": num_beams,
            "temperature": temperature,
        }

        # Call API
        response = requests.post(
            f"{API_BASE}/api/v1/caption", files=files, params=params, timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return f"**Caption:** {result['caption']}\n\n**Model:** {result['model_used']}\n**Time:** {result['elapsed_ms']}ms"
        else:
            return f"Error: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error: {str(e)}"


def check_api_health():
    """Check if API is available"""
    try:
        response = requests.get(f"{API_BASE}/api/v1/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return f"✅ API healthy | GPU: {data['gpu_available']}"
        else:
            return f"❌ API error: {response.status_code}"
    except:
        return f"❌ API unavailable at {API_BASE}"


# Gradio Interface
with gr.Blocks(title="CharaForge Multi-Modal Lab", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎨 CharaForge Multi-Modal Lab")
    gr.Markdown("Upload an image to generate AI captions using BLIP-2")

    # API Status
    with gr.Row():
        api_status = gr.Textbox(
            value=check_api_health(), label="API Status", interactive=False
        )
        refresh_btn = gr.Button("🔄 Refresh")

    # Main Interface
    with gr.Row():
        with gr.Column(scale=1):
            # Input controls
            image_input = gr.Image(label="Upload Image", type="pil", height=300)

            with gr.Accordion("Advanced Settings", open=False):
                max_length = gr.Slider(
                    minimum=10, maximum=200, value=50, step=5, label="Max Length"
                )
                num_beams = gr.Slider(
                    minimum=1, maximum=10, value=3, step=1, label="Num Beams"
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature"
                )

            generate_btn = gr.Button("🎯 Generate Caption", variant="primary")

        with gr.Column(scale=1):
            # Output
            caption_output = gr.Markdown(
                label="Generated Caption",
                value="Upload an image and click 'Generate Caption' to start.",
            )

    # Examples
    gr.Examples(
        examples=[
            ["examples/anime_girl.jpg", 50, 3, 1.0],
            ["examples/landscape.jpg", 30, 5, 0.8],
        ],
        inputs=[image_input, max_length, num_beams, temperature],
        outputs=caption_output,
        fn=generate_caption,
        cache_examples=False,
    )

    # Event handlers
    generate_btn.click(
        fn=generate_caption,
        inputs=[image_input, max_length, num_beams, temperature],
        outputs=caption_output,
    )

    refresh_btn.click(fn=check_api_health, outputs=api_status)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True, debug=True)
