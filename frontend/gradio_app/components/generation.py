# frontend/gradio_app/components/generation.py
"""
Gradio Image Generation Components
"""
import gradio as gr
import tempfile
import os
from pathlib import Path
import json

from frontend.shared.constants import DEFAULT_GENERATION_PARAMS, CONTROLNET_TYPES


def create_generation_interface(api_client):
    """Create the image generation interface"""

    with gr.Column():
        # Generation parameters
        with gr.Group():
            gr.Markdown("### 📝 提示詞設定")

            with gr.Row():
                with gr.Column(scale=3):
                    prompt = gr.Textbox(
                        label="正面提示詞",
                        placeholder="輸入描述想要生成的圖片內容...",
                        lines=3,
                        value="",
                    )

                    negative = gr.Textbox(
                        label="負面提示詞",
                        placeholder="輸入不希望出現的內容...",
                        lines=2,
                        value=DEFAULT_GENERATION_PARAMS["negative"],
                    )

        with gr.Group():
            gr.Markdown("### ⚙️ 生成參數")

            with gr.Row():
                with gr.Column():
                    width = gr.Slider(
                        minimum=256,
                        maximum=2048,
                        step=64,
                        label="寬度",
                        value=DEFAULT_GENERATION_PARAMS["width"],
                    )
                    height = gr.Slider(
                        minimum=256,
                        maximum=2048,
                        step=64,
                        label="高度",
                        value=DEFAULT_GENERATION_PARAMS["height"],
                    )

                with gr.Column():
                    steps = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        label="採樣步數",
                        value=DEFAULT_GENERATION_PARAMS["steps"],
                    )
                    cfg_scale = gr.Slider(
                        minimum=1.0,
                        maximum=30.0,
                        step=0.5,
                        label="CFG 縮放",
                        value=DEFAULT_GENERATION_PARAMS["cfg_scale"],
                    )

            with gr.Row():
                with gr.Column():
                    seed = gr.Number(
                        label="種子 (-1 為隨機)",
                        value=DEFAULT_GENERATION_PARAMS["seed"],
                        precision=0,
                    )
                    sampler = gr.Dropdown(
                        choices=[
                            "DPM++ 2M Karras",
                            "DPM++ SDE Karras",
                            "Euler a",
                            "Euler",
                            "LMS",
                            "Heun",
                            "DDIM",
                        ],
                        label="採樣器",
                        value=DEFAULT_GENERATION_PARAMS["sampler"],
                    )

                with gr.Column():
                    batch_size = gr.Slider(
                        minimum=1,
                        maximum=8,
                        step=1,
                        label="批次大小",
                        value=DEFAULT_GENERATION_PARAMS["batch_size"],
                    )

        # ControlNet settings
        with gr.Group():
            gr.Markdown("### 🎮 ControlNet 控制")

            controlnet_enabled = gr.Checkbox(label="啟用 ControlNet", value=False)

            with gr.Row():
                controlnet_type = gr.Dropdown(
                    choices=CONTROLNET_TYPES,
                    label="控制類型",
                    value="pose",
                    visible=False,
                )
                controlnet_weight = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    label="控制強度",
                    value=1.0,
                    visible=False,
                )

            control_image = gr.Image(label="控制圖片", type="filepath", visible=False)

            def toggle_controlnet(enabled):
                return [
                    gr.update(visible=enabled),  # controlnet_type
                    gr.update(visible=enabled),  # controlnet_weight
                    gr.update(visible=enabled),  # control_image
                ]

            controlnet_enabled.change(
                toggle_controlnet,
                inputs=[controlnet_enabled],
                outputs=[controlnet_type, controlnet_weight, control_image],
            )

        # Generation button and results
        with gr.Row():
            generate_btn = gr.Button(
                "🎨 生成圖片",
                variant="primary",
                size="lg",
                elem_classes=["generate-btn"],
            )
            random_seed_btn = gr.Button("🎲 隨機種子", size="sm")

        def randomize_seed():
            import random

            return random.randint(0, 2147483647)

        random_seed_btn.click(randomize_seed, outputs=[seed])

        # Results display
        with gr.Group():
            gr.Markdown("### 🖼️ 生成結果")

            with gr.Row():
                result_gallery = gr.Gallery(
                    label="生成的圖片",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height="auto",
                )

                with gr.Column():
                    result_info = gr.Textbox(
                        label="生成資訊", lines=8, max_lines=10, interactive=False
                    )

                    download_btn = gr.DownloadButton("下載圖片", visible=False)

    def generate_image_fn(
        prompt,
        negative,
        width,
        height,
        steps,
        cfg_scale,
        seed,
        sampler,
        batch_size,
        controlnet_enabled,
        controlnet_type,
        controlnet_weight,
        control_image,
    ):
        """Generate image with given parameters"""

        if not prompt.strip():
            return [], "錯誤: 請輸入正面提示詞", None

        try:
            # Prepare generation parameters
            params = {
                "prompt": prompt,
                "negative": negative,
                "width": int(width),
                "height": int(height),
                "steps": int(steps),
                "cfg_scale": float(cfg_scale),
                "seed": int(seed),
                "sampler": sampler,
                "batch_size": int(batch_size),
            }

            # Handle ControlNet
            if controlnet_enabled and control_image:
                control_params = {
                    "control_type": controlnet_type,
                    "control_image": control_image,
                    "control_weight": float(controlnet_weight),
                }
                result = api_client.controlnet_generate(
                    {**params, **control_params}, controlnet_type
                )
            else:
                result = api_client.generate_image(params)

            if "image_path" in result:
                # Handle single or multiple images
                image_paths = result["image_path"]
                if isinstance(image_paths, str):
                    image_paths = [image_paths]

                # Prepare result info
                actual_seed = result.get("seed", seed)
                info_text = f"""生成完成！

提示詞: {prompt}
負面提示詞: {negative}
種子: {actual_seed}
尺寸: {width} × {height}
步數: {steps}
CFG縮放: {cfg_scale}
採樣器: {sampler}
批次大小: {batch_size}

生成時間: {result.get('elapsed_ms', 0)} 毫秒
"""

                if controlnet_enabled:
                    info_text += (
                        f"\nControlNet: {controlnet_type} (強度: {controlnet_weight})"
                    )

                # Return first image for download
                download_file = image_paths[0] if image_paths else None

                return (
                    image_paths,
                    info_text,
                    gr.update(visible=bool(download_file), value=download_file),
                )
            else:
                return [], f"生成失敗: {result.get('message', '未知錯誤')}", None

        except Exception as e:
            return [], f"生成失敗: {str(e)}", None

    # Connect generate button
    generate_btn.click(
        generate_image_fn,
        inputs=[
            prompt,
            negative,
            width,
            height,
            steps,
            cfg_scale,
            seed,
            sampler,
            batch_size,
            controlnet_enabled,
            controlnet_type,
            controlnet_weight,
            control_image,
        ],
        outputs=[result_gallery, result_info, download_btn],
    )

    return {
        "prompt": prompt,
        "negative": negative,
        "generate_btn": generate_btn,
        "result_gallery": result_gallery,
        "result_info": result_info,
    }
