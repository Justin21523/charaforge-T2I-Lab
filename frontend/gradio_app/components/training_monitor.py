# frontend/gradio_app/components/training_monitor.py
"""
Gradio Training Monitor Components
"""
import gradio as gr
import json
import time
from datetime import datetime


def create_training_interface(api_client):
    """Create training monitoring interface"""

    with gr.Column():
        gr.Markdown("### 🔬 LoRA 訓練監控")

        # Training submission
        with gr.Group():
            gr.Markdown("#### 提交訓練任務")

            with gr.Tabs():
                with gr.TabItem("快速配置"):
                    with gr.Row():
                        preset_buttons = []

                        character_btn = gr.Button("角色訓練", variant="secondary")
                        style_btn = gr.Button("風格訓練", variant="secondary")
                        custom_btn = gr.Button("自定義配置", variant="secondary")

                        preset_buttons = [character_btn, style_btn, custom_btn]

                    # Training parameters
                    with gr.Group():
                        gr.Markdown("**訓練參數**")

                        with gr.Row():
                            with gr.Column():
                                run_id = gr.Textbox(
                                    label="任務 ID",
                                    placeholder="例如: char_alice_v1",
                                    value=f"lora_train_{int(time.time())}",
                                )
                                dataset_name = gr.Textbox(
                                    label="數據集名稱", placeholder="數據集目錄名稱"
                                )

                            with gr.Column():
                                rank = gr.Slider(
                                    minimum=4,
                                    maximum=128,
                                    step=4,
                                    label="LoRA Rank",
                                    value=16,
                                )
                                learning_rate = gr.Number(
                                    label="學習率", value=1e-4, precision=6
                                )

                        with gr.Row():
                            with gr.Column():
                                resolution = gr.Slider(
                                    minimum=512,
                                    maximum=1024,
                                    step=64,
                                    label="解析度",
                                    value=768,
                                )
                                batch_size = gr.Slider(
                                    minimum=1,
                                    maximum=8,
                                    step=1,
                                    label="批次大小",
                                    value=1,
                                )

                            with gr.Column():
                                max_steps = gr.Slider(
                                    minimum=500,
                                    maximum=10000,
                                    step=100,
                                    label="最大步數",
                                    value=2000,
                                )
                                save_every = gr.Slider(
                                    minimum=100,
                                    maximum=1000,
                                    step=100,
                                    label="保存間隔",
                                    value=500,
                                )

                    submit_training_btn = gr.Button(
                        "🚀 開始訓練", variant="primary", size="lg"
                    )

                with gr.TabItem("YAML 配置"):
                    config_editor = gr.Textbox(
                        label="訓練配置 (YAML)",
                        lines=15,
                        placeholder="在此貼上或編輯 YAML 配置文件...",
                        value="""# LoRA 訓練配置範例
run_id: "my_character_v1"
dataset_name: "my_dataset"

# LoRA 參數
rank: 16
alpha: 32
dropout: 0.1

# 訓練參數
learning_rate: 0.0001
text_encoder_lr: 0.00005
resolution: 768
batch_size: 1
gradient_accumulation_steps: 4
max_train_steps: 2000

# 進階設定
mixed_precision: "fp16"
gradient_checkpointing: true
use_ema: true
seed: 42

# 保存設定
save_every: 500
save_precision: "fp16"
""",
                    )

                    submit_yaml_btn = gr.Button(
                        "🚀 提交 YAML 配置", variant="primary", size="lg"
                    )

        # Training status
        with gr.Group():
            gr.Markdown("#### 訓練狀態監控")

            with gr.Row():
                refresh_jobs_btn = gr.Button("🔄 刷新任務", size="sm")

            # Active jobs list
            active_jobs = gr.Dropdown(
                label="進行中的訓練任務", choices=[], interactive=True
            )

            # Job status display
            job_status_display = gr.Textbox(
                label="任務詳情",
                lines=8,
                interactive=False,
                placeholder="選擇任務以查看狀態",
            )

            with gr.Row():
                cancel_job_btn = gr.Button("⏹️ 停止訓練", size="sm")
                download_model_btn = gr.DownloadButton("📥 下載模型", visible=False)

        # Training metrics and visualization
        with gr.Group():
            gr.Markdown("#### 訓練指標")

            # Metrics plot (placeholder)
            metrics_plot = gr.Plot(label="損失曲線", value=None)

            # Sample images during training
            with gr.Row():
                sample_gallery = gr.Gallery(
                    label="訓練樣本", show_label=True, columns=3, rows=2, height="300px"
                )

    # State to store current job info
    current_job_state = gr.State({})

    def load_preset_config(preset_type):
        """Load preset training configuration"""
        presets = {
            "character": {
                "rank": 16,
                "learning_rate": 1e-4,
                "text_encoder_lr": 5e-5,
                "resolution": 768,
                "batch_size": 1,
                "max_steps": 2000,
                "save_every": 500,
            },
            "style": {
                "rank": 8,
                "learning_rate": 8e-5,
                "text_encoder_lr": 0,
                "resolution": 768,
                "batch_size": 2,
                "max_steps": 1500,
                "save_every": 300,
            },
            "custom": {
                "rank": 32,
                "learning_rate": 5e-5,
                "text_encoder_lr": 1e-5,
                "resolution": 1024,
                "batch_size": 1,
                "max_steps": 3000,
                "save_every": 500,
            },
        }

        config = presets.get(preset_type, presets["character"])
        return (
            config["rank"],
            config["learning_rate"],
            config["resolution"],
            config["batch_size"],
            config["max_steps"],
            config["save_every"],
        )

    def submit_training_job(
        run_id, dataset_name, rank, lr, resolution, batch_size, max_steps, save_every
    ):
        """Submit training job"""
        if not run_id.strip() or not dataset_name.strip():
            return "", "錯誤: 請填寫任務 ID 和數據集名稱", gr.update(), {}

        try:
            config = {
                "run_id": run_id,
                "dataset_name": dataset_name,
                "rank": int(rank),
                "learning_rate": float(lr),
                "resolution": int(resolution),
                "batch_size": int(batch_size),
                "max_train_steps": int(max_steps),
                "save_every": int(save_every),
                "mixed_precision": "fp16",
                "gradient_checkpointing": True,
            }

            result = api_client.submit_training_job(config)

            if "job_id" in result or "run_id" in result:
                job_id = result.get("job_id") or result.get("run_id")
                status_text = f"訓練任務提交成功！\n任務 ID: {job_id}\n狀態: 已排隊"

                return (
                    "",
                    status_text,
                    gr.update(),
                    {"job_id": job_id, "status": "submitted"},
                )
            else:
                return (
                    "",
                    f"提交失敗: {result.get('message', 'Unknown error')}",
                    gr.update(),
                    {},
                )

        except Exception as e:
            return "", f"提交訓練任務失敗: {str(e)}", gr.update(), {}

    def submit_yaml_config(yaml_config):
        """Submit YAML configuration"""
        if not yaml_config.strip():
            return "", "錯誤: 請輸入 YAML 配置", gr.update(), {}

        try:
            import yaml

            config = yaml.safe_load(yaml_config)

            result = api_client.submit_training_job(config)

            if "job_id" in result or "run_id" in result:
                job_id = result.get("job_id") or result.get("run_id")
                status_text = f"YAML 配置提交成功！\n任務 ID: {job_id}\n狀態: 已排隊"

                return (
                    status_text,
                    gr.update(),
                    {"job_id": job_id, "status": "submitted"},
                )
            else:
                return (
                    f"提交失敗: {result.get('message', 'Unknown error')}",
                    gr.update(),
                    {},
                )

        except yaml.YAMLError as e:
            return f"YAML 格式錯誤: {str(e)}", gr.update(), {}
        except Exception as e:
            return f"提交失敗: {str(e)}", gr.update(), {}

    def refresh_training_jobs():
        """Refresh training jobs list"""
        try:
            jobs = api_client.list_training_jobs()
            if jobs:
                choices = [
                    (
                        f"{job.get('run_id', 'Unknown')} - {job.get('status', 'Unknown')}",
                        job.get("run_id", ""),
                    )
                    for job in jobs
                ]
                return gr.update(choices=choices), "訓練任務列表已更新"
            else:
                return gr.update(choices=[]), "沒有找到訓練任務"
        except Exception as e:
            return gr.update(choices=[]), f"刷新失敗: {str(e)}"

    def check_training_status(selected_job):
        """Check training job status"""
        if not selected_job:
            return "請選擇訓練任務", None, [], gr.update(visible=False), {}

        try:
            result = api_client.get_training_status(selected_job)

            status = result.get("status", "unknown")
            progress = result.get("progress", {})
            metrics = result.get("metrics", {})

            status_text = f"任務 ID: {selected_job}\n"
            status_text += f"狀態: {status}\n"

            if progress:
                current_step = progress.get("current_step", 0)
                total_steps = progress.get("total_steps", 0)
                if total_steps > 0:
                    percentage = (current_step / total_steps) * 100
                    status_text += (
                        f"進度: {current_step}/{total_steps} ({percentage:.1f}%)\n"
                    )

            status_text += f"開始時間: {result.get('started_at', 'N/A')}\n"

            if status == "completed":
                status_text += f"完成時間: {result.get('completed_at', 'N/A')}\n"
                status_text += f"總耗時: {result.get('elapsed_time', 'N/A')}\n"

            if metrics:
                status_text += "\n訓練指標:\n"
                if "loss" in metrics:
                    status_text += f"損失: {metrics['loss']:.4f}\n"
                if "learning_rate" in metrics:
                    status_text += f"學習率: {metrics['learning_rate']:.6f}\n"

            # Generate metrics plot if available
            plot_data = None
            if "loss_history" in metrics:
                import matplotlib.pyplot as plt
                import numpy as np

                steps = list(range(len(metrics["loss_history"])))
                losses = metrics["loss_history"]

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(steps, losses, "b-", linewidth=2, label="Training Loss")
                ax.set_xlabel("Step")
                ax.set_ylabel("Loss")
                ax.set_title("Training Loss Curve")
                ax.grid(True, alpha=0.3)
                ax.legend()

                plot_data = fig

            # Get sample images if available
            sample_images = []
            if "sample_images" in result:
                sample_images = result["sample_images"]

            # Show download button if completed
            download_visible = status == "completed"

            return (
                status_text,
                plot_data,
                sample_images,
                gr.update(visible=download_visible),
                result,
            )

        except Exception as e:
            return f"查詢狀態失敗: {str(e)}", None, [], gr.update(visible=False), {}

    # Connect events
    character_btn.click(
        lambda: load_preset_config("character"),
        outputs=[rank, learning_rate, resolution, batch_size, max_steps, save_every],
    )

    style_btn.click(
        lambda: load_preset_config("style"),
        outputs=[rank, learning_rate, resolution, batch_size, max_steps, save_every],
    )

    custom_btn.click(
        lambda: load_preset_config("custom"),
        outputs=[rank, learning_rate, resolution, batch_size, max_steps, save_every],
    )

    submit_training_btn.click(
        submit_training_job,
        inputs=[
            run_id,
            dataset_name,
            rank,
            learning_rate,
            resolution,
            batch_size,
            max_steps,
            save_every,
        ],
        outputs=[run_id, job_status_display, download_model_btn, current_job_state],
    )

    submit_yaml_btn.click(
        submit_yaml_config,
        inputs=[config_editor],
        outputs=[job_status_display, download_model_btn, current_job_state],
    )

    refresh_jobs_btn.click(
        refresh_training_jobs, outputs=[active_jobs, job_status_display]
    )

    active_jobs.change(
        check_training_status,
        inputs=[active_jobs],
        outputs=[
            job_status_display,
            metrics_plot,
            sample_gallery,
            download_model_btn,
            current_job_state,
        ],
    )

    return {
        "job_status_display": job_status_display,
        "metrics_plot": metrics_plot,
        "sample_gallery": sample_gallery,
        "submit_training_btn": submit_training_btn,
    }
