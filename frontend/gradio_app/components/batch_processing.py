# frontend/gradio_app/components/batch_processing.py
"""
Gradio Batch Processing Components
"""
import gradio as gr
import json
import tempfile
import os
from pathlib import Path


def create_batch_interface(api_client):
    """Create batch processing interface"""

    with gr.Column():
        gr.Markdown("### ⚡ 批次處理")

        # Job submission
        with gr.Group():
            gr.Markdown("#### 提交批次任務")

            with gr.Tabs():
                with gr.TabItem("CSV 上傳"):
                    csv_file = gr.File(
                        label="上傳 CSV 檔案", file_types=[".csv"], file_count="single"
                    )

                    gr.Markdown(
                        """
                    **CSV 格式範例:**
                    ```
                    prompt,negative,width,height,steps,seed
                    "anime girl, blue hair",lowres,768,768,25,-1
                    "cat, cute, fluffy",blurry,512,512,20,12345
                    ```
                    """
                    )

                    csv_submit_btn = gr.Button("提交 CSV 批次任務", variant="primary")

                with gr.TabItem("JSON 上傳"):
                    json_file = gr.File(
                        label="上傳 JSON 檔案",
                        file_types=[".json"],
                        file_count="single",
                    )

                    gr.Markdown(
                        """
                    **JSON 格式範例:**
                    ```json
                    {
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
                    }
                    ```
                    """
                    )

                    json_submit_btn = gr.Button("提交 JSON 批次任務", variant="primary")

                with gr.TabItem("手動設定"):
                    with gr.Row():
                        batch_prompts = gr.Textbox(
                            label="提示詞列表 (每行一個)",
                            lines=5,
                            placeholder="anime girl, blue hair\ncat, cute, fluffy\nlandscape, mountains",
                        )

                    with gr.Row():
                        with gr.Column():
                            batch_negative = gr.Textbox(
                                label="統一負面提示詞",
                                value="lowres, blurry, bad anatomy",
                                lines=2,
                            )
                            batch_width = gr.Slider(
                                256, 2048, 768, step=64, label="寬度"
                            )
                            batch_height = gr.Slider(
                                256, 2048, 768, step=64, label="高度"
                            )

                        with gr.Column():
                            batch_steps = gr.Slider(1, 100, 25, step=1, label="步數")
                            batch_cfg = gr.Slider(1.0, 30.0, 7.5, step=0.5, label="CFG")
                            batch_sampler = gr.Dropdown(
                                ["DPM++ 2M Karras", "Euler a", "DDIM"],
                                value="DPM++ 2M Karras",
                                label="採樣器",
                            )

                    manual_submit_btn = gr.Button("提交手動批次任務", variant="primary")

        # Job status
        with gr.Group():
            gr.Markdown("#### 任務狀態")

            with gr.Row():
                job_id_input = gr.Textbox(
                    label="任務 ID", placeholder="輸入任務 ID 或選擇下方任務"
                )

                check_status_btn = gr.Button("🔍 查詢狀態", size="sm")
                refresh_jobs_btn = gr.Button("🔄 刷新列表", size="sm")

            active_jobs = gr.Dropdown(label="活躍任務", choices=[], interactive=True)

            job_status_display = gr.Textbox(
                label="任務詳情",
                lines=8,
                interactive=False,
                placeholder="選擇任務以查看狀態",
            )

            with gr.Row():
                cancel_job_btn = gr.Button("❌ 取消任務", size="sm")
                download_results_btn = gr.DownloadButton("📥 下載結果", visible=False)

        # Results gallery
        with gr.Group():
            gr.Markdown("#### 生成結果")

            results_gallery = gr.Gallery(
                label="批次生成結果", show_label=True, columns=3, rows=2, height="400px"
            )

    # State to store current job info
    current_job_state = gr.State({})

    def submit_csv_batch(csv_file_path):
        """Submit CSV batch job"""
        if not csv_file_path:
            return "", "請上傳 CSV 檔案", gr.update(), {}

        try:
            import pandas as pd

            df = pd.read_csv(csv_file_path)

            # Convert DataFrame to job format
            tasks = []
            for _, row in df.iterrows():
                task = {
                    "prompt": row.get("prompt", ""),
                    "negative": row.get("negative", "lowres, blurry"),
                    "width": int(row.get("width", 768)),
                    "height": int(row.get("height", 768)),
                    "steps": int(row.get("steps", 25)),
                    "cfg_scale": float(row.get("cfg_scale", 7.5)),
                    "seed": int(row.get("seed", -1)),
                }
                tasks.append(task)

            job_data = {"job_name": f"csv_batch_{len(tasks)}_tasks", "tasks": tasks}

            result = api_client.submit_batch_job(job_data)

            if "job_id" in result:
                job_id = result["job_id"]
                status_text = (
                    f"批次任務提交成功!\n任務 ID: {job_id}\n任務數量: {len(tasks)}"
                )

                return (
                    job_id,
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
            return "", f"處理 CSV 檔案失敗: {str(e)}", gr.update(), {}

    def submit_json_batch(json_file_path):
        """Submit JSON batch job"""
        if not json_file_path:
            return "", "請上傳 JSON 檔案", gr.update(), {}

        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                job_data = json.load(f)

            result = api_client.submit_batch_job(job_data)

            if "job_id" in result:
                job_id = result["job_id"]
                task_count = len(job_data.get("tasks", []))
                status_text = (
                    f"批次任務提交成功!\n任務 ID: {job_id}\n任務數量: {task_count}"
                )

                return (
                    job_id,
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
            return "", f"處理 JSON 檔案失敗: {str(e)}", gr.update(), {}

    def submit_manual_batch(prompts, negative, width, height, steps, cfg, sampler):
        """Submit manual batch job"""
        if not prompts.strip():
            return "", "請輸入至少一個提示詞", gr.update(), {}

        try:
            prompt_list = [p.strip() for p in prompts.split("\n") if p.strip()]

            tasks = []
            for prompt in prompt_list:
                task = {
                    "prompt": prompt,
                    "negative": negative,
                    "width": int(width),
                    "height": int(height),
                    "steps": int(steps),
                    "cfg_scale": float(cfg),
                    "sampler": sampler,
                    "seed": -1,
                }
                tasks.append(task)

            job_data = {"job_name": f"manual_batch_{len(tasks)}_tasks", "tasks": tasks}

            result = api_client.submit_batch_job(job_data)

            if "job_id" in result:
                job_id = result["job_id"]
                status_text = (
                    f"批次任務提交成功!\n任務 ID: {job_id}\n任務數量: {len(tasks)}"
                )

                return (
                    job_id,
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
            return "", f"提交手動批次任務失敗: {str(e)}", gr.update(), {}

    def check_job_status(job_id):
        """Check job status"""
        if not job_id:
            return "請輸入任務 ID", [], gr.update(visible=False), {}

        try:
            result = api_client.get_job_status(job_id)

            status = result.get("status", "unknown")
            progress = result.get("progress", {})

            status_text = f"任務 ID: {job_id}\n"
            status_text += f"狀態: {status}\n"

            if progress:
                completed = progress.get("completed", 0)
                total = progress.get("total", 0)
                failed = progress.get("failed", 0)

                status_text += f"進度: {completed}/{total} 完成"
                if failed > 0:
                    status_text += f", {failed} 失敗"

                if total > 0:
                    percentage = (completed / total) * 100
                    status_text += f" ({percentage:.1f}%)"

            status_text += f"\n開始時間: {result.get('started_at', 'N/A')}"

            if status == "completed":
                status_text += f"\n完成時間: {result.get('completed_at', 'N/A')}"

            # Get result images if completed
            images = []
            download_visible = False

            if status == "completed" and "results" in result:
                images = [
                    r["image_path"] for r in result["results"] if "image_path" in r
                ]
                download_visible = True

            return status_text, images, gr.update(visible=download_visible), result

        except Exception as e:
            return f"查詢狀態失敗: {str(e)}", [], gr.update(visible=False), {}

    # Connect events
    csv_submit_btn.click(
        submit_csv_batch,
        inputs=[csv_file],
        outputs=[
            job_id_input,
            job_status_display,
            download_results_btn,
            current_job_state,
        ],
    )

    json_submit_btn.click(
        submit_json_batch,
        inputs=[json_file],
        outputs=[
            job_id_input,
            job_status_display,
            download_results_btn,
            current_job_state,
        ],
    )

    manual_submit_btn.click(
        submit_manual_batch,
        inputs=[
            batch_prompts,
            batch_negative,
            batch_width,
            batch_height,
            batch_steps,
            batch_cfg,
            batch_sampler,
        ],
        outputs=[
            job_id_input,
            job_status_display,
            download_results_btn,
            current_job_state,
        ],
    )

    check_status_btn.click(
        check_job_status,
        inputs=[job_id_input],
        outputs=[
            job_status_display,
            results_gallery,
            download_results_btn,
            current_job_state,
        ],
    )

    active_jobs.change(lambda x: x, inputs=[active_jobs], outputs=[job_id_input])

    return {
        "job_status_display": job_status_display,
        "results_gallery": results_gallery,
        "check_status_btn": check_status_btn,
    }
