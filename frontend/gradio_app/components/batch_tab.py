# frontend/gradio_app/components/batch_tab.py - Gradio Batch Processing Interface
import gradio as gr
import requests
import json
import pandas as pd
import time
import os
from typing import Optional, Tuple

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_PREFIX = os.getenv("API_PREFIX", "/api/v1")


def create_batch_tab():
    """Create batch processing Gradio interface"""

    with gr.Tab("批次處理 (Batch)"):
        gr.Markdown("## 批次任務提交與監控")

        with gr.Row():
            with gr.Column(scale=1):
                # Task submission
                gr.Markdown("### 提交批次任務")

                task_type = gr.Dropdown(
                    choices=["caption", "vqa"], label="任務類型", value="caption"
                )

                batch_name = gr.Textbox(
                    label="批次名稱 (可選)", placeholder="我的批次任務"
                )

                file_upload = gr.File(
                    label="上傳 CSV/JSON 檔案", file_types=[".csv", ".json"]
                )

                submit_btn = gr.Button("提交批次任務", variant="primary")

                # Example formats
                with gr.Accordion("檔案格式範例", open=False):
                    gr.Markdown(
                        """
                    **Caption 批次 (CSV 格式)**:
                    ```csv
                    image_path,max_length,num_beams
                    /path/to/image1.jpg,50,3
                    /path/to/image2.jpg,50,3
                    ```

                    **VQA 批次 (JSON 格式)**:
                    ```json
                    [
                      {"image_path": "/path/to/image1.jpg", "question": "What's in this image?"},
                      {"image_path": "/path/to/image2.jpg", "question": "Describe the scene"}
                    ]
                    ```
                    """
                    )

            with gr.Column(scale=1):
                # Task monitoring
                gr.Markdown("### 任務監控")

                task_id_input = gr.Textbox(
                    label="任務 ID", placeholder="輸入任務 ID 查詢狀態"
                )

                check_status_btn = gr.Button("查詢狀態")
                refresh_btn = gr.Button("刷新列表")

                # Status display
                status_json = gr.JSON(label="任務狀態")

                # Progress bar
                progress_bar = gr.HTML()

                # Task list
                gr.Markdown("### 活躍任務列表")
                task_list = gr.DataFrame(
                    headers=["Task ID", "Type", "Status", "Worker"], label="當前任務"
                )

        with gr.Row():
            # Results display
            gr.Markdown("### 批次結果")
            results_json = gr.JSON(label="批次結果", visible=False)
            download_btn = gr.File(label="下載結果檔案", visible=False)

        # Event handlers
        submit_btn.click(
            fn=submit_batch_job,
            inputs=[task_type, batch_name, file_upload],
            outputs=[status_json, task_id_input],
        )

        check_status_btn.click(
            fn=check_task_status,
            inputs=[task_id_input],
            outputs=[status_json, progress_bar, results_json, download_btn],
        )

        refresh_btn.click(fn=refresh_task_list, outputs=[task_list])

        # Auto-refresh every 10 seconds
        gr.Timer(10).tick(fn=refresh_task_list, outputs=[task_list])


def submit_batch_job(task_type: str, batch_name: str, file_upload) -> Tuple[dict, str]:
    """Submit a batch processing job"""
    try:
        if not file_upload:
            return {"error": "請選擇要上傳的檔案"}, ""

        # Prepare form data
        files = {"file": open(file_upload.name, "rb")}
        data = {
            "task_type": task_type,
            "batch_name": batch_name or f"{task_type}_batch",
        }

        # Submit to API
        response = requests.post(
            f"{API_BASE}{API_PREFIX}/batch/submit", files=files, data=data, timeout=30
        )

        files["file"].close()

        if response.status_code == 200:
            result = response.json()
            return result, result.get("task_id", "")
        else:
            return {"error": f"提交失敗: {response.text}"}, ""

    except Exception as e:
        return {"error": f"提交錯誤: {str(e)}"}, ""


def check_task_status(task_id: str) -> Tuple[dict, str, dict, Optional[str]]:
    """Check batch task status"""
    try:
        if not task_id:
            return {"error": "請輸入任務 ID"}, "", {}, None

        response = requests.get(
            f"{API_BASE}{API_PREFIX}/batch/status/{task_id}", timeout=10
        )

        if response.status_code == 200:
            status_data = response.json()

            # Generate progress bar HTML
            progress_html = ""
            if status_data.get("progress"):
                progress = status_data["progress"]
                current = progress.get("current", 0)
                total = progress.get("total", 1)
                percentage = (current / total) * 100 if total > 0 else 0

                progress_html = f"""
                <div style="width: 100%; background-color: #f0f0f0; border-radius: 5px;">
                    <div style="width: {percentage:.1f}%; background-color: #4CAF50; height: 20px; border-radius: 5px; text-align: center; line-height: 20px; color: white;">
                        {current}/{total} ({percentage:.1f}%)
                    </div>
                </div>
                <p>{progress.get('status', 'Processing...')}</p>
                """

            # Check if task completed and has results
            results = {}
            download_file = None
            if status_data.get("status") == "SUCCESS" and status_data.get("result"):
                results = status_data["result"]
                if results.get("result_file"):
                    download_file = results["result_file"]

            return status_data, progress_html, results, download_file
        else:
            return {"error": f"查詢失敗: {response.text}"}, "", {}, None

    except Exception as e:
        return {"error": f"查詢錯誤: {str(e)}"}, "", {}, None


def refresh_task_list() -> pd.DataFrame:
    """Refresh active task list"""
    try:
        response = requests.get(f"{API_BASE}{API_PREFIX}/batch/list", timeout=10)

        if response.status_code == 200:
            data = response.json()
            tasks = data.get("tasks", [])

            if not tasks:
                return pd.DataFrame(columns=["Task ID", "Type", "Status", "Worker"])

            # Convert to DataFrame
            df_data = []
            for task in tasks:
                df_data.append(
                    {
                        "Task ID": task["task_id"][:12] + "...",  # Truncate for display
                        "Type": (
                            task["name"].split(".")[-1]
                            if "." in task["name"]
                            else task["name"]
                        ),
                        "Status": task["status"],
                        "Worker": task.get("worker", "N/A"),
                    }
                )

            return pd.DataFrame(df_data)
        else:
            return pd.DataFrame(columns=["Task ID", "Type", "Status", "Worker"])

    except Exception as e:
        print(f"Error refreshing task list: {e}")
        return pd.DataFrame(columns=["Task ID", "Type", "Status", "Worker"])
