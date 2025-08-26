# frontend/gradio_app/components/lora_management.py
"""
Gradio LoRA Management Components
"""
import gradio as gr
import json


def create_lora_interface(api_client):
    """Create LoRA management interface"""

    with gr.Column():
        gr.Markdown("### 🎭 LoRA 模型管理")

        # Available LoRAs
        with gr.Group():
            gr.Markdown("#### 可用的 LoRA 模型")

            with gr.Row():
                refresh_btn = gr.Button("🔄 刷新列表", size="sm")

            available_loras = gr.Dropdown(
                label="選擇 LoRA 模型", choices=[], interactive=True
            )

            lora_info = gr.Textbox(
                label="模型資訊",
                lines=3,
                interactive=False,
                placeholder="選擇 LoRA 模型以查看詳細資訊",
            )

        # Load controls
        with gr.Group():
            gr.Markdown("#### 載入設定")

            with gr.Row():
                lora_weight = gr.Slider(
                    minimum=0.0, maximum=2.0, step=0.1, label="LoRA 權重", value=1.0
                )

                load_btn = gr.Button("📥 載入 LoRA", variant="primary")

        # Loaded LoRAs
        with gr.Group():
            gr.Markdown("#### 已載入的 LoRA")

            loaded_loras_display = gr.Textbox(
                label="已載入列表",
                lines=5,
                interactive=False,
                placeholder="尚未載入任何 LoRA 模型",
            )

            with gr.Row():
                unload_selected_btn = gr.Button("📤 卸載選中", size="sm")
                unload_all_btn = gr.Button("🗑️ 卸載全部", size="sm")

        # Selected LoRA for unloading
        selected_lora_to_unload = gr.Dropdown(
            label="選擇要卸載的 LoRA", choices=[], visible=False
        )

    # State to store loaded LoRAs
    loaded_loras_state = gr.State({})

    def refresh_lora_list():
        """Refresh available LoRA list"""
        try:
            loras = api_client.list_loras()
            if loras:
                choices = [
                    (
                        f"{lora.get('name', 'Unknown')} ({lora.get('id', 'No ID')})",
                        lora.get("id", ""),
                    )
                    for lora in loras
                ]
                return gr.update(choices=choices), "LoRA 列表已更新"
            else:
                return gr.update(choices=[]), "沒有找到 LoRA 模型"
        except Exception as e:
            return gr.update(choices=[]), f"刷新失敗: {str(e)}"

    def show_lora_info(selected_lora_id):
        """Show selected LoRA information"""
        if not selected_lora_id:
            return "選擇 LoRA 模型以查看詳細資訊"

        try:
            loras = api_client.list_loras()
            for lora in loras:
                if lora.get("id") == selected_lora_id:
                    info = f"名稱: {lora.get('name', 'Unknown')}\n"
                    info += f"ID: {lora.get('id', 'No ID')}\n"
                    info += f"描述: {lora.get('description', '無描述')}\n"
                    info += f"類型: {lora.get('type', 'character')}\n"
                    info += f"解析度: {lora.get('resolution', 'Unknown')}\n"
                    info += f"Rank: {lora.get('rank', 'Unknown')}"
                    return info
        except Exception as e:
            return f"載入 LoRA 資訊失敗: {str(e)}"

        return "找不到 LoRA 資訊"

    def load_lora(selected_lora_id, weight, loaded_loras):
        """Load selected LoRA"""
        if not selected_lora_id:
            return loaded_loras, "請選擇要載入的 LoRA 模型", gr.update(), ""

        if selected_lora_id in loaded_loras:
            return loaded_loras, f"LoRA '{selected_lora_id}' 已經載入", gr.update(), ""

        try:
            result = api_client.load_lora(selected_lora_id, weight)

            if result.get("status") == "success":
                # Update loaded LoRAs
                loaded_loras[selected_lora_id] = weight

                # Update display
                display_text = "\n".join(
                    [
                        f"{lora_id} (權重: {w:.1f})"
                        for lora_id, w in loaded_loras.items()
                    ]
                )

                # Update unload dropdown
                unload_choices = [(lora_id, lora_id) for lora_id in loaded_loras.keys()]

                return (
                    loaded_loras,
                    f"LoRA '{selected_lora_id}' 載入成功",
                    gr.update(choices=unload_choices, visible=len(unload_choices) > 0),
                    display_text,
                )
            else:
                return (
                    loaded_loras,
                    f"載入失敗: {result.get('message', 'Unknown error')}",
                    gr.update(),
                    "",
                )

        except Exception as e:
            return loaded_loras, f"載入 LoRA 時發生錯誤: {str(e)}", gr.update(), ""

    def unload_lora(lora_id, loaded_loras):
        """Unload specific LoRA"""
        if not lora_id or lora_id not in loaded_loras:
            return loaded_loras, "請選擇要卸載的 LoRA", gr.update(), ""

        try:
            result = api_client.unload_lora(lora_id)

            if result.get("status") == "success":
                # Remove from loaded LoRAs
                del loaded_loras[lora_id]

                # Update display
                if loaded_loras:
                    display_text = "\n".join(
                        [
                            f"{lora_id} (權重: {w:.1f})"
                            for lora_id, w in loaded_loras.items()
                        ]
                    )
                else:
                    display_text = "尚未載入任何 LoRA 模型"

                # Update unload dropdown
                unload_choices = [(lora_id, lora_id) for lora_id in loaded_loras.keys()]

                return (
                    loaded_loras,
                    f"LoRA '{lora_id}' 卸載成功",
                    gr.update(choices=unload_choices, visible=len(unload_choices) > 0),
                    display_text,
                )
            else:
                return (
                    loaded_loras,
                    f"卸載失敗: {result.get('message', 'Unknown error')}",
                    gr.update(),
                    "",
                )

        except Exception as e:
            return loaded_loras, f"卸載 LoRA 時發生錯誤: {str(e)}", gr.update(), ""

    def unload_all_loras(loaded_loras):
        """Unload all loaded LoRAs"""
        if not loaded_loras:
            return (
                {},
                "沒有已載入的 LoRA",
                gr.update(visible=False),
                "尚未載入任何 LoRA 模型",
            )

        failed_loras = []
        success_count = 0

        for lora_id in list(loaded_loras.keys()):
            try:
                result = api_client.unload_lora(lora_id)
                if result.get("status") == "success":
                    success_count += 1
                else:
                    failed_loras.append(lora_id)
            except Exception:
                failed_loras.append(lora_id)

        # Clear successful unloads
        for lora_id in list(loaded_loras.keys()):
            if lora_id not in failed_loras:
                del loaded_loras[lora_id]

        if failed_loras:
            message = f"已卸載 {success_count} 個 LoRA，失敗: {', '.join(failed_loras)}"
            display_text = (
                "\n".join(
                    [
                        f"{lora_id} (權重: {w:.1f})"
                        for lora_id, w in loaded_loras.items()
                    ]
                )
                if loaded_loras
                else "尚未載入任何 LoRA 模型"
            )
        else:
            message = f"所有 LoRA 已卸載 ({success_count} 個)"
            display_text = "尚未載入任何 LoRA 模型"

        unload_choices = [(lora_id, lora_id) for lora_id in loaded_loras.keys()]

        return (
            loaded_loras,
            message,
            gr.update(choices=unload_choices, visible=len(unload_choices) > 0),
            display_text,
        )

    # Connect events
    refresh_btn.click(refresh_lora_list, outputs=[available_loras, lora_info])

    available_loras.change(
        show_lora_info, inputs=[available_loras], outputs=[lora_info]
    )

    load_btn.click(
        load_lora,
        inputs=[available_loras, lora_weight, loaded_loras_state],
        outputs=[
            loaded_loras_state,
            lora_info,
            selected_lora_to_unload,
            loaded_loras_display,
        ],
    )

    unload_selected_btn.click(
        lambda: gr.update(visible=True), outputs=[selected_lora_to_unload]
    )

    selected_lora_to_unload.change(
        unload_lora,
        inputs=[selected_lora_to_unload, loaded_loras_state],
        outputs=[
            loaded_loras_state,
            lora_info,
            selected_lora_to_unload,
            loaded_loras_display,
        ],
    )

    unload_all_btn.click(
        unload_all_loras,
        inputs=[loaded_loras_state],
        outputs=[
            loaded_loras_state,
            lora_info,
            selected_lora_to_unload,
            loaded_loras_display,
        ],
    )

    # Initialize
    refresh_btn.click()

    return {
        "available_loras": available_loras,
        "loaded_loras_display": loaded_loras_display,
        "load_btn": load_btn,
    }
