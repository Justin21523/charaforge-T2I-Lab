# frontend/gradio_app/game_tab.py
import gradio as gr
import requests
import json
from typing import Dict, List, Tuple, Optional


class GameState:
    def __init__(self):
        self.session_id = None
        self.current_scene = None
        self.choices = []
        self.player_state = {}
        self.turn_number = 0


game_state = GameState()


def start_new_game(
    persona: str, setting: str, difficulty: str, player_name: str
) -> Tuple[str, str, str, str]:
    """Start a new game session"""
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/game/new",
            json={
                "persona": persona,
                "setting": setting,
                "difficulty": difficulty,
                "player_name": player_name or "冒險者",
            },
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()

        # Update global game state
        game_state.session_id = data["session_id"]
        game_state.current_scene = data["scene"]
        game_state.choices = data["choices"]
        game_state.player_state = data["player_state"]
        game_state.turn_number = data["turn_number"]

        # Format scene display
        scene_text = f"**{data['scene']['title']}**\n\n"
        scene_text += f"📍 {data['scene']['location']}\n"
        scene_text += f"🎭 情緒：{data['scene']['mood']}\n\n"
        scene_text += data["scene"]["description"]

        if data.get("narrator_message"):
            scene_text += f"\n\n💬 {data['narrator_message']}"

        # Format choices
        choices_text = "**你的選擇：**\n"
        for i, choice in enumerate(data["choices"], 1):
            choices_text += f"{i}. {choice['text']}"
            if choice.get("consequence"):
                choices_text += f" _{choice['consequence']}_"
            choices_text += "\n"

        # Format player state
        state_text = f"**玩家狀態** (回合 {data['turn_number']})\n"
        state_text += f"❤️ 健康：{data['player_state']['health']}/100\n"
        state_text += f"⚡ 精力：{data['player_state']['energy']}/100\n"
        state_text += f"😊 心情：{data['player_state']['mood']}\n"
        if data["player_state"]["inventory"]:
            state_text += f"🎒 物品：{', '.join(data['player_state']['inventory'])}\n"
        else:
            state_text += "🎒 物品：無\n"

        return (
            scene_text,
            choices_text,
            state_text,
            f"遊戲已開始！會話ID：{data['session_id'][:8]}...",
        )

    except requests.exceptions.RequestException as e:
        return "錯誤", "錯誤", "錯誤", f"API 錯誤：{str(e)}"
    except Exception as e:
        return "錯誤", "錯誤", "錯誤", f"錯誤：{str(e)}"


def take_action(
    action_text: str, custom_message: str = ""
) -> Tuple[str, str, str, str]:
    """Take an action in the game"""
    if not game_state.session_id:
        return "錯誤", "錯誤", "錯誤", "請先開始新遊戲"

    try:
        # Determine action ID from text
        action_id = "custom"
        if "1." in action_text or "第一個" in action_text:
            action_id = (
                game_state.choices[0]["id"] if game_state.choices else "choice_1"
            )
        elif "2." in action_text or "第二個" in action_text:
            action_id = (
                game_state.choices[1]["id"]
                if len(game_state.choices) > 1
                else "choice_2"
            )
        elif "3." in action_text or "第三個" in action_text:
            action_id = (
                game_state.choices[2]["id"]
                if len(game_state.choices) > 2
                else "choice_3"
            )
        else:
            action_id = action_text

        response = requests.post(
            "http://localhost:8000/api/v1/game/step",
            json={
                "session_id": game_state.session_id,
                "action": action_id,
                "message": custom_message if custom_message else None,
            },
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()

        # Update game state
        game_state.current_scene = data["scene"]
        game_state.choices = data["choices"]
        game_state.player_state = data["player_state"]
        game_state.turn_number = data["turn_number"]

        # Format response similar to start_new_game
        scene_text = f"**{data['scene']['title']}**\n\n"
        scene_text += f"📍 {data['scene']['location']}\n"
        scene_text += f"🎭 情緒：{data['scene']['mood']}\n\n"
        scene_text += data["scene"]["description"]

        if data.get("character_dialogue"):
            scene_text += f"\n\n💬 **角色回應：**\n{data['character_dialogue']}"

        # Format choices
        choices_text = "**你的選擇：**\n"
        for i, choice in enumerate(data["choices"], 1):
            choices_text += f"{i}. {choice['text']}"
            if choice.get("consequence"):
                choices_text += f" _{choice['consequence']}_"
            choices_text += "\n"

        # Format player state
        state_text = f"**玩家狀態** (回合 {data['turn_number']})\n"
        state_text += f"❤️ 健康：{data['player_state']['health']}/100\n"
        state_text += f"⚡ 精力：{data['player_state']['energy']}/100\n"
        state_text += f"😊 心情：{data['player_state']['mood']}\n"
        if data["player_state"]["inventory"]:
            state_text += f"🎒 物品：{', '.join(data['player_state']['inventory'])}\n"
        else:
            state_text += "🎒 物品：無\n"

        return scene_text, choices_text, state_text, f"行動已執行！"

    except requests.exceptions.RequestException as e:
        return "錯誤", "錯誤", "錯誤", f"API 錯誤：{str(e)}"
    except Exception as e:
        return "錯誤", "錯誤", "錯誤", f"錯誤：{str(e)}"


def save_game(save_name: str) -> str:
    """Save current game"""
    if not game_state.session_id:
        return "錯誤：沒有進行中的遊戲"

    if not save_name.strip():
        return "錯誤：請輸入存檔名稱"

    try:
        response = requests.post(
            "http://localhost:8000/api/v1/game/save",
            json={"session_id": game_state.session_id, "save_name": save_name.strip()},
            timeout=10,
        )
        response.raise_for_status()

        return f"✅ 遊戲已保存：{save_name}"

    except Exception as e:
        return f"❌ 保存失敗：{str(e)}"


def load_game(save_name: str) -> Tuple[str, str, str, str]:
    """Load saved game"""
    if not save_name:
        return "錯誤", "錯誤", "錯誤", "請選擇存檔"

    try:
        response = requests.post(
            "http://localhost:8000/api/v1/game/load",
            json={"save_name": save_name},
            timeout=10,
        )
        response.raise_for_status()

        data = response.json()

        # Update game state
        game_state.session_id = data["session_id"]
        game_state.current_scene = data["scene"]
        game_state.choices = data["choices"]
        game_state.player_state = data["player_state"]
        game_state.turn_number = data["turn_number"]

        # Format display (same as start_new_game)
        scene_text = f"**{data['scene']['title']}**\n\n"
        scene_text += f"📍 {data['scene']['location']}\n"
        scene_text += f"🎭 情緒：{data['scene']['mood']}\n\n"
        scene_text += data["scene"]["description"]

        choices_text = "**你的選擇：**\n"
        for i, choice in enumerate(data["choices"], 1):
            choices_text += f"{i}. {choice['text']}"
            if choice.get("consequence"):
                choices_text += f" _{choice['consequence']}_"
            choices_text += "\n"

        state_text = f"**玩家狀態** (回合 {data['turn_number']})\n"
        state_text += f"❤️ 健康：{data['player_state']['health']}/100\n"
        state_text += f"⚡ 精力：{data['player_state']['energy']}/100\n"
        state_text += f"😊 心情：{data['player_state']['mood']}\n"
        if data["player_state"]["inventory"]:
            state_text += f"🎒 物品：{', '.join(data['player_state']['inventory'])}\n"
        else:
            state_text += "🎒 物品：無\n"

        return scene_text, choices_text, state_text, f"✅ 遊戲已載入：{save_name}"

    except Exception as e:
        return "錯誤", "錯誤", "錯誤", f"載入失敗：{str(e)}"


def get_available_personas() -> Dict[str, str]:
    """Get available personas from API"""
    try:
        response = requests.get("http://localhost:8000/api/v1/game/personas")
        response.raise_for_status()
        return response.json()["personas"]
    except Exception as e:
        return {"wise_mentor": "智慧導師：博學、耐心、鼓勵性"}


def get_available_saves() -> List[str]:
    """Get available save files"""
    try:
        response = requests.get("http://localhost:8000/api/v1/game/saves")
        response.raise_for_status()
        return response.json()["saves"]
    except Exception as e:
        return []


def create_game_tab():
    """Create Gradio game interface"""
    with gr.Tab("🎮 文字冒險"):
        gr.Markdown("# 🎮 個性化文字冒險遊戲")
        gr.Markdown("選擇一個角色導師，開始你的冒險旅程！")

        with gr.Row():
            # Game Setup Column
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 開始新遊戲")

                personas = get_available_personas()
                persona_dropdown = gr.Dropdown(
                    choices=list(personas.keys()),
                    value=list(personas.keys())[0] if personas else None,
                    label="選擇角色導師",
                )

                # Display persona info
                persona_info = gr.Markdown(
                    value=list(personas.values())[0] if personas else "",
                    label="角色介紹",
                )

                setting_dropdown = gr.Dropdown(
                    choices=["modern", "fantasy", "forest", "future"],
                    value="modern",
                    label="遊戲背景",
                )

                difficulty_dropdown = gr.Dropdown(
                    choices=["easy", "normal", "hard"], value="normal", label="難度等級"
                )

                player_name_input = gr.Textbox(
                    label="角色名稱", placeholder="冒險者", value=""
                )

                start_game_btn = gr.Button("🚀 開始冒險", variant="primary")

                # Save/Load Section
                gr.Markdown("### 💾 存檔管理")

                save_name_input = gr.Textbox(label="存檔名稱", placeholder="我的冒險")

                with gr.Row():
                    save_btn = gr.Button("💾 保存")
                    load_saves_btn = gr.Button("🔄 刷新存檔")

                saves_dropdown = gr.Dropdown(
                    choices=get_available_saves(), label="選擇存檔"
                )

                load_btn = gr.Button("📂 載入遊戲")

            # Game Display Column
            with gr.Column(scale=2):
                gr.Markdown("### 🌟 冒險場景")

                scene_display = gr.Markdown(
                    value="點擊「開始冒險」來開始你的旅程...", label="當前場景"
                )

                choices_display = gr.Markdown(value="", label="可用選擇")

                # Action Input
                with gr.Row():
                    action_input = gr.Textbox(
                        label="你的行動",
                        placeholder="輸入選擇編號(如:1)或自定義行動",
                        scale=3,
                    )
                    action_btn = gr.Button("⚡ 執行", scale=1)

                custom_message = gr.Textbox(
                    label="額外訊息 (可選)",
                    placeholder="想要說的話或額外說明...",
                    lines=2,
                )

            # Status Column
            with gr.Column(scale=1):
                gr.Markdown("### 📊 狀態")

                player_status = gr.Markdown(value="尚未開始遊戲", label="玩家狀態")

                status_message = gr.Textbox(
                    label="系統訊息", value="準備開始冒險！", interactive=False
                )

        # Event Handlers
        def update_persona_info(persona_name):
            personas = get_available_personas()
            return personas.get(persona_name, "未知角色")

        persona_dropdown.change(
            fn=update_persona_info, inputs=[persona_dropdown], outputs=[persona_info]
        )

        start_game_btn.click(
            fn=start_new_game,
            inputs=[
                persona_dropdown,
                setting_dropdown,
                difficulty_dropdown,
                player_name_input,
            ],
            outputs=[scene_display, choices_display, player_status, status_message],
        )

        action_btn.click(
            fn=take_action,
            inputs=[action_input, custom_message],
            outputs=[scene_display, choices_display, player_status, status_message],
        )

        save_btn.click(fn=save_game, inputs=[save_name_input], outputs=[status_message])

        load_saves_btn.click(fn=get_available_saves, outputs=[saves_dropdown])

        load_btn.click(
            fn=load_game,
            inputs=[saves_dropdown],
            outputs=[scene_display, choices_display, player_status, status_message],
        )

        # Examples
        gr.Examples(
            examples=[
                ["1", "我想仔細觀察周圍的環境"],
                ["2", ""],
                ["詢問關於這個地方的歷史", ""],
                ["休息一下", "我覺得有點累了"],
            ],
            inputs=[action_input, custom_message],
        )
