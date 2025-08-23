# backend/core/content_generator.py
import random
import re
from typing import List, Dict, Tuple
from backend.core.pipeline_loader import get_chat_pipeline
from backend.core.persona_manager import PersonaManager
from backend.models.game_session import GameSession
from backend.schemas.game import GameScene, Choice


class ContentGenerator:
    def __init__(self):
        self.chat_pipeline = None
        self.persona_manager = PersonaManager()

    def _get_chat_pipeline(self):
        """Lazy load chat pipeline"""
        if self.chat_pipeline is None:
            self.chat_pipeline = get_chat_pipeline()
        return self.chat_pipeline

    def generate_opening_scene(
        self, session: GameSession
    ) -> Tuple[GameScene, List[Choice]]:
        """Generate the opening scene for a new game"""
        persona = self.persona_manager.get_persona(session.persona)
        if not persona:
            raise ValueError(f"Unknown persona: {session.persona}")

        # Create opening scene based on setting and persona
        scene_prompts = {
            "modern": f"一個現代都市的咖啡館，{persona['display_name']}正在等待你的到來",
            "fantasy": f"一座古老的圖書館，{persona['display_name']}正在研讀古籍",
            "forest": f"一片神秘的森林空地，{persona['display_name']}正在觀察自然",
            "future": f"一個未來科技實驗室，{persona['display_name']}正在分析數據",
        }

        location = scene_prompts.get(session.setting, scene_prompts["modern"])

        scene = GameScene(
            scene_id="opening",
            title="初次相遇",
            description=f"你來到了{location}。空氣中彌漫著知識與智慧的氣息。",
            mood="welcoming",
            location=location,
        )

        choices = [
            Choice(id="greet", text="主動打招呼", consequence="友善的開始"),
            Choice(id="observe", text="先觀察周圍環境", consequence="謹慎的態度"),
            Choice(
                id="ask_question", text="直接詢問對方身份", consequence="直接的交流"
            ),
        ]

        return scene, choices

    def generate_next_scene(
        self, session: GameSession, player_action: str
    ) -> Tuple[GameScene, List[Choice], str]:
        """Generate next scene based on player action"""
        persona = self.persona_manager.get_persona(session.persona)
        if not persona:
            raise ValueError(f"Unknown persona: {session.persona}")

        # Build context for LLM
        context = self._build_scene_context(session, persona, player_action)

        # Generate content using LLM
        pipeline = self._get_chat_pipeline()

        prompt = f"""作為一個文字冒險遊戲的敘述者和角色扮演者，請根據以下情境生成下一個場景：

{context}

請用以下格式回應：
SCENE_TITLE: [場景標題]
SCENE_DESCRIPTION: [場景描述，最多200字]
SCENE_MOOD: [場景情緒：welcoming/tense/mysterious/exciting/peaceful]
CHARACTER_DIALOGUE: [角色對話，符合其個性]
CHOICE_1: [選擇1]|[後果提示1]
CHOICE_2: [選擇2]|[後果提示2]
CHOICE_3: [選擇3]|[後果提示3]

請確保內容適合所有年齡，避免暴力或不當內容。"""

        try:
            response = pipeline(
                prompt,
                max_length=800,
                temperature=0.7,
                do_sample=True,
                pad_token_id=pipeline.tokenizer.eos_token_id,
            )[0]["generated_text"]

            # Parse response
            scene, choices, dialogue = self._parse_generated_content(response, session)

            # Validate content safety
            if not self.persona_manager.validate_content(scene.description + dialogue):
                # Fallback to safe content
                return self._generate_fallback_scene(session, player_action)

            return scene, choices, dialogue

        except Exception as e:
            print(f"[ContentGenerator] LLM generation failed: {e}")
            return self._generate_fallback_scene(session, player_action)

    def _build_scene_context(
        self, session: GameSession, persona: Dict, player_action: str
    ) -> str:
        """Build context string for LLM"""
        context = f"""
角色資訊：
- 名稱：{persona['display_name']}
- 個性：{persona['personality']}
- 說話風格：{persona['speaking_style']}
- 專長領域：{', '.join(persona['knowledge_areas'])}
- 背景：{persona['background']}

遊戲設定：
- 世界背景：{session.setting}
- 難度：{session.difficulty.value}
- 玩家名稱：{session.player_name}

當前狀態：
{session.get_context_summary()}

玩家行動：{player_action}

請根據角色個性和當前情境，生成合適的場景和選擇。
"""
        return context

    def _parse_generated_content(
        self, response: str, session: GameSession
    ) -> Tuple[GameScene, List[Choice], str]:
        """Parse LLM generated content into structured format"""
        try:
            # Extract components using regex
            title_match = re.search(r"SCENE_TITLE:\s*(.+)", response)
            desc_match = re.search(r"SCENE_DESCRIPTION:\s*(.+)", response, re.DOTALL)
            mood_match = re.search(r"SCENE_MOOD:\s*(\w+)", response)
            dialogue_match = re.search(r"CHARACTER_DIALOGUE:\s*(.+)", response)

            # Extract choices
            choice_pattern = r"CHOICE_(\d+):\s*([^|]+)\|([^|\n]+)"
            choice_matches = re.findall(choice_pattern, response)

            # Build scene
            scene = GameScene(
                scene_id=f"scene_{session.turn_number + 1}",
                title=title_match.group(1).strip() if title_match else "場景",
                description=(
                    desc_match.group(1).strip() if desc_match else "你繼續你的冒險..."
                ),
                mood=mood_match.group(1).strip() if mood_match else "neutral",
                location=(
                    session.current_scene.location
                    if session.current_scene
                    else "未知地點"
                ),
            )

            # Build choices
            choices = []
            for i, (num, choice_text, consequence) in enumerate(
                choice_matches[:4]
            ):  # Max 4 choices
                choices.append(
                    Choice(
                        id=f"choice_{i+1}",
                        text=choice_text.strip(),
                        consequence=consequence.strip(),
                    )
                )

            # Ensure at least 2 choices
            if len(choices) < 2:
                choices = [
                    Choice(id="continue", text="繼續探索", consequence="推進劇情"),
                    Choice(id="ask", text="詢問更多資訊", consequence="獲得線索"),
                ]

            dialogue = dialogue_match.group(1).strip() if dialogue_match else "..."

            return scene, choices, dialogue

        except Exception as e:
            print(f"[ContentGenerator] Failed to parse generated content: {e}")
            return self._generate_fallback_scene(session, "continue")

    def _generate_fallback_scene(
        self, session: GameSession, player_action: str
    ) -> Tuple[GameScene, List[Choice], str]:
        """Generate safe fallback scene when LLM fails"""
        fallback_scenes = [
            {
                "title": "休息時刻",
                "description": "你們來到了一個安靜的地方，決定稍作休息並整理思緒。",
                "mood": "peaceful",
                "dialogue": "讓我們在這裡稍作停留，思考一下接下來的方向。",
            },
            {
                "title": "新的發現",
                "description": "在探索的過程中，你注意到了一些有趣的細節。",
                "mood": "curious",
                "dialogue": "這裡似乎有些特別的地方，讓我們仔細觀察一下。",
            },
            {
                "title": "深入思考",
                "description": "你們停下腳步，開始深入思考遇到的問題。",
                "mood": "contemplative",
                "dialogue": "有時候，靜下心來思考比匆忙行動更重要。",
            },
        ]

        selected = random.choice(fallback_scenes)

        scene = GameScene(
            scene_id=f"fallback_{session.turn_number + 1}",
            title=selected["title"],
            description=selected["description"],
            mood=selected["mood"],
            location=(
                session.current_scene.location
                if session.current_scene
                else "安全的地方"
            ),
        )

        choices = [
            Choice(id="continue", text="繼續前進", consequence="推進冒險"),
            Choice(id="rest", text="休息片刻", consequence="恢復精力"),
            Choice(id="think", text="深入思考", consequence="獲得洞察"),
        ]

        return scene, choices, selected["dialogue"]
+