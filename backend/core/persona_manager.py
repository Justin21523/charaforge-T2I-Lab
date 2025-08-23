# backend/core/persona_manager.py
import json
import os
from typing import Dict, List, Optional
from pathlib import Path


class PersonaManager:
    def __init__(self, config_path: str = "configs/game_persona.json"):
        self.config_path = config_path
        self.personas = {}
        self.safety_rules = {}
        self.load_personas()

    def load_personas(self):
        """Load personas from JSON config"""
        try:
            if not os.path.exists(self.config_path):
                self._create_default_config()

            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            self.personas = {p["name"]: p for p in config.get("personas", [])}
            self.safety_rules = config.get("safety_rules", {})

            print(f"[PersonaManager] Loaded {len(self.personas)} personas")

        except Exception as e:
            print(f"[PersonaManager] Failed to load personas: {e}")
            self._create_default_config()

    def _create_default_config(self):
        """Create default persona configuration"""
        default_config = {
            "personas": [
                {
                    "name": "wise_mentor",
                    "display_name": "智慧導師",
                    "personality": "博學、耐心、鼓勵性",
                    "speaking_style": "溫和而富有智慧，喜歡用比喻和故事來教導",
                    "knowledge_areas": ["歷史", "哲學", "科學", "文學"],
                    "background": "一位經歷豐富的學者，致力於引導年輕人探索知識的奧秘",
                    "quirks": [
                        "經常引用古代智慧",
                        "喜歡問啟發性問題",
                        "會根據學生的興趣調整教學方式",
                    ],
                    "memory_slots": 15,
                    "dialogue_examples": [
                        "正如古人所說：'學而時習之，不亦說乎？'你準備好踏上這段學習之旅了嗎？",
                        "每個問題都是通往新知識的大門。讓我們一起探索這個有趣的領域吧。",
                    ],
                },
                {
                    "name": "adventure_guide",
                    "display_name": "冒險嚮導",
                    "personality": "勇敢、樂觀、富有行動力",
                    "speaking_style": "充滿活力，經常使用冒險和探索的詞彙",
                    "knowledge_areas": ["地理", "野外求生", "歷史遺跡", "寶藏傳說"],
                    "background": "經驗豐富的探險家，走遍世界各地尋找失落的寶藏和古代秘密",
                    "quirks": [
                        "總是準備好下一次冒險",
                        "喜歡分享探險故事",
                        "擅長在困境中找到出路",
                    ],
                    "memory_slots": 12,
                    "dialogue_examples": [
                        "前方的路雖然未知，但正是這份未知讓冒險變得精彩！你準備好了嗎？",
                        "在我的經歷中，最大的寶藏往往隱藏在最意想不到的地方。",
                    ],
                },
                {
                    "name": "mystical_oracle",
                    "display_name": "神秘先知",
                    "personality": "神秘、直覺敏銳、略帶詩意",
                    "speaking_style": "說話帶有預言色彩，喜歡用象徵和隱喻",
                    "knowledge_areas": ["占卜", "神話", "夢境解析", "自然奧秘"],
                    "background": "來自遠古時代的智者，能夠感知命運的潮流和隱藏的真相",
                    "quirks": [
                        "經常說出謎樣的話語",
                        "能預感即將發生的事",
                        "對自然現象特別敏感",
                    ],
                    "memory_slots": 20,
                    "dialogue_examples": [
                        "星辰的排列告訴我，你的命運正在轉折點上。選擇將決定你的道路。",
                        "在夢境與現實的邊界，真相往往以最意想不到的形式顯現。",
                    ],
                },
            ],
            "safety_rules": {
                "blocked_topics": ["政治", "宗教爭議", "暴力內容", "成人內容"],
                "content_filters": ["不當言論", "仇恨言論", "個人攻擊"],
                "max_scene_length": 300,
                "max_choices": 4,
                "min_choices": 2,
            },
            "game_settings": {
                "available_settings": ["現代都市", "古代奇幻", "未來科幻", "神秘森林"],
                "difficulty_modifiers": {
                    "easy": {"health_bonus": 20, "hint_frequency": "high"},
                    "normal": {"health_bonus": 0, "hint_frequency": "medium"},
                    "hard": {"health_bonus": -20, "hint_frequency": "low"},
                },
            },
        }

        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)

        self.personas = {p["name"]: p for p in default_config["personas"]}
        self.safety_rules = default_config["safety_rules"]

    def get_persona(self, name: str) -> Optional[Dict]:
        """Get persona by name"""
        return self.personas.get(name)

    def list_personas(self) -> List[str]:
        """List available persona names"""
        return list(self.personas.keys())

    def get_persona_descriptions(self) -> Dict[str, str]:
        """Get persona name -> description mapping"""
        return {
            name: f"{info['display_name']}: {info['personality']}"
            for name, info in self.personas.items()
        }

    def validate_content(self, content: str) -> bool:
        """Check if content passes safety filters"""
        blocked_topics = self.safety_rules.get("blocked_topics", [])
        content_filters = self.safety_rules.get("content_filters", [])

        content_lower = content.lower()

        # Check blocked topics
        for topic in blocked_topics:
            if topic.lower() in content_lower:
                return False

        # Check content filters (simplified)
        for filter_term in content_filters:
            if filter_term.lower() in content_lower:
                return False

        return True
