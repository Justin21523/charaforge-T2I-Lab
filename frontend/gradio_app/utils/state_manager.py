# frontend/gradio_app/utils/state_manager.py
"""
Gradio State Management Utilities
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional


class GradioStateManager:
    """Manage persistent state for Gradio interface"""

    def __init__(self, app_name="sagaforge_gradio"):
        self.app_name = app_name
        self.state_dir = Path.home() / f".{app_name}"
        self.state_dir.mkdir(exist_ok=True)
        self.state_file = self.state_dir / "state.json"
        self._state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load state: {e}")
        return {}

    def _save_state(self):
        """Save state to file"""
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save state: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get state value"""
        return self._state.get(key, default)

    def set(self, key: str, value: Any):
        """Set state value"""
        self._state[key] = value
        self._save_state()

    def update(self, updates: Dict[str, Any]):
        """Update multiple state values"""
        self._state.update(updates)
        self._save_state()

    def delete(self, key: str):
        """Delete state key"""
        if key in self._state:
            del self._state[key]
            self._save_state()

    def clear(self):
        """Clear all state"""
        self._state.clear()
        self._save_state()

    def get_generation_defaults(self) -> Dict[str, Any]:
        """Get default generation parameters"""
        return self.get(
            "generation_defaults",
            {
                "prompt": "",
                "negative": "lowres, blurry, bad anatomy, extra fingers",
                "width": 768,
                "height": 768,
                "steps": 25,
                "cfg_scale": 7.5,
                "seed": -1,
                "sampler": "DPM++ 2M Karras",
                "batch_size": 1,
            },
        )

    def set_generation_defaults(self, params: Dict[str, Any]):
        """Set default generation parameters"""
        self.set("generation_defaults", params)

    def get_recent_prompts(self, limit: int = 10) -> list:
        """Get recent prompts list"""
        prompts = self.get("recent_prompts", [])
        return prompts[:limit]

    def add_recent_prompt(self, prompt: str):
        """Add prompt to recent prompts"""
        if not prompt.strip():
            return

        prompts = self.get("recent_prompts", [])

        # Remove if already exists
        if prompt in prompts:
            prompts.remove(prompt)

        # Add to beginning
        prompts.insert(0, prompt)

        # Keep only last 20
        prompts = prompts[:20]

        self.set("recent_prompts", prompts)

    def get_favorite_loras(self) -> list:
        """Get favorite LoRA list"""
        return self.get("favorite_loras", [])

    def add_favorite_lora(self, lora_id: str):
        """Add LoRA to favorites"""
        favorites = self.get("favorite_loras", [])
        if lora_id not in favorites:
            favorites.append(lora_id)
            self.set("favorite_loras", favorites)

    def remove_favorite_lora(self, lora_id: str):
        """Remove LoRA from favorites"""
        favorites = self.get("favorite_loras", [])
        if lora_id in favorites:
            favorites.remove(lora_id)
            self.set("favorite_loras", favorites)


# Global state manager instance
state_manager = GradioStateManager()
