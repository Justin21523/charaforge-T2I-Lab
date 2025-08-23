# backend/core/safety_filter.py
import re
from typing import List, Tuple, Optional
import yaml
import os


class SafetyFilter:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "configs/safety_rules.yaml"
        self.blocked_patterns = []
        self.warning_patterns = []
        self.replacement_map = {}
        self.load_rules()

    def load_rules(self):
        """Load safety rules from config"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    rules = yaml.safe_load(f)

                self.blocked_patterns = [
                    re.compile(pattern, re.IGNORECASE)
                    for pattern in rules.get("blocked_patterns", [])
                ]

                self.warning_patterns = [
                    re.compile(pattern, re.IGNORECASE)
                    for pattern in rules.get("warning_patterns", [])
                ]

                self.replacement_map = rules.get("replacements", {})

        except Exception as e:
            print(f"Failed to load safety rules: {e}")
            self._load_default_rules()

    def _load_default_rules(self):
        """Load minimal default safety rules"""
        self.blocked_patterns = [
            re.compile(r"\b(極端政治|仇恨言論|暴力内容)\b", re.IGNORECASE),
            re.compile(r"\b(hate speech|violence|extreme politics)\b", re.IGNORECASE),
        ]

        self.warning_patterns = [
            re.compile(r"\b(敏感話題|personal info|私人信息)\b", re.IGNORECASE),
        ]

        self.replacement_map = {
            "不當詞彙": "適當替代詞",
            "inappropriate": "appropriate alternative",
        }

    def filter_input(
        self, text: str, level: str = "moderate"
    ) -> Tuple[bool, str, List[str]]:
        """
        Filter user input for safety
        Returns: (is_safe, filtered_text, warnings)
        """
        warnings = []
        filtered_text = text

        # Check for blocked content
        for pattern in self.blocked_patterns:
            if pattern.search(text):
                return False, "", ["Content blocked due to safety policy"]

        # Check for warning content
        if level in ["strict", "moderate"]:
            for pattern in self.warning_patterns:
                if pattern.search(text):
                    warnings.append("Content may be sensitive")

        # Apply replacements
        for bad_word, replacement in self.replacement_map.items():
            filtered_text = re.sub(
                bad_word, replacement, filtered_text, flags=re.IGNORECASE
            )

        return True, filtered_text, warnings

    def filter_output(self, text: str, level: str = "moderate") -> Tuple[bool, str]:
        """
        Filter AI output for safety
        Returns: (is_safe, filtered_text)
        """
        filtered_text = text

        # Check for blocked content in output
        for pattern in self.blocked_patterns:
            if pattern.search(text):
                return (
                    False,
                    "I cannot provide that information due to safety guidelines.",
                )

        # Apply output replacements
        for bad_word, replacement in self.replacement_map.items():
            filtered_text = re.sub(
                bad_word, replacement, filtered_text, flags=re.IGNORECASE
            )

        return True, filtered_text
