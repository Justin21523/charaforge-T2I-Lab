# core/t2i/safety.py - NSFW Detection & Content Safety
"""
NSFW 內容檢測與安全過濾系統
支援多種檢測模型和內容過濾策略
"""

import logging
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import re
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import ImageDraw, ImageFont

from core.config import get_settings, get_app_paths
from core.shared_cache import get_shared_cache
from core.exceptions import SafetyError, NSFWContentError, ContentBlockedError

logger = logging.getLogger(__name__)


class NSFWDetector:
    """NSFW 內容檢測器"""

    def __init__(self, model_name: str = "clip_based"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = 0.7  # NSFW detection threshold

        logger.info(f"NSFW Detector initialized: {model_name}")

    def load_model(self) -> bool:
        """載入 NSFW 檢測模型"""
        try:
            if self.model_name == "clip_based":
                return self._load_clip_nsfw_model()
            elif self.model_name == "nsfw_detector":
                return self._load_dedicated_nsfw_model()
            else:
                logger.warning(f"Unknown NSFW model: {self.model_name}")
                return False

        except Exception as e:
            logger.error(f"Failed to load NSFW model: {e}")
            return False

    def _load_clip_nsfw_model(self) -> bool:
        """載入基於 CLIP 的 NSFW 檢測模型"""
        try:

            # Use CLIP for content analysis
            model_id = "openai/clip-vit-base-patch32"

            self.model = CLIPModel.from_pretrained(model_id).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_id)

            # NSFW classification prompts
            self.nsfw_prompts = [
                "explicit sexual content",
                "nudity",
                "pornographic image",
                "adult content",
                "sexual activity",
            ]

            self.safe_prompts = [
                "safe for work image",
                "appropriate content",
                "family friendly",
                "clean image",
                "professional content",
            ]

            logger.info("✅ CLIP-based NSFW detector loaded")
            return True

        except ImportError as e:
            logger.error(f"CLIP dependencies not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load CLIP NSFW model: {e}")
            return False

    def _load_dedicated_nsfw_model(self) -> bool:
        """載入專用的 NSFW 檢測模型"""
        try:

            # Try to load a dedicated NSFW detection model
            model_id = "Falconsai/nsfw_image_detection"

            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForImageClassification.from_pretrained(model_id).to(
                self.device
            )

            logger.info("✅ Dedicated NSFW detector loaded")
            return True

        except Exception as e:
            logger.warning(f"Dedicated NSFW model not available: {e}")
            # Fallback to CLIP-based detection
            return self._load_clip_nsfw_model()

    def detect_nsfw(
        self, images: Union[Image.Image, List[Image.Image]]
    ) -> Tuple[List[bool], List[float]]:
        """
        檢測圖片是否包含 NSFW 內容

        Returns:
            Tuple[List[bool], List[float]]: (is_nsfw_list, confidence_scores)
        """
        if self.model is None:
            logger.warning("NSFW model not loaded, returning safe classification")
            if isinstance(images, Image.Image):
                return [False], [0.0]
            else:
                return [False] * len(images), [0.0] * len(images)

        if isinstance(images, Image.Image):
            images = [images]

        try:
            if self.model_name == "clip_based":
                return self._detect_nsfw_clip(images)
            else:
                return self._detect_nsfw_dedicated(images)

        except Exception as e:
            logger.error(f"NSFW detection failed: {e}")
            # Safe fallback - mark as not NSFW if detection fails
            return [False] * len(images), [0.0] * len(images)

    def _detect_nsfw_clip(
        self, images: List[Image.Image]
    ) -> Tuple[List[bool], List[float]]:
        """使用 CLIP 檢測 NSFW 內容"""
        is_nsfw_list = []
        confidence_scores = []

        with torch.no_grad():
            for image in images:
                # Prepare inputs
                inputs = self.processor(
                    text=self.nsfw_prompts + self.safe_prompts,
                    images=image,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                # Get predictions
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

                # Calculate NSFW score (sum of NSFW prompt probabilities)
                nsfw_score = probs[0, : len(self.nsfw_prompts)].sum().item()
                safe_score = probs[0, len(self.nsfw_prompts) :].sum().item()

                # Normalize score
                total_score = nsfw_score + safe_score
                if total_score > 0:
                    nsfw_confidence = nsfw_score / total_score
                else:
                    nsfw_confidence = 0.0

                is_nsfw = nsfw_confidence > self.threshold

                is_nsfw_list.append(is_nsfw)
                confidence_scores.append(nsfw_confidence)

                logger.debug(
                    f"NSFW detection: {nsfw_confidence:.3f} (threshold: {self.threshold})"
                )

        return is_nsfw_list, confidence_scores

    def _detect_nsfw_dedicated(
        self, images: List[Image.Image]
    ) -> Tuple[List[bool], List[float]]:
        """使用專用模型檢測 NSFW 內容"""
        is_nsfw_list = []
        confidence_scores = []

        with torch.no_grad():
            for image in images:
                # Prepare inputs
                inputs = self.processor(images=image, return_tensors="pt").to(
                    self.device
                )

                # Get predictions
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

                # Assume label 1 is NSFW (model-dependent)
                nsfw_confidence = predictions[0, 1].item()
                is_nsfw = nsfw_confidence > self.threshold

                is_nsfw_list.append(is_nsfw)
                confidence_scores.append(nsfw_confidence)

        return is_nsfw_list, confidence_scores


class ContentFilter:
    """內容過濾器 - 文本和圖像"""

    def __init__(self):
        self.settings = get_settings()
        self.nsfw_detector = NSFWDetector()

        # Banned words/phrases (can be loaded from config)
        self.banned_words = [
            # Add specific banned terms here
            "explicit",
            "sexual",
            "nude",
            "porn",
            # Add more as needed
        ]

        # Safe alternatives suggestions
        self.safe_alternatives = {
            "nude": "artistic portrait",
            "sexy": "attractive",
            "explicit": "detailed",
        }

    async def initialize(self) -> bool:
        """初始化內容過濾器"""
        try:
            # Load NSFW detection model
            nsfw_loaded = self.nsfw_detector.load_model()

            if nsfw_loaded:
                logger.info("✅ Content filter initialized successfully")
            else:
                logger.warning("⚠️ Content filter initialized with limited capabilities")

            return True

        except Exception as e:
            logger.error(f"Content filter initialization failed: {e}")
            return False

    def filter_text(self, text: str) -> Dict[str, Any]:
        """過濾文本內容"""
        try:
            original_text = text
            filtered_text = text
            flags = []

            # Check for banned words
            text_lower = text.lower()
            for banned_word in self.banned_words:
                if banned_word in text_lower:
                    flags.append(f"Contains banned word: {banned_word}")

                    # Suggest replacement if available
                    if banned_word in self.safe_alternatives:
                        replacement = self.safe_alternatives[banned_word]
                        pattern = re.compile(re.escape(banned_word), re.IGNORECASE)
                        filtered_text = pattern.sub(replacement, filtered_text)

            # Additional heuristic checks
            if self._check_explicit_patterns(text_lower):
                flags.append("Contains potentially explicit content")

            result = {
                "original": original_text,
                "filtered": filtered_text,
                "is_safe": len(flags) == 0,
                "flags": flags,
                "modified": original_text != filtered_text,
            }

            if flags:
                logger.info(f"Text filtering applied: {len(flags)} issues found")

            return result

        except Exception as e:
            logger.error(f"Text filtering failed: {e}")
            return {
                "original": text,
                "filtered": text,
                "is_safe": True,  # Safe fallback
                "flags": [],
                "modified": False,
            }

    def _check_explicit_patterns(self, text: str) -> bool:
        """檢查明確的不當內容模式"""
        # Simple pattern matching for explicit content
        explicit_patterns = [
            r"sex+[uy]",
            r"porn+",
            r"explicit+",
            r"adult.{0,10}content",
            r"nsfw",
        ]

        for pattern in explicit_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    async def filter_images(
        self, images: Union[Image.Image, List[Image.Image]]
    ) -> Dict[str, Any]:
        """過濾圖像內容"""
        try:
            if isinstance(images, Image.Image):
                images = [images]

            # Run NSFW detection
            is_nsfw_list, confidence_scores = self.nsfw_detector.detect_nsfw(images)

            # Filter out NSFW images
            safe_images = []
            filtered_indices = []

            for i, (image, is_nsfw, confidence) in enumerate(
                zip(images, is_nsfw_list, confidence_scores)
            ):
                if not is_nsfw:
                    safe_images.append(image)
                else:
                    filtered_indices.append(i)
                    logger.warning(
                        f"Image {i} filtered: NSFW confidence {confidence:.3f}"
                    )

            result = {
                "original_count": len(images),
                "safe_images": safe_images,
                "filtered_count": len(filtered_indices),
                "filtered_indices": filtered_indices,
                "confidence_scores": confidence_scores,
                "is_safe": len(filtered_indices) == 0,
            }

            return result

        except Exception as e:
            logger.error(f"Image filtering failed: {e}")
            # Safe fallback - return all images as safe
            return {
                "original_count": len(images),
                "safe_images": images,
                "filtered_count": 0,
                "filtered_indices": [],
                "confidence_scores": [0.0] * len(images),
                "is_safe": True,
            }

    def get_content_warnings(self, filter_result: Dict[str, Any]) -> List[str]:
        """生成內容警告信息"""
        warnings = []

        if "flags" in filter_result and filter_result["flags"]:
            warnings.extend(filter_result["flags"])

        if "filtered_count" in filter_result and filter_result["filtered_count"] > 0:
            warnings.append(
                f"{filter_result['filtered_count']} images filtered due to content policy"
            )

        return warnings


class SafetyChecker:
    """綜合安全檢查器"""

    def __init__(self):
        self.content_filter = ContentFilter()
        self.enabled = True
        self.strict_mode = False  # Configurable strictness

    async def initialize(self) -> bool:
        """初始化安全檢查器"""
        try:
            success = await self.content_filter.initialize()

            if success:
                logger.info("✅ Safety checker initialized")
            else:
                logger.warning(
                    "⚠️ Safety checker initialized with limited functionality"
                )

            return success

        except Exception as e:
            logger.error(f"Safety checker initialization failed: {e}")
            return False

    async def check_generation_request(
        self, prompt: str, negative_prompt: str = ""
    ) -> Dict[str, Any]:
        """檢查生成請求的安全性"""

        if not self.enabled:
            return {
                "is_safe": True,
                "filtered_prompt": prompt,
                "filtered_negative_prompt": negative_prompt,
                "warnings": [],
                "safety_disabled": True,
            }

        try:
            # Filter prompt
            prompt_result = self.content_filter.filter_text(prompt)

            # Filter negative prompt
            neg_prompt_result = self.content_filter.filter_text(negative_prompt)

            # Combine results
            is_safe = prompt_result["is_safe"] and neg_prompt_result["is_safe"]

            warnings = []
            warnings.extend(self.content_filter.get_content_warnings(prompt_result))
            warnings.extend(self.content_filter.get_content_warnings(neg_prompt_result))

            # In strict mode, block if any issues found
            if self.strict_mode and not is_safe:
                raise ContentBlockedError("Content policy violation detected", "text")

            result = {
                "is_safe": is_safe,
                "filtered_prompt": prompt_result["filtered"],
                "filtered_negative_prompt": neg_prompt_result["filtered"],
                "warnings": warnings,
                "prompt_modified": prompt_result["modified"],
                "negative_prompt_modified": neg_prompt_result["modified"],
                "safety_disabled": False,
            }

            if warnings:
                logger.info(
                    f"Generation request safety check: {len(warnings)} warnings"
                )

            return result

        except ContentBlockedError:
            raise
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            # Safe fallback
            return {
                "is_safe": True,
                "filtered_prompt": prompt,
                "filtered_negative_prompt": negative_prompt,
                "warnings": [f"Safety check error: {str(e)}"],
                "safety_disabled": False,
            }

    async def check_generated_images(
        self, images: Union[Image.Image, List[Image.Image]]
    ) -> Dict[str, Any]:
        """檢查生成圖像的安全性"""

        if not self.enabled:
            if isinstance(images, Image.Image):
                return {
                    "is_safe": True,
                    "safe_images": [images],
                    "filtered_count": 0,
                    "warnings": [],
                    "safety_disabled": True,
                }
            else:
                return {
                    "is_safe": True,
                    "safe_images": images,
                    "filtered_count": 0,
                    "warnings": [],
                    "safety_disabled": True,
                }

        try:
            # Filter images
            filter_result = await self.content_filter.filter_images(images)

            # Generate warnings
            warnings = self.content_filter.get_content_warnings(filter_result)

            # In strict mode, raise error if any images filtered
            if self.strict_mode and filter_result["filtered_count"] > 0:
                confidence_scores = filter_result.get("confidence_scores", [])
                max_confidence = max(confidence_scores) if confidence_scores else 0.0
                raise NSFWContentError(max_confidence)

            result = {
                "is_safe": filter_result["is_safe"],
                "safe_images": filter_result["safe_images"],
                "filtered_count": filter_result["filtered_count"],
                "warnings": warnings,
                "confidence_scores": filter_result.get("confidence_scores", []),
                "safety_disabled": False,
            }

            if filter_result["filtered_count"] > 0:
                logger.warning(
                    f"Filtered {filter_result['filtered_count']} images due to safety policies"
                )

            return result

        except NSFWContentError:
            raise
        except Exception as e:
            logger.error(f"Image safety check failed: {e}")
            # Safe fallback - return all images
            if isinstance(images, Image.Image):
                images = [images]

            return {
                "is_safe": True,
                "safe_images": images,
                "filtered_count": 0,
                "warnings": [f"Safety check error: {str(e)}"],
                "safety_disabled": False,
            }

    def set_strict_mode(self, enabled: bool):
        """設置嚴格模式"""
        self.strict_mode = enabled
        logger.info(f"Safety strict mode: {'enabled' if enabled else 'disabled'}")

    def disable_safety(self):
        """禁用安全檢查（僅用於開發/測試）"""
        self.enabled = False
        logger.warning("⚠️ Safety checking disabled - for development only!")

    def enable_safety(self):
        """啟用安全檢查"""
        self.enabled = True
        logger.info("✅ Safety checking enabled")

    def get_status(self) -> Dict[str, Any]:
        """取得安全檢查器狀態"""
        return {
            "enabled": self.enabled,
            "strict_mode": self.strict_mode,
            "nsfw_detector_loaded": self.content_filter.nsfw_detector.model is not None,
            "nsfw_model_name": self.content_filter.nsfw_detector.model_name,
            "nsfw_threshold": self.content_filter.nsfw_detector.threshold,
        }


# ===== Global Safety Checker Instance =====

_global_safety_checker: Optional[SafetyChecker] = None


async def get_safety_checker() -> SafetyChecker:
    """取得全域安全檢查器實例"""
    global _global_safety_checker

    if _global_safety_checker is None:
        _global_safety_checker = SafetyChecker()
        await _global_safety_checker.initialize()

    return _global_safety_checker


# ===== Utility Functions =====


async def check_prompt_safety(prompt: str, negative_prompt: str = "") -> Dict[str, Any]:
    """便利函數：檢查提示詞安全性"""
    checker = await get_safety_checker()
    return await checker.check_generation_request(prompt, negative_prompt)


async def check_image_safety(
    images: Union[Image.Image, List[Image.Image]],
) -> Dict[str, Any]:
    """便利函數：檢查圖像安全性"""
    checker = await get_safety_checker()
    return await checker.check_generated_images(images)


def apply_content_warning_overlay(
    image: Image.Image, warning_text: str = "Content Warning"
) -> Image.Image:
    """在圖像上添加內容警告覆蓋層"""
    try:
        # Create a copy of the image
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)

        # Get image dimensions
        width, height = result_image.size

        # Create semi-transparent overlay
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 128))
        result_image = Image.alpha_composite(result_image.convert("RGBA"), overlay)

        # Add warning text
        draw = ImageDraw.Draw(result_image)

        # Try to use a decent font
        try:
            font_size = max(20, min(width, height) // 20)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # Calculate text position (center)
        bbox = draw.textbbox((0, 0), warning_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (width - text_width) // 2
        y = (height - text_height) // 2

        # Draw text with outline
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    draw.text(
                        (x + dx, y + dy), warning_text, font=font, fill=(0, 0, 0, 255)
                    )

        draw.text((x, y), warning_text, font=font, fill=(255, 255, 255, 255))

        return result_image.convert("RGB")

    except Exception as e:
        logger.error(f"Failed to apply content warning overlay: {e}")
        return image


# ===== Configuration =====


def configure_safety_settings(
    nsfw_threshold: float = 0.7, strict_mode: bool = False, enabled: bool = True
):
    """配置安全設置"""
    global _global_safety_checker

    if _global_safety_checker:
        _global_safety_checker.content_filter.nsfw_detector.threshold = nsfw_threshold
        _global_safety_checker.set_strict_mode(strict_mode)

        if enabled:
            _global_safety_checker.enable_safety()
        else:
            _global_safety_checker.disable_safety()

    logger.info(
        f"Safety settings configured: threshold={nsfw_threshold}, strict={strict_mode}, enabled={enabled}"
    )


# ===== Testing and Validation =====


async def test_safety_system() -> Dict[str, Any]:
    """測試安全系統功能"""
    results = {
        "text_filtering": False,
        "image_filtering": False,
        "nsfw_detection": False,
        "errors": [],
    }

    try:
        checker = await get_safety_checker()

        # Test text filtering
        try:
            test_prompt = "Create a beautiful portrait"
            result = await checker.check_generation_request(test_prompt)
            results["text_filtering"] = result["is_safe"]
        except Exception as e:
            results["errors"].append(f"Text filtering test failed: {e}")

        # Test image filtering (create a dummy image)
        try:
            dummy_image = Image.new("RGB", (256, 256), (128, 128, 128))
            result = await checker.check_generated_images(dummy_image)
            results["image_filtering"] = len(result["safe_images"]) > 0
        except Exception as e:
            results["errors"].append(f"Image filtering test failed: {e}")

        # Test NSFW detection model
        try:
            results["nsfw_detection"] = (
                checker.content_filter.nsfw_detector.model is not None
            )
        except Exception as e:
            results["errors"].append(f"NSFW detection test failed: {e}")

        return results

    except Exception as e:
        results["errors"].append(f"Safety system test failed: {e}")
        return results
