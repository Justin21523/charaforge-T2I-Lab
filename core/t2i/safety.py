# core/t2i/safety.py - NSFW detection and content filtering
from typing import List, Dict, Tuple, Any
import hashlib
from PIL import Image, ImageFilter
import numpy as np
import torch

try:
    from transformers import pipeline, CLIPProcessor, CLIPModel

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from core.config import get_cache_paths, get_model_path


class SafetyChecker:
    """NSFW detection and content safety checks"""

    def __init__(self):
        self.nsfw_model = None
        self.clip_model = None
        self.clip_processor = None
        self.cache_paths = get_cache_paths()

        # Blocked terms and concepts
        self.blocked_terms = {
            "explicit": ["nude", "naked", "nsfw", "explicit", "porn", "sex"],
            "violence": ["blood", "gore", "violence", "weapon", "gun", "knife"],
            "harmful": ["suicide", "self-harm", "drug", "illegal"],
        }

        # Safe alternative suggestions
        self.safe_alternatives = {
            "nude": "portrait",
            "naked": "casual clothing",
            "sexy": "elegant",
            "hot": "attractive",
        }

    def load_nsfw_detector(self):
        """Load NSFW detection model"""
        if self.nsfw_model is not None:
            return

        if not TRANSFORMERS_AVAILABLE:
            print("[Safety] Transformers not available, using basic filtering")
            return

        try:
            print("[Safety] Loading NSFW detection model")

            # Use a lightweight NSFW detection model
            self.nsfw_model = pipeline(
                "image-classification",
                model="Falconsai/nsfw_image_detection",
                device=0 if torch.cuda.is_available() else -1,
            )

            print("[Safety] NSFW detector loaded")

        except Exception as e:
            print(f"[Safety] Failed to load NSFW detector: {e}")
            self.nsfw_model = None

    def load_clip_model(self):
        """Load CLIP model for content analysis"""
        if self.clip_model is not None:
            return

        if not TRANSFORMERS_AVAILABLE:
            return

        try:
            print("[Safety] Loading CLIP model for content analysis")

            model_name = "openai/clip-vit-base-patch32"
            self.clip_model = CLIPModel.from_pretrained(model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)

            if torch.cuda.is_available():
                self.clip_model = self.clip_model.to("cuda")  # type: ignore

            print("[Safety] CLIP model loaded")

        except Exception as e:
            print(f"[Safety] Failed to load CLIP model: {e}")

    def check_nsfw(self, image: Image.Image) -> Tuple[bool, Dict[str, float]]:
        """Check if image contains NSFW content"""
        self.load_nsfw_detector()

        if self.nsfw_model is None:
            # Fallback to basic analysis
            return self._basic_nsfw_check(image)

        try:
            # Run NSFW detection
            results = self.nsfw_model(image)

            # Parse results
            scores = {}
            for result in results:
                scores[result["label"].lower()] = result["score"]

            # Determine if NSFW
            nsfw_score = scores.get("nsfw", scores.get("porn", 0))
            is_nsfw = nsfw_score > 0.5

            return is_nsfw, scores

        except Exception as e:
            print(f"[Safety] NSFW detection failed: {e}")
            return self._basic_nsfw_check(image)

    def _basic_nsfw_check(self, image: Image.Image) -> Tuple[bool, Dict[str, float]]:
        """Basic NSFW check using image properties"""
        # Simple heuristic based on image characteristics
        img_array = np.array(image)

        # Check for skin-tone dominance (very basic)
        skin_pixels = self._detect_skin_pixels(img_array)
        skin_ratio = skin_pixels / (image.width * image.height)

        # Very conservative threshold
        is_nsfw = skin_ratio > 0.7

        scores = {"safe": 1.0 - skin_ratio, "nsfw": skin_ratio}

        return is_nsfw, scores

    def _detect_skin_pixels(self, img_array: np.ndarray) -> int:
        """Detect skin-tone pixels (basic HSV-based detection)"""
        import cv2

        # Convert to HSV
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Create mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        return np.sum(mask > 0)  # type: ignore

    def analyze_content_safety(
        self, image: Image.Image, prompt: str = ""
    ) -> Dict[str, Any]:
        """Comprehensive content safety analysis"""
        results = {
            "is_safe": True,
            "confidence": 1.0,
            "issues": [],
            "nsfw_scores": {},
            "prompt_issues": [],
        }

        # Check image content
        is_nsfw, nsfw_scores = self.check_nsfw(image)
        results["nsfw_scores"] = nsfw_scores

        if is_nsfw:
            results["is_safe"] = False
            results["issues"].append("NSFW content detected")
            results["confidence"] = min(
                results["confidence"], 1.0 - nsfw_scores.get("nsfw", 0)
            )

        # Check prompt content
        if prompt:
            prompt_filtered, blocked_terms = self.filter_prompt(prompt)
            if blocked_terms:
                results["is_safe"] = False
                results["prompt_issues"] = blocked_terms
                results["issues"].append("Inappropriate terms in prompt")

        # CLIP-based content analysis
        if prompt and self.clip_model is not None:
            content_scores = self._analyze_content_with_clip(image, prompt)
            results["content_alignment"] = content_scores

        return results

    def _analyze_content_with_clip(
        self, image: Image.Image, prompt: str
    ) -> Dict[str, float]:
        """Analyze content alignment using CLIP"""
        self.load_clip_model()

        if self.clip_model is None:
            return {}

        try:
            # Test prompts for safety categories
            safety_prompts = [
                "safe and appropriate content",
                "violent or harmful content",
                "adult or inappropriate content",
                "child-safe content",
            ]

            # Process image and prompts
            inputs = self.clip_processor(
                text=safety_prompts, images=image, return_tensors="pt", padding=True  # type: ignore
            )

            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Get CLIP scores
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

            # Convert to scores
            scores = {
                "safe_content": probs[0][0].item(),
                "violent_content": probs[0][1].item(),
                "adult_content": probs[0][2].item(),
                "child_safe": probs[0][3].item(),
            }

            return scores

        except Exception as e:
            print(f"[Safety] CLIP analysis failed: {e}")
            return {}

    def filter_prompt(self, prompt: str) -> Tuple[str, List[str]]:
        """Filter inappropriate terms from prompt"""
        original_prompt = prompt
        filtered_prompt = prompt.lower()
        blocked_terms_found = []

        # Check for blocked terms
        for category, terms in self.blocked_terms.items():
            for term in terms:
                if term.lower() in filtered_prompt:
                    blocked_terms_found.append(f"{term} ({category})")

                    # Replace with safe alternative if available
                    if term in self.safe_alternatives:
                        filtered_prompt = filtered_prompt.replace(
                            term.lower(), self.safe_alternatives[term]
                        )
                    else:
                        # Remove the term
                        filtered_prompt = filtered_prompt.replace(term.lower(), "")

        # Clean up extra spaces
        filtered_prompt = " ".join(filtered_prompt.split())

        return filtered_prompt, blocked_terms_found

    def apply_safety_filter(
        self, image: Image.Image, blur_strength: float = 5.0
    ) -> Image.Image:
        """Apply safety filter (blur) to image"""
        return image.filter(ImageFilter.GaussianBlur(radius=blur_strength))

    def get_content_hash(self, image: Image.Image) -> str:
        """Generate content hash for image tracking"""
        # Convert image to bytes
        img_bytes = image.tobytes()

        # Generate hash
        content_hash = hashlib.sha256(img_bytes).hexdigest()

        return content_hash[:16]  # Truncate for storage efficiency

    def is_content_allowed(
        self, image: Image.Image, prompt: str = "", strict_mode: bool = False
    ) -> Dict[str, Any]:
        """Main safety check function"""
        analysis = self.analyze_content_safety(image, prompt)

        # Determine if content is allowed
        if strict_mode:
            # Stricter thresholds for public/commercial use
            nsfw_threshold = 0.3
            confidence_threshold = 0.8
        else:
            # Normal thresholds
            nsfw_threshold = 0.5
            confidence_threshold = 0.6

        # Check NSFW scores
        nsfw_score = analysis["nsfw_scores"].get("nsfw", 0)
        is_allowed = (
            nsfw_score < nsfw_threshold
            and analysis["confidence"] > confidence_threshold
        )

        # Override if explicit issues found
        if analysis["issues"] or analysis["prompt_issues"]:
            is_allowed = False

        return {
            "allowed": is_allowed,
            "reason": (
                "Content passed safety checks"
                if is_allowed
                else f"Safety violation: {', '.join(analysis['issues'])}"
            ),
            "analysis": analysis,
            "content_hash": self.get_content_hash(image),
        }
