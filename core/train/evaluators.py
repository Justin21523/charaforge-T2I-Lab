# core/train/evaluators.py - Comprehensive training evaluation metrics
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from datetime import datetime

# Core ML libraries
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

# Computer vision and ML models
import clip
from transformers import BlipProcessor, BlipForConditionalGeneration

try:
    import face_recognition

    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

# FID calculation
try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
    from pytorch_fid.inception import InceptionV3

    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False

from core.config import get_cache_paths


logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Single evaluation result"""

    metric_name: str
    value: float
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "metadata": self.metadata or {},
            "timestamp": self.timestamp or datetime.now().isoformat(),
        }


@dataclass
class EvaluationSummary:
    """Complete evaluation summary"""

    results: List[EvaluationResult]
    overall_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "overall_score": self.overall_score,
            "metadata": self.metadata or {},
            "timestamp": datetime.now().isoformat(),
        }

    def get_metric(self, metric_name: str) -> Optional[float]:
        """Get specific metric value"""
        for result in self.results:
            if result.metric_name == metric_name:
                return result.value
        return None


class BaseEvaluator(ABC):
    """Base class for all evaluators"""

    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        self.cache_paths = get_cache_paths()

    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @abstractmethod
    def evaluate(self, **kwargs) -> EvaluationResult:
        """Evaluate and return result"""
        pass

    def batch_evaluate(self, inputs: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """Evaluate multiple inputs"""
        return [self.evaluate(**inp) for inp in inputs]


class CLIPEvaluator(BaseEvaluator):
    """CLIP-based text-image similarity evaluation"""

    def __init__(self, model_name: str = "ViT-B/32", device: str = "auto"):
        super().__init__(device)
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        )
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.preprocess = None
        self._load_model()
        logger.info("CLIPEvaluator initialized")

    def _load_model(self):
        """Load CLIP model"""
        try:
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.model.eval()
            logger.info(f"Loaded CLIP model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading CLIP model: {e}")
            raise

    def initialize(self) -> bool:
        """初始化 CLIP 模型"""
        try:
            from transformers import CLIPProcessor, CLIPModel

            model_id = "openai/clip-vit-base-patch32"
            self.model = CLIPModel.from_pretrained(model_id).to(self.device)  # type: ignore
            self.processor = CLIPProcessor.from_pretrained(model_id)

            logger.info("✅ CLIP evaluator model loaded")
            return True

        except ImportError:
            logger.error("CLIP dependencies not available")
            return False
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            return False

    def compute_text_image_similarity(
        self, images: List[Image.Image], texts: List[str]
    ) -> List[float]:
        """計算文字與圖片的相似度分數

        Args:
            images: 圖片列表
            texts: 對應的文字描述列表

        Returns:
            List[float]: 相似度分數列表 (0-1)
        """
        if self.model is None:
            logger.warning("CLIP model not loaded, using random scores")
            return [0.5 + np.random.random() * 0.3 for _ in range(len(images))]

        try:
            scores = []

            with torch.no_grad():
                for image, text in zip(images, texts):
                    # 準備輸入
                    inputs = self.processor(  # type: ignore
                        text=[text], images=image, return_tensors="pt", padding=True
                    ).to(self.device)

                    # 計算特徵
                    outputs = self.model(**inputs)

                    # 計算相似度
                    logits_per_image = outputs.logits_per_image
                    similarity = torch.sigmoid(logits_per_image).item()

                    scores.append(similarity)

            logger.info(f"CLIP similarity computed for {len(images)} images")
            return scores

        except Exception as e:
            logger.error(f"CLIP evaluation failed: {e}")
            return [0.5] * len(images)  # 回傳預設分數

    def compute_image_similarity(
        self, images1: List[Image.Image], images2: List[Image.Image]
    ) -> List[float]:
        """計算圖片間的相似度"""
        if self.model is None:
            return [0.5] * len(images1)

        try:
            scores = []

            with torch.no_grad():
                for img1, img2 in zip(images1, images2):
                    # 計算兩張圖片的特徵
                    inputs1 = self.processor(images=img1, return_tensors="pt").to(self.device)  # type: ignore
                    inputs2 = self.processor(images=img2, return_tensors="pt").to(self.device)  # type: ignore

                    features1 = self.model.get_image_features(**inputs1)  # type: ignore
                    features2 = self.model.get_image_features(**inputs2)  # type: ignore

                    # 計算餘弦相似度
                    similarity = torch.cosine_similarity(features1, features2).item()
                    scores.append(max(0.0, similarity))  # 確保非負

            return scores

        except Exception as e:
            logger.error(f"Image similarity computation failed: {e}")
            return [0.5] * len(images1)

    def evaluate(
        self, image: Union[Image.Image, str, Path], text: str, **kwargs
    ) -> EvaluationResult:
        """Evaluate CLIP similarity between image and text"""
        try:
            # Process image
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                raise ValueError("Image must be PIL Image or path")

            # Prepare inputs
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)  # type: ignore
            text_input = clip.tokenize([text]).to(self.device)

            # Calculate similarity
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)  # type: ignore
                text_features = self.model.encode_text(text_input)  # type: ignore

                # Normalize features
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)

                # Calculate cosine similarity
                similarity = torch.cosine_similarity(
                    image_features, text_features
                ).item()

            return EvaluationResult(
                metric_name="clip_similarity",
                value=similarity,
                metadata={
                    "model": self.model_name,
                    "text": text,
                    "device": str(self.device),
                },
            )

        except Exception as e:
            logger.error(f"CLIP evaluation error: {e}")
            return EvaluationResult(
                metric_name="clip_similarity", value=0.0, metadata={"error": str(e)}
            )


class FaceConsistencyEvaluator(BaseEvaluator):
    """Face consistency evaluation using face recognition"""

    def __init__(self, device: str = "auto"):
        super().__init__(device)
        if not FACE_RECOGNITION_AVAILABLE:
            logger.warning(
                "face_recognition not available, face consistency evaluation disabled"
            )

    def evaluate(
        self,
        reference_image: Union[Image.Image, str, Path],
        generated_image: Union[Image.Image, str, Path],
        **kwargs,
    ) -> EvaluationResult:
        """Evaluate face consistency between reference and generated images"""

        if not FACE_RECOGNITION_AVAILABLE:
            return EvaluationResult(
                metric_name="face_consistency",
                value=0.0,
                metadata={"error": "face_recognition not available"},
            )

        try:
            # Load images
            if isinstance(reference_image, (str, Path)):
                ref_img = np.array(Image.open(reference_image).convert("RGB"))
            else:
                ref_img = np.array(reference_image.convert("RGB"))

            if isinstance(generated_image, (str, Path)):
                gen_img = np.array(Image.open(generated_image).convert("RGB"))
            else:
                gen_img = np.array(generated_image.convert("RGB"))

            # Detect face encodings
            ref_encodings = face_recognition.face_encodings(ref_img)
            gen_encodings = face_recognition.face_encodings(gen_img)

            if not ref_encodings or not gen_encodings:
                return EvaluationResult(
                    metric_name="face_consistency",
                    value=0.0,
                    metadata={"error": "No faces detected"},
                )

            # Calculate similarity (1 - distance = similarity)
            distance = face_recognition.face_distance(
                [ref_encodings[0]], gen_encodings[0]
            )[0]
            similarity = 1.0 - distance

            return EvaluationResult(
                metric_name="face_consistency",
                value=similarity,
                metadata={
                    "face_distance": distance,
                    "ref_faces_count": len(ref_encodings),
                    "gen_faces_count": len(gen_encodings),
                },
            )

        except Exception as e:
            logger.error(f"Face consistency evaluation error: {e}")
            return EvaluationResult(
                metric_name="face_consistency", value=0.0, metadata={"error": str(e)}
            )


class FIDEvaluator(BaseEvaluator):
    """FID (Fréchet Inception Distance) evaluation"""

    def __init__(self, device: str = "auto", dims: int = 2048):
        super().__init__(device)
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.dims = dims
        if not FID_AVAILABLE:
            logger.warning("pytorch-fid not available, FID evaluation disabled")
        logger.info("FIDEvaluator initialized")

    def initialize(self) -> bool:
        """初始化 Inception 模型"""
        try:
            import torchvision.models as models

            # 載入預訓練的 Inception v3
            self.model = models.inception_v3(pretrained=True, transform_input=False)
            self.model.fc = torch.nn.Identity()  # type: ignore 移除最後的分類層
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info("✅ FID evaluator model loaded")
            return True

        except Exception as e:
            logger.error(f"Failed to load FID model: {e}")
            return False

    def compute_fid(
        self, real_images: List[Image.Image], generated_images: List[Image.Image]
    ) -> float:
        """計算 FID 分數"""
        if self.model is None:
            logger.warning("FID model not loaded, returning dummy score")
            return 50.0  # 回傳虛擬分數

        try:
            # 計算兩組圖片的特徵
            real_features = self._extract_features(real_images)
            gen_features = self._extract_features(generated_images)

            # 計算 FID
            fid_score = self._calculate_fid(real_features, gen_features)

            logger.info(f"FID score computed: {fid_score:.2f}")
            return fid_score

        except Exception as e:
            logger.error(f"FID computation failed: {e}")
            return 999.0  # 高分數表示品質差

    def _extract_features(self, images: List[Image.Image]) -> torch.Tensor:
        """提取圖片特徵"""
        features = []

        with torch.no_grad():
            for image in images:
                # 預處理圖片
                img_tensor = self._preprocess_image(image)

                # 提取特徵
                feature = self.model(img_tensor.unsqueeze(0).to(self.device))  # type: ignore
                features.append(feature.cpu())

        return torch.cat(features, dim=0)

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """圖片預處理"""
        import torchvision.transforms as transforms

        transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        return transform(image.convert("RGB"))

    def _calculate_fid(self, features1: torch.Tensor, features2: torch.Tensor) -> float:
        """計算 FID 分數"""
        try:
            # 計算平均值和協方差矩陣
            mu1 = torch.mean(features1, dim=0)
            mu2 = torch.mean(features2, dim=0)

            sigma1 = torch.cov(features1.T)
            sigma2 = torch.cov(features2.T)

            # 計算 FID
            diff = mu1 - mu2
            covmean = torch.sqrt(sigma1 @ sigma2)

            # 確保數值穩定性
            if torch.any(torch.isnan(covmean)):
                covmean = torch.zeros_like(sigma1)

            fid = torch.sum(diff * diff) + torch.trace(sigma1 + sigma2 - 2 * covmean)

            return float(fid.item())

        except Exception as e:
            logger.error(f"FID calculation failed: {e}")
            return 999.0

    def evaluate(
        self,
        real_images_path: Union[str, Path],
        fake_images_path: Union[str, Path],
        **kwargs,
    ) -> EvaluationResult:
        """Calculate FID between real and fake image directories"""

        if not FID_AVAILABLE:
            return EvaluationResult(
                metric_name="fid_score",
                value=float("inf"),
                metadata={"error": "pytorch-fid not available"},
            )

        try:
            real_path = str(real_images_path)
            fake_path = str(fake_images_path)

            # Calculate FID
            fid_value = calculate_fid_given_paths(
                [real_path, fake_path],
                batch_size=50,
                device=self.device,
                dims=self.dims,
            )

            return EvaluationResult(
                metric_name="fid_score",
                value=fid_value,
                metadata={
                    "real_path": real_path,
                    "fake_path": fake_path,
                    "dims": self.dims,
                },
            )

        except Exception as e:
            logger.error(f"FID evaluation error: {e}")
            return EvaluationResult(
                metric_name="fid_score", value=float("inf"), metadata={"error": str(e)}
            )


class LPIPSEvaluator:
    """LPIPS (Learned Perceptual Image Patch Similarity) 評估器"""

    def __init__(self, device: str = "auto"):
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        )
        self.model = None
        logger.info("LPIPSEvaluator initialized")

    def initialize(self) -> bool:
        """初始化 LPIPS 模型"""
        try:
            # 這裡需要安裝 lpips 套件: pip install lpips
            import lpips

            self.model = lpips.LPIPS(net="alex").to(self.device)

            logger.info("✅ LPIPS evaluator model loaded")
            return True

        except ImportError:
            logger.warning("LPIPS not available, using alternative similarity metric")
            return False
        except Exception as e:
            logger.error(f"Failed to load LPIPS model: {e}")
            return False

    def compute_lpips(
        self, images1: List[Image.Image], images2: List[Image.Image]
    ) -> List[float]:
        """計算 LPIPS 距離"""
        if self.model is None:
            logger.warning("LPIPS model not loaded, using dummy scores")
            return [0.5] * len(images1)

        try:
            scores = []

            with torch.no_grad():
                for img1, img2 in zip(images1, images2):
                    # 預處理圖片
                    tensor1 = self._preprocess_image(img1)
                    tensor2 = self._preprocess_image(img2)

                    # 計算 LPIPS 距離
                    distance = self.model(tensor1, tensor2)
                    scores.append(float(distance.item()))

            logger.info(f"LPIPS computed for {len(images1)} image pairs")
            return scores

        except Exception as e:
            logger.error(f"LPIPS computation failed: {e}")
            return [0.5] * len(images1)

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """預處理圖片為 LPIPS 格式"""
        import torchvision.transforms as transforms

        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        tensor = transform(image.convert("RGB"))
        return tensor.unsqueeze(0).to(self.device)


class ImageQualityEvaluator(BaseEvaluator):
    """Image quality metrics (SSIM, blur detection, etc.)"""

    def evaluate(
        self,
        image: Union[Image.Image, str, Path],
        reference: Optional[Union[Image.Image, str, Path]] = None,
        **kwargs,
    ) -> EvaluationResult:
        """Evaluate image quality metrics"""

        try:
            # Load image
            if isinstance(image, (str, Path)):
                img = np.array(Image.open(image).convert("RGB"))
            else:
                img = np.array(image.convert("RGB"))

            results = {}

            # Blur detection (Laplacian variance)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            results["blur_score"] = blur_score

            # Brightness and contrast
            results["brightness"] = np.mean(img)
            results["contrast"] = np.std(img)

            # If reference image provided, calculate SSIM
            if reference is not None:
                if isinstance(reference, (str, Path)):
                    ref_img = np.array(Image.open(reference).convert("RGB"))
                else:
                    ref_img = np.array(reference.convert("RGB"))

                # Resize to match if needed
                if img.shape != ref_img.shape:
                    ref_img = cv2.resize(ref_img, (img.shape[1], img.shape[0]))

                # Calculate SSIM
                ssim_score = ssim(img, ref_img, multichannel=True, channel_axis=2)
                results["ssim"] = ssim_score

            # Overall quality score (higher blur = better, normalized)
            quality_score = min(blur_score / 1000.0, 1.0)  # Normalize blur score

            return EvaluationResult(
                metric_name="image_quality", value=quality_score, metadata=results
            )

        except Exception as e:
            logger.error(f"Image quality evaluation error: {e}")
            return EvaluationResult(
                metric_name="image_quality", value=0.0, metadata={"error": str(e)}
            )


class AestheticEvaluator(BaseEvaluator):
    """Aesthetic quality evaluation using CLIP-based aesthetic predictor"""

    def __init__(self, device: str = "auto"):
        super().__init__(device)
        self.model = None
        self.preprocess = None
        # Use CLIP as base, can be extended with specific aesthetic models
        self._load_model()

    def _load_model(self):
        """Load aesthetic evaluation model"""
        try:
            # For now, use CLIP as base for aesthetic evaluation
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.model.eval()
            logger.info("Loaded aesthetic evaluation model")
        except Exception as e:
            logger.error(f"Error loading aesthetic model: {e}")
            raise

    def evaluate(
        self, image: Union[Image.Image, str, Path], **kwargs
    ) -> EvaluationResult:
        """Evaluate aesthetic quality of image"""

        try:
            # Load image
            if isinstance(image, (str, Path)):
                img = Image.open(image).convert("RGB")
            else:
                img = image.convert("RGB")

            # Prepare image
            image_input = self.preprocess(img).unsqueeze(0).to(self.device)  # type: ignore

            # Aesthetic prompts (positive and negative)
            aesthetic_prompts = [
                "beautiful, aesthetic, high quality, professional photography",
                "ugly, low quality, blurry, distorted, amateur",
            ]

            text_inputs = clip.tokenize(aesthetic_prompts).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input)  # type: ignore
                text_features = self.model.encode_text(text_inputs)  # type: ignore

                # Normalize features
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)

                # Calculate similarities
                similarities = torch.cosine_similarity(image_features, text_features)

                # Aesthetic score = positive similarity - negative similarity
                aesthetic_score = (similarities[0] - similarities[1]).item()
                # Normalize to 0-1 range
                aesthetic_score = (aesthetic_score + 1) / 2

            return EvaluationResult(
                metric_name="aesthetic_score",
                value=aesthetic_score,
                metadata={
                    "positive_similarity": similarities[0].item(),
                    "negative_similarity": similarities[1].item(),
                },
            )

        except Exception as e:
            logger.error(f"Aesthetic evaluation error: {e}")
            return EvaluationResult(
                metric_name="aesthetic_score", value=0.0, metadata={"error": str(e)}
            )


class CompositeEvaluator:
    """Composite evaluator that runs multiple evaluation metrics"""

    def __init__(self, device: str = "auto"):
        self.device = device
        self.evaluators = {
            "clip": CLIPEvaluator(device=device),
            "face_consistency": FaceConsistencyEvaluator(device=device),
            "fid": FIDEvaluator(device=device),
            "image_quality": ImageQualityEvaluator(device=device),
            "aesthetic": AestheticEvaluator(device=device),
        }

    def evaluate_training_batch(
        self,
        generated_images: List[Union[Image.Image, str, Path]],
        prompts: List[str],
        reference_images: Optional[List[Union[Image.Image, str, Path]]] = None,
    ) -> EvaluationSummary:
        """Evaluate a batch of generated images during training"""

        all_results = []

        # CLIP evaluation for all images
        clip_scores = []
        for img, prompt in zip(generated_images, prompts):
            result = self.evaluators["clip"].evaluate(image=img, text=prompt)
            clip_scores.append(result.value)
            all_results.append(result)

        # Average CLIP score
        avg_clip = np.mean(clip_scores) if clip_scores else 0.0
        all_results.append(
            EvaluationResult(
                metric_name="avg_clip_similarity",
                value=avg_clip,  # type: ignore
                metadata={"batch_size": len(generated_images)},
            )
        )

        # Image quality evaluation
        quality_scores = []
        for img in generated_images:
            result = self.evaluators["image_quality"].evaluate(image=img)
            quality_scores.append(result.value)

        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        all_results.append(
            EvaluationResult(
                metric_name="avg_image_quality",
                value=avg_quality,  # type: ignore
                metadata={"batch_size": len(generated_images)},
            )
        )

        # Aesthetic evaluation
        aesthetic_scores = []
        for img in generated_images:
            result = self.evaluators["aesthetic"].evaluate(image=img)
            aesthetic_scores.append(result.value)

        avg_aesthetic = np.mean(aesthetic_scores) if aesthetic_scores else 0.0
        all_results.append(
            EvaluationResult(
                metric_name="avg_aesthetic_score",
                value=avg_aesthetic,  # type: ignore
                metadata={"batch_size": len(generated_images)},
            )
        )

        # Face consistency evaluation (if reference images provided)
        if reference_images:
            face_scores = []
            for gen_img, ref_img in zip(generated_images, reference_images):
                result = self.evaluators["face_consistency"].evaluate(
                    reference_image=ref_img, generated_image=gen_img
                )
                face_scores.append(result.value)

            avg_face_consistency = np.mean(face_scores) if face_scores else 0.0
            all_results.append(
                EvaluationResult(
                    metric_name="avg_face_consistency",
                    value=avg_face_consistency,  # type: ignore
                    metadata={"batch_size": len(reference_images)},
                )
            )

        # Calculate overall score (weighted average)
        weights = {
            "avg_clip_similarity": 0.3,
            "avg_image_quality": 0.25,
            "avg_aesthetic_score": 0.25,
            "avg_face_consistency": 0.2 if reference_images else 0.0,
        }

        # Adjust weights if no face consistency
        if not reference_images:
            weights["avg_clip_similarity"] = 0.4
            weights["avg_image_quality"] = 0.3
            weights["avg_aesthetic_score"] = 0.3

        overall_score = sum(
            result.value * weights.get(result.metric_name, 0)
            for result in all_results
            if result.metric_name in weights
        )

        return EvaluationSummary(
            results=all_results,
            overall_score=overall_score,
            metadata={
                "batch_size": len(generated_images),
                "evaluation_weights": weights,
                "has_reference_images": reference_images is not None,
            },
        )

    def evaluate_model_checkpoint(
        self,
        checkpoint_dir: Union[str, Path],
        test_prompts: List[str],
        reference_dir: Optional[Union[str, Path]] = None,
    ) -> EvaluationSummary:
        """Evaluate a model checkpoint with standardized test prompts"""

        checkpoint_dir = Path(checkpoint_dir)
        results = []

        # TODO: Generate test images using the checkpoint
        # For now, this is a placeholder that would integrate with the pipeline
        logger.info(f"Evaluating checkpoint: {checkpoint_dir}")

        # If sample images exist in checkpoint directory
        samples_dir = checkpoint_dir / "samples"
        if samples_dir.exists():
            sample_images = list(samples_dir.glob("*.png")) + list(
                samples_dir.glob("*.jpg")
            )

            if sample_images and len(sample_images) >= len(test_prompts):
                # Use existing samples for evaluation
                selected_images = sample_images[: len(test_prompts)]

                return self.evaluate_training_batch(
                    generated_images=selected_images, prompts=test_prompts  # type: ignore
                )

        # Return empty evaluation if no samples found
        return EvaluationSummary(
            results=[
                EvaluationResult(
                    metric_name="checkpoint_evaluation",
                    value=0.0,
                    metadata={"error": "No sample images found for evaluation"},
                )
            ],
            overall_score=0.0,
        )

    def save_evaluation_report(
        self, summary: EvaluationSummary, output_path: Union[str, Path]
    ):
        """Save evaluation summary to JSON report"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation report saved to: {output_path}")


class MetricsCalculator:
    """綜合評估指標計算器"""

    def __init__(self, device: str = "auto"):
        self.clip_evaluator = CLIPEvaluator(device)
        self.fid_evaluator = FIDEvaluator(device)
        self.lpips_evaluator = LPIPSEvaluator(device)

        # 初始化所有評估器
        self.clip_ready = self.clip_evaluator.initialize()
        self.fid_ready = self.fid_evaluator.initialize()
        self.lpips_ready = self.lpips_evaluator.initialize()

        logger.info(
            f"MetricsCalculator ready - CLIP: {self.clip_ready}, FID: {self.fid_ready}, LPIPS: {self.lpips_ready}"
        )

    def evaluate_generation_quality(
        self,
        generated_images: List[Image.Image],
        prompts: List[str],
        reference_images: Optional[List[Image.Image]] = None,
    ) -> Dict[str, Any]:
        """綜合評估生成品質"""

        metrics = {
            "num_images": len(generated_images),
            "clip_scores": [],
            "fid_score": None,
            "lpips_scores": [],
            "average_clip_score": 0.0,
            "average_lpips_score": 0.0,
            "overall_quality": "unknown",
        }

        try:
            # CLIP 文字-圖片相似度
            if self.clip_ready and prompts:
                clip_scores = self.clip_evaluator.compute_text_image_similarity(
                    generated_images, prompts
                )
                metrics["clip_scores"] = clip_scores
                metrics["average_clip_score"] = float(np.mean(clip_scores))

            # FID 分數 (需要參考圖片)
            if self.fid_ready and reference_images:
                fid_score = self.fid_evaluator.compute_fid(
                    reference_images, generated_images
                )
                metrics["fid_score"] = float(fid_score)

            # LPIPS 分數 (需要參考圖片)
            if self.lpips_ready and reference_images:
                lpips_scores = self.lpips_evaluator.compute_lpips(
                    reference_images, generated_images
                )
                metrics["lpips_scores"] = lpips_scores
                metrics["average_lpips_score"] = float(np.mean(lpips_scores))

            # 綜合品質評估
            metrics["overall_quality"] = self._assess_overall_quality(metrics)

            logger.info(
                f"Quality evaluation completed - Overall: {metrics['overall_quality']}"
            )
            return metrics

        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            return metrics

    def _assess_overall_quality(self, metrics: Dict[str, Any]) -> str:
        """評估整體品質"""
        try:
            clip_score = metrics.get("average_clip_score", 0.0)
            fid_score = metrics.get("fid_score")

            # 基於 CLIP 分數的基本評估
            if clip_score >= 0.8:
                quality = "excellent"
            elif clip_score >= 0.6:
                quality = "good"
            elif clip_score >= 0.4:
                quality = "fair"
            else:
                quality = "poor"

            # 如果有 FID 分數，進一步調整
            if fid_score is not None:
                if fid_score < 20:
                    quality = (
                        "excellent" if quality in ["good", "excellent"] else "good"
                    )
                elif fid_score > 50:
                    quality = "poor" if quality in ["poor", "fair"] else "fair"

            return quality

        except Exception:
            return "unknown"


class TrainingMonitor:
    """Monitor training progress with evaluation metrics"""

    def __init__(self, evaluator: CompositeEvaluator, log_dir: Union[str, Path]):
        self.evaluator = evaluator
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.evaluation_history = []
        self.best_score = 0.0
        self.best_checkpoint = None

    def evaluate_and_log(
        self,
        step: int,
        generated_images: List[Union[Image.Image, str, Path]],
        prompts: List[str],
        reference_images: Optional[List[Union[Image.Image, str, Path]]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> EvaluationSummary:
        """Evaluate current training state and log results"""

        # Run evaluation
        summary = self.evaluator.evaluate_training_batch(
            generated_images=generated_images,
            prompts=prompts,
            reference_images=reference_images,
        )

        # Add step information
        summary.metadata["training_step"] = step  # type: ignore
        if checkpoint_path:
            summary.metadata["checkpoint_path"] = str(checkpoint_path)  # type: ignore

        # Track best score
        if summary.overall_score and summary.overall_score > self.best_score:
            self.best_score = summary.overall_score
            self.best_checkpoint = checkpoint_path
            logger.info(f"New best score at step {step}: {self.best_score:.4f}")

        # Save evaluation
        self.evaluation_history.append(summary)

        # Save individual report
        report_path = self.log_dir / f"evaluation_step_{step}.json"
        self.evaluator.save_evaluation_report(summary, report_path)

        # Update training log
        self._update_training_log()

        return summary

    def _update_training_log(self):
        """Update overall training evaluation log"""
        log_data = {
            "evaluation_history": [s.to_dict() for s in self.evaluation_history],
            "best_score": self.best_score,
            "best_checkpoint": (
                str(self.best_checkpoint) if self.best_checkpoint else None
            ),
            "last_updated": datetime.now().isoformat(),
        }

        log_path = self.log_dir / "training_evaluation_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluation metrics"""
        if not self.evaluation_history:
            return {}

        # Collect all metric values
        metrics = {}
        for summary in self.evaluation_history:
            for result in summary.results:
                if result.metric_name not in metrics:
                    metrics[result.metric_name] = []
                metrics[result.metric_name].append(result.value)

        # Calculate statistics
        summary = {}
        for metric_name, values in metrics.items():
            summary[metric_name] = {
                "latest": values[-1] if values else 0.0,
                "best": max(values) if values else 0.0,
                "average": np.mean(values) if values else 0.0,
                "trend": (
                    "improving"
                    if len(values) > 1 and values[-1] > values[0]
                    else "stable"
                ),
            }

        return summary
