# core/train/dataset.py - Training dataset management
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import shutil
from datetime import datetime
import json, os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from core.config import get_cache_paths

logger = logging.getLogger(__name__)


def get_dataset_path(dataset_name: str) -> Path:
    """取得資料集路徑"""
    cache_paths = get_cache_paths()
    dataset_path = cache_paths.datasets / dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)
    return dataset_path


def validate_image_dataset(dataset_path: Path) -> Tuple[bool, List[str]]:
    """驗證圖片資料集的完整性"""
    errors = []

    try:
        # 檢查資料集目錄存在
        if not dataset_path.exists():
            errors.append(f"Dataset directory not found: {dataset_path}")
            return False, errors

        # 檢查圖片檔案
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = []

        for ext in image_extensions:
            image_files.extend(dataset_path.glob(f"*{ext}"))
            image_files.extend(dataset_path.glob(f"*{ext.upper()}"))

        if len(image_files) == 0:
            errors.append("No valid image files found")
            return False, errors

        # 驗證每個圖片檔案
        valid_images = 0
        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    # 檢查圖片可以正常開啟
                    img.verify()
                    valid_images += 1
            except Exception as e:
                errors.append(f"Invalid image {img_path.name}: {str(e)}")

        if valid_images == 0:
            errors.append("No valid images found")
            return False, errors

        # 檢查 metadata.json 是否存在
        metadata_path = dataset_path / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                # 驗證 metadata 格式
                if not isinstance(metadata, dict):
                    errors.append("Invalid metadata format")
            except Exception as e:
                errors.append(f"Invalid metadata.json: {str(e)}")

        success = len(errors) == 0
        return success, errors

    except Exception as e:
        errors.append(f"Dataset validation failed: {str(e)}")
        return False, errors


def create_dataset_metadata(
    dataset_path: Path,
    name: str,
    description: str = "",
    tags: List[str] = None,  # type: ignore
    class_prompt: str = "",
    instance_prompt: str = "",
) -> bool:
    """建立資料集元資料檔案"""
    try:
        if tags is None:
            tags = []

        # 統計圖片數量
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_count = 0

        for ext in image_extensions:
            image_count += len(list(dataset_path.glob(f"*{ext}")))
            image_count += len(list(dataset_path.glob(f"*{ext.upper()}")))

        metadata = {
            "name": name,
            "description": description,
            "tags": tags,
            "image_count": image_count,
            "class_prompt": class_prompt,
            "instance_prompt": instance_prompt,
            "created_at": str(datetime.now()),
            "format_version": "1.0",
        }

        metadata_path = dataset_path / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        logger.error(f"Failed to create dataset metadata: {e}")
        return False


class DreamBoothDataset(Dataset):
    """DreamBooth 特殊訓練資料集

    支援 instance 圖片和 class 圖片的訓練
    instance: 目標主體的少量圖片 (通常 3-5 張)
    class: 同類別的大量圖片 (用於保持模型對該類別的一般性理解)
    """

    def __init__(
        self,
        instance_data_root: str,
        instance_prompt: str,
        class_data_root: Optional[str] = None,
        class_prompt: Optional[str] = None,
        resolution: int = 512,
        center_crop: bool = True,
        train_text_encoder: bool = False,
        color_jitter: bool = False,
        random_flip: bool = False,
        tokenizer: Optional[Any] = None,
        max_sequence_length: int = 77,
    ):
        """初始化 DreamBooth 資料集

        Args:
            instance_data_root: instance 圖片資料夾路徑
            instance_prompt: instance 提示詞 (如 "a photo of sks dog")
            class_data_root: class 圖片資料夾路徑 (可選)
            class_prompt: class 提示詞 (如 "a photo of dog")
            resolution: 目標解析度
            center_crop: 是否使用中心裁切
            train_text_encoder: 是否同時訓練文字編碼器
            color_jitter: 是否使用顏色抖動
            random_flip: 是否使用隨機翻轉
            tokenizer: 文字 tokenizer
            max_sequence_length: 最大序列長度
        """
        self.instance_data_root = Path(instance_data_root)
        self.instance_prompt = instance_prompt
        self.class_data_root = Path(class_data_root) if class_data_root else None
        self.class_prompt = class_prompt
        self.resolution = resolution
        self.center_crop = center_crop
        self.train_text_encoder = train_text_encoder
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

        # 建立圖片轉換管線
        self.image_transforms = self._create_transforms(color_jitter, random_flip)

        # 載入 instance 圖片
        self.instance_images = self._load_images(self.instance_data_root)
        logger.info(f"Loaded {len(self.instance_images)} instance images")

        # 載入 class 圖片 (如果提供)
        self.class_images = []
        if self.class_data_root and self.class_data_root.exists():
            self.class_images = self._load_images(self.class_data_root)
            logger.info(f"Loaded {len(self.class_images)} class images")

        # 計算重複次數以平衡資料集
        self.num_instance_images = len(self.instance_images)
        self.num_class_images = len(self.class_images)

        # 確保每個 epoch 中 instance 和 class 圖片數量平衡
        if self.class_images:
            self._calculate_repeat_counts()

        logger.info(
            f"DreamBooth dataset initialized - Instance: {self.num_instance_images}, Class: {self.num_class_images}"
        )

    def _create_transforms(self, color_jitter: bool, random_flip: bool):
        """建立圖片轉換管線"""
        import torchvision.transforms as transforms

        transform_list = []

        # 調整大小和裁切
        if self.center_crop:
            transform_list.extend(
                [
                    transforms.Resize(
                        self.resolution,
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    transforms.CenterCrop(self.resolution),
                ]
            )
        else:
            transform_list.append(
                transforms.Resize(
                    (self.resolution, self.resolution),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
            )

        # 可選的資料增強
        if random_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        if color_jitter:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                )
            )

        # 轉為 tensor 並正規化
        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # 正規化到 [-1, 1]
            ]
        )

        return transforms.Compose(transform_list)

    def _load_images(self, data_root: Path) -> List[Path]:
        """載入資料夾中的所有圖片"""
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        images = []
        for ext in image_extensions:
            images.extend(data_root.glob(f"*{ext}"))
            images.extend(data_root.glob(f"*{ext.upper()}"))

        # 過濾無效圖片
        valid_images = []
        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_images.append(img_path)
            except Exception as e:
                logger.warning(f"Invalid image {img_path}: {e}")

        return sorted(valid_images)

    def _calculate_repeat_counts(self):
        """計算重複次數以平衡 instance 和 class 圖片"""
        # 通常 class 圖片比 instance 圖片多很多
        # 我們重複 instance 圖片來平衡資料集
        if self.num_class_images > 0:
            self.instance_repeat = max(
                1, self.num_class_images // self.num_instance_images
            )
        else:
            self.instance_repeat = 1

        logger.info(f"Instance images will be repeated {self.instance_repeat} times")

    def __len__(self) -> int:
        """資料集大小"""
        if self.class_images:
            # 如果有 class 圖片，每個 epoch 包含所有 class 圖片 + 重複的 instance 圖片
            return self.num_class_images + (
                self.num_instance_images * self.instance_repeat
            )
        else:
            # 只有 instance 圖片
            return self.num_instance_images

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """取得單個訓練樣本"""
        example = {}

        if self.class_images and index >= (
            self.num_instance_images * self.instance_repeat
        ):
            # 取得 class 圖片
            class_index = index - (self.num_instance_images * self.instance_repeat)
            image_path = self.class_images[class_index]
            prompt = self.class_prompt
            is_instance = False
        else:
            # 取得 instance 圖片 (可能重複)
            instance_index = index % self.num_instance_images
            image_path = self.instance_images[instance_index]
            prompt = self.instance_prompt
            is_instance = True

        # 載入並處理圖片
        try:
            image = Image.open(image_path).convert("RGB")

            # 應用轉換
            pixel_values = self.image_transforms(image)

            example["pixel_values"] = pixel_values
            example["is_instance"] = is_instance

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # 建立空白圖片作為備案
            blank_image = Image.new(
                "RGB", (self.resolution, self.resolution), color=(128, 128, 128)  # type: ignore
            )
            example["pixel_values"] = self.image_transforms(blank_image)
            example["is_instance"] = is_instance

        # 處理文字提示
        if self.tokenizer is not None:
            # Tokenize 提示詞
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )

            example["input_ids"] = text_inputs.input_ids.flatten()
            example["attention_mask"] = text_inputs.attention_mask.flatten()
        else:
            # 如果沒有 tokenizer，直接返回文字
            example["text"] = prompt

        # 添加元資料
        example.update(
            {"image_path": str(image_path), "prompt": prompt, "index": index}
        )

        return example

    @staticmethod
    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批次整理函數"""
        batch = {}

        # 整理 pixel_values
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        batch["pixel_values"] = pixel_values

        # 整理文字相關資料
        if "input_ids" in examples[0]:
            input_ids = torch.stack([example["input_ids"] for example in examples])
            attention_mask = torch.stack(
                [example["attention_mask"] for example in examples]
            )
            batch["input_ids"] = input_ids
            batch["attention_mask"] = attention_mask
        else:
            batch["texts"] = [example["text"] for example in examples]

        # 整理其他資料
        batch["is_instance"] = torch.tensor(
            [example["is_instance"] for example in examples]
        )
        batch["prompts"] = [example["prompt"] for example in examples]
        batch["image_paths"] = [example["image_path"] for example in examples]

        return batch

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], tokenizer: Optional[Any] = None
    ) -> "DreamBoothDataset":
        """從配置建立資料集"""
        return cls(
            instance_data_root=config["instance_data_root"],
            instance_prompt=config["instance_prompt"],
            class_data_root=config.get("class_data_root"),
            class_prompt=config.get("class_prompt"),
            resolution=config.get("resolution", 512),
            center_crop=config.get("center_crop", True),
            train_text_encoder=config.get("train_text_encoder", False),
            color_jitter=config.get("color_jitter", False),
            random_flip=config.get("random_flip", False),
            tokenizer=tokenizer,
            max_sequence_length=config.get("max_sequence_length", 77),
        )

    def get_statistics(self) -> Dict[str, Any]:
        """取得資料集統計資訊"""
        stats = {
            "num_instance_images": self.num_instance_images,
            "num_class_images": self.num_class_images,
            "total_samples": len(self),
            "instance_prompt": self.instance_prompt,
            "class_prompt": self.class_prompt,
            "resolution": self.resolution,
            "instance_repeat": getattr(self, "instance_repeat", 1),
        }

        return stats


class T2IDataset(Dataset):
    """
    Minimal dataset: expects
      captions.jsonl  (each line: {"file": "images/xxx.png", "text": "..."} )
      images/<files>
    """

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        resolution: int = 768,
        caption_column: str = "caption",
        image_column: str = "image_path",
        max_samples: Optional[int] = None,
    ):

        self.dataset_name = dataset_name
        self.resolution = resolution
        self.cache_paths = get_cache_paths()

        # Load metadata
        dataset_path = get_dataset_path(dataset_name)  # type: ignore
        metadata_file = dataset_path / "metadata.parquet"  # type: ignore

        if not metadata_file.exists():
            raise FileNotFoundError(f"Dataset metadata not found: {metadata_file}")

        self.df = pd.read_parquet(metadata_file)

        # Limit samples if specified
        if max_samples:
            self.df = self.df.head(max_samples)

        self.caption_column = caption_column
        self.image_column = image_column

        print(f"[Dataset] Loaded {len(self.df)} samples from {dataset_name}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # Load image
        image_path = Path(row[self.image_column])
        if not image_path.is_absolute():
            dataset_path = get_dataset_path(self.dataset_name)  # type: ignore
            image_path = dataset_path / image_path

        try:
            image = Image.open(image_path).convert("RGB")

            # Resize to target resolution
            image = image.resize(
                (self.resolution, self.resolution), Image.Resampling.LANCZOS
            )

        except Exception as e:
            print(f"[Dataset] Error loading image {image_path}: {e}")
            # Return placeholder
            image = Image.new(
                "RGB", (self.resolution, self.resolution), color=(128, 128, 128)  # type: ignore
            )

        # Get caption
        caption = str(row.get(self.caption_column, ""))

        return {"image": image, "caption": caption, "metadata": row.to_dict()}

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for batching"""
        images = [item["image"] for item in batch]
        captions = [item["caption"] for item in batch]
        metadata = [item["metadata"] for item in batch]

        return {"images": images, "captions": captions, "metadata": metadata}


class DatasetRegistry:
    """Registry for managing available datasets"""

    def __init__(self):
        self.cache_paths = get_cache_paths()

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets"""
        datasets = []
        datasets_dir = self.cache_paths.datasets

        for dataset_dir in datasets_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            metadata_file = dataset_dir / "metadata.parquet"
            if metadata_file.exists():
                df = pd.read_parquet(metadata_file)

                dataset_info = {
                    "name": dataset_dir.name,
                    "path": str(dataset_dir),
                    "samples": len(df),
                    "splits": (
                        df["split"].unique().tolist()
                        if "split" in df.columns
                        else ["train"]
                    ),
                    "has_captions": "caption" in df.columns,
                    "has_tags": "tags" in df.columns,
                }
                datasets.append(dataset_info)

        return datasets
