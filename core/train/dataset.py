# core/train/dataset.py - Training dataset management
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json, os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

from core.config import get_cache_paths, get_dataset_path


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
                "RGB", (self.resolution, self.resolution), color=(128, 128, 128)
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
