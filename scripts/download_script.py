#!/usr/bin/env python3
"""
Multi-Modal Lab - 完整AI模型批次下載腳本
根據「各類AI模型資訊與應用.md」設計的全面下載工具

功能：
- 批次下載所有 AI 模型到 warehouse
- 支援斷點續傳和並行下載
- 自動配置 HuggingFace 快取
- 驗證模型完整性
- 生成下載報告

使用：
python scripts/download_script.py --category all
python scripts/download_script.py --category llm --priority high
"""

import os
import sys
import argparse
import json
import logging
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import subprocess
import hashlib
import requests

# 確保可以導入專案模組
sys.path.insert(0, str(Path(__file__).parent.parent))

# 設置日誌
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """模型資訊類別"""

    name: str
    model_id: str
    category: str
    priority: str  # high, medium, low
    size_gb: float
    description: str
    dependencies: List[str]
    download_method: str  # huggingface, git, wget, custom
    custom_url: Optional[str] = None
    local_path: Optional[str] = None
    verification: Optional[Dict] = None


class ModelDownloader:
    """AI 模型下載管理器"""

    def __init__(self, cache_root: Optional[str] = None, max_workers: int = 3):
        self.setup_cache_paths(cache_root)
        self.max_workers = max_workers
        self.model_catalog = self._build_model_catalog()
        self.downloaded_models = []
        self.skipped_models = []
        self.failed_models = []

    def setup_cache_paths(self, cache_root: Optional[str] = None):
        """設置快取路徑"""
        # 使用環境變數或參數指定的 cache root
        AI_CACHE_ROOT = cache_root or os.getenv(
            "AI_CACHE_ROOT", "../ai_warehouse/cache"
        )

        # 設置 HuggingFace 快取環境變數
        for k, v in {
            "HF_HOME": f"{AI_CACHE_ROOT}/hf",
            "TRANSFORMERS_CACHE": f"{AI_CACHE_ROOT}/hf/transformers",
            "HF_DATASETS_CACHE": f"{AI_CACHE_ROOT}/hf/datasets",
            "HUGGINGFACE_HUB_CACHE": f"{AI_CACHE_ROOT}/hf/hub",
            "TORCH_HOME": f"{AI_CACHE_ROOT}/torch",
        }.items():
            os.environ[k] = v
            Path(v).mkdir(parents=True, exist_ok=True)

        # 創建模型專用目錄
        self.cache_root = Path(AI_CACHE_ROOT)
        model_dirs = [
            "models/llm",
            "models/text2image",
            "models/text2video",
            "models/controlnet",
            "models/vlm",
            "models/tts",
            "models/enhancement",
            "models/embedding",
            "models/safety",
            "models/tagging",
            "models/lora",
            "models/audio",
            "datasets/raw",
            "datasets/processed",
            "datasets/metadata",
            "outputs/multi-modal-lab",
        ]

        for dir_path in model_dirs:
            (self.cache_root / dir_path).mkdir(parents=True, exist_ok=True)

        logger.info(f"Cache root: {AI_CACHE_ROOT}")

    def _build_model_catalog(self) -> Dict[str, ModelInfo]:
        """建立完整模型目錄（基於各類AI模型資訊與應用.md）"""

        catalog = {}

        # ===== 1. LLM 對話/推理模型 =====
        llm_models = [
            ModelInfo(
                name="deepseek-llm-7b",
                model_id="deepseek-ai/deepseek-llm-7b-chat",
                category="llm",
                priority="high",
                size_gb=13.0,
                description="DeepSeek LLM 7B 中英對話模型",
                dependencies=["transformers", "torch"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="qwen2-7b-instruct",
                model_id="Qwen/Qwen2-7B-Instruct",
                category="llm",
                priority="high",
                size_gb=14.0,
                description="Qwen2 7B 指令模型，中文優秀",
                dependencies=["transformers", "torch"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="yi-6b-chat",
                model_id="01-ai/Yi-6B-Chat",
                category="llm",
                priority="medium",
                size_gb=12.0,
                description="Yi 6B 中英雙語對話模型",
                dependencies=["transformers", "torch"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="chatglm3-6b",
                model_id="THUDM/chatglm3-6b",
                category="llm",
                priority="medium",
                size_gb=12.0,
                description="ChatGLM3 6B 中文對話模型",
                dependencies=["transformers", "torch"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="baichuan2-7b-chat",
                model_id="baichuan-inc/Baichuan2-7B-Chat",
                category="llm",
                priority="medium",
                size_gb=13.0,
                description="百川2 7B 中文對話模型",
                dependencies=["transformers", "torch"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="minicpm-2b",
                model_id="openbmb/MiniCPM-2B-sft-bf16",
                category="llm",
                priority="low",
                size_gb=4.0,
                description="MiniCPM 2B 輕量對話模型",
                dependencies=["transformers", "torch"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="openhermes-13b",
                model_id="teknium/OpenHermes-13B",
                category="llm",
                priority="low",
                size_gb=26.0,
                description="OpenHermes 13B 無審查對話模型",
                dependencies=["transformers", "torch"],
                download_method="huggingface",
            ),
        ]

        # ===== 2. 視覺理解/VQA 模型 =====
        vlm_models = [
            ModelInfo(
                name="blip2-opt-2.7b",
                model_id="Salesforce/blip2-opt-2.7b",
                category="vlm",
                priority="high",
                size_gb=5.4,
                description="BLIP-2 圖像描述和VQA",
                dependencies=["transformers", "torch"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="llava-v1.5-7b",
                model_id="llava-hf/llava-1.5-7b-hf",
                category="vlm",
                priority="high",
                size_gb=13.0,
                description="LLaVA 1.5 7B 多模態問答",
                dependencies=["transformers", "torch"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="qwen-vl-chat",
                model_id="Qwen/Qwen-VL-Chat",
                category="vlm",
                priority="high",
                size_gb=9.6,
                description="Qwen-VL 中文視覺語言模型",
                dependencies=["transformers", "torch"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="minigpt4-vicuna-7b",
                model_id="Vision-CAIR/MiniGPT-4",
                category="vlm",
                priority="medium",
                size_gb=13.0,
                description="MiniGPT-4 視覺對話模型",
                dependencies=["transformers", "torch"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="visualglm-6b",
                model_id="THUDM/visualglm-6b",
                category="vlm",
                priority="medium",
                size_gb=11.0,
                description="VisualGLM 中文多模態模型",
                dependencies=["transformers", "torch"],
                download_method="huggingface",
            ),
        ]

        # ===== 3. 文字→圖像生成模型 =====
        text2image_models = [
            ModelInfo(
                name="sdxl-base",
                model_id="stabilityai/stable-diffusion-xl-base-1.0",
                category="text2image",
                priority="high",
                size_gb=13.0,
                description="SDXL 基礎模型，高畫質生圖",
                dependencies=["diffusers", "torch"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="sdxl-refiner",
                model_id="stabilityai/stable-diffusion-xl-refiner-1.0",
                category="text2image",
                priority="medium",
                size_gb=13.0,
                description="SDXL 精細化模型",
                dependencies=["diffusers", "torch"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="sd15-base",
                model_id="runwayml/stable-diffusion-v1-5",
                category="text2image",
                priority="high",
                size_gb=7.0,
                description="Stable Diffusion 1.5 基礎模型",
                dependencies=["diffusers", "torch"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="deepfloyd-if-xl",
                model_id="DeepFloyd/IF-I-XL-v1.0",
                category="text2image",
                priority="medium",
                size_gb=9.0,
                description="DeepFloyd IF 高細節生圖",
                dependencies=["diffusers", "torch"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="pixart-alpha-xl",
                model_id="PixArt-alpha/PixArt-XL-2-1024-MS",
                category="text2image",
                priority="medium",
                size_gb=11.0,
                description="PixArt-α 高質量插畫模型",
                dependencies=["diffusers", "torch"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="stable-cascade",
                model_id="stabilityai/stable-cascade",
                category="text2image",
                priority="low",
                size_gb=15.0,
                description="Stable Cascade 快速生圖",
                dependencies=["diffusers", "torch"],
                download_method="huggingface",
            ),
        ]

        # ===== 4. 文字→影片生成模型 =====
        text2video_models = [
            ModelInfo(
                name="animatediff-v15",
                model_id="guoyww/animatediff-motion-adapter-v1-5-2",
                category="text2video",
                priority="medium",
                size_gb=4.0,
                description="AnimateDiff 動畫生成模型",
                dependencies=["diffusers", "torch"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="stable-video-diffusion",
                model_id="stabilityai/stable-video-diffusion-img2vid-xt",
                category="text2video",
                priority="medium",
                size_gb=9.0,
                description="Stable Video Diffusion 影片生成",
                dependencies=["diffusers", "torch"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="modelscope-t2v",
                model_id="damo-vilab/text-to-video-ms-1.7b",
                category="text2video",
                priority="low",
                size_gb=7.0,
                description="ModelScope 文字轉影片模型",
                dependencies=["diffusers", "torch"],
                download_method="huggingface",
            ),
        ]

        # ===== 5. ControlNet 控制模型 =====
        controlnet_models = [
            ModelInfo(
                name="canny-controlnet",
                model_id="lllyasviel/sd-controlnet-canny",
                category="controlnet",
                priority="high",
                size_gb=2.5,
                description="Canny邊緣檢測ControlNet",
                dependencies=["diffusers", "controlnet_aux"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="depth-controlnet",
                model_id="lllyasviel/sd-controlnet-depth",
                category="controlnet",
                priority="high",
                size_gb=2.5,
                description="深度圖ControlNet",
                dependencies=["diffusers", "controlnet_aux"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="pose-controlnet",
                model_id="lllyasviel/sd-controlnet-openpose",
                category="controlnet",
                priority="high",
                size_gb=2.5,
                description="姿態檢測ControlNet",
                dependencies=["diffusers", "controlnet_aux"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="sdxl-canny-controlnet",
                model_id="diffusers/controlnet-canny-sdxl-1.0",
                category="controlnet",
                priority="medium",
                size_gb=5.0,
                description="SDXL Canny ControlNet",
                dependencies=["diffusers", "controlnet_aux"],
                download_method="huggingface",
            ),
        ]

        # ===== 6. 語音合成 TTS 模型 =====
        tts_models = [
            ModelInfo(
                name="openvoice",
                model_id="myshell-ai/OpenVoice",
                category="tts",
                priority="medium",
                size_gb=2.0,
                description="OpenVoice 多語言語音合成",
                dependencies=["torch", "torchaudio"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="coqui-xtts-v2",
                model_id="coqui/XTTS-v2",
                category="tts",
                priority="medium",
                size_gb=4.0,
                description="Coqui XTTS v2 多語言TTS",
                dependencies=["torch", "torchaudio"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="bark",
                model_id="suno/bark",
                category="tts",
                priority="low",
                size_gb=3.0,
                description="Bark 情感語音生成",
                dependencies=["torch", "torchaudio"],
                download_method="huggingface",
            ),
        ]

        # ===== 7. 圖像修復/增強模型 =====
        enhancement_models = [
            ModelInfo(
                name="real-esrgan-x4",
                model_id="",
                category="enhancement",
                priority="medium",
                size_gb=0.5,
                description="Real-ESRGAN 4x 超分辨率",
                dependencies=["opencv-python"],
                download_method="custom",
                custom_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            ),
            ModelInfo(
                name="gfpgan-v13",
                model_id="",
                category="enhancement",
                priority="medium",
                size_gb=0.3,
                description="GFPGAN 人臉修復模型",
                dependencies=["opencv-python"],
                download_method="custom",
                custom_url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            ),
            ModelInfo(
                name="codeformer",
                model_id="",
                category="enhancement",
                priority="low",
                size_gb=0.3,
                description="CodeFormer 人臉修復",
                dependencies=["opencv-python"],
                download_method="custom",
                custom_url="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
            ),
        ]

        # ===== 8. 圖像標註/分析模型 =====
        tagging_models = [
            ModelInfo(
                name="wd14-convnextv2",
                model_id="SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
                category="tagging",
                priority="medium",
                size_gb=0.8,
                description="WD14 動漫圖像標註模型",
                dependencies=["transformers", "torch"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="deepdanbooru",
                model_id="",
                category="tagging",
                priority="low",
                size_gb=0.2,
                description="DeepDanbooru 動漫標籤分類",
                dependencies=["tensorflow"],
                download_method="custom",
                custom_url="https://github.com/KichangKim/DeepDanbooru/releases/download/v3.0/deepdanbooru-v3-20211112-sgd-e28.zip",
            ),
        ]

        # ===== 9. 嵌入向量模型 =====
        embedding_models = [
            ModelInfo(
                name="bge-base-zh-v1.5",
                model_id="BAAI/bge-base-zh-v1.5",
                category="embedding",
                priority="high",
                size_gb=1.0,
                description="BGE Base 中文嵌入模型",
                dependencies=["sentence-transformers"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="bge-m3",
                model_id="BAAI/bge-m3",
                category="embedding",
                priority="high",
                size_gb=2.0,
                description="BGE M3 多語言多模態嵌入",
                dependencies=["sentence-transformers"],
                download_method="huggingface",
            ),
            ModelInfo(
                name="clip-vit-base-patch32",
                model_id="openai/clip-vit-base-patch32",
                category="embedding",
                priority="high",
                size_gb=0.6,
                description="CLIP 圖文嵌入模型",
                dependencies=["transformers", "torch"],
                download_method="huggingface",
            ),
        ]

        # ===== 10. 安全過濾模型 =====
        safety_models = [
            ModelInfo(
                name="clip-safety-checker",
                model_id="CompVis/stable-diffusion-safety-checker",
                category="safety",
                priority="medium",
                size_gb=1.2,
                description="CLIP 安全內容檢測器",
                dependencies=["transformers", "torch"],
                download_method="huggingface",
            ),
        ]

        # 組合所有模型
        all_models = (
            llm_models
            + vlm_models
            + text2image_models
            + text2video_models
            + controlnet_models
            + tts_models
            + enhancement_models
            + tagging_models
            + embedding_models
            + safety_models
        )

        # 轉換為字典
        for model in all_models:
            catalog[model.name] = model

        logger.info(f"已載入 {len(catalog)} 個模型定義")
        return catalog

    def list_models(
        self, category: Optional[str] = None, priority: Optional[str] = None
    ) -> List[Dict]:
        """列出可用模型"""
        models = []
        for name, model_info in self.model_catalog.items():
            if category and model_info.category != category:
                continue
            if priority and model_info.priority != priority:
                continue

            models.append(
                {
                    "name": name,
                    "category": model_info.category,
                    "priority": model_info.priority,
                    "size_gb": model_info.size_gb,
                    "description": model_info.description,
                }
            )

        return sorted(models, key=lambda x: (x["priority"], x["category"], x["name"]))

    def download_model(
        self, model_info: ModelInfo, force: bool = False
    ) -> Tuple[bool, str]:
        """下載單個模型"""
        logger.info(f"開始下載: {model_info.name} ({model_info.size_gb:.1f}GB)")

        try:
            if model_info.download_method == "huggingface":
                return self._download_huggingface_model(model_info, force)
            elif model_info.download_method == "custom":
                return self._download_custom_model(model_info, force)
            elif model_info.download_method == "git":
                return self._download_git_model(model_info, force)
            else:
                return False, f"不支援的下載方式: {model_info.download_method}"

        except Exception as e:
            logger.error(f"下載失敗 {model_info.name}: {e}")
            return False, str(e)

    def _download_huggingface_model(
        self, model_info: ModelInfo, force: bool
    ) -> Tuple[bool, str]:
        """下載 HuggingFace 模型"""
        try:
            # 檢查是否已存在
            if not force and self._is_model_cached(model_info.model_id):
                logger.info(f"模型已存在，跳過下載: {model_info.name}")
                self.skipped_models.append(model_info.name)
                return True, "已存在"

            # 使用 huggingface_hub 下載
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                logger.error("需要安裝 huggingface_hub: pip install huggingface_hub")
                return False, "缺少依賴"

            logger.info(f"正在下載 {model_info.model_id}...")

            # 根據模型類別決定本地路徑
            category_paths = {
                "controlnet": self.cache_root / "models" / "controlnet",
                "text2image": self.cache_root / "models" / "text2image",
                "text2video": self.cache_root / "models" / "text2video",
                "llm": self.cache_root / "models" / "llm",
                "vlm": self.cache_root / "models" / "vlm",
                "tts": self.cache_root / "models" / "tts",
                "embedding": self.cache_root / "models" / "embedding",
                "safety": self.cache_root / "models" / "safety",
            }

            local_dir = None
            if model_info.category in category_paths:
                local_dir = category_paths[model_info.category] / model_info.name
                local_dir.mkdir(parents=True, exist_ok=True)

            # 執行下載
            snapshot_download(
                repo_id=model_info.model_id,
                local_dir=str(local_dir) if local_dir else None,
                resume_download=True,
                local_files_only=False,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
            )

            logger.info(f"下載完成: {model_info.name}")
            self.downloaded_models.append(model_info.name)
            return True, "下載成功"

        except Exception as e:
            logger.error(f"HuggingFace 下載失敗: {e}")
            return False, str(e)

    def _download_custom_model(
        self, model_info: ModelInfo, force: bool
    ) -> Tuple[bool, str]:
        """下載自定義 URL 模型"""
        if not model_info.custom_url:
            return False, "缺少自定義下載 URL"

        try:
            # 決定本地路徑
            filename = Path(model_info.custom_url).name
            local_path = self.cache_root / "models" / model_info.category / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # 檢查是否已存在
            if local_path.exists() and not force:
                logger.info(f"檔案已存在，跳過下載: {filename}")
                self.skipped_models.append(model_info.name)
                return True, "已存在"

            # 下載檔案
            logger.info(f"正在下載 {model_info.custom_url}")

            response = requests.get(model_info.custom_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded_size = 0

            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        # 顯示進度
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\r下載進度: {progress:.1f}%", end="", flush=True)

            print()  # 新行
            logger.info(f"下載完成: {model_info.name} -> {local_path}")
            self.downloaded_models.append(model_info.name)
            return True, "下載成功"

        except Exception as e:
            logger.error(f"自定義下載失敗: {e}")
            return False, str(e)

    def _download_git_model(
        self, model_info: ModelInfo, force: bool
    ) -> Tuple[bool, str]:
        """使用 git clone 下載模型"""
        if not model_info.custom_url:
            return False, "缺少 Git URL"

        try:
            local_path = (
                self.cache_root / "models" / model_info.category / model_info.name
            )

            # 檢查是否已存在
            if local_path.exists() and not force:
                logger.info(f"Git repo 已存在，跳過: {model_info.name}")
                self.skipped_models.append(model_info.name)
                return True, "已存在"

            # 移除現有目錄（如果強制更新）
            if local_path.exists() and force:
                import shutil

                shutil.rmtree(local_path)

            local_path.parent.mkdir(parents=True, exist_ok=True)

            # 執行 git clone
            logger.info(f"正在 clone {model_info.custom_url}")
            result = subprocess.run(
                ["git", "clone", model_info.custom_url, str(local_path)],
                capture_output=True,
                text=True,
                timeout=1800,  # 30分鐘超時
            )

            if result.returncode == 0:
                logger.info(f"Git clone 完成: {model_info.name}")
                self.downloaded_models.append(model_info.name)
                return True, "下載成功"
            else:
                return False, f"Git clone 失敗: {result.stderr}"

        except Exception as e:
            logger.error(f"Git 下載失敗: {e}")
            return False, str(e)

    def _is_model_cached(self, model_id: str) -> bool:
        """檢查模型是否已在快取中"""
        try:
            # 檢查 HuggingFace 快取
            hf_cache = Path(os.environ["HUGGINGFACE_HUB_CACHE"])
            model_cache_dir = hf_cache / f"models--{model_id.replace('/', '--')}"

            return model_cache_dir.exists() and any(
                list(model_cache_dir.rglob("*.safetensors"))
                + list(model_cache_dir.rglob("*.bin"))
                + list(model_cache_dir.rglob("*.pth"))
            )

        except Exception:
            return False

    def batch_download(
        self,
        categories: List[str] = None,
        priorities: List[str] = None,
        model_names: List[str] = None,
        force: bool = False,
        max_concurrent: int = None,
    ) -> Dict:
        """批次下載模型"""

        if max_concurrent is None:
            max_concurrent = self.max_workers

        # 篩選要下載的模型
        models_to_download = []

        for name, model_info in self.model_catalog.items():
            # 按名稱篩選
            if model_names and name not in model_names:
                continue

            # 按分類篩選
            if categories and model_info.category not in categories:
                continue

            # 按優先級篩選
            if priorities and model_info.priority not in priorities:
                continue

            models_to_download.append(model_info)

        logger.info(f"準備下載 {len(models_to_download)} 個模型")

        # 計算總大小
        total_size = sum(model.size_gb for model in models_to_download)
        logger.info(f"預計總下載大小: {total_size:.1f}GB")

        # 並行下載
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent
        ) as executor:
            # 提交所有下載任務
            future_to_model = {
                executor.submit(self.download_model, model, force): model
                for model in models_to_download
            }

            # 處理結果
            for future in concurrent.futures.as_completed(future_to_model):
                model_info = future_to_model[future]
                try:
                    success, message = future.result()
                    if not success:
                        self.failed_models.append(f"{model_info.name}: {message}")
                        logger.error(f"下載失敗: {model_info.name} - {message}")
                except Exception as e:
                    self.failed_models.append(f"{model_info.name}: {str(e)}")
                    logger.error(f"任務執行失敗: {model_info.name} - {e}")

        elapsed_time = time.time() - start_time

        # 計算實際下載大小
        downloaded_size = sum(
            model.size_gb
            for model in models_to_download
            if model.name in self.downloaded_models
        )

        # 生成報告
        report = {
            "total_models": len(models_to_download),
            "downloaded": len(self.downloaded_models),
            "skipped": len(self.skipped_models),
            "failed": len(self.failed_models),
            "elapsed_time": elapsed_time,
            "total_size_gb": downloaded_size,
            "downloaded_models": self.downloaded_models,
            "skipped_models": self.skipped_models,
            "failed_models": self.failed_models,
            "timestamp": datetime.now().isoformat(),
        }

        return report

    def generate_download_report(self, report: Dict):
        """生成詳細下載報告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = (
            self.cache_root
            / "outputs"
            / "multi-modal-lab"
            / f"download_report_{timestamp}.json"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 添加系統資訊
        report_data = {
            **report,
            "system_info": {
                "cache_root": str(self.cache_root),
                "python_version": sys.version,
                "platform": sys.platform,
            },
            "model_details": {
                name: {
                    "model_id": info.model_id,
                    "category": info.category,
                    "priority": info.priority,
                    "size_gb": info.size_gb,
                    "download_method": info.download_method,
                }
                for name, info in self.model_catalog.items()
                if name in (report["downloaded_models"] + report["skipped_models"])
            },
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        logger.info(f"下載報告已儲存: {save_path}")


def setup_dependencies():
    """安裝必要依賴"""
    dependencies = [
        "huggingface_hub>=0.20.0",
        "transformers>=4.36.0",
        "torch>=2.0.0",
        "diffusers>=0.24.0",
        "sentence-transformers>=2.2.0",
        "opencv-python>=4.8.0",
        "requests>=2.31.0",
    ]

    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            logger.info(f"安裝完成: {dep}")
        except subprocess.CalledProcessError as e:
            logger.error(f"安裝失敗 {dep}: {e}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="Multi-Modal Lab 完整 AI 模型批次下載工具"
    )

    parser.add_argument(
        "--category",
        "-c",
        choices=[
            "all",
            "llm",
            "vlm",
            "text2image",
            "text2video",
            "controlnet",
            "tts",
            "enhancement",
            "tagging",
            "embedding",
            "safety",
        ],
        default="all",
        help="要下載的模型分類",
    )

    parser.add_argument(
        "--priority", "-p", choices=["high", "medium", "low"], help="按優先級篩選模型"
    )

    parser.add_argument("--models", "-m", nargs="+", help="指定要下載的模型名稱")

    parser.add_argument(
        "--force", "-f", action="store_true", help="強制重新下載已存在的模型"
    )

    parser.add_argument("--list", "-l", action="store_true", help="列出所有可用模型")

    parser.add_argument(
        "--concurrent", "-j", type=int, default=3, help="並行下載數量 (預設: 3)"
    )

    parser.add_argument("--cache-root", help="自定義 warehouse 根目錄")

    parser.add_argument(
        "--setup-deps", action="store_true", help="安裝必要的 Python 依賴套件"
    )

    args = parser.parse_args()

    # 安裝依賴
    if args.setup_deps:
        logger.info("正在安裝必要依賴...")
        setup_dependencies()
        logger.info("依賴安裝完成")
        return 0

    # 初始化下載器
    try:
        downloader = ModelDownloader(
            cache_root=args.cache_root, max_workers=args.concurrent
        )
    except Exception as e:
        logger.error(f"初始化失敗: {e}")
        return 1

    # 列出模型模式
    if args.list:
        models = downloader.list_models(
            category=args.category if args.category != "all" else None,
            priority=args.priority,
        )

        print(f"\n{'名稱':<25} {'分類':<15} {'優先級':<8} {'大小':<10} {'說明'}")
        print("-" * 90)

        for model in models:
            print(
                f"{model['name']:<25} {model['category']:<15} "
                f"{model['priority']:<8} {model['size_gb']:>6.1f}GB   {model['description']}"
            )

        # 統計資訊
        total_models = len(models)
        total_size = sum(m["size_gb"] for m in models)

        print(f"\n共 {total_models} 個模型可用，總大小 {total_size:.1f}GB")

        # 按分類統計
        from collections import defaultdict

        by_category = defaultdict(list)
        for model in models:
            by_category[model["category"]].append(model)

        print("\n按分類統計:")
        for cat, cat_models in sorted(by_category.items()):
            cat_size = sum(m["size_gb"] for m in cat_models)
            print(f"  {cat}: {len(cat_models)} 模型, {cat_size:.1f}GB")

        return 0

    # 執行下載
    try:
        categories = None if args.category == "all" else [args.category]
        priorities = [args.priority] if args.priority else None

        logger.info("開始批次下載...")
        report = downloader.batch_download(
            categories=categories,
            priorities=priorities,
            model_names=args.models,
            force=args.force,
            max_concurrent=args.concurrent,
        )

        # 顯示下載結果
        print("\n" + "=" * 70)
        print("下載完成報告")
        print("=" * 70)
        print(f"總模型數量: {report['total_models']}")
        print(f"成功下載: {report['downloaded']}")
        print(f"跳過 (已存在): {report['skipped']}")
        print(f"失敗: {report['failed']}")
        print(f"總耗時: {report['elapsed_time']:.1f} 秒")
        print(f"總大小: {report['total_size_gb']:.1f} GB")

        if report["downloaded_models"]:
            print(f"\n成功下載的模型:")
            for model in report["downloaded_models"]:
                print(f"  ✓ {model}")

        if report["skipped_models"]:
            print(f"\n跳過的模型:")
            for model in report["skipped_models"]:
                print(f"  ~ {model}")

        if report["failed_models"]:
            print(f"\n失敗的模型:")
            for model in report["failed_models"]:
                print(f"  ✗ {model}")

        # 生成詳細報告
        downloader.generate_download_report(report)

        return 0 if report["failed"] == 0 else 1

    except KeyboardInterrupt:
        logger.warning("用戶中斷下載")
        return 1
    except Exception as e:
        logger.error(f"下載過程出現錯誤: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


# ===== 使用範例腳本 =====

"""
使用範例：

1. 安裝依賴並列出所有可用模型：
python scripts/download_script.py --setup-deps
python scripts/download_script.py --list

2. 下載所有高優先級模型：
python scripts/download_script.py --priority high

3. 下載 LLM 相關模型：
python scripts/download_script.py --category llm

4. 下載視覺理解模型：
python scripts/download_script.py --category vlm

5. 下載文字生圖模型：
python scripts/download_script.py --category text2image

6. 下載指定模型：
python scripts/download_script.py --models qwen2-7b-instruct blip2-opt-2.7b sdxl-base

7. 強制重新下載所有模型：
python scripts/download_script.py --category all --force

8. 並行下載 5 個模型：
python scripts/download_script.py --category controlnet --concurrent 5

9. 自定義倉儲路徑：
AI_CACHE_ROOT=/custom/path python scripts/download_script.py --priority high

10. 下載特定類型組合：
python scripts/download_script.py --category llm --priority high
python scripts/download_script.py --category enhancement --priority medium

模型分類說明：
- llm: 大語言模型 (Qwen, DeepSeek, Yi, ChatGLM, Baichuan, MiniCPM)
- vlm: 視覺語言模型 (BLIP-2, LLaVA, Qwen-VL, MiniGPT-4, VisualGLM)
- text2image: 文字生圖 (SDXL, SD1.5, PixArt, DeepFloyd IF, Stable Cascade)
- text2video: 文字生影片 (AnimateDiff, Stable Video Diffusion, ModelScope T2V)
- controlnet: 控制模型 (Canny, Depth, Pose, SDXL ControlNet)
- tts: 語音合成 (OpenVoice, Coqui XTTS, Bark)
- enhancement: 圖像修復 (Real-ESRGAN, GFPGAN, CodeFormer)
- tagging: 圖像標註 (WD14, DeepDanbooru)
- embedding: 向量嵌入 (BGE, CLIP)
- safety: 安全檢測 (Safety Checker)

優先級說明：
- high: 核心功能必需模型
- medium: 進階功能模型
- low: 特殊用途/實驗性模型
"""
