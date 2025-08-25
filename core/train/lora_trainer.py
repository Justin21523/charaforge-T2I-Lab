# core/train/lora_trainer.py - LoRA training implementation
from typing import Dict, Any, Optional, Callable, Generator
import json
import time
from pathlib import Path
import torch, os, math, time
from accelerate import Accelerator
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from peft import LoraConfig, get_peft_model

from core.config import get_run_output_dir, get_cache_paths
from core.train.dataset import T2IDataset


class LoRATrainer:
    """LoRA fine-tuning trainer for SD/SDXL models"""

    def __init__(self, base_model: str, output_dir: str, config: Dict[str, Any]):

        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.config = config
        self.cache_paths = get_cache_paths()

        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        )

        self.device = self.accelerator.device
        self.pipeline = None
        self.lora_layers = None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "samples").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

    def setup_model(self):
        """Setup base model and LoRA configuration"""
        print(f"[LoRATrainer] Saved checkpoint: {checkpoint_name}")

    def generate_samples(self, epoch: int):
        """Generate validation samples"""
        validation_prompts = self.config.get(
            "validation_prompts",
            ["anime girl with blue hair", "portrait of a character in school uniform"],
        )

        samples_dir = self.output_dir / "samples" / f"epoch_{epoch}"
        samples_dir.mkdir(parents=True, exist_ok=True)

        # TODO: Generate actual samples with current LoRA
        for i, prompt in enumerate(validation_prompts):
            # Mock sample generation
            sample_path = samples_dir / f"sample_{i:02d}.png"

            # Create placeholder image
            from PIL import Image

            placeholder = Image.new("RGB", (768, 768), color=(100, 150, 200))
            placeholder.save(sample_path)

            print(f"[LoRATrainer] Generated sample: {sample_path.name}")


def run_lora_training(
    task,
    *,
    run_id: str,
    dataset_path: str,
    base_model: str,
    rank: int = 16,
    learning_rate: float = 1e-4,
    train_steps: int = 1000,
    batch_size: int = 1,
    gradient_accumulation: int = 8,
    mixed_precision: str = "bf16",
    validation_prompts=None,
    notes: str = "",
    out_dir: str = "",
) -> Generator[Dict[str, Any], None, None]:
    """
    產生器：週期性 yield 訓練進度給 Celery
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 建立資料集 & DataLoader（最小示意）
    ds = T2IDataset(dataset_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # 2) 載入 SDXL base pipeline（可依 models.yaml 改為 SD1.5）
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16,
    )
    pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()

    # 3) 注入 LoRA（最小骨架）
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    pipe.unet = get_peft_model(pipe.unet, lora_cfg)
    pipe.unet.train()

    # 4) Optimizer
    optim = torch.optim.AdamW(pipe.unet.parameters(), lr=learning_rate)
    total_steps = train_steps
    step = 0
    loss_ema = None
    start = time.time()

    # 5) 簡化的訓練 loop（僅示意；實務需 TextEncoder/NoiseSchedule 等）
    while step < total_steps:
        for batch in dl:
            # TODO: 這裡應改為真正的 diffusion 訓練目標；此處僅作骨架佔位
            dummy_loss = torch.tensor(0.1, requires_grad=True, device=device)
            dummy_loss.backward()
            if (step + 1) % gradient_accumulation == 0:
                optim.step()
                optim.zero_grad()
            step += 1
            loss_val = float(dummy_loss.detach().item())
            loss_ema = 0.9 * (loss_ema or loss_val) + 0.1 * loss_val

            yield {
                "run_id": run_id,
                "progress": step / total_steps,
                "current_step": step,
                "total_steps": total_steps,
                "loss": loss_ema,
                "eta_minutes": int(
                    (time.time() - start) / max(step, 1) * (total_steps - step) / 60
                ),
            }
            if step >= total_steps:
                break

    # 6) 儲存 LoRA 權重（示意）
    out_path = os.path.join(out_dir, "lora.safetensors")
    # TODO: 將 PEFT/UNet LoRA 權重正確匯出
    artifacts = {"lora": out_path, "logs": os.path.join(out_dir, "train.log")}
    yield {
        "run_id": run_id,
        "progress": 1.0,
        "current_step": total_steps,
        "total_steps": total_steps,
        "loss": loss_ema,
        "artifacts": artifacts,
    }
