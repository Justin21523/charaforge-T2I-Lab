# core/train/lora_trainer.py - LoRA training implementation
from typing import Dict, Tuple, List, Any, Optional, Callable, Generator
import json
import os
import logging
from datetime import datetime
from pathlib import Path
import torch, os, math, time
import torch.nn.functional as F
import numpy as np
from accelerate import Accelerator
from torch.utils.data import DataLoader
from PIL import Image

# Diffusers and training imports
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_wandb_available
from diffusers.utils import check_min_version

# PEFT and LoRA imports
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# Accelerate for distributed training
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

# Transformers
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from core.config import get_run_output_dir, get_cache_paths
from core.train.dataset import T2IDataset
from core.train.evaluators import CLIPEvaluator, FaceConsistencyEvaluator


# Check minimum versions
check_min_version("0.28.0")

logger = get_logger(__name__, log_level="INFO")


class LoRATrainer:
    """Complete LoRA fine-tuning trainer for SD/SDXL models"""

    def __init__(self, base_model: str, output_dir: str, config: Dict[str, Any]):

        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.config = config
        self.cache_paths = get_cache_paths()

        # Determine model type
        self.is_sdxl = "xl" in base_model.lower()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "samples").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Initialize accelerator
        self._setup_accelerator()

        # Model components will be loaded in setup_model
        self.tokenizer = None
        self.text_encoder = None
        self.text_encoder_2 = None  # For SDXL
        self.vae = None
        self.unet = None
        self.noise_scheduler = None

        # Evaluators
        self.clip_evaluator = CLIPEvaluator()
        self.face_evaluator = FaceConsistencyEvaluator()

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "logs" / "training.log"
        log_file.parent.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        self.logger = logging.getLogger(__name__)

    def _setup_accelerator(self):
        """Setup Accelerate for distributed training"""
        # Accelerate configuration
        accelerator_project_config = ProjectConfiguration(
            project_dir=str(self.output_dir), logging_dir=str(self.output_dir / "logs")
        )

        # Handle potential hanging in distributed training
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.get(
                "gradient_accumulation_steps", 1
            ),
            mixed_precision=self.config.get("mixed_precision", "fp16"),
            log_with=self.config.get("report_to", None),
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs],
        )

        # Set seed for reproducibility
        if self.config.get("seed") is not None:
            set_seed(self.config["seed"])

    def setup_model(self):
        """Setup base model and LoRA configuration"""
        self.logger.info(f"Loading base model: {self.base_model}")

        # Load tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.base_model,
            subfolder="tokenizer",
            revision=self.config.get("revision", None),  # type: ignore
            use_fast=False,
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.base_model,
            subfolder="text_encoder",
            revision=self.config.get("revision", None),  # type: ignore
            torch_dtype=(
                torch.float16
                if self.config.get("mixed_precision") == "fp16"
                else torch.float32
            ),
        )

        # SDXL has second text encoder
        if self.is_sdxl:
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                self.base_model,
                subfolder="tokenizer_2",
                revision=self.config.get("revision", None),  # type: ignore
                use_fast=False,
            )

            self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                self.base_model,
                subfolder="text_encoder_2",
                revision=self.config.get("revision", None),  # type: ignore
                torch_dtype=(
                    torch.float16
                    if self.config.get("mixed_precision") == "fp16"
                    else torch.float32
                ),
            )

        # Load VAE
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

        self.vae = AutoencoderKL.from_pretrained(
            self.base_model,
            subfolder="vae",
            revision=self.config.get("revision", None),
            torch_dtype=(
                torch.float16
                if self.config.get("mixed_precision") == "fp16"
                else torch.float32
            ),
        )

        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.base_model,
            subfolder="unet",
            revision=self.config.get("revision", None),
            torch_dtype=(
                torch.float16
                if self.config.get("mixed_precision") == "fp16"
                else torch.float32
            ),
        )

        # Setup noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.base_model, subfolder="scheduler"
        )

        # Freeze models that shouldn't be trained
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        if self.is_sdxl:
            self.text_encoder_2.requires_grad_(False)  # type: ignore

        # Setup LoRA on UNet
        self._setup_lora()

        # Move to device
        self.vae.to(self.accelerator.device, dtype=torch.float32)
        self.text_encoder.to(self.accelerator.device)
        if self.is_sdxl:
            self.text_encoder_2.to(self.accelerator.device)  # type: ignore

        # Enable memory efficient attention if available
        if hasattr(self.unet, "enable_xformers_memory_efficient_attention"):
            self.unet.enable_xformers_memory_efficient_attention()
            self.logger.info("Enabled xformers memory efficient attention")

    def _setup_lora(self):
        """Setup LoRA configuration on UNet"""
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config.get("rank", 16),
            lora_alpha=self.config.get("lora_alpha", None)
            or self.config.get("rank", 16),
            target_modules=self.config.get(
                "target_modules",
                [
                    "to_k",
                    "to_q",
                    "to_v",
                    "to_out.0",
                    "proj_in",
                    "proj_out",
                    "ff.net.0.proj",
                    "ff.net.2",
                    "conv1",
                    "conv2",
                    "conv_shortcut",
                    "downsamplers.0.conv",
                    "upsamplers.0.conv",
                ],
            ),
            lora_dropout=self.config.get("lora_dropout", 0.1),
            bias="none",
            task_type=TaskType.DIFFUSION,  # type: ignore
        )

        # Apply LoRA to UNet
        self.unet = get_peft_model(self.unet, lora_config)  # type: ignore

        # Print trainable parameters
        trainable_params = sum(
            p.numel() for p in self.unet.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.unet.parameters())

        self.logger.info(f"LoRA setup complete:")
        self.logger.info(f"  - Rank: {lora_config.r}")
        self.logger.info(f"  - Alpha: {lora_config.lora_alpha}")
        self.logger.info(f"  - Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  - Total parameters: {total_params:,}")
        self.logger.info(
            f"  - Trainable ratio: {100 * trainable_params / total_params:.2f}%"
        )

    def train(
        self, dataset: T2IDataset, progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Main training loop"""

        self.logger.info(f"Starting training on {len(dataset)} samples")
        start_time = time.time()

        # Setup model
        self.setup_model()

        # Setup data loader
        train_dataloader = DataLoader(
            dataset,
            batch_size=self.config.get("train_batch_size", 1),
            shuffle=True,
            collate_fn=dataset.collate_fn,
            num_workers=self.config.get("dataloader_num_workers", 0),
            pin_memory=True,
        )

        # Setup optimizer
        optimizer = self._setup_optimizer()

        # Setup learning rate scheduler
        lr_scheduler = self._setup_lr_scheduler(optimizer, len(train_dataloader))

        # Prepare everything with accelerator
        self.unet, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            self.unet, optimizer, train_dataloader, lr_scheduler
        )

        # Calculate training steps
        num_update_steps_per_epoch = max(
            1,
            len(train_dataloader) // self.config.get("gradient_accumulation_steps", 1),
        )
        max_train_steps = self.config.get("max_train_steps")

        if max_train_steps is None:
            max_train_steps = (
                self.config.get("num_epochs", 10) * num_update_steps_per_epoch
            )
        else:
            num_epochs = max_train_steps // num_update_steps_per_epoch

        self.logger.info(f"Training configuration:")
        self.logger.info(
            f"  - Total batch size: {self.config.get('train_batch_size', 1) * self.accelerator.num_processes}"
        )
        self.logger.info(
            f"  - Gradient accumulation steps: {self.config.get('gradient_accumulation_steps', 1)}"
        )
        self.logger.info(f"  - Learning rate: {self.config.get('learning_rate', 1e-4)}")
        self.logger.info(f"  - Max training steps: {max_train_steps}")
        self.logger.info(f"  - Number of epochs: {num_epochs}")

        # Training loop
        training_logs = []
        self.global_step = 0

        for epoch in range(num_epochs):
            self.epoch = epoch
            self.unet.train()

            epoch_loss = 0.0
            progress_bar = range(len(train_dataloader))

            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    # Forward pass
                    loss = self._training_step(batch)

                    # Backward pass
                    self.accelerator.backward(loss)

                    # Gradient clipping
                    if self.config.get("max_grad_norm") is not None:
                        params_to_clip = self.unet.parameters()
                        self.accelerator.clip_grad_norm_(
                            params_to_clip, self.config["max_grad_norm"]
                        )

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Logging
                epoch_loss += loss.detach().item()

                if self.accelerator.sync_gradients:
                    self.global_step += 1

                    # Progress callback
                    if progress_callback and self.global_step % 10 == 0:
                        progress_callback(
                            {
                                "step": self.global_step,
                                "epoch": epoch,
                                "loss": loss.detach().item(),
                                "lr": lr_scheduler.get_last_lr()[0],
                                "elapsed": time.time() - start_time,
                            }
                        )

                # Save checkpoint
                if self.global_step % self.config.get("checkpointing_steps", 500) == 0:
                    self._save_checkpoint(f"checkpoint-{self.global_step}")

                # Validation and sampling
                if self.global_step % self.config.get("validation_steps", 500) == 0:
                    self._run_validation(epoch, self.global_step)

                # Early stopping check
                if self.global_step >= max_train_steps:
                    break

            # End of epoch
            avg_loss = epoch_loss / len(train_dataloader)

            log_entry = {
                "epoch": epoch,
                "global_step": self.global_step,
                "train_loss": avg_loss,
                "learning_rate": lr_scheduler.get_last_lr()[0],
                "elapsed_time": time.time() - start_time,
            }
            training_logs.append(log_entry)

            self.logger.info(
                f"Epoch {epoch}: loss={avg_loss:.4f}, lr={lr_scheduler.get_last_lr()[0]:.2e}"
            )

            # Save best checkpoint
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self._save_checkpoint("best")

            if self.global_step >= max_train_steps:
                break

        # Final save
        self._save_checkpoint("final")

        # Save training logs
        with open(self.output_dir / "training_logs.json", "w") as f:
            json.dump(training_logs, f, indent=2)

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")

        return {
            "status": "completed",
            "total_steps": self.global_step,
            "final_loss": self.best_loss,
            "training_time": total_time,
            "output_dir": str(self.output_dir),
        }

    def _training_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Single training step"""
        # Get images and captions
        images = batch["images"]
        captions = batch["captions"]

        # Convert images to tensors and move to device
        if isinstance(images[0], Image.Image):
            # Convert PIL images to tensors
            import torchvision.transforms as T

            transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
            pixel_values = torch.stack([transform(img) for img in images])
        else:
            pixel_values = torch.stack(images)

        pixel_values = pixel_values.to(self.accelerator.device, dtype=torch.float32)

        # Encode images to latents
        latents = self.vae.encode(pixel_values).latent_dist.sample()  # type: ignore
        latents = latents * self.vae.config.scaling_factor  # type: ignore

        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Sample timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,  # type: ignore
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)  # type: ignore

        # Encode text
        text_embeddings = self._encode_prompt(captions)

        # Predict noise
        noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample  # type: ignore

        # Calculate loss
        if self.config.get("snr_gamma") is not None:
            # Use SNR weighting
            snr = self._compute_snr(timesteps)
            mse_loss_weights = self._get_snr_weights(snr, self.config["snr_gamma"])
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        else:
            # Standard MSE loss
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        return loss

    def _encode_prompt(self, prompts: List[str]) -> torch.Tensor:
        """Encode text prompts"""
        # Tokenize
        text_inputs = self.tokenizer(  # type: ignore
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,  # type: ignore
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(self.accelerator.device)

        # Encode with text encoder
        text_embeddings = self.text_encoder(text_input_ids)[0]  # type: ignore

        # For SDXL, also encode with second text encoder
        if self.is_sdxl:
            text_inputs_2 = self.tokenizer_2(
                prompts,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids_2 = text_inputs_2.input_ids.to(self.accelerator.device)
            text_embeddings_2 = self.text_encoder_2(text_input_ids_2)[0]  # type: ignore

            # Concatenate embeddings
            text_embeddings = torch.cat([text_embeddings, text_embeddings_2], dim=-1)

        return text_embeddings

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer for training"""
        # Get trainable parameters
        params_to_optimize = self.unet.parameters()  # type: ignore

        # Choose optimizer
        optimizer_name = self.config.get("optimizer", "adamw")
        learning_rate = self.config.get("learning_rate", 1e-4)

        if optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=learning_rate,
                betas=(
                    self.config.get("adam_beta1", 0.9),
                    self.config.get("adam_beta2", 0.999),
                ),
                weight_decay=self.config.get("adam_weight_decay", 1e-2),
                eps=self.config.get("adam_epsilon", 1e-08),
            )
        elif optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(params_to_optimize, lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        return optimizer

    def _setup_lr_scheduler(
        self, optimizer: torch.optim.Optimizer, num_training_steps_per_epoch: int
    ):
        """Setup learning rate scheduler"""
        lr_scheduler_name = self.config.get("lr_scheduler", "constant")

        if lr_scheduler_name == "constant":
            lr_scheduler = get_scheduler(
                "constant",
                optimizer=optimizer,
                num_warmup_steps=self.config.get("lr_warmup_steps", 0),
                num_training_steps=self.config.get("max_train_steps", 1000),
            )
        elif lr_scheduler_name == "cosine":
            lr_scheduler = get_scheduler(
                "cosine",
                optimizer=optimizer,
                num_warmup_steps=self.config.get("lr_warmup_steps", 0),
                num_training_steps=self.config.get("max_train_steps", 1000),
            )
        else:
            lr_scheduler = get_scheduler(
                lr_scheduler_name,
                optimizer=optimizer,
                num_warmup_steps=self.config.get("lr_warmup_steps", 0),
                num_training_steps=self.config.get("max_train_steps", 1000),
            )

        return lr_scheduler

    def _compute_snr(self, timesteps):
        """Compute signal-to-noise ratio for timesteps"""
        alphas_cumprod = self.noise_scheduler.alphas_cumprod  # type: ignore
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
            timesteps
        ].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
            device=timesteps.device
        )[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]

        snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
        return snr

    def _get_snr_weights(self, snr, gamma):
        """Get SNR-based loss weights"""
        snr_clamped = torch.clamp(snr, min=gamma)
        weights = torch.minimum(snr, snr_clamped) / snr
        return weights

    def _save_checkpoint(self, checkpoint_name: str):
        """Save LoRA weights and training state"""
        checkpoint_dir = self.output_dir / "checkpoints" / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        unwrapped_unet = self.accelerator.unwrap_model(self.unet)
        unwrapped_unet.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "config": self.config,
            "model_name": self.base_model,
        }

        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        # Save accelerator state
        self.accelerator.save_state(checkpoint_dir / "accelerator_state")  # type: ignore

        self.logger.info(f"Saved checkpoint: {checkpoint_name}")

    def _run_validation(self, epoch: int, step: int):
        """Run validation and generate samples"""
        self.logger.info(f"Running validation at step {step}")

        validation_prompts = self.config.get(
            "validation_prompts",
            ["anime girl with blue hair", "portrait of a character in school uniform"],
        )

        samples_dir = self.output_dir / "samples" / f"step_{step}"
        samples_dir.mkdir(parents=True, exist_ok=True)

        # Create inference pipeline
        pipeline = self._create_inference_pipeline()
        if pipeline is None:
            self.logger.warning("Could not create validation pipeline")
            return

        try:
            # Generate validation samples
            validation_images = []
            for i, prompt in enumerate(validation_prompts):
                image = pipeline(
                    prompt,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    generator=torch.Generator().manual_seed(42),
                ).images[  # type: ignore
                    0
                ]

                sample_path = samples_dir / f"sample_{i:02d}.png"
                image.save(sample_path)
                validation_images.append(image)

                self.logger.info(f"Generated validation sample: {sample_path.name}")

            # Run evaluation metrics
            if len(validation_images) > 1:
                # CLIP evaluation
                try:
                    clip_scores = self.clip_evaluator.compute_text_image_similarity(
                        validation_images, validation_prompts
                    )
                    avg_clip_score = np.mean(clip_scores)

                    self.logger.info(f"Average CLIP score: {avg_clip_score:.3f}")

                    # Save metrics
                    metrics = {
                        "step": step,
                        "epoch": epoch,
                        "clip_scores": clip_scores,
                        "avg_clip_score": avg_clip_score,
                    }

                    metrics_file = samples_dir / "metrics.json"
                    with open(metrics_file, "w") as f:
                        json.dump(metrics, f, indent=2)

                except Exception as e:
                    self.logger.warning(f"CLIP evaluation failed: {e}")

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
        finally:
            # Clean up pipeline
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _create_inference_pipeline(self):
        """Create inference pipeline for validation"""
        try:
            if self.is_sdxl:
                pipeline_class = StableDiffusionXLPipeline
            else:
                pipeline_class = StableDiffusionPipeline

            # Create pipeline with current LoRA weights
            pipeline = pipeline_class.from_pretrained(
                self.base_model, torch_dtype=torch.float16, use_safetensors=True
            )

            # Load LoRA weights
            unwrapped_unet = self.accelerator.unwrap_model(self.unet)
            pipeline.unet = unwrapped_unet

            pipeline = pipeline.to(self.accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

            return pipeline

        except Exception as e:
            self.logger.error(f"Failed to create inference pipeline: {e}")
            return None
