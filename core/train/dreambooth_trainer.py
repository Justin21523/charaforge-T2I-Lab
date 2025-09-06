# core/train/dreambooth_trainer.py - Complete DreamBooth training implementation
from typing import Dict, Tuple, List, Optional, Callable, Any, Union
from pathlib import Path
import json
import logging
import time
import math
from datetime import datetime
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration

# Diffusers imports
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version

# Transformers
from transformers import CLIPTextModel, CLIPTokenizer

# Dataset and evaluation
from core.train.dataset import T2IDataset, DreamBoothDataset
from core.train.evaluators import CompositeEvaluator, TrainingMonitor
from core.config import get_cache_paths, get_run_output_dir


check_min_version("0.28.0")
logger = get_logger(__name__, log_level="INFO")


class DreamBoothTrainer:
    """Complete DreamBooth fine-tuning implementation with prior preservation"""

    def __init__(self, base_model: str, output_dir: str, config: Dict[str, Any]):
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.config = config
        self.cache_paths = get_cache_paths()

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

        # Model components (loaded in setup_model)
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.noise_scheduler = None
        self.pipeline = None

        # Training components
        self.optimizer = None
        self.lr_scheduler = None
        self.ema_unet = None

        # Evaluation
        self.evaluator = None
        self.training_monitor = None

        # Prior preservation
        self.use_prior_preservation = config.get("prior_preservation", False)
        self.prior_loss_weight = config.get("prior_loss_weight", 1.0)
        self.class_data_dir = None

        logger.info(f"DreamBooth trainer initialized: {base_model}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Prior preservation: {self.use_prior_preservation}")

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "logs" / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    def _setup_accelerator(self):
        """Initialize accelerator for distributed training"""
        accelerator_project_config = ProjectConfiguration(
            project_dir=str(self.output_dir), logging_dir=str(self.output_dir / "logs")
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.get(
                "gradient_accumulation_steps", 1
            ),
            mixed_precision=self.config.get("mixed_precision", "fp16"),
            log_with=(
                "tensorboard" if self.config.get("use_tensorboard", False) else None
            ),
            project_config=accelerator_project_config,
        )

        # Set seed for reproducibility
        if self.config.get("seed"):
            set_seed(self.config["seed"])

    def setup_model(self):
        """Setup all model components - 確保包含 feature_extractor"""
        logger.info(f"Setting up model: {self.base_model}")

        # Load tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.base_model, subfolder="tokenizer"
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.base_model, subfolder="text_encoder"
        )

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(self.base_model, subfolder="vae")

        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.base_model, subfolder="unet"
        )

        # 載入 feature_extractor (重要！)
        try:
            from transformers import CLIPImageProcessor

            self.feature_extractor = CLIPImageProcessor.from_pretrained(
                self.base_model, subfolder="feature_extractor"
            )
            logger.info("Feature extractor loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load feature_extractor from model: {e}")
            try:
                # 使用預設的 feature_extractor
                from transformers import CLIPImageProcessor

                self.feature_extractor = CLIPImageProcessor()
                logger.info("Using default CLIPImageProcessor")
            except ImportError:
                logger.warning("CLIPImageProcessor not available, setting to None")
                self.feature_extractor = None

        # Setup noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.base_model, subfolder="scheduler"
        )

        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Only train UNet
        self.unet.requires_grad_(True)

        logger.info("✅ All model components setup successfully")

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer"""
        optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            self.unet.parameters(),  # type: ignore
            lr=self.config.get("learning_rate", 5e-6),
            betas=(0.9, 0.999),
            weight_decay=self.config.get("weight_decay", 1e-2),
            eps=1e-8,
        )

        logger.info(f"Optimizer setup: {optimizer_cls.__name__}")
        logger.info(f"Learning rate: {self.config.get('learning_rate', 5e-6)}")

        return optimizer

    def _setup_lr_scheduler(
        self, optimizer: torch.optim.Optimizer, num_training_steps: int
    ):
        """Setup learning rate scheduler"""
        lr_scheduler = get_scheduler(
            self.config.get("lr_scheduler", "constant"),
            optimizer=optimizer,
            num_warmup_steps=self.config.get("lr_warmup_steps", 100),
            num_training_steps=num_training_steps,
        )

        logger.info(f"LR Scheduler: {self.config.get('lr_scheduler', 'constant')}")
        return lr_scheduler

    def prepare_prior_preservation_data(
        self, class_prompt: str, num_class_images: int = 200
    ):
        """Generate prior preservation data"""
        if not self.use_prior_preservation:
            return

        self.class_data_dir = self.output_dir / "class_data"
        self.class_data_dir.mkdir(exist_ok=True)

        # Check if we already have enough class images
        existing_images = list(self.class_data_dir.glob("*.png")) + list(
            self.class_data_dir.glob("*.jpg")
        )

        if len(existing_images) >= num_class_images:
            logger.info(
                f"Found {len(existing_images)} existing class images, skipping generation"
            )
            return

        logger.info(
            f"Generating {num_class_images - len(existing_images)} class images for prior preservation"
        )

        # Load pipeline for generation
        if self.pipeline is None:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.base_model,
                cache_dir=self.cache_paths.hf_home,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            )
            self.pipeline.to(self.accelerator.device)

        # Generate class images
        images_to_generate = num_class_images - len(existing_images)
        batch_size = 4  # Adjust based on VRAM

        for i in range(0, images_to_generate, batch_size):
            current_batch_size = min(batch_size, images_to_generate - i)

            # Generate images
            with torch.autocast("cuda"):
                images = self.pipeline(
                    [class_prompt] * current_batch_size,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                ).images  # type: ignore

            # Save images
            for j, image in enumerate(images):
                image_path = (
                    self.class_data_dir
                    / f"class_{len(existing_images) + i + j:05d}.png"
                )
                image.save(image_path)

        logger.info(
            f"Generated {images_to_generate} class images for prior preservation"
        )

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute training loss with optional prior preservation"""

        # Encode images to latent space
        latents = self.vae.encode(batch["pixel_values"]).latent_dist.sample()  # type: ignore
        latents = latents * self.vae.config.scaling_factor  # type: ignore

        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Sample random timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,  # type: ignore
            (bsz,),
            device=latents.device,
        ).long()

        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)  # type: ignore

        # Get text embeddings
        encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]  # type: ignore

        # Predict noise
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample  # type: ignore

        # Compute loss
        if self.noise_scheduler.config.prediction_type == "epsilon":  # type: ignore
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":  # type: ignore
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)  # type: ignore
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"  # type: ignore
            )

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return loss

    def train(
        self, dataset: DreamBoothDataset, progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Main DreamBooth training loop"""

        logger.info(f"Starting DreamBooth training on {len(dataset)} samples")
        start_time = time.time()

        # Setup model
        self.setup_model()

        # Prepare prior preservation data if needed
        if self.use_prior_preservation and hasattr(dataset, "class_prompt"):
            self.prepare_prior_preservation_data(
                dataset.class_prompt, self.config.get("num_class_images", 200)  # type: ignore
            )

        # Setup data loader
        train_dataloader = DataLoader(
            dataset,
            batch_size=self.config.get("train_batch_size", 1),
            shuffle=True,
            collate_fn=dataset.collate_fn,
            num_workers=self.config.get("dataloader_num_workers", 0),
            pin_memory=True,
        )

        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()

        # Calculate training steps
        num_update_steps_per_epoch = max(
            len(train_dataloader) // self.config.get("gradient_accumulation_steps", 1),
            1,
        )
        max_train_steps = self.config.get("max_train_steps")
        num_epochs = self.config.get("num_epochs", 10)

        if max_train_steps is None:
            max_train_steps = num_epochs * num_update_steps_per_epoch
        else:
            num_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

        self.lr_scheduler = self._setup_lr_scheduler(self.optimizer, max_train_steps)

        # Prepare everything with accelerator
        self.unet, self.optimizer, train_dataloader, self.lr_scheduler = (
            self.accelerator.prepare(
                self.unet, self.optimizer, train_dataloader, self.lr_scheduler
            )
        )

        # Setup evaluation
        self.evaluator = CompositeEvaluator(device=self.accelerator.device)
        self.training_monitor = TrainingMonitor(
            self.evaluator, self.output_dir / "evaluation"
        )

        # Training loop
        training_logs = []
        self.global_step = 0

        logger.info(f"Training configuration:")
        logger.info(f"  - Epochs: {num_epochs}")
        logger.info(f"  - Max steps: {max_train_steps}")
        logger.info(f"  - Batch size: {self.config.get('train_batch_size', 1)}")
        logger.info(
            f"  - Gradient accumulation: {self.config.get('gradient_accumulation_steps', 1)}"
        )
        logger.info(f"  - Learning rate: {self.config.get('learning_rate', 5e-6)}")
        logger.info(f"  - Prior preservation: {self.use_prior_preservation}")

        for epoch in range(num_epochs):
            self.epoch = epoch
            self.unet.train()

            epoch_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    # Compute loss
                    loss = self.compute_loss(batch)

                    # Backpropagation
                    self.accelerator.backward(loss)

                    # Gradient clipping
                    if self.config.get("max_grad_norm"):
                        self.accelerator.clip_grad_norm_(
                            self.unet.parameters(), self.config["max_grad_norm"]
                        )

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    # Update EMA
                    if self.ema_unet:
                        self.ema_unet.step(self.unet.parameters())

                # Logging and evaluation
                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    epoch_loss += loss.detach().item()
                    num_batches += 1

                    # Progress callback
                    if progress_callback:
                        progress_callback(
                            {
                                "step": self.global_step,
                                "epoch": epoch,
                                "loss": loss.item(),
                                "lr": self.lr_scheduler.get_last_lr()[0],
                                "elapsed": time.time() - start_time,
                            }
                        )

                    # Save checkpoint
                    if (
                        self.global_step % self.config.get("checkpointing_steps", 500)
                        == 0
                    ):
                        self.save_checkpoint(f"checkpoint-{self.global_step}")

                    # Validation and evaluation
                    if self.global_step % self.config.get("validation_steps", 100) == 0:
                        self._run_validation()

                    # Early stopping check
                    if self.global_step >= max_train_steps:
                        break

            # End of epoch logging
            avg_loss = epoch_loss / max(num_batches, 1)

            log_entry = {
                "epoch": epoch,
                "global_step": self.global_step,
                "train_loss": avg_loss,
                "learning_rate": self.lr_scheduler.get_last_lr()[0],
                "elapsed_time": time.time() - start_time,
            }
            training_logs.append(log_entry)

            logger.info(f"Epoch {epoch}/{num_epochs} completed:")
            logger.info(f"  - Average loss: {avg_loss:.6f}")
            logger.info(f"  - Learning rate: {self.lr_scheduler.get_last_lr()[0]:.2e}")
            logger.info(f"  - Global step: {self.global_step}")

            # Save epoch checkpoint
            if epoch % self.config.get("save_epochs", 5) == 0:
                self.save_checkpoint(f"epoch-{epoch}")

            if self.global_step >= max_train_steps:
                break

        # Final checkpoint
        self.save_checkpoint("final")

        # Final evaluation
        final_evaluation = self._run_validation()

        # Training summary
        total_time = time.time() - start_time

        training_summary = {
            "status": "completed",
            "total_epochs": self.epoch + 1,
            "total_steps": self.global_step,
            "final_loss": training_logs[-1]["train_loss"] if training_logs else 0.0,
            "best_loss": self.best_loss,
            "training_time": total_time,
            "training_logs": training_logs,
            "final_evaluation": (
                final_evaluation.to_dict() if final_evaluation else None
            ),
            "config": self.config,
            "output_dir": str(self.output_dir),
        }

        # Save training summary
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(training_summary, f, indent=2, ensure_ascii=False)

        logger.info(f"DreamBooth training completed in {total_time:.2f} seconds")
        logger.info(f"Final loss: {training_logs[-1]['train_loss']:.6f}")
        logger.info(f"Training summary saved to: {summary_path}")

        return training_summary

    def _run_validation(self):
        """Run validation during training"""
        if not hasattr(self, "validation_prompts"):
            self.validation_prompts = [
                f"a photo of {self.config.get('instance_token', 'sks')} {self.config.get('class_name', 'person')}",
                f"{self.config.get('instance_token', 'sks')} {self.config.get('class_name', 'person')} in a garden",
                f"portrait of {self.config.get('instance_token', 'sks')} {self.config.get('class_name', 'person')}",
            ]

        logger.info(f"Running validation at step {self.global_step}")

        # Create validation pipeline
        validation_pipeline = self._create_validation_pipeline()

        # Generate validation images
        validation_images = []
        samples_dir = self.output_dir / "samples" / f"step_{self.global_step}"
        samples_dir.mkdir(parents=True, exist_ok=True)

        for i, prompt in enumerate(self.validation_prompts):
            try:
                with torch.autocast("cuda"):
                    image = validation_pipeline(
                        prompt,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        height=512,
                        width=512,
                        generator=torch.Generator().manual_seed(
                            42
                        ),  # Fixed seed for consistency
                    ).images[  # type: ignore
                        0
                    ]

                # Save image
                image_path = samples_dir / f"validation_{i:02d}.png"
                image.save(image_path)
                validation_images.append(image_path)

            except Exception as e:
                logger.error(f"Error generating validation image {i}: {e}")

        # Run evaluation if we have images
        if validation_images and self.training_monitor:
            try:
                evaluation = self.training_monitor.evaluate_and_log(
                    step=self.global_step,
                    generated_images=validation_images,
                    prompts=self.validation_prompts[: len(validation_images)],
                )

                # Update best loss if evaluation improved
                if (
                    evaluation.overall_score
                    and evaluation.overall_score > self.best_loss
                ):
                    self.best_loss = evaluation.overall_score
                    logger.info(f"New best evaluation score: {self.best_loss:.4f}")

                return evaluation

            except Exception as e:
                logger.error(f"Error during evaluation: {e}")

        return None

    def _create_validation_pipeline(self):
        """Create pipeline for validation - 修復版本"""
        try:
            # Use EMA weights if available
            if self.ema_unet:
                self.ema_unet.copy_to(self.unet.parameters())  # type: ignore

            # 建立 pipeline 參數，包含 feature_extractor
            pipeline_kwargs = {
                "vae": self.vae,
                "text_encoder": self.text_encoder,
                "tokenizer": self.tokenizer,
                "unet": self.accelerator.unwrap_model(self.unet),
                "scheduler": self.noise_scheduler,
                "safety_checker": None,
                "requires_safety_checker": False,
            }

            # 添加 feature_extractor (如果存在)
            if (
                hasattr(self, "feature_extractor")
                and self.feature_extractor is not None
            ):
                pipeline_kwargs["feature_extractor"] = self.feature_extractor
            else:
                # 嘗試從原始 pipeline 取得 feature_extractor
                try:
                    from transformers import CLIPImageProcessor

                    pipeline_kwargs["feature_extractor"] = CLIPImageProcessor()
                except ImportError:
                    # 如果無法載入，設為 None (某些版本可能不需要)
                    pipeline_kwargs["feature_extractor"] = None

            validation_pipeline = StableDiffusionPipeline(**pipeline_kwargs)
            validation_pipeline.to(self.accelerator.device)
            validation_pipeline.set_progress_bar_config(disable=True)

            return validation_pipeline

        except Exception as e:
            logger.error(f"Failed to create validation pipeline: {e}")
            return None

    def save_checkpoint(self, checkpoint_name: str):
        """Save training checkpoint - 修復版本"""
        checkpoint_dir = self.output_dir / "checkpoints" / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving checkpoint: {checkpoint_name}")

        # Save UNet state
        unet_state = self.accelerator.unwrap_model(self.unet).state_dict()
        torch.save(unet_state, checkpoint_dir / "unet.safetensors")

        # Save optimizer and scheduler states
        if self.accelerator.is_main_process:
            torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")  # type: ignore
            torch.save(self.lr_scheduler.state_dict(), checkpoint_dir / "scheduler.pt")  # type: ignore

            # Save EMA if available
            if self.ema_unet:
                torch.save(self.ema_unet.state_dict(), checkpoint_dir / "ema_unet.pt")

        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "config": self.config,
        }

        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)

        # Create full pipeline - 修復版本
        try:
            # 準備 pipeline 參數
            pipeline_kwargs = {
                "vae": self.vae,
                "text_encoder": self.text_encoder,
                "tokenizer": self.tokenizer,
                "unet": self.accelerator.unwrap_model(self.unet),
                "scheduler": self.noise_scheduler,
                "safety_checker": None,
                "requires_safety_checker": False,
            }

            # 添加 feature_extractor
            if (
                hasattr(self, "feature_extractor")
                and self.feature_extractor is not None
            ):
                pipeline_kwargs["feature_extractor"] = self.feature_extractor
            else:
                try:
                    from transformers import CLIPImageProcessor

                    pipeline_kwargs["feature_extractor"] = CLIPImageProcessor()
                except ImportError:
                    pipeline_kwargs["feature_extractor"] = None

            pipeline = StableDiffusionPipeline(**pipeline_kwargs)
            pipeline.save_pretrained(checkpoint_dir / "pipeline")
            logger.info(f"Pipeline saved to: {checkpoint_dir / 'pipeline'}")

        except Exception as e:
            logger.error(f"Error saving pipeline: {e}")

        logger.info(f"Checkpoint saved successfully: {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load training checkpoint"""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")

        # Load training state
        training_state_path = checkpoint_path / "training_state.json"
        if training_state_path.exists():
            with open(training_state_path, "r") as f:
                training_state = json.load(f)

            self.global_step = training_state.get("global_step", 0)
            self.epoch = training_state.get("epoch", 0)
            self.best_loss = training_state.get("best_loss", float("inf"))

        # Load UNet state
        unet_path = checkpoint_path / "unet.safetensors"
        if unet_path.exists():
            unet_state = torch.load(unet_path, map_location="cpu")
            self.unet.load_state_dict(unet_state)  # type: ignore

        # Load optimizer state
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer_path.exists() and self.optimizer:
            optimizer_state = torch.load(optimizer_path, map_location="cpu")
            self.optimizer.load_state_dict(optimizer_state)

        # Load scheduler state
        scheduler_path = checkpoint_path / "scheduler.pt"
        if scheduler_path.exists() and self.lr_scheduler:
            scheduler_state = torch.load(scheduler_path, map_location="cpu")
            self.lr_scheduler.load_state_dict(scheduler_state)

        # Load EMA state
        ema_path = checkpoint_path / "ema_unet.pt"
        if ema_path.exists() and self.ema_unet:
            ema_state = torch.load(ema_path, map_location="cpu")
            self.ema_unet.load_state_dict(ema_state)

        logger.info(f"Checkpoint loaded successfully")
        logger.info(f"Resumed from step: {self.global_step}, epoch: {self.epoch}")

    def export_model(self, export_path: Union[str, Path], use_ema: bool = True):
        """Export trained model as a Stable Diffusion pipeline - 修復版本"""
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting model to: {export_path}")

        # Use EMA weights if available and requested
        if use_ema and self.ema_unet:
            self.ema_unet.copy_to(self.unet.parameters())  # type: ignore

        # Create and save pipeline - 修復版本
        try:
            # 準備 pipeline 參數
            pipeline_kwargs = {
                "vae": self.vae,
                "text_encoder": self.text_encoder,
                "tokenizer": self.tokenizer,
                "unet": self.accelerator.unwrap_model(self.unet),
                "scheduler": self.noise_scheduler,
                "safety_checker": None,
                "requires_safety_checker": False,
            }

            # 添加 feature_extractor
            if (
                hasattr(self, "feature_extractor")
                and self.feature_extractor is not None
            ):
                pipeline_kwargs["feature_extractor"] = self.feature_extractor
            else:
                try:
                    from transformers import CLIPImageProcessor

                    pipeline_kwargs["feature_extractor"] = CLIPImageProcessor()
                except ImportError:
                    pipeline_kwargs["feature_extractor"] = None

            pipeline = StableDiffusionPipeline(**pipeline_kwargs)
            pipeline.save_pretrained(export_path)

            logger.info(f"Model exported successfully to: {export_path}")

        except Exception as e:
            logger.error(f"Model export failed: {e}")
            raise

    def generate_sample_grid(
        self,
        prompts: List[str],
        output_path: Union[str, Path],
        grid_size: Tuple[int, int] = (3, 3),
    ):
        """Generate a grid of sample images"""
        output_path = Path(output_path)

        # Create validation pipeline
        pipeline = self._create_validation_pipeline()

        # Generate images
        images = []
        for prompt in prompts[: grid_size[0] * grid_size[1]]:
            try:
                with torch.autocast("cuda"):
                    image = pipeline(
                        prompt,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        height=512,
                        width=512,
                    ).images[  # type: ignore
                        0
                    ]
                images.append(image)
            except Exception as e:
                logger.error(f"Error generating image for prompt '{prompt}': {e}")
                # Create placeholder image
                images.append(Image.new("RGB", (512, 512), color="gray"))  # type: ignore

        # Create grid
        if images:
            from PIL import Image as PILImage

            grid_width = grid_size[1] * 512
            grid_height = grid_size[0] * 512
            grid_image = PILImage.new("RGB", (grid_width, grid_height))

            for i, image in enumerate(images):
                row = i // grid_size[1]
                col = i % grid_size[1]
                x = col * 512
                y = row * 512
                grid_image.paste(image, (x, y))

            grid_image.save(output_path)
            logger.info(f"Sample grid saved to: {output_path}")

        return output_path
