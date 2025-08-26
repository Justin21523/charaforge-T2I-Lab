# frontend/shared/api_client.py
"""
Shared API client for SagaForge T2I Lab
Supports both sync and async operations
"""
import requests
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SagaForgeAPIClient:
    """Sync API client for SagaForge T2I Lab"""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "message": str(e)}

    def generate_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate single image"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/t2i/generate",
                json=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise

    def controlnet_generate(
        self, params: Dict[str, Any], control_type: str = "pose"
    ) -> Dict[str, Any]:
        """Generate image with ControlNet"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/controlnet/{control_type}",
                json=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"ControlNet generation failed: {e}")
            raise

    def list_loras(self) -> List[Dict[str, Any]]:
        """List available LoRA models"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/lora/list")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to list LoRAs: {e}")
            return []

    def load_lora(self, lora_id: str, weight: float = 1.0) -> Dict[str, Any]:
        """Load LoRA model"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/lora/load",
                json={"lora_id": lora_id, "weight": weight},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to load LoRA: {e}")
            raise

    def unload_lora(self, lora_id: str) -> Dict[str, Any]:
        """Unload LoRA model"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/lora/unload", json={"lora_id": lora_id}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to unload LoRA: {e}")
            raise

    def submit_batch_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit batch generation job"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/batch/submit",
                json=job_data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to submit batch job: {e}")
            raise

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get batch job status"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/batch/status/{job_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            raise

    def submit_training_job(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Submit LoRA training job"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/finetune/lora/train",
                json=config,
                timeout=10,  # Training job submission should be quick
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to submit training job: {e}")
            raise

    def get_training_status(self, run_id: str) -> Dict[str, Any]:
        """Get training job status"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/finetune/lora/status/{run_id}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get training status: {e}")
            raise

    def upload_file(self, file_path: Path, file_type: str = "image") -> Dict[str, Any]:
        """Upload file to server"""
        try:
            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f, f"image/{file_path.suffix[1:]}")}
                data = {"file_type": file_type}
                response = self.session.post(
                    f"{self.base_url}/api/v1/upload",
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            raise


class AsyncSagaForgeAPIClient:
    """Async API client for SagaForge T2I Lab"""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(f"{self.base_url}/api/v1/health") as response:
                    response.raise_for_status()
                    return await response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "message": str(e)}

    async def generate_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate single image"""
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.base_url}/api/v1/t2i/generate", json=params
                ) as response:
                    response.raise_for_status()
                    return await response.json()
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise
