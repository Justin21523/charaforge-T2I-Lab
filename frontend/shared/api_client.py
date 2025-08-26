# frontend/shared/api_client.py
"""
SagaForge T2I Lab - Python API Client
Shared API client for PyQt and Gradio frontends
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
        self.timeout = timeout  # 30 seconds
        self.session = requests.Session()

    def request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to API"""
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(
                method=method, url=url, timeout=self.timeout, **kwargs
            )

            if response.status_code == 200:
                return response.json()
            else:
                error_data = response.json() if response.content else {}
                error_message = error_data.get(
                    "message", f"HTTP {response.status_code}"
                )
                return {"status": "error", "message": error_message}

        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}
        except json.JSONDecodeError:
            return {"status": "error", "message": "Invalid JSON response"}

    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        return self.request("GET", "/api/v1/health")

    def generate_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image with T2I"""
        return self.request("POST", "/api/v1/t2i/generate", json=params)

    def controlnet_generate(
        self, params: Dict[str, Any], control_type: str = "pose"
    ) -> Dict[str, Any]:
        """Generate image with ControlNet"""
        return self.request("POST", f"/api/v1/controlnet/{control_type}", json=params)

    def list_loras(self) -> List[Dict[str, Any]]:
        """List available LoRA models"""
        response = self.request("GET", "/api/v1/lora/list")
        if response.get("status") == "error":
            return []
        return response.get("loras", response if isinstance(response, list) else [])

    def load_lora(self, lora_id: str, weight: float = 1.0) -> Dict[str, Any]:
        """Load LoRA model"""
        return self.request(
            "POST", "/api/v1/lora/load", json={"lora_id": lora_id, "weight": weight}
        )

    def unload_lora(self, lora_id: str) -> Dict[str, Any]:
        """Unload LoRA model"""
        return self.request("POST", "/api/v1/lora/unload", json={"lora_id": lora_id})

    def get_lora_status(self) -> Dict[str, Any]:
        """Get loaded LoRA status"""
        return self.request("GET", "/api/v1/lora/status")

    def submit_batch_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit batch generation job"""
        return self.request("POST", "/api/v1/batch/submit", json=job_data)

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get batch job status"""
        return self.request("GET", f"/api/v1/batch/status/{job_id}")

    def list_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List batch jobs"""
        endpoint = "/api/v1/batch/jobs"
        if status:
            endpoint += f"?status={status}"
        response = self.request("GET", endpoint)
        if response.get("status") == "error":
            return []
        return response.get("jobs", response if isinstance(response, list) else [])

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel batch job"""
        return self.request("POST", f"/api/v1/batch/cancel/{job_id}")

    def submit_training_job(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Submit LoRA training job"""
        return self.request("POST", "/api/v1/finetune/lora/train", json=config)

    def get_training_status(self, run_id: str) -> Dict[str, Any]:
        """Get training job status"""
        return self._request("GET", f"/api/v1/finetune/lora/status/{run_id}")

    def list_training_jobs(self) -> List[Dict[str, Any]]:
        """List training jobs"""
        response = self.request("GET", "/api/v1/finetune/lora/jobs")
        if response.get("status") == "error":
            return []
        return response.get("jobs", response if isinstance(response, list) else [])

    def cancel_training(self, run_id: str) -> Dict[str, Any]:
        """Cancel training job"""
        return self.request("POST", f"/api/v1/finetune/lora/cancel/{run_id}")

    def get_training_metrics(self, run_id: str) -> Dict[str, Any]:
        """Get training metrics"""
        return self.request("GET", f"/api/v1/finetune/lora/metrics/{run_id}")

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List available datasets"""
        response = self.request("GET", "/api/v1/datasets/list")
        if response.get("status") == "error":
            return []
        return response.get("datasets", response if isinstance(response, list) else [])

    def upload_file(self, file_path: str, file_type: str = "image") -> Dict[str, Any]:
        """Upload file"""
        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                data = {"file_type": file_type}
                response = self.session.post(
                    f"{self.base_url}/api/v1/upload",
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "status": "error",
                    "message": f"Upload failed: {response.status_code}",
                }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return self.request("GET", "/api/v1/monitoring/status")

    def get_resource_usage(self) -> Dict[str, Any]:
        """Get resource usage"""
        return self.request("GET", "/api/v1/monitoring/resources")


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
