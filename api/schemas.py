# api/schemas.py - Updated Pydantic Schemas
"""
API 資料模型定義 - 根據最新架構調整
支援 T2I 生成、LoRA 訓練、批次處理等功能
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum


# ===== Base Models =====


class StatusEnum(str, Enum):
    """狀態枚舉"""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelTypeEnum(str, Enum):
    """模型類型枚舉"""

    SD15 = "sd15"
    SDXL = "sdxl"


class TrainingTypeEnum(str, Enum):
    """訓練類型枚舉"""

    LORA = "lora"
    DREAMBOOTH = "dreambooth"


# ===== Health & System Schemas =====


class HealthResponse(BaseModel):
    """健康檢查回應"""

    status: str = Field(..., description="System health status")
    timestamp: str = Field(..., description="Check timestamp")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Runtime environment")
    components: Dict[str, Any] = Field(default={}, description="Component status")
    performance: Optional[Dict[str, Any]] = Field(default=None)
    cache: Optional[Dict[str, Any]] = Field(default=None)
    errors: Optional[Dict[str, Any]] = Field(default=None)


class SystemStatusResponse(BaseModel):
    """系統狀態回應"""

    status: str = Field(..., description="Overall system status")
    device: Dict[str, Any] = Field(..., description="Device configuration")
    memory_usage: Dict[str, float] = Field(default={}, description="Memory usage stats")
    active_jobs: int = Field(default=0, description="Number of active jobs")
    queue_length: int = Field(default=0, description="Queue length")
    cache_size_gb: float = Field(default=0.0, description="Cache size in GB")


# ===== T2I Generation Schemas =====


class LoRAWeight(BaseModel):
    """LoRA 權重配置"""

    id: str = Field(..., description="LoRA model ID")
    weight: float = Field(default=1.0, ge=0.0, le=2.0, description="LoRA weight")


class GenerationRequest(BaseModel):
    """圖片生成請求"""

    # Core parameters
    prompt: str = Field(
        ..., min_length=1, max_length=2000, description="Generation prompt"
    )
    negative_prompt: Optional[str] = Field(
        default="", max_length=1000, description="Negative prompt"
    )

    # Model selection
    model_type: ModelTypeEnum = Field(
        default=ModelTypeEnum.SD15, description="Base model type"
    )
    model_name: Optional[str] = Field(default=None, description="Specific model name")
    lora_weights: Optional[List[LoRAWeight]] = Field(
        default=[], description="LoRA weights to apply"
    )

    # Generation parameters
    width: int = Field(default=512, ge=256, le=2048, description="Image width")
    height: int = Field(default=512, ge=256, le=2048, description="Image height")
    num_inference_steps: int = Field(
        default=20, ge=5, le=150, description="Inference steps"
    )
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="CFG scale")
    seed: Optional[int] = Field(
        default=None, ge=0, le=2**32 - 1, description="Random seed"
    )

    # Batch generation
    num_images: int = Field(
        default=1, ge=1, le=4, description="Number of images to generate"
    )

    # Advanced options
    scheduler: str = Field(
        default="euler_a",
        pattern="^(euler_a|ddim|dpm|lms)$",
        description="Scheduler type",
    )
    clip_skip: int = Field(default=1, ge=1, le=12, description="CLIP layers to skip")

    # Quality settings
    enable_vae_slicing: bool = Field(default=True, description="Enable VAE slicing")
    enable_xformers: bool = Field(
        default=True, description="Enable xFormers optimization"
    )

    @field_validator("width", "height")
    def validate_dimensions(cls, v):
        if v % 8 != 0:
            raise ValueError(f"Dimensions must be multiples of 8, got {v}")
        return v


class GenerationResponse(BaseModel):
    """圖片生成回應"""

    job_id: str = Field(..., description="Unique job identifier")
    status: StatusEnum = Field(..., description="Generation status")

    # Generation info
    prompt: str = Field(..., description="Used prompt")
    model_used: str = Field(..., description="Model used for generation")
    generation_time_seconds: Optional[float] = Field(
        default=None, description="Generation time"
    )

    # Results
    images: List[str] = Field(default=[], description="Generated image URLs")
    metadata: Dict[str, Any] = Field(default={}, description="Generation metadata")

    # Progress info
    progress_percent: int = Field(
        default=0, ge=0, le=100, description="Progress percentage"
    )
    current_step: Optional[int] = Field(default=0, description="Current step")
    total_steps: Optional[int] = Field(default=0, description="Total steps")

    # Error info
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ControlNetRequest(BaseModel):
    """ControlNet 生成請求"""

    # Base generation parameters
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: Optional[str] = Field(default="")

    # ControlNet specific
    controlnet_type: str = Field(..., pattern="^(canny|depth|pose|scribble|normal)$")
    control_image: str = Field(
        ..., description="Base64 encoded control image or image ID"
    )
    controlnet_conditioning_scale: float = Field(default=1.0, ge=0.0, le=2.0)

    # Generation parameters
    width: int = Field(default=512, ge=256, le=2048)
    height: int = Field(default=512, ge=256, le=2048)
    num_inference_steps: int = Field(default=20, ge=5, le=50)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: Optional[int] = Field(default=None)


class StylePreset(BaseModel):
    """風格預設配置"""

    name: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = Field(default="")

    # Style parameters
    prompt_prefix: str = Field(default="")
    prompt_suffix: str = Field(default="")
    negative_prompt: str = Field(default="")

    # Generation settings
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    num_inference_steps: int = Field(default=20, ge=5, le=150)

    # Model preferences
    preferred_model: Optional[str] = Field(default=None)
    recommended_lora: List[str] = Field(default=[])


# ===== LoRA Management Schemas =====


class LoRALoadRequest(BaseModel):
    """LoRA 載入請求"""

    lora_id: str = Field(..., description="LoRA model ID")
    weight: float = Field(default=1.0, ge=0.0, le=2.0, description="LoRA weight")
    model_type: ModelTypeEnum = Field(
        default=ModelTypeEnum.SD15, description="Target model type"
    )


class LoRAUnloadRequest(BaseModel):
    """LoRA 卸載請求"""

    lora_id: str = Field(..., description="LoRA model ID to unload")
    model_type: ModelTypeEnum = Field(
        default=ModelTypeEnum.SD15, description="Target model type"
    )


class LoRAStatusResponse(BaseModel):
    """LoRA 狀態回應"""

    status: str = Field(..., description="Operation status")
    lora_id: Optional[str] = Field(default=None, description="LoRA ID")
    weight: Optional[float] = Field(default=None, description="Applied weight")
    loaded_loras: Dict[str, float] = Field(
        default={}, description="Currently loaded LoRAs"
    )
    message: Optional[str] = Field(default=None, description="Status message")


class ModelInfo(BaseModel):
    """模型資訊"""

    id: str = Field(..., description="Model ID")
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    size_mb: float = Field(..., description="Model size in MB")
    description: Optional[str] = Field(default="", description="Model description")
    cached_at: Optional[str] = Field(default=None, description="Cache timestamp")


class ModelsResponse(BaseModel):
    """模型列表回應"""

    sd15: List[ModelInfo] = Field(default=[], description="SD 1.5 models")
    sdxl: List[ModelInfo] = Field(default=[], description="SDXL models")
    lora: List[ModelInfo] = Field(default=[], description="LoRA models")
    controlnet: List[ModelInfo] = Field(default=[], description="ControlNet models")
    embedding: List[ModelInfo] = Field(default=[], description="Embedding models")


# ===== Training Schemas =====


class LoRATrainingRequest(BaseModel):
    """LoRA 訓練請求"""

    # Project info
    project_name: str = Field(
        ..., min_length=1, max_length=100, description="Project name"
    )
    description: Optional[str] = Field(
        default="", max_length=500, description="Project description"
    )

    # Base model
    base_model: ModelTypeEnum = Field(
        default=ModelTypeEnum.SD15, description="Base model type"
    )
    base_model_name: Optional[str] = Field(
        default=None, description="Specific base model"
    )

    # Dataset configuration
    dataset_type: str = Field(default="folder", pattern="^(folder|csv|huggingface)$")
    dataset_path: Optional[str] = Field(default=None, description="Dataset path")
    instance_prompt: str = Field(
        ..., min_length=1, max_length=200, description="Instance prompt"
    )
    class_prompt: Optional[str] = Field(
        default=None, max_length=200, description="Class prompt"
    )

    # LoRA parameters
    lora_rank: int = Field(default=16, ge=4, le=128, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=8, le=256, description="LoRA alpha")
    lora_dropout: float = Field(default=0.1, ge=0.0, le=0.5, description="LoRA dropout")
    target_modules: Optional[List[str]] = Field(
        default=None, description="Target modules"
    )

    # Training hyperparameters
    learning_rate: float = Field(
        default=1e-4, ge=1e-6, le=1e-2, description="Learning rate"
    )
    train_batch_size: int = Field(
        default=1, ge=1, le=8, description="Training batch size"
    )
    num_train_epochs: int = Field(
        default=10, ge=1, le=1000, description="Number of epochs"
    )
    max_train_steps: Optional[int] = Field(
        default=None, ge=1, description="Max training steps"
    )

    # Optimization
    gradient_accumulation_steps: int = Field(
        default=1, ge=1, le=32, description="Gradient accumulation"
    )
    gradient_checkpointing: bool = Field(
        default=True, description="Enable gradient checkpointing"
    )
    mixed_precision: str = Field(
        default="fp16", pattern="^(no|fp16|bf16)$", description="Mixed precision"
    )

    # Validation
    validation_steps: int = Field(
        default=100, ge=10, le=1000, description="Validation frequency"
    )
    save_steps: int = Field(default=500, ge=50, le=5000, description="Save frequency")
    validation_prompt: Optional[str] = Field(
        default=None, description="Validation prompt"
    )

    # Advanced options
    use_8bit_adam: bool = Field(default=True, description="Use 8-bit Adam")
    use_xformers: bool = Field(default=True, description="Enable xFormers")
    enable_cpu_offload: bool = Field(default=False, description="Enable CPU offload")


class DreamBoothTrainingRequest(BaseModel):
    """DreamBooth 訓練請求"""

    # Project info
    project_name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(default="", max_length=500)

    # Base model
    base_model: ModelTypeEnum = Field(default=ModelTypeEnum.SD15)
    base_model_name: Optional[str] = Field(default=None)

    # Instance and class configuration
    instance_data_dir: str = Field(..., description="Instance images directory")
    instance_prompt: str = Field(..., min_length=1, max_length=200)
    class_data_dir: Optional[str] = Field(
        default=None, description="Class images directory"
    )
    class_prompt: Optional[str] = Field(default=None, max_length=200)

    # Training parameters
    learning_rate: float = Field(default=5e-6, ge=1e-7, le=1e-3)
    train_batch_size: int = Field(default=1, ge=1, le=4)
    num_train_epochs: int = Field(default=10, ge=1, le=500)
    max_train_steps: Optional[int] = Field(default=None)

    # DreamBooth specific
    prior_loss_weight: float = Field(default=1.0, ge=0.0, le=2.0)
    num_class_images: int = Field(default=50, ge=0, le=200)
    with_prior_preservation: bool = Field(default=True)

    # Advanced options
    gradient_checkpointing: bool = Field(default=True)
    mixed_precision: str = Field(default="fp16", pattern="^(no|fp16|bf16)$")
    use_8bit_adam: bool = Field(default=True)


class TrainingJobResponse(BaseModel):
    """訓練任務回應"""

    job_id: str = Field(..., description="Unique job identifier")
    project_name: str = Field(..., description="Project name")
    training_type: TrainingTypeEnum = Field(..., description="Training type")
    status: StatusEnum = Field(..., description="Training status")

    # Progress information
    progress_percent: int = Field(
        default=0, ge=0, le=100, description="Progress percentage"
    )
    current_epoch: Optional[int] = Field(default=0, description="Current epoch")
    total_epochs: Optional[int] = Field(default=0, description="Total epochs")
    current_step: Optional[int] = Field(default=0, description="Current step")
    total_steps: Optional[int] = Field(default=0, description="Total steps")

    # Training metrics
    current_loss: Optional[float] = Field(default=None, description="Current loss")
    learning_rate: Optional[float] = Field(default=None, description="Learning rate")
    train_time_seconds: Optional[float] = Field(
        default=None, description="Training time"
    )
    estimated_remaining_seconds: Optional[float] = Field(
        default=None, description="Estimated remaining time"
    )

    # Results
    output_model_path: Optional[str] = Field(
        default=None, description="Output model path"
    )
    validation_images: List[str] = Field(
        default=[], description="Validation image URLs"
    )
    training_logs: Optional[str] = Field(default=None, description="Training logs")

    # System info
    gpu_memory_usage_gb: Optional[float] = Field(
        default=None, description="GPU memory usage"
    )

    # Error info
    error_message: Optional[str] = Field(default=None, description="Error message")
    error_traceback: Optional[str] = Field(default=None, description="Error traceback")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class TrainingConfig(BaseModel):
    """訓練配置模板"""

    name: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = Field(default="")
    training_type: TrainingTypeEnum = Field(..., description="Training type")

    # Template parameters
    config: Dict[str, Any] = Field(..., description="Configuration template")
    recommended_for: List[str] = Field(default=[], description="Recommended use cases")


# ===== Dataset Schemas =====


class DatasetValidationResult(BaseModel):
    """資料集驗證結果"""

    valid: bool = Field(..., description="Is dataset valid")
    total_images: int = Field(..., description="Total number of images")
    valid_images: int = Field(..., description="Number of valid images")
    invalid_images: int = Field(..., description="Number of invalid images")

    # Validation details
    missing_captions: int = Field(default=0, description="Images missing captions")
    unsupported_formats: int = Field(default=0, description="Unsupported image formats")
    corrupted_files: int = Field(default=0, description="Corrupted files")

    # Recommendations
    warnings: List[str] = Field(default=[], description="Validation warnings")
    suggestions: List[str] = Field(default=[], description="Improvement suggestions")


class DatasetUploadResponse(BaseModel):
    """資料集上傳回應"""

    project_name: str = Field(..., description="Project name")
    upload_path: str = Field(..., description="Upload directory path")
    files_uploaded: int = Field(..., description="Number of files uploaded")
    total_size_mb: float = Field(..., description="Total size in MB")
    files: List[Dict[str, Any]] = Field(default=[], description="Uploaded file details")


# ===== Batch Processing Schemas =====


class BatchGenerationRequest(BaseModel):
    """批次生成請求"""

    batch_name: str = Field(..., min_length=1, max_length=100)
    prompts: List[str] = Field(..., min_items=1, max_items=100)  # type: ignore

    # Common generation parameters
    model_type: ModelTypeEnum = Field(default=ModelTypeEnum.SD15)
    width: int = Field(default=512, ge=256, le=2048)
    height: int = Field(default=512, ge=256, le=2048)
    num_inference_steps: int = Field(default=20, ge=5, le=150)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)

    # Batch-specific parameters
    seeds: Optional[List[int]] = Field(
        default=None, description="Seeds for each prompt"
    )
    lora_weights: List[LoRAWeight] = Field(default=[])

    # Output options
    save_individually: bool = Field(default=True)
    create_zip: bool = Field(default=False)


class BatchJobResponse(BaseModel):
    """批次任務回應"""

    batch_id: str = Field(..., description="Unique batch identifier")
    batch_name: str = Field(..., description="Batch name")
    status: StatusEnum = Field(..., description="Batch status")

    # Progress information
    total_items: int = Field(..., description="Total items to process")
    completed_items: int = Field(default=0, description="Completed items")
    failed_items: int = Field(default=0, description="Failed items")
    progress_percent: int = Field(
        default=0, ge=0, le=100, description="Progress percentage"
    )

    # Results
    output_directory: Optional[str] = Field(
        default=None, description="Output directory"
    )
    generated_files: List[str] = Field(default=[], description="Generated file paths")
    zip_file: Optional[str] = Field(default=None, description="ZIP file path")

    # Timing
    estimated_time_remaining: Optional[int] = Field(
        default=None, description="ETA in seconds"
    )

    # Error handling
    errors: List[str] = Field(default=[], description="Error messages")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# ===== Safety Schemas =====


class SafetyCheckRequest(BaseModel):
    """安全檢查請求"""

    content: str = Field(..., description="Content to check")
    content_type: str = Field(default="text", pattern="^(text|image)$")
    strict_mode: bool = Field(default=False, description="Enable strict checking")


class SafetyCheckResponse(BaseModel):
    """安全檢查回應"""

    is_safe: bool = Field(..., description="Is content safe")
    risk_level: str = Field(..., pattern="^(low|medium|high|critical)$")

    # Analysis details
    content_hash: str = Field(..., description="Content hash")
    issues: List[str] = Field(default=[], description="Identified issues")
    recommendations: List[str] = Field(default=[], description="Recommendations")

    # NSFW specific (for images)
    nsfw_confidence: Optional[float] = Field(
        default=None, description="NSFW confidence score"
    )

    # Timestamps
    checked_at: datetime = Field(default_factory=datetime.now)


# ===== Export Schemas =====


class ExportRequest(BaseModel):
    """匯出請求"""

    model_id: str = Field(..., description="Model ID to export")
    export_format: str = Field(
        default="safetensors", pattern="^(safetensors|ckpt|onnx|zip)$"
    )
    include_metadata: bool = Field(default=True, description="Include metadata")


class ExportResponse(BaseModel):
    """匯出回應"""

    export_id: str = Field(..., description="Export ID")
    model_id: str = Field(..., description="Source model ID")
    export_format: str = Field(..., description="Export format")

    # Results
    export_path: Optional[str] = Field(default=None, description="Export file path")
    download_url: Optional[str] = Field(default=None, description="Download URL")
    export_size_mb: Optional[float] = Field(
        default=None, description="Export size in MB"
    )

    # Status
    status: StatusEnum = Field(default=StatusEnum.QUEUED, description="Export status")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = Field(default=None)


# ===== Monitoring Schemas =====


class ResourceUsage(BaseModel):
    """資源使用情況"""

    cpu_percent: float = Field(..., description="CPU usage percentage")
    memory_percent: float = Field(..., description="Memory usage percentage")
    memory_used_gb: float = Field(..., description="Memory used in GB")
    memory_available_gb: float = Field(..., description="Available memory in GB")

    # GPU info (optional)
    gpu_memory_allocated_gb: Optional[float] = Field(default=None)
    gpu_memory_total_gb: Optional[float] = Field(default=None)
    gpu_utilization_percent: Optional[float] = Field(default=None)


class QueueStatus(BaseModel):
    """隊列狀態"""

    queue_name: str = Field(..., description="Queue name")
    active_jobs: int = Field(default=0, description="Active jobs")
    pending_jobs: int = Field(default=0, description="Pending jobs")
    completed_jobs: int = Field(default=0, description="Completed jobs")
    failed_jobs: int = Field(default=0, description="Failed jobs")


class MonitoringResponse(BaseModel):
    """監控回應"""

    timestamp: datetime = Field(default_factory=datetime.now)

    # System resources
    system: ResourceUsage = Field(..., description="System resource usage")

    # Queue status
    queues: List[QueueStatus] = Field(default=[], description="Queue statuses")

    # Performance metrics
    api_response_time_ms: Optional[float] = Field(default=None)
    generation_throughput: Optional[float] = Field(default=None)

    # Health indicators
    health_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Overall health score"
    )
    alerts: List[str] = Field(default=[], description="Active alerts")


# ===== Error Schemas =====


class ErrorResponse(BaseModel):
    """錯誤回應"""

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")

    # Context
    request_id: Optional[str] = Field(default=None, description="Request ID")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Recovery suggestions
    suggestions: Optional[List[str]] = Field(
        default=None, description="Recovery suggestions"
    )


# ===== Utility Schemas =====


class SuccessResponse(BaseModel):
    """成功回應"""

    status: str = Field(default="success", description="Operation status")
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.now)


class PaginatedResponse(BaseModel):
    """分頁回應"""

    items: List[Any] = Field(..., description="Items in current page")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    pages: int = Field(..., description="Total number of pages")

    @field_validator("pages")
    def calculate_pages(cls, v, values):
        total = values.get("total", 0)
        per_page = values.get("per_page", 1)
        return max(1, (total + per_page - 1) // per_page)


# ===== Response Model Aliases =====

# 為向後兼容性提供別名
GenerationJobResponse = GenerationResponse
TrainingResponse = TrainingJobResponse
BatchResponse = BatchJobResponse
