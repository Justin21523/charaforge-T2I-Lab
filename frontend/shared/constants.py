# frontend/shared/constants.py
"""
Shared constants for SagaForge T2I Lab frontends
Constants used across all frontend interfaces
"""
# API endpoints
API_BASE_URL = "http://localhost:8000"
WS_BASE_URL = "ws://localhost:8000"

# Default generation parameters
DEFAULT_GENERATION_PARAMS = {
    "prompt": "",
    "negative": "lowres, blurry, bad anatomy, extra fingers, worst quality",
    "width": 768,
    "height": 768,
    "steps": 25,
    "cfg_scale": 7.5,
    "seed": -1,
    "sampler": "DPM++ 2M Karras",
    "batch_size": 1,
}

# Available samplers
SAMPLERS = [
    "DPM++ 2M Karras",
    "DPM++ SDE Karras",
    "Euler a",
    "Euler",
    "LMS",
    "Heun",
    "DDIM",
    "PLMS",
    "UniPC",
]

# ControlNet types
CONTROLNET_TYPES = [
    "pose",
    "depth",
    "canny",
    "lineart",
    "scribble",
    "softedge",
    "seg",
    "normal",
    "mlsd",
]

# Image resolution presets
RESOLUTION_PRESETS = {
    "SD 1.5": [(512, 512), (512, 768), (768, 512), (640, 640), (512, 640), (640, 512)],
    "SDXL": [
        (1024, 1024),
        (1152, 896),
        (896, 1152),
        (1216, 832),
        (832, 1216),
        (1344, 768),
        (768, 1344),
        (1536, 640),
        (640, 1536),
    ],
}

# LoRA training presets
LORA_TRAINING_PRESETS = {
    "character": {
        "rank": 16,
        "alpha": 32,
        "learning_rate": 1e-4,
        "text_encoder_lr": 5e-5,
        "resolution": 768,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_train_steps": 2000,
        "save_every": 500,
        "mixed_precision": "fp16",
        "gradient_checkpointing": True,
        "use_ema": True,
    },
    "style": {
        "rank": 8,
        "alpha": 16,
        "learning_rate": 8e-5,
        "text_encoder_lr": 0,  # Disable text encoder training for style
        "resolution": 768,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_train_steps": 1500,
        "save_every": 300,
        "mixed_precision": "fp16",
        "gradient_checkpointing": True,
        "use_ema": False,
    },
    "concept": {
        "rank": 32,
        "alpha": 64,
        "learning_rate": 5e-5,
        "text_encoder_lr": 1e-5,
        "resolution": 1024,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_train_steps": 3000,
        "save_every": 500,
        "mixed_precision": "fp16",
        "gradient_checkpointing": True,
        "use_ema": True,
    },
}

# Supported model types
MODEL_TYPES = {
    "base": [
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/stable-diffusion-2-1",
        "CompVis/stable-diffusion-v1-4",
    ],
    "anime": [
        "hakurei/waifu-diffusion",
        "andite/anything-v4.0",
        "Linaqruf/anything-v3.0",
        "dreamlike-art/dreamlike-anime-1.0",
    ],
    "realistic": [
        "SG161222/Realistic_Vision_V2.0",
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1-base",
    ],
}

# File format constants
SUPPORTED_IMAGE_FORMATS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]

SUPPORTED_MODEL_FORMATS = [".safetensors", ".ckpt", ".pt", ".pth", ".bin"]

SUPPORTED_DATASET_FORMATS = [".csv", ".json", ".jsonl", ".parquet"]

# API endpoints
API_ENDPOINTS = {
    "health": "/api/v1/health",
    "generate": "/api/v1/t2i/generate",
    "controlnet": "/api/v1/controlnet",
    "lora_list": "/api/v1/lora/list",
    "lora_load": "/api/v1/lora/load",
    "lora_unload": "/api/v1/lora/unload",
    "lora_status": "/api/v1/lora/status",
    "batch_submit": "/api/v1/batch/submit",
    "batch_status": "/api/v1/batch/status",
    "batch_jobs": "/api/v1/batch/jobs",
    "batch_cancel": "/api/v1/batch/cancel",
    "batch_download": "/api/v1/batch/download",
    "train_submit": "/api/v1/finetune/lora/train",
    "train_status": "/api/v1/finetune/lora/status",
    "train_jobs": "/api/v1/finetune/lora/jobs",
    "train_cancel": "/api/v1/finetune/lora/cancel",
    "train_metrics": "/api/v1/finetune/lora/metrics",
    "datasets_list": "/api/v1/datasets/list",
    "datasets_upload": "/api/v1/datasets/upload",
    "datasets_info": "/api/v1/datasets/info",
    "models_list": "/api/v1/models/list",
    "export_lora": "/api/v1/export/lora",
    "monitoring_status": "/api/v1/monitoring/status",
    "monitoring_resources": "/api/v1/monitoring/resources",
    "monitoring_queue": "/api/v1/monitoring/queue",
    "upload": "/api/v1/upload",
}

# Job status constants
JOB_STATUSES = {
    "PENDING": "pending",
    "RUNNING": "running",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "CANCELLED": "cancelled",
    "PAUSED": "paused",
}

# Training status constants
TRAINING_STATUSES = {
    "PREPARING": "preparing",
    "TRAINING": "training",
    "VALIDATING": "validating",
    "SAVING": "saving",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "CANCELLED": "cancelled",
}

# Default negative prompts by style
DEFAULT_NEGATIVE_PROMPTS = {
    "general": "lowres, blurry, bad anatomy, extra fingers, worst quality",
    "anime": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    "realistic": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime)",
    "portrait": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, multiple people, crowd",
}

# Style presets
STYLE_PRESETS = {
    "anime": {
        "name": "Anime Style",
        "positive_suffix": ", anime style, detailed, high quality",
        "negative": DEFAULT_NEGATIVE_PROMPTS["anime"],
        "cfg_scale": 7.5,
        "steps": 25,
    },
    "realistic": {
        "name": "Realistic Style",
        "positive_suffix": ", photorealistic, detailed, high quality",
        "negative": DEFAULT_NEGATIVE_PROMPTS["realistic"],
        "cfg_scale": 8.0,
        "steps": 30,
    },
    "portrait": {
        "name": "Portrait Style",
        "positive_suffix": ", portrait, detailed face, professional lighting",
        "negative": DEFAULT_NEGATIVE_PROMPTS["portrait"],
        "cfg_scale": 7.0,
        "steps": 28,
    },
    "landscape": {
        "name": "Landscape Style",
        "positive_suffix": ", landscape, scenic, detailed, cinematic",
        "negative": "lowres, worst quality, low quality, blurry, people, person",
        "cfg_scale": 6.5,
        "steps": 25,
    },
}

# UI themes
UI_THEMES = {
    "dark": {
        "name": "深色主題",
        "primary": "#3b82f6",
        "background": "#1f2937",
        "surface": "#374151",
        "text": "#f9fafb",
    },
    "light": {
        "name": "淺色主題",
        "primary": "#2563eb",
        "background": "#ffffff",
        "surface": "#f9fafb",
        "text": "#111827",
    },
    "system": {
        "name": "跟隨系統",
        "primary": "#3b82f6",
        "background": "auto",
        "surface": "auto",
        "text": "auto",
    },
}

# Validation limits
VALIDATION_LIMITS = {
    "prompt_max_length": 2000,
    "negative_max_length": 1000,
    "batch_size_max": 8,
    "steps_max": 100,
    "cfg_scale_max": 30.0,
    "lora_weight_max": 2.0,
    "file_size_max": 100 * 1024 * 1024,  # 100MB
    "dataset_images_max": 10000,
    "training_jobs_max": 10,
}

# Error messages
ERROR_MESSAGES = {
    "connection_failed": "無法連接到 API 服務器",
    "invalid_prompt": "提示詞格式無效",
    "file_too_large": "文件過大",
    "unsupported_format": "不支援的文件格式",
    "generation_failed": "圖片生成失敗",
    "training_failed": "訓練任務失敗",
    "model_not_found": "找不到指定模型",
    "insufficient_resources": "系統資源不足",
}

# Success messages
SUCCESS_MESSAGES = {
    "generation_complete": "圖片生成完成",
    "training_submitted": "訓練任務提交成功",
    "model_loaded": "模型載入成功",
    "file_uploaded": "檔案上傳成功",
    "settings_saved": "設定已保存",
}

# Default cache directories (relative to AI_CACHE_ROOT)
CACHE_DIRECTORIES = {
    "models": "models",
    "datasets": "datasets",
    "outputs": "outputs",
    "cache": "cache",
    "logs": "logs",
    "temp": "temp",
}

# Progress update intervals (in seconds)
UPDATE_INTERVALS = {
    "job_status": 3,
    "training_metrics": 5,
    "resource_monitor": 10,
    "health_check": 30,
}

# Default timeout values (in seconds)
TIMEOUT_VALUES = {
    "api_request": 30,
    "file_upload": 300,
    "generation": 60,
    "training_submit": 10,
}

# Feature flags
FEATURE_FLAGS = {
    "enable_controlnet": True,
    "enable_training": True,
    "enable_batch": True,
    "enable_safety_filter": True,
    "enable_watermark": False,
    "enable_metrics": True,
    "enable_export": True,
}
