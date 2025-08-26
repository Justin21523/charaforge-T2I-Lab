# frontend/shared/constants.py
"""
Shared constants for SagaForge T2I Lab frontends
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

# ControlNet types
CONTROLNET_TYPES = [
    "pose",
    "depth",
    "canny",
    "lineart",
]

# Image formats
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

# LoRA training presets
LORA_TRAINING_PRESETS = {
    "character": {
        "rank": 16,
        "learning_rate": 1e-4,
        "text_encoder_lr": 5e-5,
        "resolution": 768,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_train_steps": 2000,
    },
    "style": {
        "rank": 8,
        "learning_rate": 8e-5,
        "text_encoder_lr": 0,
        "resolution": 768,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_train_steps": 1500,
    },
}
