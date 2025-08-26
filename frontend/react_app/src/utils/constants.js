// frontend/react_app/src/utils/constants.js
export const DEFAULT_GENERATION_PARAMS = {
  prompt: "",
  negative: "lowres, blurry, bad anatomy, extra fingers, worst quality",
  width: 768,
  height: 768,
  steps: 25,
  cfg_scale: 7.5,
  seed: -1,
  sampler: "DPM++ 2M Karras",
  batch_size: 1,
};

export const CONTROLNET_TYPES = ["pose", "depth", "canny", "lineart"];

export const SAMPLERS = [
  "DPM++ 2M Karras",
  "DPM++ SDE Karras",
  "Euler a",
  "Euler",
  "LMS",
  "Heun",
  "DDIM",
];

export const LORA_TRAINING_PRESETS = {
  character: {
    rank: 16,
    learning_rate: 1e-4,
    text_encoder_lr: 5e-5,
    resolution: 768,
    batch_size: 1,
    gradient_accumulation_steps: 8,
    max_train_steps: 2000,
  },
  style: {
    rank: 8,
    learning_rate: 8e-5,
    text_encoder_lr: 0,
    resolution: 768,
    batch_size: 2,
    gradient_accumulation_steps: 4,
    max_train_steps: 1500,
  },
};
