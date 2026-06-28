const scenarios = {
  portrait: {
    status: "Mock job succeeded",
    title: "Character portrait generation",
    description:
      "Submit a reproducible image job with a controlled seed, scheduler, negative prompt, and optional LoRA stack.",
    image: "./assets/scenario-portrait.png",
    config: {
      Prompt: "anime character portrait, detailed eyes",
      Seed: "418237",
      Scheduler: "DPM++ 2M Karras",
      Size: "768 x 768",
      Steps: "28",
      LoRA: "alice_v2 @ 0.75",
    },
    meta: ["seed 418237", "2.8s mock", "succeeded"],
    timeline: [
      ["Queued", "Client posts /api/v1/t2i/submit and receives a job id."],
      ["Running", "Worker loads the selected model and streams step progress."],
      ["Saved", "Result image and metadata are written under AI_WAREHOUSE outputs."],
      ["Visible", "Dashboard polls /status and displays signed image URLs."],
    ],
    api: {
      method: "POST",
      path: "/api/v1/t2i/submit",
      body: { prompt: "...", seed: 418237, width: 768, height: 768, steps: 28 },
    },
  },
  controlnet: {
    status: "Control image processed",
    title: "ControlNet guided generation",
    description:
      "Use a pose/depth/canny/lineart control image to keep composition stable while changing style and prompt.",
    image: "./assets/scenario-controlnet.png",
    config: {
      Control: "canny",
      Weight: "0.85",
      Preprocess: "enabled",
      Model: "SDXL",
      Steps: "24",
      Batch: "1 image",
    },
    meta: ["control canny", "seed 90211", "succeeded"],
    timeline: [
      ["Upload", "Browser sends a data URL control image with generation parameters."],
      ["Preprocess", "Backend normalizes the control map and enforces image size limits."],
      ["Generate", "ControlNet pipeline shares the loaded base model components."],
      ["Return", "Image endpoint streams the final PNG with access control."],
    ],
    api: {
      method: "POST",
      path: "/api/v1/controlnet/canny",
      body: { prompt: "...", control_image: "data:image/png;base64,...", weight: 0.85 },
    },
  },
  batch: {
    status: "Batch archive ready",
    title: "Batch prompt processing",
    description:
      "Submit many prompt rows from CSV/JSON, monitor each task, and download generated outputs as a zip.",
    image: "./assets/scenario-batch.png",
    config: {
      Source: "CSV prompt sheet",
      Tasks: "24",
      Completed: "24 / 24",
      Failed: "0",
      Archive: "batch_results.zip",
      Mode: "serial GPU lock",
    },
    meta: ["24 tasks", "zip ready", "completed"],
    timeline: [
      ["Parsed", "Frontend converts CSV rows into typed batch task payloads."],
      ["Submitted", "Backend creates an in-process batch job with task-level results."],
      ["Generated", "Tasks serialize access to the shared diffusion pipeline lock."],
      ["Archived", "Download endpoint packages output images into a zip stream."],
    ],
    api: {
      method: "POST",
      path: "/api/v1/batch/submit",
      body: { job_name: "style_sweep", tasks: ["prompt row 1", "prompt row 2"] },
    },
  },
  lora: {
    status: "Training checkpoint exported",
    title: "LoRA fine-tuning monitor",
    description:
      "Validate a dataset, enqueue a Celery training job, stream progress, and register exported LoRA weights.",
    image: "./assets/scenario-lora.png",
    config: {
      Dataset: "raw/alice_portraits",
      Preset: "character",
      Rank: "16",
      Steps: "4000",
      Loss: "0.092",
      Output: "lora/alice_v2",
    },
    meta: ["64% progress", "loss 0.092", "exported"],
    timeline: [
      ["Validated", "Dataset endpoint checks image files and captions before training."],
      ["Queued", "FastAPI submits a Celery task to the training queue."],
      ["Streaming", "Worker publishes progress via Redis PubSub to WebSocket clients."],
      ["Registered", "Final safetensors file is copied into the model registry."],
    ],
    api: {
      method: "POST",
      path: "/api/v1/finetune/lora/train",
      body: { project_name: "alice_v2", dataset_path: "alice_portraits", lora_rank: 16 },
    },
  },
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function renderScenario(key) {
  const scenario = scenarios[key] || scenarios.portrait;
  $("#scenario-status").textContent = scenario.status;
  $("#scenario-title").textContent = scenario.title;
  $("#scenario-description").textContent = scenario.description;
  $("#scenario-image").src = scenario.image;
  $("#scenario-image").alt = `${scenario.title} mock output`;
  $("#scenario-config").innerHTML = Object.entries(scenario.config)
    .map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`)
    .join("");
  $("#scenario-meta").innerHTML = scenario.meta.map((item) => `<span>${item}</span>`).join("");
  $("#scenario-timeline").innerHTML = scenario.timeline
    .map(([title, text]) => `<li><strong>${title}</strong>${text}</li>`)
    .join("");
  $("#scenario-api").textContent = JSON.stringify(scenario.api, null, 2);

  $$(".scenario-card").forEach((button) => {
    button.classList.toggle("active", button.dataset.scenario === key);
  });
}

$$(".scenario-card").forEach((button) => {
  button.addEventListener("click", () => renderScenario(button.dataset.scenario));
});

$("#replay-button").addEventListener("click", () => {
  const active = $(".scenario-card.active")?.dataset.scenario || "portrait";
  const statuses = ["Queued", "Running", "Saving output", scenarios[active].status];
  let index = 0;
  $("#scenario-status").textContent = statuses[index];
  const timer = window.setInterval(() => {
    index += 1;
    $("#scenario-status").textContent = statuses[index];
    if (index >= statuses.length - 1) {
      window.clearInterval(timer);
    }
  }, 650);
});

renderScenario("portrait");
