// frontend/react_app/src/services/apiService.js
import axios from "axios";

const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL || "http://localhost:8000";

class APIService {
  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        "Content-Type": "application/json",
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log(
          `API Request: ${config.method?.toUpperCase()} ${config.url}`
        );
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        return response.data;
      },
      (error) => {
        console.error("API Error:", error);
        const message =
          error.response?.data?.message || error.message || "Unknown error";
        return Promise.reject(new Error(message));
      }
    );
  }

  // Health check
  async healthCheck() {
    try {
      return await this.client.get("/api/v1/health");
    } catch (error) {
      return { status: "error", message: error.message };
    }
  }

  // Image generation
  async generateImage(params) {
    return await this.client.post("/api/v1/t2i/generate", params);
  }

  async controlnetGenerate(params, controlType = "pose") {
    return await this.client.post(`/api/v1/controlnet/${controlType}`, params);
  }

  // LoRA management
  async listLoras() {
    try {
      return await this.client.get("/api/v1/lora/list");
    } catch (error) {
      console.error("Failed to list LoRAs:", error);
      return [];
    }
  }

  async loadLora(loraId, weight = 1.0) {
    return await this.client.post("/api/v1/lora/load", {
      lora_id: loraId,
      weight: weight,
    });
  }

  async unloadLora(loraId) {
    return await this.client.post("/api/v1/lora/unload", {
      lora_id: loraId,
    });
  }

  // Batch processing
  async submitBatchJob(jobData) {
    return await this.client.post("/api/v1/batch/submit", jobData);
  }

  async getJobStatus(jobId) {
    return await this.client.get(`/api/v1/batch/status/${jobId}`);
  }

  async cancelJob(jobId) {
    return await this.client.post(`/api/v1/batch/cancel/${jobId}`);
  }

  // Training
  async submitTrainingJob(config) {
    return await this.client.post("/api/v1/finetune/lora/train", config);
  }

  async getTrainingStatus(runId) {
    return await this.client.get(`/api/v1/finetune/lora/status/${runId}`);
  }

  // File upload
  async uploadFile(file, fileType = "image") {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("file_type", fileType);

    return await this.client.post("/api/v1/upload", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
  }
}

export default new APIService();
