// frontend/react_app/src/services/apiService.js
import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const API_KEY_HEADER = import.meta.env.VITE_API_KEY_HEADER || "X-API-Key";

const STORAGE_API_KEY = "charaforge.apiKey";
const STORAGE_API_KEY_HEADER = "charaforge.apiKeyHeader";

const readStoredJson = (key, fallback) => {
  try {
    if (typeof window === "undefined") return fallback;
    const raw = window.localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw);
  } catch (e) {
    return fallback;
  }
};

const writeStoredJson = (key, value) => {
  try {
    if (typeof window === "undefined") return;
    if (value === null || value === undefined) {
      window.localStorage.removeItem(key);
      return;
    }
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch (e) {
    // ignore
  }
};

class APIService {
  constructor() {
    this.apiKeyHeader = readStoredJson(STORAGE_API_KEY_HEADER, API_KEY_HEADER) || API_KEY_HEADER;
    this.apiKey = readStoredJson(STORAGE_API_KEY, "") || "";

    const defaultHeaders = {
      "Content-Type": "application/json",
    };
    if (this.apiKey) {
      defaultHeaders[this.apiKeyHeader] = this.apiKey;
    }

    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: defaultHeaders,
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
        const data = error.response?.data || {};
        const requestId =
          data?.request_id || error.response?.headers?.["x-request-id"] || "";

        const status = error.response?.status || 0;
        const code = typeof data?.error === "string" ? data.error : "";
        const details =
          data && typeof data === "object" && data.details && typeof data.details === "object"
            ? data.details
            : {};

        const message =
          (typeof data?.message === "string" && data.message) ||
          (typeof data?.detail === "string" && data.detail) ||
          error.message ||
          "Unknown error";

        const withRequestId = requestId ? `${message} (request_id=${requestId})` : message;
        const apiError = new Error(withRequestId);
        apiError.requestId = requestId;
        apiError.status = status;
        apiError.code = code;
        apiError.details = details;
        return Promise.reject(apiError);
      }
    );
  }

  getApiKey() {
    return this.apiKey || "";
  }

  getApiKeyHeader() {
    return this.apiKeyHeader || API_KEY_HEADER;
  }

  setApiKey(apiKey, apiKeyHeader = this.apiKeyHeader) {
    const nextHeader = apiKeyHeader || API_KEY_HEADER;
    const prevHeader = this.apiKeyHeader;

    this.apiKey = apiKey || "";
    this.apiKeyHeader = nextHeader;

    if (prevHeader && prevHeader !== nextHeader) {
      delete this.client.defaults.headers.common[prevHeader];
    }

    if (this.apiKey) {
      this.client.defaults.headers.common[this.apiKeyHeader] = this.apiKey;
    } else {
      delete this.client.defaults.headers.common[this.apiKeyHeader];
    }

    writeStoredJson(STORAGE_API_KEY, this.apiKey);
    writeStoredJson(STORAGE_API_KEY_HEADER, this.apiKeyHeader);
  }

  clearApiKey() {
    this.setApiKey("");
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

  async submitT2IJob(params) {
    return await this.client.post("/api/v1/t2i/submit", params);
  }

  async getT2IJobStatus(jobId) {
    return await this.client.get(`/api/v1/t2i/status/${jobId}`);
  }

  async cancelT2IJob(jobId) {
    return await this.client.post(`/api/v1/t2i/cancel/${jobId}`);
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

  async listJobs(limit = 50) {
    return await this.client.get(`/api/v1/batch/list?limit=${limit}`);
  }

  async downloadJobResults(jobId) {
    const response = await this.client.get(`/api/v1/batch/download/${jobId}`, {
      responseType: "blob",
    });
    return response;
  }

  // Training
  async submitTrainingJob(config) {
    return await this.client.post("/api/v1/finetune/lora/train", config);
  }

  async getTrainingStatus(runId) {
    return await this.client.get(`/api/v1/finetune/lora/status/${runId}`);
  }

  async cancelTraining(runId) {
    return await this.client.post(`/api/v1/finetune/lora/cancel/${runId}`);
  }

  // File upload
  async uploadFile(file, fileType = "image") {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("file_type", fileType);

    const uploadHeaders = {
      "Content-Type": "multipart/form-data",
    };
    if (this.apiKey) {
      uploadHeaders[this.apiKeyHeader] = this.apiKey;
    }

    return await this.client.post("/api/v1/upload", formData, {
      headers: {
        ...uploadHeaders,
      },
    });
  }
}

export default new APIService();
