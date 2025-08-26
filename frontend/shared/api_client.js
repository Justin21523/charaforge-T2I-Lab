/** frontend/shared/api_client.js
 *
 * JavaScript API client for SagaForge T2I Lab
 * Supports both browser and Node.js environments
 */

const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL || "http://localhost:8000";

class SagaForgeAPIClient {
  constructor(baseUrl = "http://localhost:8000", timeout = 30000) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.timeout = timeout; // 30 seconds
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const config = {
      timeout: this.timeout,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
      ...options,
    };

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await fetch(url, {
        ...config,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.message || `HTTP ${response.status}: ${response.statusText}`
        );
      }

      const data = await response.json();
      return data;
    } catch (error) {
      if (error.name === "AbortError") {
        throw new Error("Request timeout");
      }
      throw error;
    }
  }

  // Health check
  async healthCheck() {
    try {
      return await this.request("/api/v1/health");
    } catch (error) {
      return { status: "error", message: error.message };
    }
  }

  // Image generation
  async generateImage(params) {
    return await this.request("/api/v1/t2i/generate", {
      method: "POST",
      body: JSON.stringify(params),
    });
  }

  async controlnetGenerate(params, controlType = "pose") {
    return await this.request(`/api/v1/controlnet/${controlType}`, {
      method: "POST",
      body: JSON.stringify(params),
    });
  }

  // LoRA management
  async listLoras() {
    try {
      const response = await this.request("/api/v1/lora/list");
      return response.loras || response || [];
    } catch (error) {
      console.error("Failed to list LoRAs:", error);
      return [];
    }
  }

  async loadLora(loraId, weight = 1.0) {
    return await this.request("/api/v1/lora/load", {
      method: "POST",
      body: JSON.stringify({
        lora_id: loraId,
        weight: weight,
      }),
    });
  }

  async unloadLora(loraId) {
    return await this.request("/api/v1/lora/unload", {
      method: "POST",
      body: JSON.stringify({ lora_id: loraId }),
    });
  }
  // Batch processing
  async submitBatchJob(jobData) {
    return await this.request("/api/v1/batch/submit", {
      method: "POST",
      body: JSON.stringify(jobData),
    });
  }

  async getJobStatus(jobId) {
    return await this.request(`/api/v1/batch/status/${jobId}`);
  }

  async listJobs(status = null) {
    const query = status ? `?status=${status}` : "";
    return await this.request(`/api/v1/batch/jobs${query}`);
  }

  async cancelJob(jobId) {
    return await this.request(`/api/v1/batch/cancel/${jobId}`, {
      method: "POST",
    });
  }

  async downloadJobResults(jobId) {
    const response = await fetch(
      `${this.baseUrl}/api/v1/batch/download/${jobId}`
    );
    if (!response.ok) {
      throw new Error(`Failed to download results: ${response.statusText}`);
    }
    return response.blob();
  }

  // Training
  async submitTrainingJob(config) {
    return await this.request("/api/v1/finetune/lora/train", {
      method: "POST",
      body: JSON.stringify(config),
    });
  }

  async getTrainingStatus(runId) {
    return await this.request(`/api/v1/finetune/lora/status/${runId}`);
  }

  async listTrainingJobs() {
    return await this.request("/api/v1/finetune/lora/jobs");
  }

  async cancelTraining(runId) {
    return await this.request(`/api/v1/finetune/lora/cancel/${runId}`, {
      method: "POST",
    });
  }

  async getTrainingMetrics(runId) {
    return await this.request(`/api/v1/finetune/lora/metrics/${runId}`);
  }

  // Dataset management
  async listDatasets() {
    return await this.request("/api/v1/datasets/list");
  }

  async uploadDataset(formData) {
    return await this.request("/api/v1/datasets/upload", {
      method: "POST",
      body: formData,
      headers: {}, // Let browser set Content-Type for FormData
    });
  }

  async getDatasetInfo(datasetName) {
    return await this.request(`/api/v1/datasets/info/${datasetName}`);
  }

  // File upload
  async uploadFile(file, fileType = "image") {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("file_type", fileType);

    return await this.request("/api/v1/upload", {
      method: "POST",
      body: formData,
      headers: {}, // Remove Content-Type to let browser set boundary
    });
  }
  // Safety and monitoring
  async getSystemStatus() {
    return await this.request("/api/v1/monitoring/status");
  }

  async getResourceUsage() {
    return await this.request("/api/v1/monitoring/resources");
  }

  async getQueueStatus() {
    return await this.request("/api/v1/monitoring/queue");
  }

  // Export and model management
  async exportLora(runId, format = "safetensors") {
    const response = await fetch(
      `${this.baseUrl}/api/v1/export/lora/${runId}?format=${format}`
    );
    if (!response.ok) {
      throw new Error(`Export failed: ${response.statusText}`);
    }
    return response.blob();
  }

  async listModels(modelType = "lora") {
    return await this.request(`/api/v1/models/list?type=${modelType}`);
  }
}

// Export for different environments
if (typeof module !== "undefined" && module.exports) {
  module.exports = SagaForgeAPIClient;
} else if (typeof window !== "undefined") {
  window.SagaForgeAPIClient = SagaForgeAPIClient;
}
