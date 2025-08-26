/** frontend/shared/api_client.js
 *
 * JavaScript API client for SagaForge T2I Lab
 * Supports both browser and Node.js environments
 */
class SagaForgeAPIClient {
  constructor(baseUrl = "http://localhost:8000", timeout = 30000) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.timeout = timeout;
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
      const response = await fetch(url, config);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${error.message}`);
      throw error;
    }
  }

  async healthCheck() {
    try {
      return await this.request("/api/v1/health");
    } catch (error) {
      return { status: "error", message: error.message };
    }
  }

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

  async listLoras() {
    try {
      return await this.request("/api/v1/lora/list");
    } catch (error) {
      console.error("Failed to list LoRAs:", error);
      return [];
    }
  }

  async loadLora(loraId, weight = 1.0) {
    return await this.request("/api/v1/lora/load", {
      method: "POST",
      body: JSON.stringify({ lora_id: loraId, weight }),
    });
  }

  async unloadLora(loraId) {
    return await this.request("/api/v1/lora/unload", {
      method: "POST",
      body: JSON.stringify({ lora_id: loraId }),
    });
  }

  async submitBatchJob(jobData) {
    return await this.request("/api/v1/batch/submit", {
      method: "POST",
      body: JSON.stringify(jobData),
    });
  }

  async getJobStatus(jobId) {
    return await this.request(`/api/v1/batch/status/${jobId}`);
  }

  async submitTrainingJob(config) {
    return await this.request("/api/v1/finetune/lora/train", {
      method: "POST",
      body: JSON.stringify(config),
    });
  }

  async getTrainingStatus(runId) {
    return await this.request(`/api/v1/finetune/lora/status/${runId}`);
  }

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
}

// Export for different environments
if (typeof module !== "undefined" && module.exports) {
  module.exports = SagaForgeAPIClient;
} else if (typeof window !== "undefined") {
  window.SagaForgeAPIClient = SagaForgeAPIClient;
}
