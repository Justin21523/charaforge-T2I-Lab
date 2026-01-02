// frontend/react_app/src/services/apiService.js
import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const API_KEY_HEADER = import.meta.env.VITE_API_KEY_HEADER || "X-API-Key";

const STORAGE_API_KEY = "charaforge.apiKey";
const STORAGE_API_KEY_HEADER = "charaforge.apiKeyHeader";
const STORAGE_USE_JWT = "charaforge.auth.useJwt";
const STORAGE_JWT_ACCESS_TOKEN = "charaforge.jwt.accessToken";
const STORAGE_JWT_EXPIRES_AT = "charaforge.jwt.expiresAt";
const STORAGE_JWT_REFRESH_EXPIRES_AT = "charaforge.jwt.refreshExpiresAt";
const STORAGE_JWT_ROLE = "charaforge.jwt.role";
const STORAGE_JWT_SCOPES = "charaforge.jwt.scopes";
const LEGACY_STORAGE_JWT_REFRESH_TOKEN = "charaforge.jwt.refreshToken";
const CSRF_COOKIE_NAME = import.meta.env.VITE_JWT_CSRF_COOKIE_NAME || "cfr_csrf";
const CSRF_HEADER_NAME = "X-CSRF-Token";

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
    this.useJwt = Boolean(readStoredJson(STORAGE_USE_JWT, false));
    this.accessToken = readStoredJson(STORAGE_JWT_ACCESS_TOKEN, "") || "";
    this.accessExpiresAt = Number(readStoredJson(STORAGE_JWT_EXPIRES_AT, 0) || 0);
    this.refreshExpiresAt = Number(readStoredJson(STORAGE_JWT_REFRESH_EXPIRES_AT, 0) || 0);
    this.jwtRole = readStoredJson(STORAGE_JWT_ROLE, "") || "";
    this.jwtScopes = readStoredJson(STORAGE_JWT_SCOPES, []) || [];
    this._refreshPromise = null;
    writeStoredJson(LEGACY_STORAGE_JWT_REFRESH_TOKEN, null);

    const defaultHeaders = { "Content-Type": "application/json" };
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: defaultHeaders,
    });
    this.authClient = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: defaultHeaders,
      withCredentials: true,
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        const url = String(config?.url || "");
        const isTokenEndpoint = url.includes("/api/v1/auth/token");
        const isRefreshEndpoint = url.includes("/api/v1/auth/refresh");
        const isLogoutEndpoint = url.includes("/api/v1/auth/logout");

        const headers = config.headers || {};

        const dropHeader = (name) => {
          try {
            delete headers[name];
          } catch (e) {
            // ignore
          }
        };

        if (isTokenEndpoint) {
          dropHeader("Authorization");
          if (this.apiKey) {
            headers[this.apiKeyHeader] = this.apiKey;
          } else {
            dropHeader(this.apiKeyHeader);
          }
        } else if (isRefreshEndpoint || isLogoutEndpoint) {
          dropHeader("Authorization");
          dropHeader(this.apiKeyHeader);
        } else if (this.useJwt && this.accessToken) {
          headers.Authorization = `Bearer ${this.accessToken}`;
          dropHeader(this.apiKeyHeader);
        } else if (this.apiKey) {
          headers[this.apiKeyHeader] = this.apiKey;
          dropHeader("Authorization");
        } else {
          dropHeader("Authorization");
          dropHeader(this.apiKeyHeader);
        }

        config.headers = headers;
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
      async (error) => {
        console.error("API Error:", error);

        const status = error.response?.status || 0;
        const originalConfig = error.config || null;
        const url = String(originalConfig?.url || "");
        const isAuthEndpoint =
          url.includes("/api/v1/auth/token") ||
          url.includes("/api/v1/auth/refresh") ||
          url.includes("/api/v1/auth/logout");

        if (
          status === 401 &&
          !isAuthEndpoint &&
          originalConfig &&
          !originalConfig._jwtRetry &&
          this.useJwt &&
          this.accessToken
        ) {
          originalConfig._jwtRetry = true;
          try {
            await this.refreshJwtSession();
            return await this.client.request(originalConfig);
          } catch (refreshError) {
            this.clearJwtSession();
            if (this.apiKey && !originalConfig._apiKeyRetry) {
              originalConfig._apiKeyRetry = true;
              return await this.client.request(originalConfig);
            }
            // fall through to formatted error
          }
        }

        const data = error.response?.data || {};
        const requestId =
          data?.request_id || error.response?.headers?.["x-request-id"] || "";

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

  setUseJwt(enabled) {
    this.useJwt = Boolean(enabled);
    writeStoredJson(STORAGE_USE_JWT, this.useJwt);
    if (!this.useJwt) {
      this.clearJwtSession();
    }
  }

  getUseJwt() {
    return Boolean(this.useJwt);
  }

  hasJwtSession() {
    return Boolean(this.accessToken);
  }

  getJwtInfo() {
    return {
      enabled: Boolean(this.useJwt),
      accessToken: this.accessToken || "",
      expiresAt: Number(this.accessExpiresAt || 0),
      refreshExpiresAt: Number(this.refreshExpiresAt || 0),
      role: this.jwtRole || "",
      scopes: Array.isArray(this.jwtScopes) ? this.jwtScopes : [],
    };
  }

  _readCookie(name) {
    try {
      if (typeof document === "undefined") return "";
      const needle = `${name}=`;
      const parts = String(document.cookie || "")
        .split(";")
        .map((chunk) => chunk.trim());
      for (const part of parts) {
        if (!part.startsWith(needle)) continue;
        return decodeURIComponent(part.slice(needle.length));
      }
      return "";
    } catch (e) {
      return "";
    }
  }

  _getCsrfToken() {
    return this._readCookie(CSRF_COOKIE_NAME) || "";
  }

  getAuthHeaders() {
    const headers = {};
    if (this.useJwt && this.accessToken) {
      headers.Authorization = `Bearer ${this.accessToken}`;
      return headers;
    }
    if (this.apiKey) {
      headers[this.apiKeyHeader] = this.apiKey;
    }
    return headers;
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

    writeStoredJson(STORAGE_API_KEY, this.apiKey);
    writeStoredJson(STORAGE_API_KEY_HEADER, this.apiKeyHeader);
  }

  clearApiKey() {
    this.setApiKey("");
  }

  _setJwtSession(payload) {
    const accessToken = payload?.access_token || "";
    const expiresAt = Number(payload?.expires_at || 0);
    const refreshExpiresAt = Number(payload?.refresh_expires_at || 0);
    const role = payload?.role || "";
    const scopes = Array.isArray(payload?.scopes) ? payload.scopes : [];

    this.accessToken = accessToken;
    this.accessExpiresAt = expiresAt;
    this.refreshExpiresAt = refreshExpiresAt;
    this.jwtRole = role;
    this.jwtScopes = scopes;

    writeStoredJson(STORAGE_JWT_ACCESS_TOKEN, accessToken);
    writeStoredJson(STORAGE_JWT_EXPIRES_AT, expiresAt);
    writeStoredJson(STORAGE_JWT_REFRESH_EXPIRES_AT, refreshExpiresAt);
    writeStoredJson(STORAGE_JWT_ROLE, role);
    writeStoredJson(STORAGE_JWT_SCOPES, scopes);
  }

  clearJwtSession() {
    this.accessToken = "";
    this.accessExpiresAt = 0;
    this.refreshExpiresAt = 0;
    this.jwtRole = "";
    this.jwtScopes = [];
    writeStoredJson(STORAGE_JWT_ACCESS_TOKEN, "");
    writeStoredJson(STORAGE_JWT_EXPIRES_AT, 0);
    writeStoredJson(STORAGE_JWT_REFRESH_EXPIRES_AT, 0);
    writeStoredJson(STORAGE_JWT_ROLE, "");
    writeStoredJson(STORAGE_JWT_SCOPES, []);
    writeStoredJson(LEGACY_STORAGE_JWT_REFRESH_TOKEN, null);
  }

  async exchangeApiKeyForToken() {
    if (!this.apiKey) {
      throw new Error("Missing API key");
    }
    try {
      const headers = { [this.apiKeyHeader]: this.apiKey };
      const res = await this.authClient.post("/api/v1/auth/token", null, { headers });
      this._setJwtSession(res.data);
      return res.data;
    } catch (error) {
      const data = error.response?.data || {};
      const requestId =
        data?.request_id || error.response?.headers?.["x-request-id"] || "";
      const message =
        (typeof data?.message === "string" && data.message) ||
        error.message ||
        "JWT token exchange failed";
      const withRequestId = requestId ? `${message} (request_id=${requestId})` : message;
      throw new Error(withRequestId);
    }
  }

  async refreshJwtSession() {
    if (this._refreshPromise) {
      return await this._refreshPromise;
    }

    this._refreshPromise = (async () => {
      const csrf = this._getCsrfToken();
      const headers = csrf ? { [CSRF_HEADER_NAME]: csrf } : {};
      const res = await this.authClient.post("/api/v1/auth/refresh", null, { headers });
      this._setJwtSession(res.data);
      return res.data;
    })();

    try {
      return await this._refreshPromise;
    } finally {
      this._refreshPromise = null;
    }
  }

  async logoutJwt({ all = false } = {}) {
    this.clearJwtSession();
    try {
      const csrf = this._getCsrfToken();
      const headers = csrf ? { [CSRF_HEADER_NAME]: csrf } : {};
      const res = await this.authClient.post("/api/v1/auth/logout", {
        all: Boolean(all),
      }, { headers });
      return res.data;
    } catch (error) {
      return { status: "error", message: error.message };
    }
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

  async listT2IJobs({ limit = 50, status = null, all = false } = {}) {
    const params = { limit };
    if (status) params.status = status;
    if (all) params.all = true;
    return await this.client.get("/api/v1/t2i/jobs", { params });
  }

  async deleteT2IJob(jobId, { deleteOutputs = true } = {}) {
    return await this.client.delete(`/api/v1/t2i/jobs/${jobId}`, {
      params: { delete_outputs: Boolean(deleteOutputs) },
    });
  }

  async cleanupT2IJobs({
    ttlSeconds,
    dryRun = true,
    deleteRecords = false,
    all = false,
    onlyTerminal = true,
    limit = 200,
  } = {}) {
    const params = {
      dry_run: Boolean(dryRun),
      delete_records: Boolean(deleteRecords),
      all: Boolean(all),
      only_terminal: Boolean(onlyTerminal),
      limit: Number(limit || 200),
    };
    if (ttlSeconds !== undefined && ttlSeconds !== null) {
      params.ttl_seconds = Number(ttlSeconds);
    }
    return await this.client.post("/api/v1/t2i/jobs/cleanup", null, { params });
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

  async authMe() {
    return await this.client.get("/api/v1/auth/me");
  }

  // File upload
  async uploadFile(file, fileType = "image") {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("file_type", fileType);

    const uploadHeaders = {
      "Content-Type": "multipart/form-data",
    };

    return await this.client.post("/api/v1/upload", formData, {
      headers: {
        ...uploadHeaders,
      },
    });
  }
}

export default new APIService();
