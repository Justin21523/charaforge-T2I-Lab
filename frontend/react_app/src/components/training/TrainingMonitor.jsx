// frontend/react_app/src/components/training/TrainingMonitor.jsx
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Brain, Play, RefreshCw, XCircle } from "lucide-react";
import { useAPI } from "../../hooks/useAPI";
import { useLocalStorage } from "../../hooks/useLocalStorage";
import { buildTrainProgressWsUrl, useWebSocket } from "../../hooks/useWebSocket";
import { LORA_TRAINING_PRESETS } from "../../utils/constants";
import Loading from "../common/Loading";
import toast from "react-hot-toast";
import "../../styles/components/Training.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const defaultConfig = {
  project_name: "",
  dataset_path: "",
  instance_prompt: "",
  model_type: "sdxl",
  base_model: "",
  lora_rank: 16,
  lora_alpha: 32,
  lora_dropout: 0.1,
  learning_rate: 1e-4,
  train_batch_size: 1,
  gradient_accumulation_steps: 8,
  mixed_precision: "fp16",
  resolution: 768,
  max_train_steps: 2000,
  num_train_epochs: 10,
  save_steps: 500,
  validation_steps: 100,
};

const TrainingMonitor = () => {
  const { apiCall, apiService, isLoading } = useAPI();

  const [selectedPreset, setSelectedPreset] = useState("character");
  const [trainingConfig, setTrainingConfig] = useLocalStorage(
    "training-config",
    defaultConfig
  );

  const [jobs, setJobs] = useLocalStorage("training-jobs", []);
  const [selectedJobId, setSelectedJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);

  const presetKeys = useMemo(() => Object.keys(LORA_TRAINING_PRESETS), []);

  const applyPreset = useCallback(
    (presetName) => {
      const preset = LORA_TRAINING_PRESETS[presetName];
      if (!preset) return;

      setSelectedPreset(presetName);
      setTrainingConfig((prev) => ({
        ...prev,
        lora_rank: preset.rank,
        lora_alpha: preset.rank * 2,
        learning_rate: preset.learning_rate,
        resolution: preset.resolution,
        train_batch_size: preset.batch_size,
        gradient_accumulation_steps: preset.gradient_accumulation_steps,
        max_train_steps: preset.max_train_steps,
      }));
    },
    [setTrainingConfig]
  );

  useEffect(() => {
    if (!trainingConfig.project_name) {
      applyPreset(selectedPreset);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const refreshStatus = useCallback(
    async (jobId) => {
      if (!jobId) return;
      try {
        const status = await apiCall(
          () => apiService.getTrainingStatus(jobId),
          null,
          { showLoading: false, showError: false }
        );
        setJobStatus(status);
      } catch (error) {
        console.error("Failed to fetch training status:", error);
      }
    },
    [apiCall, apiService]
  );

  const wsUrl = useMemo(
    () => (selectedJobId ? buildTrainProgressWsUrl(API_BASE_URL, selectedJobId) : null),
    [selectedJobId]
  );

  const wsProtocols = useCallback(() => {
    const getFallbackProtocols = () => {
      const protocols = ["charaforge"];
      const now = Math.floor(Date.now() / 1000);
      const jwtInfo = apiService.getJwtInfo();
      if (apiService.getUseJwt() && jwtInfo?.accessToken && jwtInfo?.expiresAt > now + 10) {
        protocols.push(`access_token.${jwtInfo.accessToken}`);
        return protocols;
      }
      const apiKey = apiService.getApiKey();
      if (apiKey) {
        protocols.push(`api_key.${apiKey}`);
      }
      return protocols;
    };

    return (async () => {
      const protocols = ["charaforge"];
      try {
        const ticket = await apiService.issueTrainingWsTicket(selectedJobId);
        if (ticket?.ws_ticket) {
          protocols.push(`ws_ticket.${ticket.ws_ticket}`);
          return protocols;
        }
      } catch (error) {
        // ignore and fall back
      }
      return getFallbackProtocols();
    })();
  }, [apiService, selectedJobId]);

  const { status: wsStatus, lastMessage: wsMessage } = useWebSocket(wsUrl, {
    enabled: Boolean(selectedJobId),
    reconnect: true,
    protocols: wsProtocols,
  });

  useEffect(() => {
    if (!selectedJobId || !wsMessage) return;
    if (wsMessage.topic === "ws.subscribed") return;
    if (wsMessage.job_id && wsMessage.job_id !== selectedJobId) return;

    const state = wsMessage.state;
    const progressPayload = wsMessage.progress || null;
    if (!state) return;

    if (state === "PROGRESS") {
      setJobStatus({ job_id: selectedJobId, status: "PROGRESS", progress: progressPayload });
      return;
    }
    if (state === "SUCCESS") {
      setJobStatus({
        job_id: selectedJobId,
        status: "SUCCESS",
        progress: progressPayload,
        result: progressPayload,
      });
      return;
    }
    if (state === "FAILURE") {
      setJobStatus({
        job_id: selectedJobId,
        status: "FAILURE",
        progress: progressPayload,
        error: progressPayload?.error || progressPayload?.message || "Training failed",
      });
    }
  }, [selectedJobId, wsMessage]);

  useEffect(() => {
    if (!selectedJobId) return;
    refreshStatus(selectedJobId);
    if (wsStatus === "connected") return;
    const interval = setInterval(() => refreshStatus(selectedJobId), 5000);
    return () => clearInterval(interval);
  }, [selectedJobId, refreshStatus, wsStatus]);

  const updateConfig = useCallback(
    (key, value) => setTrainingConfig((prev) => ({ ...prev, [key]: value })),
    [setTrainingConfig]
  );

  const submitTraining = useCallback(async () => {
    if (!trainingConfig.project_name.trim()) {
      toast.error("請填寫專案名稱");
      return;
    }
    if (!trainingConfig.dataset_path.trim()) {
      toast.error("請填寫資料集路徑（或資料夾名稱）");
      return;
    }
    if (!trainingConfig.instance_prompt.trim()) {
      toast.error("請填寫 instance prompt");
      return;
    }

    try {
      const payload = {
        ...trainingConfig,
        base_model: trainingConfig.base_model?.trim() || null,
      };

      const result = await apiCall(() => apiService.submitTrainingJob(payload), null, {
        showLoading: true,
        showSuccess: true,
        successMessage: "訓練任務提交成功",
      });

      if (result?.job_id) {
        const job = {
          job_id: result.job_id,
          project_name: trainingConfig.project_name,
          submitted_at: result.submitted_at || new Date().toISOString(),
        };
        setJobs((prev) => [job, ...prev]);
        setSelectedJobId(result.job_id);
        setJobStatus(null);
      }
    } catch (error) {
      console.error("Training submission failed:", error);
    }
  }, [apiCall, apiService, setJobs, trainingConfig]);

  const cancelTraining = useCallback(
    async (jobId) => {
      if (!jobId) return;
      try {
        await apiCall(() => apiService.cancelTraining(jobId), null, {
          showLoading: true,
          showSuccess: true,
          successMessage: "已送出取消請求",
        });
        refreshStatus(jobId);
      } catch (error) {
        console.error("Cancel training failed:", error);
      }
    },
    [apiCall, apiService, refreshStatus]
  );

  const progress = jobStatus?.progress || null;
  const percent =
    typeof progress?.percent === "number" ? Math.round(progress.percent) : null;

  return (
    <div className="training-monitor">
      <div className="panel-header">
        <h2 className="panel-title">
          <Brain className="title-icon" />
          LoRA 訓練
        </h2>
        <button
          className="btn btn-secondary"
          onClick={() => refreshStatus(selectedJobId)}
          disabled={!selectedJobId}
        >
          <RefreshCw size={16} />
          刷新狀態
        </button>
      </div>

      <div className="training-content">
        <div className="training-controls">
          <div className="preset-selector">
            {presetKeys.map((key) => {
              const preset = LORA_TRAINING_PRESETS[key];
              return (
                <button
                  key={key}
                  className={`preset-button ${selectedPreset === key ? "active" : ""}`}
                  onClick={() => applyPreset(key)}
                  type="button"
                >
                  <div className="preset-title">{key}</div>
                  <div className="preset-description">
                    Rank {preset.rank}, {preset.max_train_steps} steps
                  </div>
                </button>
              );
            })}
          </div>

          <div className="training-form">
            <div className="form-group">
              <label>專案名稱</label>
              <input
                type="text"
                value={trainingConfig.project_name}
                onChange={(e) => updateConfig("project_name", e.target.value)}
                placeholder="例如: char_alice_v1"
                className="input"
              />
            </div>

            <div className="form-group">
              <label>資料集路徑 / 名稱</label>
              <input
                type="text"
                value={trainingConfig.dataset_path}
                onChange={(e) => updateConfig("dataset_path", e.target.value)}
                placeholder="例如: /mnt/data/datasets/charaforge-t2i-lab/myset 或 myset"
                className="input"
              />
            </div>

            <div className="form-group">
              <label>Instance Prompt</label>
              <input
                type="text"
                value={trainingConfig.instance_prompt}
                onChange={(e) => updateConfig("instance_prompt", e.target.value)}
                placeholder="例如: a photo of <alice> person"
                className="input"
              />
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>模型類型</label>
                <select
                  className="select"
                  value={trainingConfig.model_type}
                  onChange={(e) => updateConfig("model_type", e.target.value)}
                >
                  <option value="sdxl">SDXL</option>
                  <option value="sd15">SD1.5</option>
                </select>
              </div>

              <div className="form-group">
                <label>Base Model（可選）</label>
                <input
                  type="text"
                  value={trainingConfig.base_model}
                  onChange={(e) => updateConfig("base_model", e.target.value)}
                  placeholder="留空使用預設"
                  className="input"
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>LoRA Rank</label>
                <input
                  type="range"
                  min="4"
                  max="128"
                  step="4"
                  value={trainingConfig.lora_rank}
                  onChange={(e) =>
                    updateConfig("lora_rank", parseInt(e.target.value, 10))
                  }
                  className="slider"
                />
                <span className="slider-value">{trainingConfig.lora_rank}</span>
              </div>

              <div className="form-group">
                <label>學習率</label>
                <input
                  type="number"
                  value={trainingConfig.learning_rate}
                  onChange={(e) =>
                    updateConfig("learning_rate", parseFloat(e.target.value))
                  }
                  step="0.000001"
                  min="0.000001"
                  max="0.01"
                  className="input"
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>解析度</label>
                <input
                  type="range"
                  min="512"
                  max="1024"
                  step="64"
                  value={trainingConfig.resolution}
                  onChange={(e) =>
                    updateConfig("resolution", parseInt(e.target.value, 10))
                  }
                  className="slider"
                />
                <span className="slider-value">{trainingConfig.resolution}px</span>
              </div>

              <div className="form-group">
                <label>批次大小</label>
                <input
                  type="range"
                  min="1"
                  max="8"
                  value={trainingConfig.train_batch_size}
                  onChange={(e) =>
                    updateConfig("train_batch_size", parseInt(e.target.value, 10))
                  }
                  className="slider"
                />
                <span className="slider-value">{trainingConfig.train_batch_size}</span>
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>最大步數</label>
                <input
                  type="range"
                  min="200"
                  max="10000"
                  step="100"
                  value={trainingConfig.max_train_steps}
                  onChange={(e) =>
                    updateConfig("max_train_steps", parseInt(e.target.value, 10))
                  }
                  className="slider"
                />
                <span className="slider-value">{trainingConfig.max_train_steps}</span>
              </div>

              <div className="form-group">
                <label>梯度累積</label>
                <input
                  type="range"
                  min="1"
                  max="16"
                  value={trainingConfig.gradient_accumulation_steps}
                  onChange={(e) =>
                    updateConfig(
                      "gradient_accumulation_steps",
                      parseInt(e.target.value, 10)
                    )
                  }
                  className="slider"
                />
                <span className="slider-value">
                  {trainingConfig.gradient_accumulation_steps}
                </span>
              </div>
            </div>

            <button
              className="btn btn-primary btn-lg"
              onClick={submitTraining}
              disabled={isLoading}
              type="button"
            >
              {isLoading ? <Loading size="small" text="提交中..." /> : (
                <>
                  <Play size={16} />
                  開始訓練
                </>
              )}
            </button>
          </div>

          <div className="training-jobs">
            {jobs.length === 0 ? (
              <div className="training-job-item">
                <div className="job-name">尚無訓練任務</div>
                <div className="job-progress">提交後會在此顯示</div>
              </div>
            ) : (
              jobs.map((job) => (
                <div
                  key={job.job_id}
                  className={`training-job-item ${selectedJobId === job.job_id ? "selected" : ""}`}
                  onClick={() => setSelectedJobId(job.job_id)}
                >
                  <div className="job-header">
                    <p className="job-name">{job.project_name}</p>
                    <span className="job-progress">{job.job_id}</span>
                  </div>
                  <div className="job-progress">{job.submitted_at}</div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="training-monitor-panel">
          {!selectedJobId ? (
            <div className="empty-state">
              <Brain className="empty-icon" />
              <p>選擇任務以查看詳細狀態</p>
            </div>
          ) : !jobStatus ? (
            <Loading size="medium" text="載入任務狀態..." />
          ) : (
            <>
              <div className="metrics-grid">
                <div className="metric-card">
                  <p className="metric-value">{jobStatus.status || "UNKNOWN"}</p>
                  <p className="metric-label">狀態</p>
                </div>
                <div className="metric-card">
                  <p className="metric-value">{percent ?? "--"}%</p>
                  <p className="metric-label">進度</p>
                </div>
                <div className="metric-card">
                  <p className="metric-value">{progress?.step ?? "--"}</p>
                  <p className="metric-label">Step</p>
                </div>
                <div className="metric-card">
                  <p className="metric-value">{progress?.lr ? Number(progress.lr).toExponential(2) : "--"}</p>
                  <p className="metric-label">LR</p>
                </div>
              </div>

              {progress?.message && (
                <div className="alert alert-info">{progress.message}</div>
              )}

              {jobStatus.error && (
                <div className="alert alert-danger">
                  <XCircle size={16} /> {jobStatus.error}
                </div>
              )}

              {jobStatus.result?.model_id && (
                <div className="alert alert-success">
                  LoRA 已輸出：{jobStatus.result.model_id}
                </div>
              )}

              <div style={{ display: "flex", gap: 8 }}>
                <button
                  className="btn btn-danger"
                  type="button"
                  onClick={() => cancelTraining(selectedJobId)}
                >
                  <XCircle size={16} />
                  取消訓練
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default TrainingMonitor;
