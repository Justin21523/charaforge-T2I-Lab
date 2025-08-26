// frontend/react_app/src/components/training/TrainingMonitor.jsx
import React, { useState, useCallback, useEffect } from 'react';
import { Brain, Play, Pause, Download, TrendingUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useAPI } from '../../hooks/useAPI';
import { useLocalStorage } from '../../hooks/useLocalStorage';
import { LORA_TRAINING_PRESETS } from '../../utils/constants';
import { formatDuration } from '../../utils/helpers';
import Loading from '../common/Loading';
import toast from 'react-hot-toast';
import '../../styles/components/Training.css';

const TrainingMonitor = () => {
  const { apiCall, isLoading } = useAPI();
  const [selectedPreset, setSelectedPreset] = useState('character');
  const [trainingJobs, setTrainingJobs] = useState([]);
  const [selectedJob, setSelectedJob] = useState(null);
  const [jobDetails, setJobDetails] = useState(null);
  const [trainingConfig, setTrainingConfig] = useLocalStorage('training-config', {
    run_id: '',
    dataset_name: '',
    ...LORA_TRAINING_PRESETS.character
  });

  // Fetch training jobs
  const refreshJobs = useCallback(async () => {
    try {
      const jobs = await apiCall(
        () => apiService.listTrainingJobs(),
        null,
        { showLoading: false, showError: false }
      );
      setTrainingJobs(jobs || []);
    } catch (error) {
      console.error('Failed to refresh training jobs:', error);
    }
  }, [apiCall]);

  useEffect(() => {
    refreshJobs();
    const interval = setInterval(refreshJobs, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, [refreshJobs]);

  // Fetch job details when selection changes
  useEffect(() => {
    if (selectedJob) {
      fetchJobDetails(selectedJob);
      const interval = setInterval(() => fetchJobDetails(selectedJob), 5000);
      return () => clearInterval(interval);
    }
  }, [selectedJob]);

  const fetchJobDetails = useCallback(async (runId) => {
    try {
      const details = await apiCall(
        () => apiService.getTrainingStatus(runId),
        null,
        { showLoading: false, showError: false }
      );
      setJobDetails(details);
    } catch (error) {
      console.error('Failed to fetch job details:', error);
    }
  }, [apiCall]);

  const handlePresetChange = useCallback((presetName) => {
    setSelectedPreset(presetName);
    const preset = LORA_TRAINING_PRESETS[presetName];
    setTrainingConfig(prev => ({
      ...prev,
      ...preset,
      run_id: prev.run_id,
      dataset_name: prev.dataset_name
    }));
  }, [setTrainingConfig]);

  const submitTrainingJob = useCallback(async () => {
    if (!trainingConfig.run_id.trim() || !trainingConfig.dataset_name.trim()) {
      toast.error('請填寫任務 ID 和數據集名稱');
      return;
    }

    try {
      const result = await apiCall(
        () => apiService.submitTrainingJob(trainingConfig),
        null,
        {
          showLoading: true,
          showSuccess: true,
          successMessage: '訓練任務提交成功'
        }
      );

      if (result.run_id || result.job_id) {
        const jobId = result.run_id || result.job_id;
        setSelectedJob(jobId);
        refreshJobs();

        // Generate new run_id for next job
        setTrainingConfig(prev => ({
          ...prev,
          run_id: `lora_train_${Date.now()}`
        }));
      }
    } catch (error) {
      console.error('Training submission failed:', error);
    }
  }, [trainingConfig, apiCall, setTrainingConfig]);

  const cancelTraining = useCallback(async (runId) => {
    try {
      await apiCall(
        () => apiService.cancelTraining(runId),
        null,
        {
          showSuccess: true,
          successMessage: '訓練任務已取消'
        }
      );
      refreshJobs();
    } catch (error) {
      console.error('Failed to cancel training:', error);
    }
  }, [apiCall]);

  const downloadModel = useCallback(async (runId) => {
    try {
      const blob = await apiCall(
        () => apiService.exportLora(runId),
        null,
        {
          showLoading: true,
          showSuccess: true,
          successMessage: '模型下載完成'
        }
      );

      downloadBlob(blob, `lora_${runId}.safetensors`);
    } catch (error) {
      console.error('Failed to download model:', error);
    }
  }, [apiCall]);

  const updateConfig = useCallback((key, value) => {
    setTrainingConfig(prev => ({ ...prev, [key]: value }));
  }, [setTrainingConfig]);

  return (
    <div className="training-monitor">
      <div className="panel-header">
        <h2 className="panel-title">
          <Brain className="title-icon" />
          LoRA 訓練監控
        </h2>
        <button
          className="btn btn-secondary"
          onClick={refreshJobs}
        >
          <TrendingUp size={16} />
          刷新任務
        </button>
      </div>

      <div className="training-content">
        <div className="training-controls">
          <h3 className="section-title">提交訓練任務</h3>

          <div className="preset-selector">
            {Object.entries(LORA_TRAINING_PRESETS).map(([key, preset]) => (
              <button
                key={key}
                className={`preset-button ${selectedPreset === key ? 'active' : ''}`}
                onClick={() => handlePresetChange(key)}
              >
                <div className="preset-title">
                  {key === 'character' ? '角色訓練' : '風格訓練'}
                </div>
                <div className="preset-description">
                  Rank {preset.rank}, {preset.max_train_steps} 步
                </div>
              </button>
            ))}
          </div>

          <div className="training-form">
            <div className="form-group">
              <label>任務 ID</label>
              <input
                type="text"
                value={trainingConfig.run_id}
                onChange={(e) => updateConfig('run_id', e.target.value)}
                placeholder="例如: char_alice_v1"
                className="input"
              />
            </div>

            <div className="form-group">
              <label>數據集名稱</label>
              <input
                type="text"
                value={trainingConfig.dataset_name}
                onChange={(e) => updateConfig('dataset_name', e.target.value)}
                placeholder="數據集目錄名稱"
                className="input"
              />
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>LoRA Rank</label>
                <input
                  type="range"
                  min="4"
                  max="128"
                  step="4"
                  value={trainingConfig.rank}
                  onChange={(e) => updateConfig('rank', parseInt(e.target.value))}
                  className="slider"
                />
                <span className="slider-value">{trainingConfig.rank}</span>
              </div>

              <div className="form-group">
                <label>學習率</label>
                <input
                  type="number"
                  value={trainingConfig.learning_rate}
                  onChange={(e) => updateConfig('learning_rate', parseFloat(e.target.value))}
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
                  onChange={(e) => updateConfig('resolution', parseInt(e.target.value))}
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
                  value={trainingConfig.batch_size}
                  onChange={(e) => updateConfig('batch_size', parseInt(e.target.value))}
                  className="slider"
                />
                <span className="slider-value">{trainingConfig.batch_size}</span>
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>最大步數</label>
                <input
                  type="range"
                  min="500"
                  max="10000"
                  step="100"
                  value={trainingConfig.max_train_steps}
                  onChange={(e) => updateConfig('max_train_steps', parseInt(e.target.value))}
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
                  onChange={(e) => updateConfig('gradient_accumulation_steps', parseInt(e.target.value))}
                  className="slider"
                />
                <span className="slider-value">{trainingConfig.gradient_accumulation_steps}</span>
              </div>
            </div>

            <button
              className="btn btn-primary btn-lg"
              onClick={submitTrainingJob}
              disabled={isLoading}
            >
              {isLoading ? (
                <Loading size="small" text="提交中..." />
              ) : (
                <>
                  <Play size={16} />
                  開始訓練
                </>
              )}
            </button>
          </div>

          <div className="training-jobs">
            <h4>訓練任務列表</h4>
            {trainingJobs.length === 0 ? (
              <div className="empty-state">
                <Brain className="empty-icon" />
                <p>沒有訓練任務</p>
              </div>
            ) : (
              <div className="job-list">
                {trainingJobs.map((job) => (
                  <div
                    key={job.run_id}
                    className={`training-job-item ${selectedJob === job.run_id ? 'selected' : ''}`}
                    onClick={() => setSelectedJob(job.run_id)}
                  >
                    <div className="job-header">
                      <h5 className="job-name">{job.run_id}</h5>
                      <span className={`job-status ${job.status}`}>
                        {job.status}
                      </span>
                    </div>
                    <div className="job-progress">
                      {job.current_step || 0} / {job.max_steps || 0} 步
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="training-monitor-panel">
          <h3 className="section-title">訓練監控</h3>

          {!selectedJob ? (
            <div className="empty-state">
              <TrendingUp className="empty-icon" />
              <p>選擇訓練任務查看詳情</p>
            </div>
          ) : !jobDetails ? (
            <Loading size="medium" text="載入訓練詳情..." />
          ) : (
            <div className="job-details">
              <div className="job-info">
                <h4>任務: {jobDetails.run_id}</h4>
                <div className="metrics-grid">
                  <div className="metric-card">
                    <div className="metric-value">{jobDetails.status}</div>
                    <div className="metric-label">狀態</div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-value">
                      {jobDetails.current_step || 0}
                    </div>
                    <div className="metric-label">當前步數</div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-value">
                      {jobDetails.loss ? jobDetails.loss.toFixed(4) : 'N/A'}
                    </div>
                    <div className="metric-label">當前損失</div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-value">
                      {jobDetails.elapsed_time ? formatDuration(jobDetails.elapsed_time) : 'N/A'}
                    </div>
                    <div className="metric-label">運行時間</div>
                  </div>
                </div>

                {jobDetails.progress && (
                  <div className="progress-section">
                    <div className="progress-bar-large">
                      <div
                        className="progress-fill"
                        style={{
                          width: `${(jobDetails.current_step / jobDetails.max_steps) * 100}%`
                        }}
                      />
                    </div>
                    <div className="progress-text">
                      {Math.round((jobDetails.current_step / jobDetails.max_steps) * 100)}% 完成
                    </div>
                  </div>
                )}
              </div>

              {jobDetails.loss_history && jobDetails.loss_history.length > 0 && (
                <div className="metrics-chart">
                  <h4>損失曲線</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={jobDetails.loss_history.map((loss, index) => ({ step: index, loss }))}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="step" />
                      <YAxis />
                      <Tooltip formatter={(value) => [value.toFixed(4), '損失']} />
                      <Line
                        type="monotone"
                        dataKey="loss"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              {jobDetails.sample_images && jobDetails.sample_images.length > 0 && (
                <div className="sample-images">
                  <h4>訓練樣本</h4>
                  <div className="sample-gallery">
                    {jobDetails.sample_images.map((image, index) => (
                      <img
                        key={index}
                        src={image.url || image}
                        alt={`Sample ${index + 1}`}
                        className="sample-image"
                        loading="lazy"
                      />
                    ))}
                  </div>
                </div>
              )}

              <div className="job-actions">
                {jobDetails.status === 'running' && (
                  <button
                    className="btn btn-danger"
                    onClick={() => cancelTraining(jobDetails.run_id)}
                  >
                    <Pause size={16} />
                    停止訓練
                  </button>
                )}

                {jobDetails.status === 'completed' && (
                  <button
                    className="btn btn-primary"
                    onClick={() => downloadModel(jobDetails.run_id)}
                  >
                    <Download size={16} />
                    下載模型
                  </button>
                )}
              </div>

              {jobDetails.error && (
                <div className="error-section">
                  <h4>錯誤信息</h4>
                  <div className="error-message">
                    <pre>{jobDetails.error}</pre>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TrainingMonitor;