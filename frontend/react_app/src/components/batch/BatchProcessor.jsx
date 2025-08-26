// frontend/react_app/src/components/batch/BatchProcessor.jsx
import React, { useState, useCallback, useEffect } from 'react';
import { Upload, Download, Play, Pause, Trash2, FileText, Clock } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { useAPI } from '../../hooks/useAPI';
import { formatDuration, downloadBlob } from '../../utils/helpers';
import JobStatus from './JobStatus';
import Loading from '../common/Loading';
import toast from 'react-hot-toast';
import '../../styles/components/Batch.css';

const BatchProcessor = () => {
  const { apiCall, isLoading } = useAPI();
  const [activeTab, setActiveTab] = useState('csv');
  const [jobs, setJobs] = useState([]);
  const [selectedJob, setSelectedJob] = useState(null);
  const [batchParams, setBatchParams] = useState({
    prompts: '',
    negative: 'lowres, blurry, bad anatomy',
    width: 768,
    height: 768,
    steps: 25,
    cfg_scale: 7.5,
    sampler: 'DPM++ 2M Karras'
  });

  // File upload handlers
  const onCsvDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('file_type', 'csv');

      const result = await apiCall(
        () => apiService.submitBatchJob({ file: formData, type: 'csv' }),
        null,
        {
          showLoading: true,
          showSuccess: true,
          successMessage: 'CSV 批次任務提交成功'
        }
      );

      if (result.job_id) {
        refreshJobs();
        setSelectedJob(result.job_id);
      }
    } catch (error) {
      console.error('CSV upload failed:', error);
    }
  }, [apiCall]);

  const onJsonDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    try {
      const text = await file.text();
      const jobData = JSON.parse(text);

      const result = await apiCall(
        () => apiService.submitBatchJob(jobData),
        null,
        {
          showLoading: true,
          showSuccess: true,
          successMessage: 'JSON 批次任務提交成功'
        }
      );

      if (result.job_id) {
        refreshJobs();
        setSelectedJob(result.job_id);
      }
    } catch (error) {
      toast.error('JSON 格式錯誤或提交失敗');
      console.error('JSON upload failed:', error);
    }
  }, [apiCall]);

  const {
    getRootProps: getCsvRootProps,
    getInputProps: getCsvInputProps,
    isDragActive: isCsvDragActive
  } = useDropzone({
    onDrop: onCsvDrop,
    accept: { 'text/csv': ['.csv'] },
    multiple: false
  });

  const {
    getRootProps: getJsonRootProps,
    getInputProps: getJsonInputProps,
    isDragActive: isJsonDragActive
  } = useDropzone({
    onDrop: onJsonDrop,
    accept: { 'application/json': ['.json'] },
    multiple: false
  });

  // Job management
  const refreshJobs = useCallback(async () => {
    try {
      const jobList = await apiCall(
        () => apiService.listJobs(),
        null,
        { showLoading: false, showError: false }
      );
      setJobs(jobList || []);
    } catch (error) {
      console.error('Failed to refresh jobs:', error);
    }
  }, [apiCall]);

  useEffect(() => {
    refreshJobs();
    const interval = setInterval(refreshJobs, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, [refreshJobs]);

  const submitManualBatch = useCallback(async () => {
    if (!batchParams.prompts.trim()) {
      toast.error('請輸入至少一個提示詞');
      return;
    }

    const promptList = batchParams.prompts
      .split('\n')
      .map(p => p.trim())
      .filter(p => p);

    const tasks = promptList.map((prompt, index) => ({
      id: index + 1,
      prompt: prompt,
      negative: batchParams.negative,
      width: batchParams.width,
      height: batchParams.height,
      steps: batchParams.steps,
      cfg_scale: batchParams.cfg_scale,
      sampler: batchParams.sampler,
      seed: -1
    }));

    const jobData = {
      job_name: `manual_batch_${tasks.length}_tasks`,
      tasks: tasks
    };

    try {
      const result = await apiCall(
        () => apiService.submitBatchJob(jobData),
        null,
        {
          showLoading: true,
          showSuccess: true,
          successMessage: `批次任務提交成功 (${tasks.length} 個任務)`
        }
      );

      if (result.job_id) {
        refreshJobs();
        setSelectedJob(result.job_id);
        setBatchParams(prev => ({ ...prev, prompts: '' }));
      }
    } catch (error) {
      console.error('Manual batch submission failed:', error);
    }
  }, [batchParams, apiCall]);

  const cancelJob = useCallback(async (jobId) => {
    try {
      await apiCall(
        () => apiService.cancelJob(jobId),
        null,
        {
          showSuccess: true,
          successMessage: '任務已取消'
        }
      );
      refreshJobs();
    } catch (error) {
      console.error('Failed to cancel job:', error);
    }
  }, [apiCall]);

  const downloadResults = useCallback(async (jobId) => {
    try {
      const blob = await apiCall(
        () => apiService.downloadJobResults(jobId),
        null,
        {
          showLoading: true,
          showSuccess: true,
          successMessage: '結果下載完成'
        }
      );

      downloadBlob(blob, `batch_results_${jobId}.zip`);
    } catch (error) {
      console.error('Failed to download results:', error);
    }
  }, [apiCall]);

  return (
    <div className="batch-processor">
      <div className="panel-header">
        <h2 className="panel-title">
          <Upload className="title-icon" />
          批次處理
        </h2>
        <button
          className="btn btn-secondary"
          onClick={refreshJobs}
        >
          <Clock size={16} />
          刷新任務
        </button>
      </div>

      <div className="batch-content">
        <div className="batch-section">
          <h3 className="section-title">提交批次任務</h3>

          <div className="batch-tabs">
            <div className="tab-buttons">
              <button
                className={`tab-button ${activeTab === 'csv' ? 'active' : ''}`}
                onClick={() => setActiveTab('csv')}
              >
                CSV 上傳
              </button>
              <button
                className={`tab-button ${activeTab === 'json' ? 'active' : ''}`}
                onClick={() => setActiveTab('json')}
              >
                JSON 上傳
              </button>
              <button
                className={`tab-button ${activeTab === 'manual' ? 'active' : ''}`}
                onClick={() => setActiveTab('manual')}
              >
                手動設定
              </button>
            </div>
          </div>

          {activeTab === 'csv' && (
            <div className="tab-content">
              <div
                {...getCsvRootProps()}
                className={`file-upload-area ${isCsvDragActive ? 'active' : ''}`}
              >
                <input {...getCsvInputProps()} />
                <div className="upload-content">
                  <FileText className="upload-icon" />
                  {isCsvDragActive ? (
                    <p>放開以上傳 CSV 檔案</p>
                  ) : (
                    <div>
                      <p>拖放 CSV 檔案或點擊選擇</p>
                      <p className="upload-hint">支援標準的批次生成格式</p>
                    </div>
                  )}
                </div>
              </div>

              <div className="upload-instructions">
                <h4>CSV 格式範例:</h4>
                <pre>{`prompt,negative,width,height,steps,seed
"anime girl, blue hair",lowres,768,768,25,-1
"cat, cute, fluffy",blurry,512,512,20,12345`}</pre>
              </div>
            </div>
          )}

          {activeTab === 'json' && (
            <div className="tab-content">
              <div
                {...getJsonRootProps()}
                className={`file-upload-area ${isJsonDragActive ? 'active' : ''}`}
              >
                <input {...getJsonInputProps()} />
                <div className="upload-content">
                  <FileText className="upload-icon" />
                  {isJsonDragActive ? (
                    <p>放開以上傳 JSON 檔案</p>
                  ) : (
                    <div>
                      <p>拖放 JSON 檔案或點擊選擇</p>
                      <p className="upload-hint">支援結構化的任務配置</p>
                    </div>
                  )}
                </div>
              </div>

              <div className="upload-instructions">
                <h4>JSON 格式範例:</h4>
                <pre>{`{
  "job_name": "batch_generation_001",
  "tasks": [
    {
      "prompt": "anime girl, blue hair",
      "negative": "lowres, blurry",
      "width": 768,
      "height": 768,
      "steps": 25
    }
  ]
}`}</pre>
              </div>
            </div>
          )}

          {activeTab === 'manual' && (
            <div className="tab-content">
              <div className="manual-config">
                <div className="control-group">
                  <label>提示詞列表 (每行一個)</label>
                  <textarea
                    value={batchParams.prompts}
                    onChange={(e) => setBatchParams(prev => ({ ...prev, prompts: e.target.value }))}
                    placeholder="anime girl, blue hair&#10;cat, cute, fluffy&#10;landscape, mountains"
                    rows={6}
                    className="textarea"
                  />
                </div>

                <div className="control-group">
                  <label>統一負面提示詞</label>
                  <textarea
                    value={batchParams.negative}
                    onChange={(e) => setBatchParams(prev => ({ ...prev, negative: e.target.value }))}
                    rows={2}
                    className="textarea"
                  />
                </div>

                <div className="control-row">
                  <div className="control-group">
                    <label>寬度</label>
                    <input
                      type="range"
                      min="256"
                      max="2048"
                      step="64"
                      value={batchParams.width}
                      onChange={(e) => setBatchParams(prev => ({ ...prev, width: parseInt(e.target.value) }))}
                      className="slider"
                    />
                    <span className="slider-value">{batchParams.width}px</span>
                  </div>

                  <div className="control-group">
                    <label>高度</label>
                    <input
                      type="range"
                      min="256"
                      max="2048"
                      step="64"
                      value={batchParams.height}
                      onChange={(e) => setBatchParams(prev => ({ ...prev, height: parseInt(e.target.value) }))}
                      className="slider"
                    />
                    <span className="slider-value">{batchParams.height}px</span>
                  </div>
                </div>

                <div className="control-row">
                  <div className="control-group">
                    <label>步數</label>
                    <input
                      type="range"
                      min="1"
                      max="100"
                      value={batchParams.steps}
                      onChange={(e) => setBatchParams(prev => ({ ...prev, steps: parseInt(e.target.value) }))}
                      className="slider"
                    />
                    <span className="slider-value">{batchParams.steps}</span>
                  </div>

                  <div className="control-group">
                    <label>CFG 縮放</label>
                    <input
                      type="range"
                      min="1"
                      max="30"
                      step="0.5"
                      value={batchParams.cfg_scale}
                      onChange={(e) => setBatchParams(prev => ({ ...prev, cfg_scale: parseFloat(e.target.value) }))}
                      className="slider"
                    />
                    <span className="slider-value">{batchParams.cfg_scale}</span>
                  </div>
                </div>

                <div className="control-group">
                  <label>採樣器</label>
                  <select
                    value={batchParams.sampler}
                    onChange={(e) => setBatchParams(prev => ({ ...prev, sampler: e.target.value }))}
                    className="select"
                  >
                    <option value="DPM++ 2M Karras">DPM++ 2M Karras</option>
                    <option value="DPM++ SDE Karras">DPM++ SDE Karras</option>
                    <option value="Euler a">Euler a</option>
                    <option value="Euler">Euler</option>
                    <option value="DDIM">DDIM</option>
                  </select>
                </div>

                <button
                  className="btn btn-primary"
                  onClick={submitManualBatch}
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <Loading size="small" text="提交中..." />
                  ) : (
                    <>
                      <Play size={16} />
                      提交手動批次任務
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>

        <div className="batch-section">
          <h3 className="section-title">任務狀態監控</h3>

          {jobs.length === 0 ? (
            <div className="empty-state">
              <Upload className="empty-icon" />
              <p className="empty-text">沒有批次任務</p>
              <p className="empty-hint">提交第一個批次任務開始</p>
            </div>
          ) : (
            <div className="job-list">
              {jobs.map((job) => (
                <div
                  key={job.job_id}
                  className={`job-item ${selectedJob === job.job_id ? 'selected' : ''}`}
                  onClick={() => setSelectedJob(job.job_id)}
                >
                  <div className="job-header">
                    <h4 className="job-name">{job.job_name || job.job_id}</h4>
                    <span className={`job-status ${job.status}`}>
                      {job.status}
                    </span>
                  </div>

                  <div className="job-info">
                    <span className="job-tasks">
                      {job.completed || 0}/{job.total || 0} 任務
                    </span>
                    <span className="job-time">
                      {job.created_at && new Date(job.created_at).toLocaleString()}
                    </span>
                  </div>

                  {job.progress && (
                    <div className="progress-bar">
                      <div
                        className="progress-fill"
                        style={{ width: `${(job.completed / job.total) * 100}%` }}
                      />
                    </div>
                  )}

                  <div className="job-actions">
                    {job.status === 'running' && (
                      <button
                        className="btn btn-danger btn-sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          cancelJob(job.job_id);
                        }}
                        title="取消任務"
                      >
                        <Pause size={14} />
                      </button>
                    )}

                    {job.status === 'completed' && (
                      <button
                        className="btn btn-primary btn-sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          downloadResults(job.job_id);
                        }}
                        title="下載結果"
                      >
                        <Download size={14} />
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {selectedJob && (
            <JobStatus
              jobId={selectedJob}
              onRefresh={refreshJobs}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default BatchProcessor;