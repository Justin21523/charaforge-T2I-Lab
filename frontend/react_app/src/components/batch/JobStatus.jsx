// frontend/react_app/src/components/batch/JobStatus.jsx
import React, { useState, useEffect, useCallback } from 'react';
import { Clock, CheckCircle, XCircle, AlertCircle, Download } from 'lucide-react';
import { useAPI } from '../../hooks/useAPI';
import { formatDuration } from '../../utils/helpers';
import Loading from '../common/Loading';

const JobStatus = ({ jobId, onRefresh }) => {
  const { apiCall } = useAPI();
  const [jobDetails, setJobDetails] = useState(null);
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const fetchJobStatus = useCallback(async () => {
    if (!jobId) return;

    setIsLoading(true);
    try {
      const status = await apiCall(
        () => apiService.getJobStatus(jobId),
        null,
        { showLoading: false, showError: false }
      );

      setJobDetails(status);

      if (status.results) {
        setResults(status.results);
      }
    } catch (error) {
      console.error('Failed to fetch job status:', error);
    } finally {
      setIsLoading(false);
    }
  }, [jobId, apiCall]);

  useEffect(() => {
    fetchJobStatus();

    // Auto refresh for running jobs
    let interval;
    if (jobDetails?.status === 'running' || jobDetails?.status === 'pending') {
      interval = setInterval(fetchJobStatus, 3000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [fetchJobStatus, jobDetails?.status]);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'pending':
        return <Clock className="status-icon pending" />;
      case 'running':
        return <AlertCircle className="status-icon running" />;
      case 'completed':
        return <CheckCircle className="status-icon completed" />;
      case 'failed':
        return <XCircle className="status-icon failed" />;
      default:
        return <Clock className="status-icon" />;
    }
  };

  if (isLoading && !jobDetails) {
    return <Loading size="medium" text="載入任務狀態..." />;
  }

  if (!jobDetails) {
    return (
      <div className="job-status-empty">
        <p>選擇任務以查看詳細狀態</p>
      </div>
    );
  }

  return (
    <div className="job-status-details">
      <div className="status-header">
        <div className="status-info">
          {getStatusIcon(jobDetails.status)}
          <div>
            <h4>任務詳情</h4>
            <p>ID: {jobId}</p>
          </div>
        </div>
        <button
          className="btn btn-secondary btn-sm"
          onClick={fetchJobStatus}
        >
          刷新
        </button>
      </div>

      <div className="status-content">
        <div className="status-metrics">
          <div className="metric">
            <span className="metric-label">狀態</span>
            <span className={`metric-value status-${jobDetails.status}`}>
              {jobDetails.status}
            </span>
          </div>

          <div className="metric">
            <span className="metric-label">進度</span>
            <span className="metric-value">
              {jobDetails.completed || 0} / {jobDetails.total || 0}
            </span>
          </div>

          {jobDetails.elapsed_time && (
            <div className="metric">
              <span className="metric-label">耗時</span>
              <span className="metric-value">
                {formatDuration(jobDetails.elapsed_time)}
              </span>
            </div>
          )}
        </div>

        {jobDetails.total > 0 && (
          <div className="progress-section">
            <div className="progress-bar-large">
              <div
                className="progress-fill"
                style={{
                  width: `${(jobDetails.completed / jobDetails.total) * 100}%`
                }}
              />
            </div>
            <div className="progress-text">
              {Math.round((jobDetails.completed / jobDetails.total) * 100)}% 完成
            </div>
          </div>
        )}

        {jobDetails.error && (
          <div className="error-message">
            <XCircle className="error-icon" />
            <span>錯誤: {jobDetails.error}</span>
          </div>
        )}

        {results.length > 0 && (
          <div className="results-section">
            <h4>生成結果</h4>
            <div className="results-gallery">
              {results.slice(0, 12).map((result, index) => (
                <div key={index} className="result-item">
                  {result.image_path && (
                    <img
                      src={result.image_path}
                      alt={`Result ${index + 1}`}
                      className="result-image"
                      loading="lazy"
                    />
                  )}
                  {result.error && (
                    <div className="result-error">
                      <XCircle className="error-icon" />
                      <span>失敗</span>
                    </div>
                  )}
                </div>
              ))}
            </div>

            {results.length > 12 && (
              <p className="results-more">
                還有 {results.length - 12} 個結果...
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default JobStatus;