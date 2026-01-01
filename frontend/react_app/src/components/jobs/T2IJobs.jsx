import React, { useCallback, useEffect, useMemo, useState } from "react";
import { ClipboardList, RefreshCw, Trash2, XCircle } from "lucide-react";
import toast from "react-hot-toast";
import { useAPI } from "../../hooks/useAPI";
import Loading from "../common/Loading";
import "../../styles/components/T2IJobs.css";

const STATUS_OPTIONS = [
  { value: "", label: "全部" },
  { value: "queued", label: "queued" },
  { value: "running", label: "running" },
  { value: "succeeded", label: "succeeded" },
  { value: "failed", label: "failed" },
  { value: "canceled", label: "canceled" },
];

const formatTime = (value) => {
  const ts = Number(value || 0);
  if (!Number.isFinite(ts) || ts <= 0) return "";
  try {
    return new Date(ts * 1000).toLocaleString();
  } catch (e) {
    return "";
  }
};

const shortId = (jobId) => {
  const raw = String(jobId || "");
  return raw.length > 10 ? `${raw.slice(0, 8)}…` : raw;
};

const T2IJobs = () => {
  const { apiCall, apiService, isLoading } = useAPI();

  const [authRole, setAuthRole] = useState("");
  const isAdmin = useMemo(() => authRole === "admin", [authRole]);

  const [jobs, setJobs] = useState([]);
  const [count, setCount] = useState(0);
  const [limit, setLimit] = useState(50);
  const [status, setStatus] = useState("");
  const [all, setAll] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState({});
  const [detailsById, setDetailsById] = useState({});

  const [cleanupTtl, setCleanupTtl] = useState(0);
  const [cleanupDryRun, setCleanupDryRun] = useState(true);
  const [cleanupDeleteRecords, setCleanupDeleteRecords] = useState(false);
  const [cleanupOnlyTerminal, setCleanupOnlyTerminal] = useState(true);
  const [cleanupAll, setCleanupAll] = useState(false);
  const [cleanupLimit, setCleanupLimit] = useState(200);
  const [cleanupResult, setCleanupResult] = useState(null);

  const refreshAuthMe = useCallback(async () => {
    try {
      const me = await apiCall(() => apiService.authMe(), null, {
        showLoading: false,
        showError: false,
      });
      setAuthRole(String(me?.role || ""));
    } catch (e) {
      setAuthRole("");
    }
  }, [apiCall, apiService]);

  const refreshJobs = useCallback(async () => {
    const useAll = Boolean(all && isAdmin);
    const result = await apiCall(
      () => apiService.listT2IJobs({ limit, status: status || null, all: useAll }),
      null,
      { showLoading: true, showError: true }
    );
    const list = Array.isArray(result?.jobs) ? result.jobs : [];
    setJobs(list);
    setCount(Number(result?.count || list.length || 0));
  }, [all, apiCall, apiService, isAdmin, limit, status]);

  useEffect(() => {
    refreshAuthMe();
  }, [refreshAuthMe]);

  useEffect(() => {
    refreshJobs();
  }, [refreshJobs]);

  const toggleDetails = useCallback(
    async (jobId) => {
      const id = String(jobId || "");
      if (!id) return;
      setDetailsOpen((prev) => ({ ...prev, [id]: !prev[id] }));
      if (detailsById[id]) return;

      setDetailsById((prev) => ({ ...prev, [id]: { loading: true } }));
      try {
        const snapshot = await apiCall(
          () => apiService.getT2IJobStatus(id),
          null,
          { showLoading: false, showError: true }
        );
        setDetailsById((prev) => ({ ...prev, [id]: { loading: false, snapshot } }));
      } catch (e) {
        setDetailsById((prev) => ({ ...prev, [id]: { loading: false, error: e?.message || "failed" } }));
      }
    },
    [apiCall, apiService, detailsById]
  );

  const cancelJob = useCallback(
    async (jobId) => {
      const id = String(jobId || "");
      if (!id) return;
      await apiCall(() => apiService.cancelT2IJob(id), null, {
        showLoading: true,
        showSuccess: true,
        successMessage: "已送出取消請求",
      });
      await refreshJobs();
    },
    [apiCall, apiService, refreshJobs]
  );

  const deleteJob = useCallback(
    async (jobId) => {
      const id = String(jobId || "");
      if (!id) return;
      if (!window.confirm(`確定要刪除 job ${id}？（預設會連同輸出一起刪除）`)) return;

      await apiCall(() => apiService.deleteT2IJob(id, { deleteOutputs: true }), null, {
        showLoading: true,
        showSuccess: true,
        successMessage: "已刪除 job",
      });
      setDetailsById((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
      await refreshJobs();
    },
    [apiCall, apiService, refreshJobs]
  );

  const runCleanup = useCallback(
    async (dryRunOverride) => {
      const dryRun = dryRunOverride !== undefined ? Boolean(dryRunOverride) : Boolean(cleanupDryRun);
      const useAll = Boolean(cleanupAll && isAdmin);
      const ttlSeconds = Number(cleanupTtl || 0) || null;
      if (!ttlSeconds || ttlSeconds <= 0) {
        toast.error("請填寫 ttl_seconds（> 0）");
        return;
      }

      const result = await apiCall(
        () =>
          apiService.cleanupT2IJobs({
            ttlSeconds,
            dryRun,
            deleteRecords: cleanupDeleteRecords,
            all: useAll,
            onlyTerminal: cleanupOnlyTerminal,
            limit: cleanupLimit,
          }),
        null,
        {
          showLoading: true,
          showSuccess: true,
          successMessage: dryRun ? "Dry-run 完成" : "清理完成",
        }
      );
      setCleanupResult(result);
      if (!dryRun) {
        await refreshJobs();
      }
    },
    [
      apiCall,
      apiService,
      cleanupAll,
      cleanupDeleteRecords,
      cleanupDryRun,
      cleanupLimit,
      cleanupOnlyTerminal,
      cleanupTtl,
      isAdmin,
      refreshJobs,
    ]
  );

  return (
    <div className="t2i-jobs">
      <div className="panel-header">
        <h2 className="panel-title">
          <ClipboardList className="title-icon" />
          T2I Jobs ({count})
        </h2>
        <button className="btn btn-secondary" onClick={refreshJobs}>
          <RefreshCw size={16} />
          刷新
        </button>
      </div>

      <div className="t2i-jobs-controls">
        <div className="t2i-jobs-filter card">
          <div className="t2i-jobs-filter-row">
            <div className="form-group">
              <label>狀態</label>
              <select className="select" value={status} onChange={(e) => setStatus(e.target.value)}>
                {STATUS_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label>limit</label>
              <input
                className="input"
                type="number"
                min={1}
                max={200}
                value={limit}
                onChange={(e) => setLimit(Number(e.target.value || 50))}
              />
            </div>

            <div className="form-group t2i-jobs-checkbox">
              <label>admin all</label>
              <input
                type="checkbox"
                checked={Boolean(all)}
                onChange={(e) => setAll(e.target.checked)}
                disabled={!isAdmin}
                title={!isAdmin ? "需要 admin" : "列出全部 owner 的 jobs"}
              />
            </div>
          </div>
        </div>

        <div className="t2i-jobs-cleanup card">
          <div className="t2i-jobs-cleanup-title">清理輸出 (POST /t2i/jobs/cleanup)</div>
          <div className="t2i-jobs-cleanup-row">
            <div className="form-group">
              <label>ttl_seconds</label>
              <input
                className="input"
                type="number"
                min={1}
                value={cleanupTtl}
                onChange={(e) => setCleanupTtl(Number(e.target.value || 0))}
                placeholder="例如 86400"
              />
            </div>

            <div className="form-group t2i-jobs-checkbox">
              <label>dry_run</label>
              <input
                type="checkbox"
                checked={Boolean(cleanupDryRun)}
                onChange={(e) => setCleanupDryRun(e.target.checked)}
              />
            </div>

            <div className="form-group t2i-jobs-checkbox">
              <label>delete_records</label>
              <input
                type="checkbox"
                checked={Boolean(cleanupDeleteRecords)}
                onChange={(e) => setCleanupDeleteRecords(e.target.checked)}
              />
            </div>

            <div className="form-group t2i-jobs-checkbox">
              <label>only_terminal</label>
              <input
                type="checkbox"
                checked={Boolean(cleanupOnlyTerminal)}
                onChange={(e) => setCleanupOnlyTerminal(e.target.checked)}
              />
            </div>

            <div className="form-group t2i-jobs-checkbox">
              <label>admin all</label>
              <input
                type="checkbox"
                checked={Boolean(cleanupAll)}
                onChange={(e) => setCleanupAll(e.target.checked)}
                disabled={!isAdmin}
              />
            </div>

            <div className="form-group">
              <label>limit</label>
              <input
                className="input"
                type="number"
                min={1}
                max={2000}
                value={cleanupLimit}
                onChange={(e) => setCleanupLimit(Number(e.target.value || 200))}
              />
            </div>

            <button className="btn btn-secondary" type="button" onClick={() => runCleanup(true)}>
              Dry run
            </button>
            <button className="btn btn-primary" type="button" onClick={() => runCleanup(false)}>
              執行清理
            </button>
          </div>

          {cleanupResult && (
            <pre className="t2i-jobs-cleanup-result">
              {JSON.stringify(cleanupResult, null, 2)}
            </pre>
          )}
        </div>
      </div>

      {isLoading && (
        <div className="t2i-jobs-loading">
          <Loading />
        </div>
      )}

      <div className="t2i-jobs-list card">
        {jobs.length === 0 ? (
          <div className="t2i-jobs-empty">沒有 jobs</div>
        ) : (
          jobs.map((job) => {
            const jobId = String(job?.job_id || "");
            const isOpen = Boolean(detailsOpen[jobId]);
            const detail = detailsById[jobId] || null;
            const progress = job?.progress;
            const progressText =
              typeof progress?.step === "number" && typeof progress?.total === "number"
                ? `${progress.step}/${progress.total}`
                : "";

            return (
              <div key={jobId} className="t2i-job-row">
                <div className="t2i-job-main">
                  <button
                    type="button"
                    className="t2i-job-id"
                    onClick={() => toggleDetails(jobId)}
                    title={jobId}
                  >
                    {shortId(jobId)}
                  </button>
                  <span className={`t2i-job-status t2i-job-status-${job?.status || "unknown"}`}>
                    {String(job?.status || "")}
                  </span>
                  {progressText && <span className="t2i-job-progress">{progressText}</span>}
                  <span className="t2i-job-time">{formatTime(job?.created_at)}</span>

                  <div className="t2i-job-actions">
                    {(job?.status === "queued" || job?.status === "running") && (
                      <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        onClick={() => cancelJob(jobId)}
                      >
                        <XCircle size={14} />
                        取消
                      </button>
                    )}
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={() => deleteJob(jobId)}
                    >
                      <Trash2 size={14} />
                      刪除
                    </button>
                  </div>
                </div>

                {isOpen && (
                  <div className="t2i-job-details">
                    {detail?.loading && <div className="t2i-job-details-loading">載入中…</div>}
                    {detail?.error && (
                      <div className="t2i-job-details-error">{String(detail.error)}</div>
                    )}
                    {detail?.snapshot && (
                      <div className="t2i-job-details-body">
                        {Array.isArray(detail.snapshot?.image_path) &&
                          detail.snapshot.image_path.length > 0 && (
                            <div className="t2i-job-images">
                              {detail.snapshot.image_path.map((url) => (
                                <a
                                  key={url}
                                  href={url}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="t2i-job-image"
                                >
                                  <img src={url} alt={jobId} />
                                </a>
                              ))}
                            </div>
                          )}
                        {detail.snapshot?.error && (
                          <pre className="t2i-job-error">
                            {JSON.stringify(detail.snapshot.error, null, 2)}
                          </pre>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default T2IJobs;

