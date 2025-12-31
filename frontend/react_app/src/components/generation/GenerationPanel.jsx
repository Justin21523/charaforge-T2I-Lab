// frontend/react_app/src/components/generation/GenerationPanel.jsx
import React, { useCallback, useEffect, useRef, useState } from "react";
import { Shuffle, Wand2, X } from "lucide-react";
import toast from "react-hot-toast";
import { useAPI } from "../../hooks/useAPI";
import { useLocalStorage } from "../../hooks/useLocalStorage";
import { DEFAULT_GENERATION_PARAMS } from "../../utils/constants";
import { downloadBlob, generateRandomSeed } from "../../utils/helpers";
import Loading from "../common/Loading";
import ImagePreview from "./ImagePreview";
import ParameterControls from "./ParameterControls";
import "../../styles/components/Generation.css";

const GenerationPanel = () => {
  const { apiCall, apiService, isLoading } = useAPI();
  const [params, setParams] = useLocalStorage('generation-params', DEFAULT_GENERATION_PARAMS);
  const [asyncMode, setAsyncMode] = useLocalStorage("t2i-async-mode", true);
  const [controlnetParams, setControlnetParams] = useState({
    enabled: false,
    type: 'pose',
    image: null,
    weight: 1.0,
  });

  const [generatedImages, setGeneratedImages] = useState([]);
  const [currentImage, setCurrentImage] = useState(null);
  const [generationInfo, setGenerationInfo] = useState(null);
  const [activeJobId, setActiveJobId] = useState(null);
  const [activeJobStatus, setActiveJobStatus] = useState(null);
  const activeJobRef = useRef(null);

  const updateParam = useCallback((key, value) => {
    setParams(prev => ({ ...prev, [key]: value }));
  }, [setParams]);

  const randomizeSeed = useCallback(() => {
    updateParam('seed', generateRandomSeed());
  }, [updateParam]);

  const handleGenerate = useCallback(async () => {
    if (!params.prompt.trim()) {
      toast.error('請輸入提示詞');
      return;
    }

    if (activeJobId) {
      toast.error("目前有任務進行中，請先取消或等待完成");
      return;
    }

    let jobId = null;
    try {
      const generationParams = { ...params };

      let result;
      if (controlnetParams.enabled) {
        if (!controlnetParams.image) {
          toast.error('請先上傳 ControlNet 控制圖片');
          return;
        }

        const payload = {
          ...generationParams,
          control_image: controlnetParams.image,
          weight: controlnetParams.weight,
          preprocess: true,
        };

        result = await apiCall(
          () => apiService.controlnetGenerate(payload, controlnetParams.type),
          null,
          {
            showLoading: true,
            showSuccess: true,
            successMessage: 'ControlNet 生成完成！'
          }
        );
      } else {
        if (!asyncMode) {
          result = await apiCall(
            () => apiService.generateImage(generationParams),
            null,
            {
              showLoading: true,
              showSuccess: true,
              successMessage: "圖片生成完成！",
            }
          );
        } else {
          setActiveJobStatus({ status: "queued" });
          const submit = await apiCall(
            () => apiService.submitT2IJob(generationParams),
            null,
            {
              showLoading: true,
              showSuccess: false,
            }
          );

          jobId = submit.job_id;
          activeJobRef.current = jobId;
          setActiveJobId(jobId);
          toast.loading("任務已送出，生成中...", { id: "t2i-job" });

          while (activeJobRef.current === jobId) {
            const status = await apiService.getT2IJobStatus(jobId);
            if (activeJobRef.current !== jobId) {
              break;
            }
            setActiveJobStatus(status);

            if (status.status === "succeeded") {
              toast.success("圖片生成完成！", { id: "t2i-job" });
              result = status;
              break;
            }
            if (status.status === "failed") {
              toast.error(status.error?.message || "生成失敗", { id: "t2i-job" });
              break;
            }
            if (status.status === "canceled") {
              toast("已取消", { id: "t2i-job" });
              break;
            }

            // Poll every ~1s.
            // eslint-disable-next-line no-await-in-loop
            await new Promise((r) => setTimeout(r, 1000));
          }
        }
      }

      if (result && result.image_path) {
        const imagePaths = Array.isArray(result.image_path) ? result.image_path : [result.image_path];

        const newImages = imagePaths.map((path, index) => ({
          id: Date.now() + index,
          path: path,
          metadata: {
            ...params,
            controlnet: controlnetParams.enabled
              ? { type: controlnetParams.type, weight: controlnetParams.weight }
              : null,
            seed: result.seed || params.seed,
            elapsed_ms: result.elapsed_ms,
            generated_at: new Date().toISOString(),
          },
        }));

        setGeneratedImages(prev => [...newImages, ...prev]);
        setCurrentImage(newImages[0]);
        setGenerationInfo({
          ...params,
          seed: result.seed || params.seed,
          elapsed_ms: result.elapsed_ms,
        });

        // Update seed for next generation
        if (result.seed) {
          updateParam('seed', result.seed);
        }
      }
    } catch (error) {
      console.error('Generation failed:', error);
    } finally {
      if (jobId) {
        if (activeJobRef.current === jobId) {
          activeJobRef.current = null;
        }
        setActiveJobId(null);
        setActiveJobStatus(null);
      }
    }
  }, [activeJobId, asyncMode, params, controlnetParams, apiCall, updateParam]);

  useEffect(() => {
    return () => {
      activeJobRef.current = null;
    };
  }, []);

  const handleCancelJob = useCallback(async () => {
    if (!activeJobId) return;
    try {
      const status = await apiService.cancelT2IJob(activeJobId);
      setActiveJobStatus(status);
      toast("已送出取消", { id: "t2i-job" });
    } catch (error) {
      toast.error(error.message || "取消失敗");
    }
  }, [activeJobId, apiService]);

  const handleDownload = useCallback(async (image) => {
    if (!image) return;

    try {
      const response = await fetch(image.path);
      const blob = await response.blob();
      const filename = `generated-${image.metadata.seed}-${Date.now()}.png`;
      downloadBlob(blob, filename);
      toast.success('圖片下載完成');
    } catch (error) {
      toast.error('下載失敗');
      console.error('Download failed:', error);
    }
  }, []);

  const isJobActive = Boolean(activeJobId);
  const isBusy = isLoading || isJobActive;
  const progressStep = activeJobStatus?.progress?.step;
  const progressTotal = activeJobStatus?.progress?.total;
  const progressText =
    Number.isFinite(progressStep) && Number.isFinite(progressTotal)
      ? `${progressStep}/${progressTotal}`
      : "";

  return (
    <div className="generation-panel">
      <div className="panel-header">
        <h2 className="panel-title">
          <Wand2 className="title-icon" />
          圖片生成
        </h2>
        <div className="header-actions">
          {!controlnetParams.enabled && (
            <label className="checkbox-label" title="使用非同步任務（submit/status/cancel）">
              <input
                type="checkbox"
                checked={asyncMode}
                onChange={(e) => setAsyncMode(e.target.checked)}
                className="checkbox"
              />
              非同步
            </label>
          )}
          <button
            className="btn btn-secondary"
            onClick={randomizeSeed}
            title="隨機種子"
          >
            <Shuffle size={16} />
          </button>
          <button
            className="btn btn-primary"
            onClick={handleGenerate}
            disabled={isBusy}
          >
            {isBusy ? (
              <Loading
                size="small"
                text={progressText ? `生成中... (${progressText})` : "生成中..."}
              />
            ) : (
              <>
                <Wand2 size={16} />
                生成圖片
              </>
            )}
          </button>
          {isJobActive && (
            <button
              className="btn btn-danger"
              onClick={handleCancelJob}
              title="取消任務"
            >
              <X size={16} />
              取消
            </button>
          )}
        </div>
      </div>

      <div className="generation-content">
        <div className="controls-section">
          <ParameterControls
            params={params}
            onParamChange={updateParam}
            controlnetParams={controlnetParams}
            onControlnetChange={setControlnetParams}
          />
        </div>

        <div className="preview-section">
          <ImagePreview
            currentImage={currentImage}
            generatedImages={generatedImages}
            generationInfo={generationInfo}
            onImageSelect={setCurrentImage}
            onDownload={handleDownload}
          />
        </div>
      </div>
    </div>
  );
};

export default GenerationPanel;
