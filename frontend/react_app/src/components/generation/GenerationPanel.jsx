// frontend/react_app/src/components/generation/GenerationPanel.jsx
import React, { useState, useCallback } from 'react';
import { Wand2, Shuffle, Download, Settings } from 'lucide-react';
import { useAPI } from '../../hooks/useAPI';
import { useLocalStorage } from '../../hooks/useLocalStorage';
import { DEFAULT_GENERATION_PARAMS, SAMPLERS, CONTROLNET_TYPES } from '../../utils/constants';
import { generateRandomSeed, downloadBlob } from '../../utils/helpers';
import ParameterControls from './ParameterControls';
import ImagePreview from './ImagePreview';
import Loading from '../common/Loading';
import toast from 'react-hot-toast';
import '../../styles/components/Generation.css';

const GenerationPanel = () => {
  const { apiCall, isLoading } = useAPI();
  const [params, setParams] = useLocalStorage('generation-params', DEFAULT_GENERATION_PARAMS);
  const [controlnetParams, setControlnetParams] = useState({
    enabled: false,
    type: 'pose',
    image: null,
    weight: 1.0,
  });

  const [generatedImages, setGeneratedImages] = useState([]);
  const [currentImage, setCurrentImage] = useState(null);
  const [generationInfo, setGenerationInfo] = useState(null);

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

    try {
      const generationParams = { ...params };

      let result;
      if (controlnetParams.enabled && controlnetParams.image) {
        const controlParams = {
          control_type: controlnetParams.type,
          control_image: controlnetParams.image,
          control_weight: controlnetParams.weight,
        };

        result = await apiCall(
          () => apiService.controlnetGenerate({ ...generationParams, ...controlParams }, controlnetParams.type),
          null,
          {
            showLoading: false,
            showSuccess: true,
            successMessage: '圖片生成完成！'
          }
        );
      } else {
        result = await apiCall(
          () => apiService.generateImage(generationParams),
          null,
          {
            showLoading: false,
            showSuccess: true,
            successMessage: '圖片生成完成！'
          }
        );
      }

      if (result && result.image_path) {
        const imagePaths = Array.isArray(result.image_path) ? result.image_path : [result.image_path];

        const newImages = imagePaths.map((path, index) => ({
          id: Date.now() + index,
          path: path,
          metadata: {
            ...params,
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
    }
  }, [params, controlnetParams, apiCall, updateParam]);

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

  return (
    <div className="generation-panel">
      <div className="panel-header">
        <h2 className="panel-title">
          <Wand2 className="title-icon" />
          圖片生成
        </h2>
        <div className="header-actions">
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
            disabled={isLoading}
          >
            {isLoading ? (
              <Loading size="small" text="生成中..." />
            ) : (
              <>
                <Wand2 size={16} />
                生成圖片
              </>
            )}
          </button>
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