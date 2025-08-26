// frontend/react_app/src/components/generation/ParameterControls.jsx
import React from 'react';
import { Upload, X } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { SAMPLERS, CONTROLNET_TYPES } from '../../utils/constants';

const ParameterControls = ({ params, onParamChange, controlnetParams, onControlnetChange }) => {
  const onDrop = (acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      const reader = new FileReader();
      reader.onload = () => {
        onControlnetChange(prev => ({
          ...prev,
          image: reader.result,
        }));
      };
      reader.readAsDataURL(file);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
    },
    multiple: false,
  });

  return (
    <div className="parameter-controls">
      {/* Prompt Section */}
      <div className="control-group">
        <h3 className="group-title">提示詞設定</h3>

        <div className="control-item">
          <label htmlFor="prompt">正面提示詞</label>
          <textarea
            id="prompt"
            value={params.prompt}
            onChange={(e) => onParamChange('prompt', e.target.value)}
            placeholder="描述想要生成的圖片內容..."
            rows={3}
            className="textarea"
          />
        </div>

        <div className="control-item">
          <label htmlFor="negative">負面提示詞</label>
          <textarea
            id="negative"
            value={params.negative}
            onChange={(e) => onParamChange('negative', e.target.value)}
            placeholder="描述不希望出現的內容..."
            rows={2}
            className="textarea"
          />
        </div>
      </div>

      {/* Generation Parameters */}
      <div className="control-group">
        <h3 className="group-title">生成參數</h3>

        <div className="control-row">
          <div className="control-item">
            <label htmlFor="width">寬度</label>
            <input
              id="width"
              type="range"
              min="256"
              max="2048"
              step="64"
              value={params.width}
              onChange={(e) => onParamChange('width', parseInt(e.target.value))}
              className="slider"
            />
            <span className="slider-value">{params.width}px</span>
          </div>

          <div className="control-item">
            <label htmlFor="height">高度</label>
            <input
              id="height"
              type="range"
              min="256"
              max="2048"
              step="64"
              value={params.height}
              onChange={(e) => onParamChange('height', parseInt(e.target.value))}
              className="slider"
            />
            <span className="slider-value">{params.height}px</span>
          </div>
        </div>

        <div className="control-row">
          <div className="control-item">
            <label htmlFor="steps">採樣步數</label>
            <input
              id="steps"
              type="range"
              min="1"
              max="100"
              value={params.steps}
              onChange={(e) => onParamChange('steps', parseInt(e.target.value))}
              className="slider"
            />
            <span className="slider-value">{params.steps}</span>
          </div>

          <div className="control-item">
            <label htmlFor="cfg_scale">CFG 縮放</label>
            <input
              id="cfg_scale"
              type="range"
              min="1"
              max="30"
              step="0.5"
              value={params.cfg_scale}
              onChange={(e) => onParamChange('cfg_scale', parseFloat(e.target.value))}
              className="slider"
            />
            <span className="slider-value">{params.cfg_scale}</span>
          </div>
        </div>

        <div className="control-row">
          <div className="control-item">
            <label htmlFor="seed">種子</label>
            <input
              id="seed"
              type="number"
              value={params.seed}
              onChange={(e) => onParamChange('seed', parseInt(e.target.value) || -1)}
              placeholder="-1 為隨機"
              className="input"
            />
          </div>

          <div className="control-item">
            <label htmlFor="sampler">採樣器</label>
            <select
              id="sampler"
              value={params.sampler}
              onChange={(e) => onParamChange('sampler', e.target.value)}
              className="select"
            >
              {SAMPLERS.map(sampler => (
                <option key={sampler} value={sampler}>{sampler}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="control-item">
          <label htmlFor="batch_size">批次大小</label>
          <input
            id="batch_size"
            type="range"
            min="1"
            max="8"
            value={params.batch_size}
            onChange={(e) => onParamChange('batch_size', parseInt(e.target.value))}
            className="slider"
          />
          <span className="slider-value">{params.batch_size}</span>
        </div>
      </div>

      {/* ControlNet Section */}
      <div className="control-group">
        <h3 className="group-title">ControlNet 控制</h3>

        <div className="control-item">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={controlnetParams.enabled}
              onChange={(e) => onControlnetChange(prev => ({
                ...prev,
                enabled: e.target.checked,
              }))}
              className="checkbox"
            />
            啟用 ControlNet
          </label>
        </div>

        {controlnetParams.enabled && (
          <>
            <div className="control-row">
              <div className="control-item">
                <label htmlFor="controlnet_type">控制類型</label>
                <select
                  id="controlnet_type"
                  value={controlnetParams.type}
                  onChange={(e) => onControlnetChange(prev => ({
                    ...prev,
                    type: e.target.value,
                  }))}
                  className="select"
                >
                  {CONTROLNET_TYPES.map(type => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>
              </div>

              <div className="control-item">
                <label htmlFor="controlnet_weight">控制強度</label>
                <input
                  id="controlnet_weight"
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={controlnetParams.weight}
                  onChange={(e) => onControlnetChange(prev => ({
                    ...prev,
                    weight: parseFloat(e.target.value),
                  }))}
                  className="slider"
                />
                <span className="slider-value">{controlnetParams.weight}</span>
              </div>
            </div>

            <div className="control-item">
              <label>控制圖片</label>
              <div
                {...getRootProps()}
                className={`dropzone ${isDragActive ? 'active' : ''}`}
              >
                <input {...getInputProps()} />
                {controlnetParams.image ? (
                  <div className="dropzone-preview">
                    <img
                      src={controlnetParams.image}
                      alt="Control"
                      className="control-image-preview"
                    />
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onControlnetChange(prev => ({ ...prev, image: null }));
                      }}
                      className="remove-image-btn"
                    >
                      <X size={16} />
                    </button>
                  </div>
                ) : (
                  <div className="dropzone-content">
                    <Upload className="upload-icon" />
                    <p className="dropzone-text">
                      {isDragActive ? '放開以上傳圖片' : '拖放圖片或點擊選擇'}
                    </p>
                    <p className="dropzone-hint">
                      支援 PNG, JPG, JPEG, BMP, WEBP
                    </p>
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default ParameterControls;