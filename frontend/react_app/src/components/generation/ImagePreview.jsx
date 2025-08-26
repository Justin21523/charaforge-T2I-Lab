// frontend/react_app/src/components/generation/ImagePreview.jsx
import React from 'react';
import { Download, Copy, Image as ImageIcon, Clock } from 'lucide-react';
import { copyToClipboard, formatDuration } from '../../utils/helpers';
import toast from 'react-hot-toast';

const ImagePreview = ({
  currentImage,
  generatedImages,
  generationInfo,
  onImageSelect,
  onDownload
}) => {
  const handleCopyPrompt = async () => {
    if (generationInfo && generationInfo.prompt) {
      const success = await copyToClipboard(generationInfo.prompt);
      if (success) {
        toast.success('提示詞已複製');
      }
    }
  };

  return (
    <div className="image-preview">
      <div className="preview-header">
        <h3 className="preview-title">
          <ImageIcon size={18} />
          生成結果
        </h3>
        {currentImage && (
          <div className="preview-actions">
            <button
              onClick={handleCopyPrompt}
              className="btn btn-secondary"
              title="複製提示詞"
            >
              <Copy size={16} />
            </button>
            <button
              onClick={() => onDownload(currentImage)}
              className="btn btn-primary"
              title="下載圖片"
            >
              <Download size={16} />
            </button>
          </div>
        )}
      </div>

      <div className="preview-content">
        {currentImage ? (
          <div className="main-image-container">
            <img
              src={currentImage.path}
              alt="Generated"
              className="main-image"
              loading="lazy"
            />
          </div>
        ) : (
          <div className="empty-preview">
            <ImageIcon className="empty-icon" />
            <p className="empty-text">尚未生成圖片</p>
            <p className="empty-hint">輸入提示詞並點擊生成按鈕開始</p>
          </div>
        )}

        {generationInfo && (
          <div className="generation-info">
            <h4 className="info-title">生成資訊</h4>
            <div className="info-content">
              <div className="info-item">
                <span className="info-label">提示詞:</span>
                <span className="info-value prompt-text">{generationInfo.prompt}</span>
              </div>
              {generationInfo.negative && (
                <div className="info-item">
                  <span className="info-label">負面提示詞:</span>
                  <span className="info-value">{generationInfo.negative}</span>
                </div>
              )}
              <div className="info-row">
                <div className="info-item">
                  <span className="info-label">種子:</span>
                  <span className="info-value">{generationInfo.seed}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">尺寸:</span>
                  <span className="info-value">
                    {generationInfo.width} × {generationInfo.height}
                  </span>
                </div>
              </div>
              <div className="info-row">
                <div className="info-item">
                  <span className="info-label">步數:</span>
                  <span className="info-value">{generationInfo.steps}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">CFG:</span>
                  <span className="info-value">{generationInfo.cfg_scale}</span>
                </div>
              </div>
              <div className="info-row">
                <div className="info-item">
                  <span className="info-label">採樣器:</span>
                  <span className="info-value">{generationInfo.sampler}</span>
                </div>
                {generationInfo.elapsed_ms && (
                  <div className="info-item">
                    <span className="info-label">
                      <Clock size={14} />
                      耗時:
                    </span>
                    <span className="info-value">
                      {formatDuration(generationInfo.elapsed_ms)}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {generatedImages.length > 0 && (
          <div className="image-gallery">
            <h4 className="gallery-title">歷史生成 ({generatedImages.length})</h4>
            <div className="gallery-grid">
              {generatedImages.map((image) => (
                <div
                  key={image.id}
                  className={`gallery-item ${currentImage?.id === image.id ? 'active' : ''}`}
                  onClick={() => onImageSelect(image)}
                >
                  <img
                    src={image.path}
                    alt={`Generated ${image.id}`}
                    className="gallery-image"
                    loading="lazy"
                  />
                  <div className="gallery-overlay">
                    <span className="gallery-seed">#{image.metadata.seed}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImagePreview;