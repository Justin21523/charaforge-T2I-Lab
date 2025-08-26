/*frontend/react_app/src/components/gallery/ImageGallery.jsx*/
import React, { useState, useEffect, useCallback } from 'react';
import { Image as ImageIcon, Download, Trash2, Eye, Grid, List } from 'lucide-react';
import { useLocalStorage } from '../../hooks/useLocalStorage';
import { downloadBlob } from '../../utils/helpers';
import ImageCard from './ImageCard';
import Loading from '../common/Loading';
import toast from 'react-hot-toast';
import '../../styles/components/Gallery.css';

const ImageGallery = () => {
  const [images, setImages] = useLocalStorage('gallery-images', []);
  const [currentImage, setCurrentImage] = useState(null);
  const [viewMode, setViewMode] = useState('grid'); // 'grid' or 'list'
  const [selectedImages, setSelectedImages] = useState(new Set());
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (images.length > 0 && !currentImage) {
      setCurrentImage(images[0]);
    }
  }, [images, currentImage]);

  const addImage = useCallback((imagePath, metadata) => {
    const newImage = {
      id: Date.now(),
      path: imagePath,
      metadata: metadata || {},
      addedAt: new Date().toISOString()
    };

    setImages(prev => [newImage, ...prev]);
    setCurrentImage(newImage);
    toast.success('圖片已添加到畫廊');
  }, [setImages]);

  const removeImage = useCallback((imageId) => {
    setImages(prev => prev.filter(img => img.id !== imageId));

    if (currentImage && currentImage.id === imageId) {
      const remainingImages = images.filter(img => img.id !== imageId);
      setCurrentImage(remainingImages.length > 0 ? remainingImages[0] : null);
    }

    setSelectedImages(prev => {
      const newSet = new Set(prev);
      newSet.delete(imageId);
      return newSet;
    });

    toast.success('圖片已從畫廊移除');
  }, [setImages, currentImage, images]);

  const removeSelectedImages = useCallback(() => {
    if (selectedImages.size === 0) {
      toast.error('請先選擇要刪除的圖片');
      return;
    }

    const remainingImages = images.filter(img => !selectedImages.has(img.id));
    setImages(remainingImages);

    // Update current image if it was deleted
    if (currentImage && selectedImages.has(currentImage.id)) {
      setCurrentImage(remainingImages.length > 0 ? remainingImages[0] : null);
    }

    setSelectedImages(new Set());
    toast.success(`已刪除 ${selectedImages.size} 張圖片`);
  }, [selectedImages, images, setImages, currentImage]);

  const downloadCurrentImage = useCallback(async () => {
    if (!currentImage) {
      toast.error('沒有選中的圖片');
      return;
    }

    try {
      const response = await fetch(currentImage.path);
      const blob = await response.blob();
      const filename = `gallery-${currentImage.id}-${Date.now()}.png`;
      downloadBlob(blob, filename);
      toast.success('圖片下載完成');
    } catch (error) {
      toast.error('下載失敗');
      console.error('Download failed:', error);
    }
  }, [currentImage]);

  const downloadSelectedImages = useCallback(async () => {
    if (selectedImages.size === 0) {
      toast.error('請先選擇要下載的圖片');
      return;
    }

    setIsLoading(true);
    try {
      // Note: This would need a backend endpoint to create a zip file
      // For now, download images individually
      for (const imageId of selectedImages) {
        const image = images.find(img => img.id === imageId);
        if (image) {
          const response = await fetch(image.path);
          const blob = await response.blob();
          const filename = `gallery-${image.id}.png`;
          downloadBlob(blob, filename);
        }
      }
      toast.success(`已下載 ${selectedImages.size} 張圖片`);
    } catch (error) {
      toast.error('批量下載失敗');
      console.error('Batch download failed:', error);
    } finally {
      setIsLoading(false);
    }
  }, [selectedImages, images]);

  const toggleImageSelection = useCallback((imageId) => {
    setSelectedImages(prev => {
      const newSet = new Set(prev);
      if (newSet.has(imageId)) {
        newSet.delete(imageId);
      } else {
        newSet.add(imageId);
      }
      return newSet;
    });
  }, []);

  const selectAllImages = useCallback(() => {
    setSelectedImages(new Set(images.map(img => img.id)));
  }, [images]);

  const clearSelection = useCallback(() => {
    setSelectedImages(new Set());
  }, []);

  const clearGallery = useCallback(() => {
    if (images.length === 0) {
      toast.info('畫廊已經是空的');
      return;
    }

    if (window.confirm('確定要清空整個畫廊嗎？此操作無法撤銷。')) {
      setImages([]);
      setCurrentImage(null);
      setSelectedImages(new Set());
      toast.success('畫廊已清空');
    }
  }, [images, setImages]);

  return (
    <div className="image-gallery">
      <div className="panel-header">
        <h2 className="panel-title">
          <ImageIcon className="title-icon" />
          圖片畫廊 ({images.length})
        </h2>

        <div className="gallery-actions">
          <div className="view-mode-buttons">
            <button
              className={`btn btn-sm ${viewMode === 'grid' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setViewMode('grid')}
              title="網格檢視"
            >
              <Grid size={16} />
            </button>
            <button
              className={`btn btn-sm ${viewMode === 'list' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setViewMode('list')}
              title="列表檢視"
            >
              <List size={16} />
            </button>
          </div>

          {selectedImages.size > 0 && (
            <div className="selection-actions">
              <span className="selection-count">
                已選擇 {selectedImages.size} 張
              </span>
              <button
                className="btn btn-secondary btn-sm"
                onClick={downloadSelectedImages}
                disabled={isLoading}
              >
                <Download size={14} />
                下載選中
              </button>
              <button
                className="btn btn-danger btn-sm"
                onClick={removeSelectedImages}
              >
                <Trash2 size={14} />
                刪除選中
              </button>
              <button
                className="btn btn-secondary btn-sm"
                onClick={clearSelection}
              >
                取消選擇
              </button>
            </div>
          )}

          <div className="gallery-controls">
            {images.length > 0 && (
              <>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={selectedImages.size === images.length ? clearSelection : selectAllImages}
                >
                  {selectedImages.size === images.length ? '取消全選' : '全選'}
                </button>
                <button
                  className="btn btn-danger btn-sm"
                  onClick={clearGallery}
                >
                  <Trash2 size={14} />
                  清空畫廊
                </button>
              </>
            )}
          </div>
        </div>
      </div>

      <div className="gallery-content">
        <div className="gallery-main">
          {currentImage ? (
            <div className="current-image-container">
              <img
                src={currentImage.path}
                alt={`Image ${currentImage.id}`}
                className="current-image"
                onError={(e) => {
                  console.error('Image load error:', e);
                  toast.error('圖片載入失敗');
                }}
              />

              <div className="image-overlay">
                <div className="image-info">
                  <span>ID: {currentImage.id}</span>
                  {currentImage.metadata.seed && (
                    <span>種子: {currentImage.metadata.seed}</span>
                  )}
                  {currentImage.metadata.width && currentImage.metadata.height && (
                    <span>{currentImage.metadata.width} × {currentImage.metadata.height}</span>
                  )}
                </div>

                <div className="image-actions">
                  <button
                    className="btn btn-primary btn-sm"
                    onClick={downloadCurrentImage}
                    title="下載此圖片"
                  >
                    <Download size={14} />
                  </button>
                  <button
                    className="btn btn-danger btn-sm"
                    onClick={() => removeImage(currentImage.id)}
                    title="刪除此圖片"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <ImageIcon className="empty-icon" />
              <p className="empty-text">畫廊是空的</p>
              <p className="empty-hint">生成圖片後會自動添加到這裡</p>
            </div>
          )}
        </div>

        <div className="gallery-sidebar">
          <h3 className="sidebar-title">圖片列表</h3>

          {images.length === 0 ? (
            <div className="empty-state">
              <ImageIcon className="empty-icon" />
              <p>暫無圖片</p>
            </div>
          ) : (
            <>
              <div className={`gallery-grid ${viewMode}`}>
                {images.map((image) => (
                  <ImageCard
                    key={image.id}
                    image={image}
                    isSelected={selectedImages.has(image.id)}
                    isActive={currentImage && currentImage.id === image.id}
                    onSelect={() => setCurrentImage(image)}
                    onToggleSelection={() => toggleImageSelection(image.id)}
                    onRemove={() => removeImage(image.id)}
                    viewMode={viewMode}
                  />
                ))}
              </div>

              {currentImage && currentImage.metadata && (
                <div className="image-info">
                  <h4>圖片資訊</h4>

                  <div className="info-section">
                    <div className="info-title">基本資訊</div>
                    <div className="info-grid">
                      <div className="info-item">
                        <span className="info-label">ID:</span>
                        <span className="info-value">{currentImage.id}</span>
                      </div>
                      <div className="info-item">
                        <span className="info-label">添加時間:</span>
                        <span className="info-value">
                          {new Date(currentImage.addedAt).toLocaleString()}
                        </span>
                      </div>
                    </div>
                  </div>

                  {currentImage.metadata.prompt && (
                    <div className="info-section">
                      <div className="info-title">提示詞</div>
                      <div className="info-text">{currentImage.metadata.prompt}</div>
                    </div>
                  )}

                  {currentImage.metadata.negative && (
                    <div className="info-section">
                      <div className="info-title">負面提示詞</div>
                      <div className="info-text">{currentImage.metadata.negative}</div>
                    </div>
                  )}

                  <div className="info-section">
                    <div className="info-title">生成參數</div>
                    <div className="info-grid">
                      {currentImage.metadata.seed && (
                        <div className="info-item">
                          <span className="info-label">種子:</span>
                          <span className="info-value">{currentImage.metadata.seed}</span>
                        </div>
                      )}
                      {currentImage.metadata.width && currentImage.metadata.height && (
                        <div className="info-item">
                          <span className="info-label">尺寸:</span>
                          <span className="info-value">
                            {currentImage.metadata.width} × {currentImage.metadata.height}
                          </span>
                        </div>
                      )}
                      {currentImage.metadata.steps && (
                        <div className="info-item">
                          <span className="info-label">步數:</span>
                          <span className="info-value">{currentImage.metadata.steps}</span>
                        </div>
                      )}
                      {currentImage.metadata.cfg_scale && (
                        <div className="info-item">
                          <span className="info-label">CFG:</span>
                          <span className="info-value">{currentImage.metadata.cfg_scale}</span>
                        </div>
                      )}
                      {currentImage.metadata.sampler && (
                        <div className="info-item">
                          <span className="info-label">採樣器:</span>
                          <span className="info-value">{currentImage.metadata.sampler}</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default ImageGallery;