// frontend/react_app/src/components/gallery/ImageCard.jsx
import React from 'react';
import { Check, Download, Trash2, Eye } from 'lucide-react';

const ImageCard = ({
  image,
  isSelected,
  isActive,
  onSelect,
  onToggleSelection,
  onRemove,
  viewMode = 'grid'
}) => {
  return (
    <div className={`image-card ${isActive ? 'active' : ''} ${viewMode}`}>
      <div className="card-checkbox">
        <input
          type="checkbox"
          checked={isSelected}
          onChange={onToggleSelection}
          className="checkbox"
        />
      </div>

      <div className="card-image" onClick={onSelect}>
        <img
          src={image.path}
          alt={`Generated ${image.id}`}
          className="gallery-thumbnail"
          loading="lazy"
          onError={(e) => {
            e.target.style.display = 'none';
          }}
        />

        {viewMode === 'grid' && (
          <div className="card-overlay">
            <div className="card-actions">
              <button
                className="card-action-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  onSelect();
                }}
                title="查看圖片"
              >
                <Eye size={16} />
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="card-info">
        <div className="card-title">#{image.id}</div>
        {image.metadata.seed && (
          <div className="card-seed">種子: {image.metadata.seed}</div>
        )}
        {viewMode === 'list' && image.metadata.prompt && (
          <div className="card-prompt">
            {image.metadata.prompt.substring(0, 50)}
            {image.metadata.prompt.length > 50 ? '...' : ''}
          </div>
        )}
        <div className="card-date">
          {new Date(image.addedAt).toLocaleDateString()}
        </div>
      </div>

      {viewMode === 'list' && (
        <div className="card-actions-list">
          <button
            className="btn btn-sm btn-danger"
            onClick={(e) => {
              e.stopPropagation();
              onRemove();
            }}
            title="刪除圖片"
          >
            <Trash2 size={14} />
          </button>
        </div>
      )}
    </div>
  );
};

export default ImageCard;