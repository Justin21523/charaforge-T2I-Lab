// frontend/react_app/src/components/common/Loading.jsx
import React from 'react';
import { Loader2 } from 'lucide-react';
import '../../styles/components/Loading.css';

const Loading = ({ size = 'medium', text = '載入中...', className = '' }) => {
  const sizeClasses = {
    small: 'loading-small',
    medium: 'loading-medium',
    large: 'loading-large'
  };

  return (
    <div className={`loading ${sizeClasses[size]} ${className}`}>
      <Loader2 className="loading-spinner" />
      {text && <span className="loading-text">{text}</span>}
    </div>
  );
};

export default Loading;