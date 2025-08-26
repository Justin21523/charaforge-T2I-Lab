// frontend/react_app/src/components/lora/LoRAManager.jsx
import React, { useState, useEffect, useCallback } from 'react';
import { Layers, Download, Upload, Trash2, RefreshCw } from 'lucide-react';
import { useAPI } from '../../hooks/useAPI';
import Loading from '../common/Loading';
import toast from 'react-hot-toast';
import '../../styles/components/LoRA.css';

const LoRAManager = () => {
  const { apiCall, isLoading } = useAPI();
  const [availableLoras, setAvailableLoras] = useState([]);
  const [loadedLoras, setLoadedLoras] = useState({});
  const [selectedLora, setSelectedLora] = useState(null);
  const [loraWeight, setLoraWeight] = useState(1.0);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const refreshLoraList = useCallback(async () => {
    setIsRefreshing(true);
    try {
      const loras = await apiCall(
        () => apiService.listLoras(),
        null,
        { showLoading: false, showError: true }
      );
      setAvailableLoras(loras || []);
    } catch (error) {
      console.error('Failed to refresh LoRA list:', error);
    } finally {
      setIsRefreshing(false);
    }
  }, [apiCall]);

  useEffect(() => {
    refreshLoraList();
  }, [refreshLoraList]);

  const handleLoadLora = useCallback(async () => {
    if (!selectedLora) {
      toast.error('請選擇要載入的 LoRA 模型');
      return;
    }

    if (loadedLoras[selectedLora.id]) {
      toast.error('該 LoRA 已經載入');
      return;
    }

    try {
      const result = await apiCall(
        () => apiService.loadLora(selectedLora.id, loraWeight),
        null,
        {
          showLoading: true,
          showSuccess: true,
          successMessage: `LoRA '${selectedLora.name}' 載入成功`,
          errorMessage: 'LoRA 載入失敗'
        }
      );

      if (result.status === 'success') {
        setLoadedLoras(prev => ({
          ...prev,
          [selectedLora.id]: {
            ...selectedLora,
            weight: loraWeight
          }
        }));
      }
    } catch (error) {
      console.error('Failed to load LoRA:', error);
    }
  }, [selectedLora, loraWeight, loadedLoras, apiCall]);

  const handleUnloadLora = useCallback(async (loraId) => {
    try {
      const result = await apiCall(
        () => apiService.unloadLora(loraId),
        null,
        {
          showLoading: true,
          showSuccess: true,
          successMessage: `LoRA '${loraId}' 卸載成功`,
          errorMessage: 'LoRA 卸載失敗'
        }
      );

      if (result.status === 'success') {
        setLoadedLoras(prev => {
          const newLoaded = { ...prev };
          delete newLoaded[loraId];
          return newLoaded;
        });
      }
    } catch (error) {
      console.error('Failed to unload LoRA:', error);
    }
  }, [apiCall]);

  const handleUnloadAll = useCallback(async () => {
    const loraIds = Object.keys(loadedLoras);
    if (loraIds.length === 0) {
      toast.error('沒有已載入的 LoRA');
      return;
    }

    let successCount = 0;
    let failedLoras = [];

    for (const loraId of loraIds) {
      try {
        const result = await apiCall(
          () => apiService.unloadLora(loraId),
          null,
          { showLoading: false, showError: false }
        );

        if (result.status === 'success') {
          successCount++;
        } else {
          failedLoras.push(loraId);
        }
      } catch (error) {
        failedLoras.push(loraId);
      }
    }

    // Update loaded LoRAs
    if (successCount > 0) {
      setLoadedLoras(prev => {
        const newLoaded = { ...prev };
        loraIds.forEach(id => {
          if (!failedLoras.includes(id)) {
            delete newLoaded[id];
          }
        });
        return newLoaded;
      });
    }

    if (failedLoras.length === 0) {
      toast.success(`所有 LoRA 已卸載 (${successCount} 個)`);
    } else {
      toast.error(`已卸載 ${successCount} 個 LoRA，失敗: ${failedLoras.join(', ')}`);
    }
  }, [loadedLoras, apiCall]);

  return (
    <div className="lora-manager">
      <div className="panel-header">
        <h2 className="panel-title">
          <Layers className="title-icon" />
          LoRA 模型管理
        </h2>
        <button
          className="btn btn-secondary"
          onClick={refreshLoraList}
          disabled={isRefreshing}
        >
          <RefreshCw className={isRefreshing ? 'spinning' : ''} size={16} />
          刷新列表
        </button>
      </div>

      <div className="lora-content">
        {/* Available LoRAs */}
        <div className="lora-section">
          <h3 className="section-title">可用 LoRA 模型</h3>

          {isRefreshing ? (
            <Loading size="medium" text="載入 LoRA 列表..." />
          ) : availableLoras.length === 0 ? (
            <div className="empty-state">
              <Layers className="empty-icon" />
              <p className="empty-text">沒有找到 LoRA 模型</p>
              <p className="empty-hint">請檢查模型目錄或重新整理列表</p>
            </div>
          ) : (
            <div className="lora-list">
              {availableLoras.map((lora) => (
                <div
                  key={lora.id}
                  className={`lora-item ${selectedLora?.id === lora.id ? 'selected' : ''} ${loadedLoras[lora.id] ? 'loaded' : ''}`}
                  onClick={() => setSelectedLora(lora)}
                >
                  <div className="lora-info">
                    <h4 className="lora-name">{lora.name || lora.id}</h4>
                    <p className="lora-description">
                      {lora.description || '無描述'}
                    </p>
                    <div className="lora-meta">
                      <span className="lora-type">類型: {lora.type || 'character'}</span>
                      <span className="lora-resolution">解析度: {lora.resolution || 'Unknown'}</span>
                      <span className="lora-rank">Rank: {lora.rank || 'Unknown'}</span>
                    </div>
                  </div>
                  {loadedLoras[lora.id] && (
                    <div className="loaded-indicator">
                      已載入 (權重: {loadedLoras[lora.id].weight.toFixed(1)})
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Load Controls */}
          {selectedLora && (
            <div className="load-controls">
              <div className="weight-control">
                <label htmlFor="lora-weight">LoRA 權重</label>
                <input
                  id="lora-weight"
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={loraWeight}
                  onChange={(e) => setLoraWeight(parseFloat(e.target.value))}
                  className="slider"
                />
                <span className="slider-value">{loraWeight.toFixed(1)}</span>
              </div>

              <button
                className="btn btn-primary"
                onClick={handleLoadLora}
                disabled={isLoading || loadedLoras[selectedLora.id]}
              >
                <Download size={16} />
                {loadedLoras[selectedLora.id] ? '已載入' : '載入 LoRA'}
              </button>
            </div>
          )}
        </div>

        {/* Loaded LoRAs */}
        <div className="lora-section">
          <div className="section-header">
            <h3 className="section-title">已載入 LoRA ({Object.keys(loadedLoras).length})</h3>
            {Object.keys(loadedLoras).length > 0 && (
              <button
                className="btn btn-danger btn-sm"
                onClick={handleUnloadAll}
              >
                <Trash2 size={14} />
                卸載全部
              </button>
            )}
          </div>

          {Object.keys(loadedLoras).length === 0 ? (
            <div className="empty-state">
              <Upload className="empty-icon" />
              <p className="empty-text">尚未載入任何 LoRA</p>
              <p className="empty-hint">從上方列表選擇並載入 LoRA 模型</p>
            </div>
          ) : (
            <div className="loaded-lora-list">
              {Object.values(loadedLoras).map((lora) => (
                <div key={lora.id} className="loaded-lora-item">
                  <div className="loaded-lora-info">
                    <h4 className="loaded-lora-name">{lora.name || lora.id}</h4>
                    <p className="loaded-lora-weight">權重: {lora.weight.toFixed(1)}</p>
                  </div>
                  <button
                    className="btn btn-danger btn-sm"
                    onClick={() => handleUnloadLora(lora.id)}
                    title="卸載這個 LoRA"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LoRAManager;