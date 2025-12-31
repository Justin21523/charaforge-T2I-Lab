// frontend/react_app/src/components/common/Header.jsx
import React from 'react';
import { Link } from 'react-router-dom';
import { Palette, Activity } from 'lucide-react';
import { useAPI } from '../../hooks/useAPI';
import '../../styles/components/Header.css';

const Header = () => {
  const { isConnected, checkHealth } = useAPI();

  return (
    <header className="header">
      <div className="header-left">
        <Link to="/" className="logo">
          <Palette className="logo-icon" />
          <span className="logo-text">CharaForge T2I Lab</span>
        </Link>
      </div>

      <div className="header-center">
        <h1 className="page-title">角色生成與 LoRA 訓練工作台</h1>
      </div>

      <div className="header-right">
        <div className="api-status" onClick={checkHealth}>
          <Activity
            className={`status-icon ${isConnected ? 'connected' : 'disconnected'}`}
          />
          <span className={`status-text ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? 'API 已連線' : 'API 離線'}
          </span>
        </div>
      </div>
    </header>
  );
};

export default Header;
