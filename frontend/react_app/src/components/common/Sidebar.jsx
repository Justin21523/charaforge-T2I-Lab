// frontend/react_app/src/components/common/Sidebar.jsx
import React from 'react';
import { Link } from 'react-router-dom';
import {
  Image,
  Layers,
  Zap,
  Brain,
  Images,
  ClipboardList,
  ChevronRight
} from 'lucide-react';
import '../../styles/components/Sidebar.css';

const Sidebar = ({ currentPath }) => {
  const normalizedPath = currentPath === "/" ? "/generation" : currentPath;
  const menuItems = [
    {
      path: '/generation',
      icon: Image,
      label: '圖片生成',
      description: '文字轉圖片生成'
    },
    {
      path: '/batch',
      icon: Zap,
      label: '批次生成',
      description: '大量任務生成'
    },
    {
      path: '/training',
      icon: Brain,
      label: '訓練 / 監控',
      description: 'LoRA 訓練進度'
    },
    {
      path: '/lora',
      icon: Layers,
      label: 'LoRA 管理',
      description: '載入 / 卸載 LoRA'
    },
    {
      path: '/gallery',
      icon: Images,
      label: '圖片畫廊',
      description: '生成結果瀏覽'
    },
    {
      path: '/jobs',
      icon: ClipboardList,
      label: '任務管理',
      description: 'T2I jobs 列表 / 清理'
    },
  ];

  return (
    <aside className="sidebar">
      <nav className="sidebar-nav">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = normalizedPath === item.path;

          return (
            <Link
              key={item.path}
              to={item.path}
              className={`nav-item ${isActive ? 'active' : ''}`}
            >
              <div className="nav-item-content">
                <Icon className="nav-icon" />
                <div className="nav-text">
                  <span className="nav-label">{item.label}</span>
                  <span className="nav-description">{item.description}</span>
                </div>
              </div>
              {isActive && <ChevronRight className="nav-arrow" />}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
};

export default Sidebar;
