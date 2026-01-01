// frontend/react_app/src/components/common/Header.jsx
import React, { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Activity, KeyRound, LogIn, LogOut, Palette, X } from "lucide-react";
import toast from "react-hot-toast";
import { useAPI } from "../../hooks/useAPI";
import { useLocalStorage } from "../../hooks/useLocalStorage";
import apiService from "../../services/apiService";
import "../../styles/components/Header.css";

const Header = () => {
  const { isConnected, checkHealth } = useAPI();
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const [storedApiKey, setStoredApiKey] = useLocalStorage("charaforge.apiKey", "");
  const [storedHeaderName, setStoredHeaderName] = useLocalStorage(
    "charaforge.apiKeyHeader",
    "X-API-Key"
  );
  const [useJwt, setUseJwt] = useLocalStorage("charaforge.auth.useJwt", false);

  const maskedKey = useMemo(() => {
    if (!storedApiKey) return "";
    if (storedApiKey.length <= 6) return "••••••";
    return `${storedApiKey.slice(0, 3)}••••${storedApiKey.slice(-2)}`;
  }, [storedApiKey]);

  const toggleAuthPanel = () => setIsAuthOpen((prev) => !prev);

  const handleSaveAuth = async () => {
    apiService.setApiKey(storedApiKey, storedHeaderName);
    apiService.setUseJwt(Boolean(useJwt));

    if (useJwt && storedApiKey) {
      try {
        await apiService.exchangeApiKeyForToken();
        toast.success("JWT 已登入");
      } catch (error) {
        toast.error(error?.message || "JWT 登入失敗");
      }
    } else if (!useJwt) {
      toast.success(storedApiKey ? "API Key 已更新" : "API Key 已清除");
    }

    checkHealth();
    setIsAuthOpen(false);
  };

  const handleClearAuth = () => {
    setStoredApiKey("");
    apiService.clearApiKey();
    toast.success("API Key 已清除");
    checkHealth();
  };

  const handleLogoutJwt = async (all) => {
    try {
      await apiService.logoutJwt({ all: Boolean(all) });
      toast.success(all ? "JWT 已登出（全部）" : "JWT 已登出");
      checkHealth();
    } catch (error) {
      toast.error(error?.message || "登出失敗");
    }
  };

  const jwtInfo = apiService.getJwtInfo();
  const jwtLoggedIn = Boolean(jwtInfo?.accessToken && jwtInfo?.refreshToken);

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
        <div className="header-auth">
          <button
            type="button"
            className="btn btn-secondary btn-sm header-auth-btn"
            onClick={toggleAuthPanel}
            title={storedApiKey ? `API Key: ${maskedKey}` : "設定 API Key"}
          >
            <KeyRound size={16} />
          </button>

          {isAuthOpen && (
            <div className="auth-panel card">
              <div className="auth-panel-header">
                <div className="auth-panel-title">API Key</div>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={() => setIsAuthOpen(false)}
                  title="關閉"
                >
                  <X size={14} />
                </button>
              </div>

              <div className="auth-panel-body">
                <label className="auth-label" htmlFor="api_key_header">
                  Header
                </label>
                <input
                  id="api_key_header"
                  className="input"
                  value={storedHeaderName}
                  onChange={(e) => setStoredHeaderName(e.target.value)}
                  placeholder="X-API-Key"
                />

                <label className="auth-label" htmlFor="api_key_value">
                  Key
                </label>
                <input
                  id="api_key_value"
                  className="input"
                  value={storedApiKey}
                  onChange={(e) => setStoredApiKey(e.target.value)}
                  placeholder="(留空代表不送 API Key)"
                  type="password"
                  autoComplete="off"
                />

                <label className="auth-label" htmlFor="auth_use_jwt">
                  JWT 模式
                </label>
                <div className="auth-panel-toggle">
                  <input
                    id="auth_use_jwt"
                    type="checkbox"
                    checked={Boolean(useJwt)}
                    onChange={(e) => setUseJwt(e.target.checked)}
                  />
                  <span className="auth-panel-toggle-text">
                    使用 JWT（建議：前端改送 Bearer，不再每次送 API Key）
                  </span>
                </div>

                {useJwt && (
                  <div className="auth-jwt-card">
                    <div className="auth-jwt-status">
                      狀態：{jwtLoggedIn ? "已登入" : "未登入"}
                      {jwtLoggedIn && jwtInfo?.role ? `（${jwtInfo.role}）` : ""}
                    </div>
                    <div className="auth-jwt-actions">
                      <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        onClick={async () => {
                          try {
                            await apiService.exchangeApiKeyForToken();
                            toast.success("JWT 已登入");
                            checkHealth();
                          } catch (error) {
                            toast.error(error?.message || "JWT 登入失敗");
                          }
                        }}
                        disabled={!storedApiKey}
                        title={!storedApiKey ? "請先填入 API key" : "用 API key 換取 JWT"}
                      >
                        <LogIn size={14} />
                        登入
                      </button>
                      <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        onClick={() => handleLogoutJwt(false)}
                        disabled={!jwtLoggedIn}
                      >
                        <LogOut size={14} />
                        登出
                      </button>
                      <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        onClick={() => handleLogoutJwt(true)}
                        disabled={!jwtLoggedIn}
                        title="撤銷此帳號下的所有 refresh token"
                      >
                        <LogOut size={14} />
                        全部登出
                      </button>
                    </div>
                  </div>
                )}

                <div className="auth-panel-actions">
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={handleClearAuth}
                  >
                    清除
                  </button>
                  <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    onClick={handleSaveAuth}
                  >
                    儲存
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

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
