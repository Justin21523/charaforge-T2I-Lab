// frontend/react_app/src/hooks/useAPI.js
import { useState, useEffect, useCallback } from "react";
import apiService from "../services/apiService";
import toast from "react-hot-toast";

export const useAPI = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const checkHealth = useCallback(async () => {
    try {
      const connected = health.status === "ok";
      setIsConnected(connected);

      if (!connected) {
        console.warn("API health check failed:", health.message);
      }

      return connected;
    } catch (error) {
      setIsConnected(false);
      console.error("Health check error:", error);
      return false;
    }
  }, []);

  const apiCall = useCallback(
    async (apiFunction, params = null, options = {}) => {
      const {
        showLoading = true,
        showSuccess = false,
        showError = true,
      } = options;

      if (showLoading) setIsLoading(true);

      try {
        const result = params ? await apiFunction(params) : await apiFunction();

        if (showSuccess) {
          toast.success(
            options.successMessage || "Operation completed successfully"
          );
        }

        return result;
      } catch (error) {
        if (showError) {
          toast.error(options.errorMessage || error.message);
        }
        throw error;
      } finally {
        if (showLoading) setIsLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    checkHealth();

    // Check health every 30 seconds
    const interval = setInterval(checkHealth, 30000);

    return () => clearInterval(interval);
  }, [checkHealth]);

  return {
    isConnected,
    isLoading,
    checkHealth,
    apiCall,
    apiService,
  };
};

export const useWebSocket = (url, options = {}) => {
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    if (!url) return;

    const ws = new WebSocket(url);

    ws.onopen = () => {
      setIsConnected(true);
      setSocket(ws);
      console.log("WebSocket connected");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setMessages((prev) => [...prev, data]);

        if (options.onMessage) {
          options.onMessage(data);
        }
      } catch (error) {
        console.error("WebSocket message parse error:", error);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      setSocket(null);
      console.log("WebSocket disconnected");
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    return () => {
      ws.close();
    };
  }, [url, options]);

  const sendMessage = useCallback(
    (message) => {
      if (socket && isConnected) {
        socket.send(JSON.stringify(message));
      }
    },
    [socket, isConnected]
  );

  return {
    socket,
    isConnected,
    messages,
    sendMessage,
  };
};
