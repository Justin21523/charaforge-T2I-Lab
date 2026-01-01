import { useCallback, useEffect, useRef, useState } from "react";

const DEFAULT_RECONNECT_DELAY_MS = 1500;
const STORAGE_API_KEY = "charaforge.apiKey";
const STORAGE_USE_JWT = "charaforge.auth.useJwt";
const STORAGE_JWT_ACCESS_TOKEN = "charaforge.jwt.accessToken";
const STORAGE_JWT_EXPIRES_AT = "charaforge.jwt.expiresAt";

const readStoredJson = (key, fallback) => {
  try {
    if (typeof window === "undefined") return fallback;
    const raw = window.localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw);
  } catch (e) {
    return fallback;
  }
};

const toWsUrl = (httpBaseUrl, path) => {
  const url = new URL(httpBaseUrl);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = path.startsWith("/") ? path : `/${path}`;
  url.search = "";
  url.hash = "";
  return url.toString();
};

export const buildTrainProgressWsUrl = (apiBaseUrl, jobId) =>
  (() => {
    const url = new URL(toWsUrl(apiBaseUrl, `/api/v1/ws/train/${jobId}`));
    const apiKey = readStoredJson(STORAGE_API_KEY, "") || "";
    const useJwt = Boolean(readStoredJson(STORAGE_USE_JWT, false));
    const accessToken = readStoredJson(STORAGE_JWT_ACCESS_TOKEN, "") || "";
    const expiresAt = Number(readStoredJson(STORAGE_JWT_EXPIRES_AT, 0) || 0);
    const now = Math.floor(Date.now() / 1000);

    if (useJwt && accessToken && expiresAt > now + 10) {
      url.searchParams.set("access_token", accessToken);
    } else if (apiKey) {
      url.searchParams.set("api_key", apiKey);
    }
    return url.toString();
  })();

export const useWebSocket = (
  url,
  { enabled = true, reconnect = true, reconnectDelayMs = DEFAULT_RECONNECT_DELAY_MS } = {}
) => {
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const [status, setStatus] = useState("disconnected");
  const [lastMessage, setLastMessage] = useState(null);

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch (e) {
        // ignore
      }
      wsRef.current = null;
    }
    setStatus("disconnected");
  }, []);

  const sendJson = useCallback((payload) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    ws.send(JSON.stringify(payload));
    return true;
  }, []);

  useEffect(() => {
    if (!enabled || !url) return () => {};

    let cancelled = false;

    const connect = () => {
      if (cancelled) return;
      disconnect();
      setStatus("connecting");

      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelled) return;
        setStatus("connected");
      };

      ws.onmessage = (event) => {
        if (cancelled) return;
        try {
          setLastMessage(JSON.parse(event.data));
        } catch (e) {
          setLastMessage({ topic: "ws.message", data: event.data });
        }
      };

      ws.onerror = () => {
        if (cancelled) return;
        setStatus("error");
      };

      ws.onclose = () => {
        if (cancelled) return;
        setStatus("disconnected");
        if (reconnect) {
          reconnectTimerRef.current = setTimeout(connect, reconnectDelayMs);
        }
      };
    };

    connect();

    return () => {
      cancelled = true;
      disconnect();
    };
  }, [disconnect, enabled, reconnect, reconnectDelayMs, url]);

  return { status, lastMessage, sendJson, disconnect };
};
