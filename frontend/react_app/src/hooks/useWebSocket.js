import { useCallback, useEffect, useRef, useState } from "react";

const DEFAULT_RECONNECT_DELAY_MS = 1500;

const toWsUrl = (httpBaseUrl, path) => {
  const url = new URL(httpBaseUrl);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = path.startsWith("/") ? path : `/${path}`;
  url.search = "";
  url.hash = "";
  return url.toString();
};

export const buildTrainProgressWsUrl = (apiBaseUrl, jobId) =>
  toWsUrl(apiBaseUrl, `/api/v1/ws/train/${jobId}`);

export const useWebSocket = (
  url,
  {
    enabled = true,
    reconnect = true,
    reconnectDelayMs = DEFAULT_RECONNECT_DELAY_MS,
    protocols = null,
  } = {}
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
    let connectAttempt = 0;

    const connect = () => {
      if (cancelled) return;
      connectAttempt += 1;
      const attemptId = connectAttempt;
      disconnect();
      setStatus("connecting");

      (async () => {
        let selectedProtocols = null;
        try {
          selectedProtocols = typeof protocols === "function" ? protocols() : protocols;
          selectedProtocols = await Promise.resolve(selectedProtocols);
        } catch (e) {
          selectedProtocols = null;
        }

        if (cancelled || attemptId !== connectAttempt) return;

        const ws = selectedProtocols ? new WebSocket(url, selectedProtocols) : new WebSocket(url);
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
      })();
    };

    connect();

    return () => {
      cancelled = true;
      disconnect();
    };
  }, [disconnect, enabled, reconnect, reconnectDelayMs, url]);

  return { status, lastMessage, sendJson, disconnect };
};
