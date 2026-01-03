"""Simple in-process metrics for Prometheus scraping."""

from __future__ import annotations

import threading
from typing import Dict, Tuple


def _escape_label(value: str) -> str:
    return (
        (value or "")
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace('"', '\\"')
    )


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._in_flight = 0
        self._ws_active = 0
        self._requests_total: Dict[Tuple[str, str, int], int] = {}
        self._duration_sum: Dict[Tuple[str, str], float] = {}
        self._duration_count: Dict[Tuple[str, str], int] = {}
        self._rate_limit_denied_total: Dict[str, int] = {}
        self._auth_refresh_total: Dict[Tuple[str, str], int] = {}
        self._auth_revoke_total: Dict[str, int] = {}
        self._ws_connections_total: Dict[Tuple[str, str], int] = {}
        self._ws_disconnect_total: Dict[Tuple[str, str], int] = {}

    def inc_in_flight(self) -> None:
        with self._lock:
            self._in_flight += 1

    def dec_in_flight(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    def inc_ws_active(self) -> None:
        with self._lock:
            self._ws_active += 1

    def dec_ws_active(self) -> None:
        with self._lock:
            self._ws_active = max(0, self._ws_active - 1)

    def observe_request(self, *, method: str, route: str, status_code: int, duration_s: float) -> None:
        method = (method or "").upper()
        route = route or ""
        status_code = int(status_code)
        duration_s = float(duration_s or 0.0)

        with self._lock:
            key_total = (method, route, status_code)
            self._requests_total[key_total] = self._requests_total.get(key_total, 0) + 1

            key_latency = (method, route)
            self._duration_sum[key_latency] = self._duration_sum.get(key_latency, 0.0) + duration_s
            self._duration_count[key_latency] = self._duration_count.get(key_latency, 0) + 1

    def inc_rate_limited(self, *, bucket: str) -> None:
        bucket = str(bucket or "global")
        with self._lock:
            self._rate_limit_denied_total[bucket] = self._rate_limit_denied_total.get(bucket, 0) + 1

    def inc_auth_refresh(self, *, result: str, source: str) -> None:
        result = str(result or "unknown")
        source = str(source or "unknown")
        with self._lock:
            key = (result, source)
            self._auth_refresh_total[key] = self._auth_refresh_total.get(key, 0) + 1

    def inc_auth_revoke(self, *, reason: str) -> None:
        reason = str(reason or "unknown")
        with self._lock:
            self._auth_revoke_total[reason] = self._auth_revoke_total.get(reason, 0) + 1

    def inc_ws_connection(self, *, endpoint: str, outcome: str) -> None:
        endpoint = str(endpoint or "unknown")
        outcome = str(outcome or "unknown")
        with self._lock:
            key = (endpoint, outcome)
            self._ws_connections_total[key] = self._ws_connections_total.get(key, 0) + 1

    def inc_ws_disconnect(self, *, endpoint: str, code: str) -> None:
        endpoint = str(endpoint or "unknown")
        code = str(code or "unknown")
        with self._lock:
            key = (endpoint, code)
            self._ws_disconnect_total[key] = self._ws_disconnect_total.get(key, 0) + 1

    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            in_flight = int(self._in_flight)
            ws_active = int(self._ws_active)
            requests_total = dict(self._requests_total)
            duration_sum = dict(self._duration_sum)
            duration_count = dict(self._duration_count)
            rate_limit_denied_total = dict(self._rate_limit_denied_total)
            auth_refresh_total = dict(self._auth_refresh_total)
            auth_revoke_total = dict(self._auth_revoke_total)
            ws_connections_total = dict(self._ws_connections_total)
            ws_disconnect_total = dict(self._ws_disconnect_total)

        lines.append("# HELP charaforge_http_in_flight_requests In-flight HTTP requests.")
        lines.append("# TYPE charaforge_http_in_flight_requests gauge")
        lines.append(f"charaforge_http_in_flight_requests {in_flight}")

        lines.append("# HELP charaforge_ws_active_connections Active WebSocket connections.")
        lines.append("# TYPE charaforge_ws_active_connections gauge")
        lines.append(f"charaforge_ws_active_connections {ws_active}")

        lines.append("# HELP charaforge_http_requests_total Total HTTP requests.")
        lines.append("# TYPE charaforge_http_requests_total counter")
        for (method, route, status), value in sorted(requests_total.items()):
            lines.append(
                "charaforge_http_requests_total"
                f'{{method="{_escape_label(method)}",route="{_escape_label(route)}",status="{status}"}} {value}'
            )

        lines.append(
            "# HELP charaforge_http_request_duration_seconds Request duration (seconds)."
        )
        lines.append("# TYPE charaforge_http_request_duration_seconds summary")
        for (method, route), value in sorted(duration_sum.items()):
            lines.append(
                "charaforge_http_request_duration_seconds_sum"
                f'{{method="{_escape_label(method)}",route="{_escape_label(route)}"}} {value}'
            )
        for (method, route), value in sorted(duration_count.items()):
            lines.append(
                "charaforge_http_request_duration_seconds_count"
                f'{{method="{_escape_label(method)}",route="{_escape_label(route)}"}} {value}'
            )

        lines.append("# HELP charaforge_rate_limit_denied_total Rate-limited requests.")
        lines.append("# TYPE charaforge_rate_limit_denied_total counter")
        for bucket, value in sorted(rate_limit_denied_total.items()):
            lines.append(
                "charaforge_rate_limit_denied_total"
                f'{{bucket="{_escape_label(bucket)}"}} {value}'
            )

        lines.append("# HELP charaforge_auth_refresh_total Refresh token attempts.")
        lines.append("# TYPE charaforge_auth_refresh_total counter")
        for (result, source), value in sorted(auth_refresh_total.items()):
            lines.append(
                "charaforge_auth_refresh_total"
                f'{{result="{_escape_label(result)}",source="{_escape_label(source)}"}} {value}'
            )

        lines.append("# HELP charaforge_auth_revoke_total Refresh token/session revocations.")
        lines.append("# TYPE charaforge_auth_revoke_total counter")
        for reason, value in sorted(auth_revoke_total.items()):
            lines.append(
                "charaforge_auth_revoke_total"
                f'{{reason="{_escape_label(reason)}"}} {value}'
            )

        lines.append("# HELP charaforge_ws_connections_total WebSocket connection attempts.")
        lines.append("# TYPE charaforge_ws_connections_total counter")
        for (endpoint, outcome), value in sorted(ws_connections_total.items()):
            lines.append(
                "charaforge_ws_connections_total"
                f'{{endpoint="{_escape_label(endpoint)}",outcome="{_escape_label(outcome)}"}} {value}'
            )

        lines.append("# HELP charaforge_ws_disconnect_total WebSocket disconnects.")
        lines.append("# TYPE charaforge_ws_disconnect_total counter")
        for (endpoint, code), value in sorted(ws_disconnect_total.items()):
            lines.append(
                "charaforge_ws_disconnect_total"
                f'{{endpoint="{_escape_label(endpoint)}",code="{_escape_label(code)}"}} {value}'
            )

        return "\n".join(lines) + "\n"
