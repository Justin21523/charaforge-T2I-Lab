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
        self._requests_total: Dict[Tuple[str, str, int], int] = {}
        self._duration_sum: Dict[Tuple[str, str], float] = {}
        self._duration_count: Dict[Tuple[str, str], int] = {}

    def inc_in_flight(self) -> None:
        with self._lock:
            self._in_flight += 1

    def dec_in_flight(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

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

    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            in_flight = int(self._in_flight)
            requests_total = dict(self._requests_total)
            duration_sum = dict(self._duration_sum)
            duration_count = dict(self._duration_count)

        lines.append("# HELP charaforge_http_in_flight_requests In-flight HTTP requests.")
        lines.append("# TYPE charaforge_http_in_flight_requests gauge")
        lines.append(f"charaforge_http_in_flight_requests {in_flight}")

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

        return "\n".join(lines) + "\n"

