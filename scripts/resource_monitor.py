# scripts/resource_monitor.py - System Resource Monitoring Script
"""
System resource monitoring script for production deployment
"""

import os
import sys
import time
import json
from datetime import datetime
import psutil
import signal
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    import GPUtil

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: GPUtil not available, GPU monitoring disabled")

# Shared Cache Bootstrap
import pathlib, torch

AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "/mnt/ai_warehouse/cache")
for k, v in {
    "HF_HOME": f"{AI_CACHE_ROOT}/hf",
    "TRANSFORMERS_CACHE": f"{AI_CACHE_ROOT}/hf/transformers",
    "HF_DATASETS_CACHE": f"{AI_CACHE_ROOT}/hf/datasets",
    "HUGGINGFACE_HUB_CACHE": f"{AI_CACHE_ROOT}/hf/hub",
    "TORCH_HOME": f"{AI_CACHE_ROOT}/torch",
}.items():
    os.environ[k] = v
    pathlib.Path(v).mkdir(parents=True, exist_ok=True)

from ..utils.logging import setup_logger, get_logger

logger = get_logger(__name__)


class SystemMonitor:
    """System resource monitor with alerting"""

    def __init__(self, interval: int = 30):
        self.interval = interval
        self.running = True
        self.metrics_file = f"{AI_CACHE_ROOT}/logs/metrics.json"
        self.alert_history = {}

        # Alert thresholds
        self.thresholds = {
            "cpu_warning": 80,
            "cpu_critical": 95,
            "memory_warning": 80,
            "memory_critical": 95,
            "disk_warning": 85,
            "disk_critical": 95,
            "gpu_memory_warning": 80,
            "gpu_memory_critical": 95,
            "memory_available_min": 500,  # MB
        }

        # Ensure metrics directory exists
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down monitor...")
        self.running = False

    def collect_metrics(self) -> dict:
        """Collect system metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {},
            "memory": {},
            "disk": {},
            "gpu": [],
            "processes": {},
        }

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        load_avg = os.getloadavg() if hasattr(os, "getloadavg") else [0, 0, 0]  # type: ignore

        metrics["cpu"] = {
            "usage_percent": cpu_percent,
            "count": cpu_count,
            "load_1m": load_avg[0],
            "load_5m": load_avg[1],
            "load_15m": load_avg[2],
        }

        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        metrics["memory"] = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "usage_percent": memory.percent,
            "swap_total_gb": swap.total / (1024**3),
            "swap_used_gb": swap.used / (1024**3),
            "swap_percent": swap.percent,
        }

        # Disk metrics
        disk = psutil.disk_usage("/")
        metrics["disk"] = {
            "total_gb": disk.total / (1024**3),
            "used_gb": disk.used / (1024**3),
            "free_gb": disk.free / (1024**3),
            "usage_percent": (disk.used / disk.total) * 100,
        }

        # Cache disk usage
        if os.path.exists(AI_CACHE_ROOT):
            cache_disk = psutil.disk_usage(AI_CACHE_ROOT)
            metrics["disk"]["cache_total_gb"] = cache_disk.total / (1024**3)
            metrics["disk"]["cache_used_gb"] = cache_disk.used / (1024**3)
            metrics["disk"]["cache_free_gb"] = cache_disk.free / (1024**3)
            metrics["disk"]["cache_usage_percent"] = (
                cache_disk.used / cache_disk.total
            ) * 100

        # GPU metrics
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    metrics["gpu"].append(
                        {
                            "id": gpu.id,
                            "name": gpu.name,
                            "load_percent": gpu.load * 100,
                            "memory_used_mb": gpu.memoryUsed,
                            "memory_total_mb": gpu.memoryTotal,
                            "memory_percent": gpu.memoryUtil * 100,
                            "temperature_c": gpu.temperature,
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to collect GPU metrics: {e}")

        # Process metrics
        python_processes = []
        for proc in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_percent", "cmdline"]
        ):
            try:
                if "python" in proc.info["name"].lower():
                    cmdline = " ".join(proc.info["cmdline"][:3])  # First 3 args
                    python_processes.append(
                        {
                            "pid": proc.info["pid"],
                            "name": proc.info["name"],
                            "cpu_percent": proc.info["cpu_percent"],
                            "memory_percent": proc.info["memory_percent"],
                            "cmdline": cmdline,
                        }
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        metrics["processes"]["python"] = python_processes[:10]  # Top 10

        return metrics

    def check_alerts(self, metrics: dict):
        """Check metrics against thresholds and generate alerts"""
        alerts = []
        current_time = datetime.now()

        # CPU alerts
        cpu_usage = metrics["cpu"]["usage_percent"]
        if cpu_usage >= self.thresholds["cpu_critical"]:
            alerts.append(
                {
                    "level": "CRITICAL",
                    "type": "cpu",
                    "message": f"Critical CPU usage: {cpu_usage:.1f}%",
                    "value": cpu_usage,
                    "threshold": self.thresholds["cpu_critical"],
                }
            )
        elif cpu_usage >= self.thresholds["cpu_warning"]:
            alerts.append(
                {
                    "level": "WARNING",
                    "type": "cpu",
                    "message": f"High CPU usage: {cpu_usage:.1f}%",
                    "value": cpu_usage,
                    "threshold": self.thresholds["cpu_warning"],
                }
            )

        # Memory alerts
        memory_usage = metrics["memory"]["usage_percent"]
        memory_available = metrics["memory"]["available_gb"] * 1024  # Convert to MB

        if memory_usage >= self.thresholds["memory_critical"]:
            alerts.append(
                {
                    "level": "CRITICAL",
                    "type": "memory",
                    "message": f"Critical memory usage: {memory_usage:.1f}%",
                    "value": memory_usage,
                    "threshold": self.thresholds["memory_critical"],
                }
            )
        elif memory_usage >= self.thresholds["memory_warning"]:
            alerts.append(
                {
                    "level": "WARNING",
                    "type": "memory",
                    "message": f"High memory usage: {memory_usage:.1f}%",
                    "value": memory_usage,
                    "threshold": self.thresholds["memory_warning"],
                }
            )

        if memory_available < self.thresholds["memory_available_min"]:
            alerts.append(
                {
                    "level": "CRITICAL",
                    "type": "memory",
                    "message": f"Low available memory: {memory_available:.0f}MB",
                    "value": memory_available,
                    "threshold": self.thresholds["memory_available_min"],
                }
            )

        # Disk alerts
        disk_usage = metrics["disk"]["usage_percent"]
        if disk_usage >= self.thresholds["disk_critical"]:
            alerts.append(
                {
                    "level": "CRITICAL",
                    "type": "disk",
                    "message": f"Critical disk usage: {disk_usage:.1f}%",
                    "value": disk_usage,
                    "threshold": self.thresholds["disk_critical"],
                }
            )
        elif disk_usage >= self.thresholds["disk_warning"]:
            alerts.append(
                {
                    "level": "WARNING",
                    "type": "disk",
                    "message": f"High disk usage: {disk_usage:.1f}%",
                    "value": disk_usage,
                    "threshold": self.thresholds["disk_warning"],
                }
            )

        # GPU alerts
        for gpu in metrics["gpu"]:
            gpu_memory = gpu["memory_percent"]
            gpu_name = gpu["name"]

            if gpu_memory >= self.thresholds["gpu_memory_critical"]:
                alerts.append(
                    {
                        "level": "CRITICAL",
                        "type": "gpu",
                        "message": f"Critical GPU memory usage on {gpu_name}: {gpu_memory:.1f}%",
                        "value": gpu_memory,
                        "threshold": self.thresholds["gpu_memory_critical"],
                    }
                )
            elif gpu_memory >= self.thresholds["gpu_memory_warning"]:
                alerts.append(
                    {
                        "level": "WARNING",
                        "type": "gpu",
                        "message": f"High GPU memory usage on {gpu_name}: {gpu_memory:.1f}%",
                        "value": gpu_memory,
                        "threshold": self.thresholds["gpu_memory_warning"],
                    }
                )

        # Log alerts (with cooldown to avoid spam)
        for alert in alerts:
            alert_key = f"{alert['type']}_{alert['level']}"
            last_alert = self.alert_history.get(alert_key)

            # 5-minute cooldown between same alerts
            if not last_alert or (current_time - last_alert).total_seconds() > 300:
                if alert["level"] == "CRITICAL":
                    logger.critical(alert["message"])
                else:
                    logger.warning(alert["message"])
                self.alert_history[alert_key] = current_time

        return alerts

    def save_metrics(self, metrics: dict):
        """Save metrics to file"""
        try:
            # Append to metrics file
            with open(self.metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def run(self):
        """Main monitoring loop"""
        logger.info(f"Starting system monitor (interval: {self.interval}s)")
        logger.info(f"Metrics file: {self.metrics_file}")
        logger.info(f"GPU monitoring: {'enabled' if GPU_AVAILABLE else 'disabled'}")

        while self.running:
            try:
                # Collect metrics
                metrics = self.collect_metrics()

                # Check for alerts
                alerts = self.check_alerts(metrics)
                metrics["alerts"] = alerts

                # Save metrics
                self.save_metrics(metrics)

                # Log summary every 10 minutes
                current_time = datetime.now()
                if (
                    current_time.minute % 10 == 0
                    and current_time.second < self.interval
                ):
                    cpu = metrics["cpu"]["usage_percent"]
                    memory = metrics["memory"]["usage_percent"]
                    disk = metrics["disk"]["usage_percent"]
                    gpu_info = ""
                    if metrics["gpu"]:
                        gpu_memory = metrics["gpu"][0]["memory_percent"]
                        gpu_info = f", GPU: {gpu_memory:.1f}%"

                    logger.info(
                        f"System health - CPU: {cpu:.1f}%, Memory: {memory:.1f}%, Disk: {disk:.1f}%{gpu_info}"
                    )

                time.sleep(self.interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.interval)

        logger.info("System monitor stopped")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="System resource monitor")
    parser.add_argument(
        "--interval", "-i", type=int, default=30, help="Monitoring interval in seconds"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    # setup_logger()
    logger = get_logger(__name__)

    if args.verbose:
        logger.setLevel("DEBUG")

    # Start monitoring
    monitor = SystemMonitor(interval=args.interval)
    try:
        monitor.run()
    except KeyboardInterrupt:
        logger.info("Monitor stopped by user")
    except Exception as e:
        logger.error(f"Monitor failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
