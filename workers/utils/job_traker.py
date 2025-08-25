# workers/utils/job_tracker.py - Job status tracking
import redis
import json
from datetime import datetime
from typing import Dict, Any, Optional
from core.config import get_settings


class JobTracker:
    """Track job status and progress in Redis"""

    def __init__(self):
        settings = get_settings()
        self.redis_client = redis.from_url(settings.redis_url)
        self.prefix = "charaforge:jobs:"

    def set_job_status(self, job_id: str, status: str, metadata: Dict[str, Any] = None):
        """Set job status"""
        key = f"{self.prefix}{job_id}"

        data = {
            "job_id": job_id,
            "status": status,
            "updated_at": datetime.now().isoformat(),
        }

        if metadata:
            data.update(metadata)

        self.redis_client.setex(key, 3600 * 24, json.dumps(data))  # 24 hour TTL

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        key = f"{self.prefix}{job_id}"
        data = self.redis_client.get(key)

        if data:
            return json.loads(data)
        return None

    def update_job_progress(self, job_id: str, progress: Dict[str, Any]):
        """Update job progress"""
        current_data = self.get_job_status(job_id) or {}
        current_data.update(progress)
        current_data["updated_at"] = datetime.now().isoformat()

        key = f"{self.prefix}{job_id}"
        self.redis_client.setex(key, 3600 * 24, json.dumps(current_data))


# workers/utils/queue_monitor.py - Queue monitoring utilities
from workers.celery_app import celery_app
from typing import Dict, Any


class QueueMonitor:
    """Monitor Celery queue status"""

    def __init__(self):
        self.celery_app = celery_app

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        try:
            inspect = self.celery_app.control.inspect()

            # Get active tasks
            active = inspect.active() or {}
            active_count = sum(len(tasks) for tasks in active.values())

            # Get queued tasks (reserved)
            reserved = inspect.reserved() or {}
            queued_count = sum(len(tasks) for tasks in reserved.values())

            # Get worker stats
            stats = inspect.stats() or {}
            worker_count = len(stats)

            return {
                "active_tasks": active_count,
                "queued_tasks": queued_count,
                "workers_online": worker_count,
                "queues": {
                    "training": self._get_queue_length("training"),
                    "generation": self._get_queue_length("generation"),
                    "batch": self._get_queue_length("batch"),
                },
            }

        except Exception as e:
            return {
                "error": f"Failed to get queue stats: {str(e)}",
                "active_tasks": 0,
                "queued_tasks": 0,
                "workers_online": 0,
            }

    def _get_queue_length(self, queue_name: str) -> int:
        """Get length of specific queue"""
        try:
            # This would require additional Redis queries
            # For now, return 0 as placeholder
            return 0
        except:
            return 0
