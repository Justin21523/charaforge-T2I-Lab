# workers/utils/queue_monitor.py - Queue monitoring utilities
from workers.celery_app import celery_app
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class QueueMonitor:
    """Monitor Celery queue status and worker health"""

    def __init__(self):
        self.celery_app = celery_app

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics"""
        try:
            inspect = self.celery_app.control.inspect()

            # Get active tasks
            active_tasks = inspect.active()
            active_count = 0
            if active_tasks:
                for worker, tasks in active_tasks.items():
                    active_count += len(tasks)

            # Get reserved tasks (queued)
            reserved_tasks = inspect.reserved()
            reserved_count = 0
            if reserved_tasks:
                for worker, tasks in reserved_tasks.items():
                    reserved_count += len(tasks)

            # Get scheduled tasks
            scheduled_tasks = inspect.scheduled()
            scheduled_count = 0
            if scheduled_tasks:
                for worker, tasks in scheduled_tasks.items():
                    scheduled_count += len(tasks)

            # Get worker stats
            worker_stats = inspect.stats()
            worker_count = len(worker_stats) if worker_stats else 0

            # Get registered tasks
            registered_tasks = inspect.registered()

            # Queue-specific stats
            queue_stats = {}
            for queue in ["training", "generation", "batch", "default"]:
                queue_stats[queue] = {
                    "length": self._get_queue_length(queue),
                    "workers": self._get_queue_workers(queue, worker_stats),
                }

            return {
                "active_tasks": active_count,
                "reserved_tasks": reserved_count,
                "scheduled_tasks": scheduled_count,
                "total_queued": reserved_count + scheduled_count,
                "workers_online": worker_count,
                "queue_details": queue_stats,
                "worker_details": self._format_worker_details(worker_stats),
                "registered_tasks": (
                    list(registered_tasks.values())[0] if registered_tasks else []
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {
                "error": f"Failed to get queue stats: {str(e)}",
                "active_tasks": 0,
                "reserved_tasks": 0,
                "workers_online": 0,
                "queue_details": {},
            }

    def get_active_tasks_details(self) -> List[Dict[str, Any]]:
        """Get detailed information about active tasks"""
        try:
            inspect = self.celery_app.control.inspect()
            active_tasks = inspect.active()

            if not active_tasks:
                return []

            tasks_details = []
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    tasks_details.append(
                        {
                            "worker": worker,
                            "task_id": task.get("id"),
                            "task_name": task.get("name"),
                            "args": task.get("args", []),
                            "kwargs": task.get("kwargs", {}),
                            "time_start": task.get("time_start"),
                            "acknowledged": task.get("acknowledged"),
                            "delivery_info": task.get("delivery_info", {}),
                        }
                    )

            return tasks_details

        except Exception as e:
            logger.error(f"Failed to get active task details: {e}")
            return []

    def get_worker_health(self) -> Dict[str, Any]:
        """Check worker health status"""
        try:
            inspect = self.celery_app.control.inspect()

            # Ping workers
            ping_result = inspect.ping()
            online_workers = list(ping_result.keys()) if ping_result else []

            # Get worker stats
            worker_stats = inspect.stats() or {}

            # Check each worker
            worker_health = {}
            for worker in online_workers:
                stats = worker_stats.get(worker, {})

                worker_health[worker] = {
                    "online": True,
                    "load": stats.get("rusage", {}).get("stime", 0),
                    "memory": stats.get("rusage", {}).get("maxrss", 0),
                    "total_tasks": stats.get("total", 0),
                    "pool_processes": stats.get("pool", {}).get("max-concurrency", 0),
                    "prefetch_count": stats.get("prefetch_count", 0),
                }

            return {
                "total_workers": len(online_workers),
                "online_workers": online_workers,
                "worker_details": worker_health,
                "overall_health": "healthy" if len(online_workers) > 0 else "unhealthy",
            }

        except Exception as e:
            logger.error(f"Failed to get worker health: {e}")
            return {
                "total_workers": 0,
                "online_workers": [],
                "overall_health": "unknown",
                "error": str(e),
            }

    def _get_queue_length(self, queue_name: str) -> int:
        """Get length of specific queue (approximate)"""
        try:
            # This is a simplified approach
            # In production, you might want to query the broker directly
            inspect = self.celery_app.control.inspect()

            # Get reserved tasks and filter by routing key
            reserved = inspect.reserved()
            if not reserved:
                return 0

            queue_length = 0
            for worker, tasks in reserved.items():
                for task in tasks:
                    delivery_info = task.get("delivery_info", {})
                    routing_key = delivery_info.get("routing_key", "")

                    if queue_name in routing_key or (
                        queue_name == "default" and not routing_key
                    ):
                        queue_length += 1

            return queue_length

        except Exception as e:
            logger.warning(f"Failed to get queue length for {queue_name}: {e}")
            return 0

    def _get_queue_workers(self, queue_name: str, worker_stats: Dict) -> List[str]:
        """Get workers assigned to specific queue"""
        if not worker_stats:
            return []

        # This is simplified - in practice you'd check worker routing
        return list(worker_stats.keys())

    def _format_worker_details(self, worker_stats: Dict) -> Dict[str, Any]:
        """Format worker statistics for display"""
        if not worker_stats:
            return {}

        formatted = {}
        for worker, stats in worker_stats.items():
            formatted[worker] = {
                "broker": stats.get("broker", {}),
                "clock": stats.get("clock", 0),
                "pid": stats.get("pid", 0),
                "pool": stats.get("pool", {}),
                "prefetch_count": stats.get("prefetch_count", 0),
                "rusage": stats.get("rusage", {}),
                "total_tasks": stats.get("total", 0),
            }

        return formatted

    def restart_workers(self, worker_names: List[str] = None) -> Dict[str, Any]:  # type: ignore
        """Restart specific workers or all workers"""
        try:
            control = self.celery_app.control

            if worker_names:
                # Restart specific workers
                result = {}
                for worker in worker_names:
                    try:
                        control.broadcast("pool_restart", destination=[worker])
                        result[worker] = "restart_requested"
                    except Exception as e:
                        result[worker] = f"restart_failed: {e}"

                return {"status": "partial", "results": result}
            else:
                # Restart all workers
                control.broadcast("pool_restart")
                return {
                    "status": "success",
                    "message": "Restart requested for all workers",
                }

        except Exception as e:
            return {"status": "failed", "error": f"Failed to restart workers: {e}"}

    def purge_queue(self, queue_name: str) -> Dict[str, Any]:
        """Purge all tasks from a specific queue"""
        try:
            # This is dangerous - implement with caution
            purged = self.celery_app.control.purge()

            return {
                "status": "success",
                "purged_tasks": purged,
                "message": f"Queue {queue_name} purged",
            }

        except Exception as e:
            return {"status": "failed", "error": f"Failed to purge queue: {e}"}
