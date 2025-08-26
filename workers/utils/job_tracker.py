# workers/utils/job_tracker.py - Job status tracking
import redis
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from core.config import get_settings
import logging

logger = logging.getLogger(__name__)


class JobTracker:
    """Track job status and progress in Redis"""

    def __init__(self):
        settings = get_settings()
        try:
            self.redis_client = redis.from_url(settings.redis_url)
            # Test connection
            self.redis_client.ping()
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None

        self.prefix = "charaforge:jobs:"
        self.stats_key = "charaforge:stats"
        self.default_ttl = 3600 * 24 * 7  # 7 days

    def set_job_status(self, job_id: str, status: str, metadata: Dict[str, Any] = None):  # type: ignore
        """Set job status with metadata"""
        if not self.redis_client:
            logger.warning("Redis not available, job status not saved")
            return

        try:
            key = f"{self.prefix}{job_id}"

            data = {
                "job_id": job_id,
                "status": status,
                "updated_at": datetime.now().isoformat(),
            }

            if metadata:
                data.update(metadata)

            # Save job data
            self.redis_client.setex(key, self.default_ttl, json.dumps(data))

            # Update job statistics
            self._update_job_stats(status)

            logger.debug(f"Job status updated: {job_id} -> {status}")

        except Exception as e:
            logger.error(f"Failed to set job status: {e}")

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and metadata"""
        if not self.redis_client:
            return None

        try:
            key = f"{self.prefix}{job_id}"
            data = self.redis_client.get(key)

            if data:
                return json.loads(data)  # type: ignore
            return None

        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return None

    def update_job_progress(self, job_id: str, progress: Dict[str, Any]):
        """Update job progress information"""
        current_data = self.get_job_status(job_id) or {}
        current_data.update(progress)
        current_data["updated_at"] = datetime.now().isoformat()

        self.set_job_status(job_id, current_data.get("status", "running"), current_data)

    def list_jobs(
        self,
        status_filter: Optional[str] = None,
        job_type_filter: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List jobs with optional filters"""
        if not self.redis_client:
            return []

        try:
            pattern = f"{self.prefix}*"
            keys = self.redis_client.keys(pattern)

            jobs = []
            for key in keys[:limit]:  # type: ignore Limit results
                try:
                    data = self.redis_client.get(key)
                    if data:
                        job_data = json.loads(data)  # type: ignore

                        # Apply filters
                        if status_filter and job_data.get("status") != status_filter:
                            continue

                        if (
                            job_type_filter
                            and job_data.get("job_type") != job_type_filter
                        ):
                            continue

                        jobs.append(job_data)

                except Exception as e:
                    logger.warning(f"Failed to parse job data for {key}: {e}")
                    continue

            # Sort by updated_at descending
            jobs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            return jobs

        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            return []

    def cleanup_old_jobs(self, days: int = 7):
        """Clean up jobs older than specified days"""
        if not self.redis_client:
            return

        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            pattern = f"{self.prefix}*"
            keys = self.redis_client.keys(pattern)

            deleted_count = 0
            for key in keys:  # type: ignore
                try:
                    data = self.redis_client.get(key)
                    if data:
                        job_data = json.loads(data)  # type: ignore
                        updated_at = datetime.fromisoformat(
                            job_data.get("updated_at", "")
                        )

                        if updated_at < cutoff_date:
                            self.redis_client.delete(key)
                            deleted_count += 1

                except Exception as e:
                    logger.warning(f"Failed to process job cleanup for {key}: {e}")

            logger.info(f"Cleaned up {deleted_count} old jobs")

        except Exception as e:
            logger.error(f"Job cleanup failed: {e}")

    def _update_job_stats(self, status: str):
        """Update job statistics"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            stats_key = f"{self.stats_key}:{today}"

            # Increment counter for this status
            self.redis_client.hincrby(stats_key, f"{status}_count", 1)  # type: ignore
            self.redis_client.hincrby(stats_key, "total_count", 1)  # type: ignore

            # Set expiry for stats (30 days)
            self.redis_client.expire(stats_key, 3600 * 24 * 30)  # type: ignore

        except Exception as e:
            logger.warning(f"Failed to update job stats: {e}")

    def get_job_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get job statistics for the last N days"""
        if not self.redis_client:
            return {}

        try:
            stats = {}
            total_jobs = 0

            for i in range(days):
                date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                stats_key = f"{self.stats_key}:{date}"

                day_stats = self.redis_client.hgetall(stats_key)
                if day_stats:
                    # Convert bytes to string and int
                    day_stats = {
                        k.decode(): int(v.decode()) for k, v in day_stats.items()  # type: ignore
                    }
                    stats[date] = day_stats
                    total_jobs += day_stats.get("total_count", 0)

            return {
                "daily_stats": stats,
                "total_jobs_period": total_jobs,
                "period_days": days,
            }

        except Exception as e:
            logger.error(f"Failed to get job stats: {e}")
            return {}
