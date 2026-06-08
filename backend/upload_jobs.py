"""上传任务进度管理。

默认使用进程内存保存任务状态，适合单进程开发部署。
设置 UPLOAD_JOBS_STORE=redis 启用 Redis 持久化，支持进程重启恢复。
"""
from __future__ import annotations

import json
import logging
import os

from copy import deepcopy
from datetime import UTC, datetime
from threading import Lock
from typing import Literal
from uuid import uuid4

logger = logging.getLogger(__name__)


StepStatus = Literal["pending", "running", "completed", "failed"]
JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


DEFAULT_STEPS = [
    ("upload", "文档上传"),
    ("cleanup", "清理旧版本"),
    ("parse", "解析与分块"),
    ("parent_store", "父级分块入库"),
    ("vector_store", "向量化入库"),
]

DELETE_STEPS = [
    ("prepare", "准备删除"),
    ("bm25", "同步 BM25 统计"),
    ("milvus", "删除向量数据"),
    ("parent_store", "删除父级分块"),
]

REDIS_JOB_KEY = "upload_job"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class UploadJobManager:
    """线程安全的上传任务状态容器，可选 Redis 持久化。"""

    def __init__(self, redis_enabled: bool | None = None, max_jobs: int = 100):
        self._jobs: dict[str, dict] = {}
        self._lock = Lock()
        self._max_jobs = max_jobs

        if redis_enabled is None:
            # Auto-detect: prefer Redis when available
            if os.getenv("UPLOAD_JOBS_STORE", "").lower() == "redis":
                redis_enabled = True
            else:
                redis_enabled = self._is_redis_available()

        self._redis_enabled = redis_enabled
        self._cache = None
        if self._redis_enabled:
            try:
                from cache import RedisCache
                self._cache = RedisCache()
                # Verify the connection actually works
                self._cache._get_client().ping()
            except Exception:
                self._redis_enabled = False
                self._cache = None
                logger.warning("Redis 连接失败，回退到单实例内存模式（进程重启后任务状态将丢失）")
        else:
            logger.warning("未启用 Redis 持久化，运行在单实例内存模式（进程重启后任务状态将丢失）")

    @staticmethod
    def _is_redis_available() -> bool:
        """Check whether Redis is reachable."""
        try:
            from cache import RedisCache
            cache = RedisCache()
            cache._get_client().ping()
            return True
        except Exception:
            return False

    def _redis_job_key(self, job_id: str) -> str:
        return f"{REDIS_JOB_KEY}:{job_id}"

    def _persist(self, job: dict) -> None:
        if self._cache is not None:
            self._cache.set_json(self._redis_job_key(job["job_id"]), job, ttl=7200)

    def _load_from_redis(self, job_id: str) -> dict | None:
        if self._cache is None:
            return None
        return self._cache.get_json(self._redis_job_key(job_id))

    def _cleanup_old_jobs(self) -> None:
        """Remove completed/failed jobs older than 1 hour when dict exceeds max size.

        Must be called while self._lock is held.
        """
        if len(self._jobs) <= self._max_jobs:
            return
        now = datetime.now(UTC)
        to_remove: list[str] = []
        for job_id, job in self._jobs.items():
            if job.get("status") in ("completed", "failed"):
                updated = job.get("updated_at", "")
                try:
                    dt = datetime.fromisoformat(updated)
                    if (now - dt).total_seconds() > 3600:
                        to_remove.append(job_id)
                except (ValueError, TypeError):
                    to_remove.append(job_id)
        for jid in to_remove:
            del self._jobs[jid]
        # If still over max, evict oldest completed/failed regardless of age
        if len(self._jobs) > self._max_jobs:
            candidates = [
                (jid, j) for jid, j in self._jobs.items()
                if j.get("status") in ("completed", "failed")
            ]
            candidates.sort(key=lambda x: x[1].get("updated_at", ""))
            excess = len(self._jobs) - self._max_jobs
            for jid, _ in candidates[:excess]:
                del self._jobs[jid]

    def create_job(
        self,
        filename: str,
        *,
        steps: list[tuple[str, str]] | None = None,
        current_step: str = "upload",
        message: str = "等待上传",
        completion_step: str = "vector_store",
    ) -> dict:
        steps = steps or DEFAULT_STEPS
        job_id = uuid4().hex
        now = _now_iso()
        job = {
            "job_id": job_id,
            "filename": filename,
            "status": "pending",
            "current_step": current_step,
            "message": message,
            "completion_step": completion_step,
            "total_chunks": 0,
            "processed_chunks": 0,
            "error": None,
            "created_at": now,
            "updated_at": now,
            "steps": [
                {
                    "key": key,
                    "label": label,
                    "percent": 0,
                    "status": "pending",
                    "message": "",
                }
                for key, label in steps
            ],
        }
        with self._lock:
            self._jobs[job_id] = job
            self._persist(job)
            self._cleanup_old_jobs()
            return deepcopy(job)

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None and self._redis_enabled:
                job = self._load_from_redis(job_id)
                if job is not None:
                    self._jobs[job_id] = job
            return deepcopy(job) if job else None

    def update_step(
        self,
        job_id: str,
        step_key: str,
        percent: int,
        status: StepStatus = "running",
        message: str = "",
        *,
        total_chunks: int | None = None,
        processed_chunks: int | None = None,
    ) -> dict | None:
        percent = max(0, min(100, int(percent)))
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None and self._redis_enabled:
                job = self._load_from_redis(job_id)
                if job is not None:
                    self._jobs[job_id] = job
            if not job:
                return None

            step = self._find_step(job, step_key)
            if not step:
                return None

            step["percent"] = percent
            step["status"] = status
            step["message"] = message
            if job["status"] not in ("completed", "cancelled"):
                job["status"] = "failed" if status == "failed" else "running"
            job["current_step"] = step_key
            job["message"] = message
            job["updated_at"] = _now_iso()

            if total_chunks is not None:
                job["total_chunks"] = int(total_chunks)
            if processed_chunks is not None:
                job["processed_chunks"] = int(processed_chunks)

            self._persist(job)
            return deepcopy(job)

    def complete_step(self, job_id: str, step_key: str, message: str = "") -> dict | None:
        return self.update_step(job_id, step_key, 100, "completed", message)

    def complete_job(self, job_id: str, message: str = "文档入库完成") -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None and self._redis_enabled:
                job = self._load_from_redis(job_id)
            if not job:
                return None
            for step in job["steps"]:
                if step["status"] != "failed":
                    step["percent"] = 100
                    step["status"] = "completed"
            job["status"] = "completed"
            job["current_step"] = job.get("completion_step") or job["current_step"]
            job["message"] = message
            job["error"] = None
            job["updated_at"] = _now_iso()
            self._persist(job)
            return deepcopy(job)

    def cancel_job(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None and self._redis_enabled:
                job = self._load_from_redis(job_id)
            if not job:
                return None
            job["status"] = "cancelled"
            job["message"] = "任务已取消，正在回退..."
            job["updated_at"] = _now_iso()
            self._persist(job)
            return deepcopy(job)

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None and self._redis_enabled:
                job = self._load_from_redis(job_id)
            return job["status"] == "cancelled" if job else False

    def fail_job(self, job_id: str, step_key: str, error: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None and self._redis_enabled:
                job = self._load_from_redis(job_id)
            if not job:
                return None
            step = self._find_step(job, step_key)
            if step:
                step["status"] = "failed"
                step["message"] = error
            job["status"] = "failed"
            job["current_step"] = step_key
            job["message"] = error
            job["error"] = error
            job["updated_at"] = _now_iso()
            self._persist(job)
            return deepcopy(job)

    def list_jobs(self) -> list[dict]:
        with self._lock:
            if self._redis_enabled and self._cache is not None:
                try:
                    client = self._cache._get_client()
                    prefix = self._cache.key_prefix + ":"
                    pattern = f"{prefix}{REDIS_JOB_KEY}:*"
                    keys = list(client.scan_iter(match=pattern))
                    if keys:
                        jobs = []
                        for key in keys:
                            raw = client.get(key)
                            if raw:
                                try:
                                    jobs.append(json.loads(raw))
                                except Exception:
                                    pass
                        return jobs
                except Exception:
                    pass
            return [deepcopy(job) for job in self._jobs.values()]

    @staticmethod
    def _find_step(job: dict, step_key: str) -> dict | None:
        for step in job["steps"]:
            if step["key"] == step_key:
                return step
        return None


upload_job_manager = UploadJobManager()
delete_job_manager = UploadJobManager()
