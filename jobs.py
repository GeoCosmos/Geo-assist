"""
In-memory job registry for background ingest tasks.

Each bulk ingest creates a Job and returns its ID immediately. The background
task updates prepared/status as work progresses. Clients poll /ingest/status/{id}.

Jobs live only in memory — they don't survive a server restart. That's fine:
a restarted server can't resume in-flight embedding anyway.
"""
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Job:
    id: str
    total: int                        # files accepted for processing
    prepared: int = 0                 # files through parse+embed (the slow phase)
    status: str = "running"           # running | done | failed
    results: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # pre-flight rejections + runtime errors
    created_at: float = field(default_factory=time.time)


_registry: dict[str, Job] = {}
_JOB_TTL = 2 * 3600  # seconds — completed jobs older than this are evicted on next create()


def _evict_old() -> None:
    cutoff = time.time() - _JOB_TTL
    stale = [jid for jid, j in _registry.items() if j.created_at < cutoff]
    for jid in stale:
        del _registry[jid]


def create(total: int, errors: list[str] | None = None) -> Job:
    _evict_old()
    job = Job(id=uuid.uuid4().hex[:8], total=total, errors=list(errors or []))
    _registry[job.id] = job
    return job


def get(job_id: str) -> Job | None:
    return _registry.get(job_id)
