"""Shared context cache for DAG-Native execution.

Provides a TTL-based cache for shared prerequisite knowledge, with
file-hash-based invalidation for stale entries.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class CacheEntry:
    """A cached context entry with TTL and hash-based validation."""
    key: str
    content: str
    token_size: int = 0
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 300.0  # 5 min default TTL
    source_file_hash: str = ""   # hash of the source file for invalidation
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        """Check if the entry has exceeded its TTL."""
        return (time.time() - self.created_at) > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def access(self) -> None:
        """Record an access to this entry."""
        self.access_count += 1
        self.last_accessed = time.time()


class ContextCache:
    """In-memory cache for shared context entries.

    Features:
    - TTL-based expiration
    - File-hash-based invalidation
    - LRU eviction when capacity is exceeded
    - Hit/miss statistics
    """

    def __init__(self, max_entries: int = 100, default_ttl: float = 300.0) -> None:
        self._cache: dict[str, CacheEntry] = {}
        self._max_entries = max_entries
        self._default_ttl = default_ttl

        # Statistics
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._stale_evictions: int = 0

    def get(self, key: str) -> str | None:
        """Retrieve content from cache. Returns None on miss or expiration."""
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired:
            del self._cache[key]
            self._stale_evictions += 1
            self._misses += 1
            return None

        entry.access()
        self._hits += 1
        return entry.content

    def put(
        self,
        key: str,
        content: str,
        ttl: float | None = None,
        source_file_hash: str = "",
    ) -> None:
        """Store content in the cache."""
        # Evict if at capacity
        if len(self._cache) >= self._max_entries:
            self._evict_lru()

        token_size = len(content) // 4  # estimation

        entry = CacheEntry(
            key=key,
            content=content,
            token_size=token_size,
            ttl_seconds=ttl or self._default_ttl,
            source_file_hash=source_file_hash,
        )
        self._cache[key] = entry

    def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache entry."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def invalidate_by_file_hash(self, file_path: str, new_hash: str) -> int:
        """Invalidate all entries whose source file hash differs from new_hash."""
        count = 0
        stale_keys = [
            k for k, e in self._cache.items()
            if e.source_file_hash and e.source_file_hash != new_hash
        ]
        for key in stale_keys:
            del self._cache[key]
            count += 1
        return count

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._cache:
            return
        lru_key = min(
            self._cache,
            key=lambda k: self._cache[k].last_accessed,
        )
        del self._cache[lru_key]
        self._evictions += 1

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def total_tokens_cached(self) -> int:
        return sum(e.token_size for e in self._cache.values())

    def get_stats(self) -> dict[str, Any]:
        return {
            "cache_size": self.size,
            "max_entries": self._max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "evictions": self._evictions,
            "stale_evictions": self._stale_evictions,
            "total_tokens_cached": self.total_tokens_cached,
            "default_ttl": self._default_ttl,
        }


def compute_file_hash(file_path: str | Path) -> str:
    """Compute a SHA256 hash of a file for cache invalidation."""
    path = Path(file_path)
    if not path.exists():
        return ""
    try:
        content = path.read_bytes()
        return hashlib.sha256(content).hexdigest()
    except Exception:
        return ""
