"""
Intelligent Caching Layer

LRU cache with TTL support for expensive intelligence operations.
Dramatically improves performance for repeated queries.

Features:
- LRU eviction policy
- TTL (time-to-live) support
- Statistics tracking
- Automatic cleanup
- Thread-safe operations

Author: AI System
Version: 3.0
"""

from typing import Any, Optional, Dict, Callable
from datetime import datetime, timedelta
from collections import OrderedDict
import threading
from .base_types import CacheEntry, hash_content


class IntelligentCache:
    """
    LRU Cache with TTL and statistics

    Thread-safe caching layer for expensive intelligence operations.
    Uses LRU eviction and optional TTL for entries.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl_seconds: Optional[float] = 300,  # 5 minutes default
        verbose: bool = False
    ):
        """
        Initialize cache

        Args:
            max_size: Maximum number of entries
            default_ttl_seconds: Default TTL for entries (None = no expiration)
            verbose: Enable verbose logging
        """
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds
        self.verbose = verbose

        # Use OrderedDict for LRU behavior
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value if found and not expired, None otherwise
        """
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None

            entry = self._cache[key]

            # Check if expired
            if entry.is_expired():
                del self._cache[key]
                self.expirations += 1
                self.misses += 1
                if self.verbose:
                    print(f"[CACHE] Key expired: {key[:30]}...")
                return None

            # Touch entry and move to end (most recently used)
            entry.touch()
            self._cache.move_to_end(key)

            self.hits += 1
            if self.verbose:
                print(f"[CACHE] Hit: {key[:30]}... (accessed {entry.access_count} times)")

            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None
    ):
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL for this entry (uses default if None)
        """
        with self._lock:
            now = datetime.now()

            # Use provided TTL or default
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                access_count=0,
                ttl_seconds=ttl
            )

            # If key exists, remove it (will be re-added at end)
            if key in self._cache:
                del self._cache[key]

            # Add to end (most recently used)
            self._cache[key] = entry

            # Evict oldest if over capacity
            if len(self._cache) > self.max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                self.evictions += 1
                if self.verbose:
                    print(f"[CACHE] Evicted: {oldest_key[:30]}...")

            if self.verbose:
                print(f"[CACHE] Set: {key[:30]}... (TTL: {ttl}s)")

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        ttl_seconds: Optional[float] = None
    ) -> Any:
        """
        Get from cache or compute and cache

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            ttl_seconds: TTL for computed value

        Returns:
            Cached or computed value
        """
        # Try to get from cache
        value = self.get(key)
        if value is not None:
            return value

        # Compute value
        if self.verbose:
            print(f"[CACHE] Computing: {key[:30]}...")

        value = compute_fn()

        # Cache computed value
        self.set(key, value, ttl_seconds)

        return value

    def invalidate(self, key: str) -> bool:
        """
        Invalidate (remove) cache entry

        Args:
            key: Cache key

        Returns:
            True if entry was removed, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if self.verbose:
                    print(f"[CACHE] Invalidated: {key[:30]}...")
                return True
            return False

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern

        Args:
            pattern: Pattern to match (substring)

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            keys_to_remove = [
                key for key in self._cache.keys()
                if pattern in key
            ]

            for key in keys_to_remove:
                del self._cache[key]

            if self.verbose and keys_to_remove:
                print(f"[CACHE] Invalidated {len(keys_to_remove)} entries matching: {pattern}")

            return len(keys_to_remove)

    def clear(self):
        """Clear entire cache"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            if self.verbose:
                print(f"[CACHE] Cleared {count} entries")

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                del self._cache[key]

            self.expirations += len(expired_keys)

            if self.verbose and expired_keys:
                print(f"[CACHE] Cleaned up {len(expired_keys)} expired entries")

            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions,
                'expirations': self.expirations,
                'total_requests': total_requests,
            }

    def reset_stats(self):
        """Reset statistics counters"""
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.expirations = 0
            if self.verbose:
                print("[CACHE] Statistics reset")

    def __len__(self) -> int:
        """Get number of entries in cache"""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache"""
        with self._lock:
            if key not in self._cache:
                return False
            entry = self._cache[key]
            return not entry.is_expired()


class CacheKeyBuilder:
    """Helper to build cache keys consistently"""

    @staticmethod
    def for_intent_classification(message: str) -> str:
        """Build cache key for intent classification"""
        return f"intent:{hash_content(message)}"

    @staticmethod
    def for_entity_extraction(message: str) -> str:
        """Build cache key for entity extraction"""
        return f"entity:{hash_content(message)}"

    @staticmethod
    def for_task_decomposition(message: str, intent_types: str) -> str:
        """Build cache key for task decomposition"""
        return f"task:{hash_content(message)}:{hash_content(intent_types)}"

    @staticmethod
    def for_confidence_score(message: str, intents: str, entities: str) -> str:
        """Build cache key for confidence scoring"""
        components = f"{message}|{intents}|{entities}"
        return f"confidence:{hash_content(components)}"

    @staticmethod
    def for_llm_call(prompt: str, model: str) -> str:
        """Build cache key for LLM calls"""
        return f"llm:{model}:{hash_content(prompt)}"

    @staticmethod
    def for_semantic_similarity(text: str) -> str:
        """Build cache key for semantic embeddings"""
        return f"embedding:{hash_content(text)}"


# Global cache instance (can be configured)
_global_cache: Optional[IntelligentCache] = None


def get_global_cache() -> IntelligentCache:
    """Get or create global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = IntelligentCache(
            max_size=1000,
            default_ttl_seconds=300,  # 5 minutes
            verbose=False
        )
    return _global_cache


def configure_global_cache(
    max_size: int = 1000,
    default_ttl_seconds: Optional[float] = 300,
    verbose: bool = False
):
    """Configure global cache instance"""
    global _global_cache
    _global_cache = IntelligentCache(
        max_size=max_size,
        default_ttl_seconds=default_ttl_seconds,
        verbose=verbose
    )
