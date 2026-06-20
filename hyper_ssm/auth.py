"""
API key authentication and rate limiting for ICM server.

Usage:
    store = ApiKeyStore("icm_keys.json")
    key = store.create_key("admin user")
    assert store.validate(key)

    limiter = RateLimiter(rpm=60)
    assert limiter.check(key)
"""

import json
import os
import secrets
import threading
import time
from typing import Dict, List, Optional


class ApiKeyStore:
    """Thread-safe API key store backed by a JSON file."""

    def __init__(self, path: Optional[str] = None):
        self._keys: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._path = path
        if path and os.path.exists(path):
            self._load()

    def create_key(self, description: str = "") -> str:
        key = "icm_" + secrets.token_hex(16)
        with self._lock:
            self._keys[key] = {
                "description": description,
                "created_at": time.time(),
                "is_active": True,
            }
            self._save()
        return key

    def validate(self, key: str) -> bool:
        with self._lock:
            entry = self._keys.get(key)
            return entry is not None and entry.get("is_active", False)

    def revoke(self, key: str) -> bool:
        with self._lock:
            if key not in self._keys:
                return False
            self._keys[key]["is_active"] = False
            self._save()
            return True

    def delete_key(self, key: str) -> bool:
        with self._lock:
            if key not in self._keys:
                return False
            del self._keys[key]
            self._save()
            return True

    def get_key_info(self, key: str) -> Optional[dict]:
        with self._lock:
            entry = self._keys.get(key)
            if entry is None:
                return None
            return {**entry, "key_prefix": key[:12] + "..."}

    def list_keys(self) -> List[dict]:
        with self._lock:
            return [
                {
                    "key_prefix": k[:12] + "...",
                    "description": v.get("description", ""),
                    "created_at": v.get("created_at", 0),
                    "is_active": v.get("is_active", True),
                }
                for k, v in self._keys.items()
            ]

    def count(self) -> int:
        with self._lock:
            return len(self._keys)

    def _load(self):
        with open(self._path, "r") as f:
            self._keys = json.load(f)

    def _save(self):
        if not self._path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self._path)) or ".", exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._keys, f, indent=2)


class RateLimiter:
    """Sliding-window rate limiter per key."""

    def __init__(self, requests_per_minute: int = 60):
        self._buckets: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self._rpm = requests_per_minute

    def check(self, key: str) -> bool:
        now = time.time()
        cutoff = now - 60.0
        with self._lock:
            bucket = self._buckets.setdefault(key, [])
            bucket[:] = [t for t in bucket if t > cutoff]
            if len(bucket) >= self._rpm:
                return False
            bucket.append(now)
            return True

    def remaining(self, key: str) -> int:
        now = time.time()
        cutoff = now - 60.0
        with self._lock:
            bucket = self._buckets.get(key, [])
            bucket[:] = [t for t in bucket if t > cutoff]
            return max(0, self._rpm - len(bucket))

    def reset(self):
        with self._lock:
            self._buckets.clear()
