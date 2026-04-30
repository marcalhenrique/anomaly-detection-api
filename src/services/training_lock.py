import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

import redis.asyncio as aioredis

from src.core.config import get_settings

_settings = get_settings()

class TrainingLockProtocol(Protocol):
    @asynccontextmanager
    async def acquire(self, series_id: str) -> AsyncIterator[None]: ...

class LocalTrainingLock:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    @asynccontextmanager
    async def acquire(self, series_id: str) -> AsyncIterator[None]:
        if series_id not in self._locks:
            self._locks[series_id] = asyncio.Lock()

        async with self._locks[series_id]:
            yield

class RedisTrainingLock:
    def __init__(self, redis_client: aioredis.Redis | None = None) -> None:
        self._redis = redis_client or aioredis.Redis(
            host=_settings.redis_host,
            port=_settings.redis_port,
            db=_settings.redis_db,
            decode_responses=True,
        )
        self._timeout = 60
        self._blocking_timeout = 120

    def _lock_key(self, series_id: str) -> str:
        return f"training_lock:{series_id}"

    @asynccontextmanager
    async def acquire(self, series_id: str) -> AsyncIterator[None]:
        key = self._lock_key(series_id)
        token = uuid.uuid4().hex
        acquired = False
        deadline = asyncio.get_event_loop().time() + self._blocking_timeout

        while not acquired:
            acquired = await self._redis.set(key, token, nx=True, ex=self._timeout)
            if not acquired:
                if asyncio.get_event_loop().time() >= deadline:
                    raise TimeoutError(
                        f"Could not acquire training lock for {series_id}"
                    )
                await asyncio.sleep(0.05)

        try:
            yield
        finally:
            current = await self._redis.get(key)
            if current == token:
                await self._redis.delete(key)

