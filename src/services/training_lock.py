import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol


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
