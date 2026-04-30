import asyncio

import fakeredis.aioredis
import pytest

from src.services.training_lock import LocalTrainingLock, RedisTrainingLock


@pytest.fixture
def fake_redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def lock(fake_redis):
    return RedisTrainingLock(redis_client=fake_redis)


@pytest.fixture
def local_lock():
    return LocalTrainingLock()


@pytest.mark.asyncio
async def test_same_series_id_executes_sequentially(lock) -> None:
    """Two concurrent acquires for the same series_id must not overlap."""
    execution_order: list[str] = []

    async def task(label: str) -> None:
        async with lock.acquire("sensor-1"):
            execution_order.append(f"{label}_start")
            await asyncio.sleep(0.05)
            execution_order.append(f"{label}_end")

    await asyncio.gather(task("A"), task("B"))

    # A must fully complete before B starts (or vice versa — never interleaved)
    assert execution_order.index("A_end") < execution_order.index(
        "B_start"
    ) or execution_order.index("B_end") < execution_order.index("A_start")


@pytest.mark.asyncio
async def test_different_series_ids_run_concurrently(lock) -> None:
    """Acquires for different series_ids must not block each other."""
    execution_order: list[str] = []

    async def task(series_id: str, label: str) -> None:
        async with lock.acquire(series_id):
            execution_order.append(f"{label}_start")
            await asyncio.sleep(0.05)
            execution_order.append(f"{label}_end")

    await asyncio.gather(task("sensor-1", "A"), task("sensor-2", "B"))

    # Both must have started before either finishes (true concurrency)
    assert execution_order.index("A_start") < execution_order.index("B_end")
    assert execution_order.index("B_start") < execution_order.index("A_end")


@pytest.mark.asyncio
async def test_lock_released_after_exception(lock) -> None:
    """Lock must be released even if the block raises an exception."""
    with pytest.raises(ValueError):
        async with lock.acquire("sensor-1"):
            raise ValueError("training failed")

    # If the lock was properly released, this acquires without hanging
    acquired = False
    async with lock.acquire("sensor-1"):
        acquired = True

    assert acquired


@pytest.mark.asyncio
async def test_independent_locks_per_series_id(lock) -> None:
    """Each series_id gets its own independent lock."""
    async with lock.acquire("sensor-1"):
        async with lock.acquire("sensor-2"):
            pass  # must not deadlock


# ---------------------------------------------------------------------------
# LocalTrainingLock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_lock_same_series_id_executes_sequentially(local_lock) -> None:
    """Two concurrent acquires for the same series_id must not overlap."""
    execution_order: list[str] = []

    async def task(label: str) -> None:
        async with local_lock.acquire("sensor-1"):
            execution_order.append(f"{label}_start")
            await asyncio.sleep(0.05)
            execution_order.append(f"{label}_end")

    await asyncio.gather(task("A"), task("B"))

    assert execution_order.index("A_end") < execution_order.index(
        "B_start"
    ) or execution_order.index("B_end") < execution_order.index("A_start")


@pytest.mark.asyncio
async def test_local_lock_different_series_ids_run_concurrently(local_lock) -> None:
    """Acquires for different series_ids must not block each other."""
    execution_order: list[str] = []

    async def task(series_id: str, label: str) -> None:
        async with local_lock.acquire(series_id):
            execution_order.append(f"{label}_start")
            await asyncio.sleep(0.05)
            execution_order.append(f"{label}_end")

    await asyncio.gather(task("sensor-1", "A"), task("sensor-2", "B"))

    assert execution_order.index("A_start") < execution_order.index("B_end")
    assert execution_order.index("B_start") < execution_order.index("A_end")


@pytest.mark.asyncio
async def test_local_lock_released_after_exception(local_lock) -> None:
    """LocalTrainingLock must release the lock even if the block raises."""
    with pytest.raises(ValueError):
        async with local_lock.acquire("sensor-1"):
            raise ValueError("oops")

    acquired = False
    async with local_lock.acquire("sensor-1"):
        acquired = True

    assert acquired


# ---------------------------------------------------------------------------
# RedisTrainingLock — timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_redis_lock_raises_timeout_when_blocked_too_long(fake_redis) -> None:
    """RedisTrainingLock must raise TimeoutError when the lock cannot be acquired."""
    lock = RedisTrainingLock(redis_client=fake_redis)
    lock._blocking_timeout = 0.1  # very short timeout for the test

    async with lock.acquire("sensor-timeout"):
        with pytest.raises(TimeoutError):
            async with lock.acquire("sensor-timeout"):
                pass  # pragma: no cover
