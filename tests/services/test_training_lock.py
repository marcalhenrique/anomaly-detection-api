import asyncio

import pytest

from src.services.training_lock import LocalTrainingLock


@pytest.mark.asyncio
async def test_same_series_id_executes_sequentially() -> None:
    """Two concurrent acquires for the same series_id must not overlap."""
    lock = LocalTrainingLock()
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
async def test_different_series_ids_run_concurrently() -> None:
    """Acquires for different series_ids must not block each other."""
    lock = LocalTrainingLock()
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
async def test_lock_released_after_exception() -> None:
    """Lock must be released even if the block raises an exception."""
    lock = LocalTrainingLock()

    with pytest.raises(ValueError):
        async with lock.acquire("sensor-1"):
            raise ValueError("training failed")

    # If the lock was properly released, this acquires without hanging
    acquired = False
    async with lock.acquire("sensor-1"):
        acquired = True

    assert acquired


@pytest.mark.asyncio
async def test_independent_locks_per_series_id() -> None:
    """Each series_id gets its own independent lock."""
    lock = LocalTrainingLock()

    async with lock.acquire("sensor-1"):
        async with lock.acquire("sensor-2"):
            pass  # must not deadlock

    assert "sensor-1" in lock._locks
    assert "sensor-2" in lock._locks
