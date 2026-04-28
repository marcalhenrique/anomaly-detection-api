from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
import time

import numpy as np
from prometheus_client import Histogram, Gauge

PREDICT_TOTAL = Histogram(
    "predict_latency_ms",
    "Latência total do predict (ms)",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
)
PREDICT_METADATA = Histogram(
    "predict_metadata_latency_ms",
    "Tempo de lookup de metadata (ms)",
    buckets=[0.1, 0.5, 1, 5, 10, 25, 50],
)
PREDICT_MODEL_LOAD = Histogram(
    "predict_model_load_latency_ms",
    "Tempo de load do modelo (ms)",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500],
)
PREDICT_INFERENCE = Histogram(
    "predict_inference_latency_ms",
    "Tempo de inferência pura (ms)",
    buckets=[0.1, 0.5, 1, 5, 10, 25, 50],
)
PREDICT_OPERATION = Histogram(
    "predict_operation_latency_ms",
    "Latência da operação de predict — metadata + model load + inference (ms)",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
)
TRAIN_TOTAL = Histogram(
    "train_latency_ms",
    "Latência total do treinamento (ms)",
    buckets=[50, 100, 500, 1000, 5000, 10000, 30000],
)
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_ms",
    "Latência HTTP de ponta a ponta — do receive ao send (ms)",
    labelnames=["method", "path", "status_code"],
    buckets=[5, 10, 25, 50, 100, 150, 200, 300, 500, 1000, 2500],
)
SERIES_TRAINED = Gauge(
    "series_trained_total",
    "Number of distinct series with at least one trained model",
)


@contextmanager
def timed(label: str, timings: dict[str, float]) -> Generator[None, None, None]:
    t = time.perf_counter()
    yield
    timings[label] = round((time.perf_counter() - t) * 1000, 3)


class MetricsCollector:
    def __init__(self, latency_window: int = 1000) -> None:
        self._predict_latencies: deque[float] = deque(maxlen=latency_window)
        self._train_latencies: deque[float] = deque(maxlen=latency_window)

    def record_predict(self, timings: dict[str, float]) -> None:
        PREDICT_METADATA.observe(timings["metadata_ms"])
        PREDICT_MODEL_LOAD.observe(timings["model_load_ms"])
        PREDICT_INFERENCE.observe(timings["inference_ms"])
        PREDICT_OPERATION.observe(timings["predict_operation_ms"])
        PREDICT_TOTAL.observe(timings["total_ms"])
        self._predict_latencies.append(timings["total_ms"])

    def record_training(self, ms: float) -> None:
        TRAIN_TOTAL.observe(ms)
        self._train_latencies.append(ms)

    def set_series_trained(self, count: int) -> None:
        SERIES_TRAINED.set(count)

    def inc_series_trained(self) -> None:
        SERIES_TRAINED.inc()

    def get_inference_stats(self) -> dict[str, float]:
        if not self._predict_latencies:
            return {"avg": 0.0, "p95": 0.0}
        arr = np.array(self._predict_latencies)
        return {"avg": float(np.mean(arr)), "p95": float(np.percentile(arr, 95))}

    def get_training_stats(self) -> dict[str, float]:
        if not self._train_latencies:
            return {"avg": 0.0, "p95": 0.0}
        arr = np.array(self._train_latencies)
        return {"avg": float(np.mean(arr)), "p95": float(np.percentile(arr, 95))}

    def get_series_trained(self) -> int:
        return int(SERIES_TRAINED._value.get() or 0)
