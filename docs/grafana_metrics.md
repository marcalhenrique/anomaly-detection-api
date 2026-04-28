# Grafana Dashboard Metrics

This document describes all metrics exposed by the **Anomaly Detection API** and consumed by the Grafana dashboard.

All metrics are collected via Prometheus and follow the [OpenMetrics](https://openmetrics.io/) format.

---

## Table of Contents

- [Application Metrics](#application-metrics)
  - [Prediction Metrics](#prediction-metrics)
  - [Training Metrics](#training-metrics)
  - [HTTP Metrics](#http-metrics)
  - [Business Metrics](#business-metrics)
- [System Metrics](#system-metrics)
  - [CPU & Memory](#cpu--memory)
  - [Process & Runtime](#process--runtime)

---

## Application Metrics

Custom metrics emitted by the application via `prometheus_client`.

### Prediction Metrics

| Metric Name                     | Type      | Description                                                                                   | Labels | Unit |
| ------------------------------- | --------- | --------------------------------------------------------------------------------------------- | ------ | ---- |
| `predict_latency_ms`            | Histogram | Total latency of a predict request (metadata lookup + model load + inference + serialization) | —      | ms   |
| `predict_metadata_latency_ms`   | Histogram | Time spent looking up model metadata in the cache or database                                 | —      | ms   |
| `predict_model_load_latency_ms` | Histogram | Time spent loading the model from MLflow/MinIO into memory                                    | —      | ms   |
| `predict_inference_latency_ms`  | Histogram | Pure inference time (model prediction only)                                                   | —      | ms   |
| `predict_operation_latency_ms`  | Histogram | Combined operation latency (metadata + model load + inference)                                | —      | ms   |

**Derived Queries:**

- **Average Predict Latency**

  ```promql
  rate(predict_latency_ms_sum[5m]) / rate(predict_latency_ms_count[5m])
  ```

- **Predict Latency Percentiles**

  ```promql
  histogram_quantile(0.50, rate(predict_latency_ms_bucket[1m]))  # p50
  histogram_quantile(0.95, rate(predict_latency_ms_bucket[1m]))  # p95
  histogram_quantile(0.99, rate(predict_latency_ms_bucket[1m]))  # p99
  ```

- **Predict Throughput**
  ```promql
  rate(predict_latency_ms_count[1m])
  ```

### Training Metrics

| Metric Name        | Type      | Description                                                                     | Labels | Unit |
| ------------------ | --------- | ------------------------------------------------------------------------------- | ------ | ---- |
| `train_latency_ms` | Histogram | Total latency of a training request (data fetch + model fit + artifact logging) | —      | ms   |

**Derived Queries:**

- **Average Training Latency**

  ```promql
  rate(train_latency_ms_sum[5m]) / rate(train_latency_ms_count[5m])
  ```

- **Training Latency Percentiles**

  ```promql
  histogram_quantile(0.50, rate(train_latency_ms_bucket[1m]))  # p50
  histogram_quantile(0.95, rate(train_latency_ms_bucket[1m]))  # p95
  ```

- **Training Throughput**
  ```promql
  rate(train_latency_ms_count[1m])
  ```

### HTTP Metrics

| Metric Name                | Type      | Description                                                             | Labels                          | Unit |
| -------------------------- | --------- | ----------------------------------------------------------------------- | ------------------------------- | ---- |
| `http_request_duration_ms` | Histogram | End-to-end HTTP request latency (from request receive to response send) | `method`, `path`, `status_code` | ms   |

**Derived Queries:**

- **HTTP Latency by Endpoint (p50 / p95)**
  ```promql
  histogram_quantile(0.50, rate(http_request_duration_ms_bucket[1m]))
  histogram_quantile(0.95, rate(http_request_duration_ms_bucket[1m]))
  ```

> The `path` label is normalized to the route pattern (e.g., `/predict/{series_id}`) to keep cardinality low.

### Business Metrics

| Metric Name            | Type  | Description                                                         | Labels | Unit  |
| ---------------------- | ----- | ------------------------------------------------------------------- | ------ | ----- |
| `series_trained_total` | Gauge | Number of distinct time series that have at least one trained model | —      | count |

**Query:**

```promql
series_trained_total
```

---

## System Metrics

Default process and runtime metrics exposed automatically by `prometheus_client`.

### CPU & Memory

| Metric Name                     | Type    | Description                                                   | Unit    |
| ------------------------------- | ------- | ------------------------------------------------------------- | ------- |
| `process_cpu_seconds_total`     | Counter | Total user and system CPU time spent by the process           | seconds |
| `process_resident_memory_bytes` | Gauge   | Resident set size (RSS) — physical memory used by the process | bytes   |
| `process_virtual_memory_bytes`  | Gauge   | Virtual memory size of the process                            | bytes   |

**Derived Queries:**

- **CPU Usage %**

  ```promql
  rate(process_cpu_seconds_total[1m]) * 100
  ```

- **Memory Usage (MB)**
  ```promql
  process_resident_memory_bytes / 1024 / 1024
  process_virtual_memory_bytes / 1024 / 1024
  ```

### Process & Runtime

| Metric Name                         | Type    | Description                                               | Unit    |
| ----------------------------------- | ------- | --------------------------------------------------------- | ------- |
| `process_open_fds`                  | Gauge   | Number of open file descriptors                           | count   |
| `process_max_fds`                   | Gauge   | Maximum number of open file descriptors                   | count   |
| `process_start_time_seconds`        | Gauge   | Unix timestamp when the process started                   | seconds |
| `python_gc_objects_collected_total` | Counter | Objects collected during garbage collection by generation | count   |
| `python_gc_collections_total`       | Counter | Number of times each GC generation was collected          | count   |
| `python_info`                       | Gauge   | Python platform information (version, implementation)     | —       |

---

## Dashboard Layout

The Grafana dashboard is organized into the following sections:

1. **Overview** — Key performance indicators (KPIs) displayed as stat panels with color thresholds and area sparklines.
2. **System Resources** — CPU and memory timeseries with smooth lines and fill opacity.
3. **Predict Performance** — Latency percentiles (p50 / p95 / p99) for total predict and inference times.
4. **Predict Breakdown** — Granular latency views: metadata lookup, model load, and operation latency.
5. **Training Performance** — Training latency percentiles and training throughput.
6. **Throughput** — Predict and HTTP request throughput with request rates.

---

## Prometheus Scrape Configuration

```yaml
scrape_configs:
  - job_name: anomaly-api
    static_configs:
      - targets:
          - api:8000
    metrics_path: /metrics/
    scrape_interval: 15s
```

---

## Notes

- All histograms use predefined buckets tailored to the expected latency ranges of each operation.
- The `rate()` function requires at least two scrape intervals of data; therefore, panels using `rate()` may show "No Data" immediately after a Prometheus restart until enough samples are collected.
- The `http_request_duration_ms` metric excludes paths such as `/metrics`, `/healthcheck`, `/docs`, and `/openapi.json` to avoid skewing results with health-check traffic.
