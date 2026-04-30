# Tests

Test coverage for the training flow (`POST /fit/{series_id}`), prediction flow (`POST /predict/{series_id}`), healthcheck, and end-to-end integration.

## Structure

```
tests/
├── api/
│   ├── test_dependencies.py
│   ├── test_fit_route.py
│   ├── test_healthcheck_route.py
│   └── test_predict_route.py
├── core/
│   ├── test_database.py
│   └── test_structlog_config.py
├── e2e/
│   ├── test_fit.py
│   └── test_predict.py
├── repositories/
│   └── test_model_metadata_repository.py
├── schemas/
│   └── test_train_request.py
└── services/
    ├── test_anomaly_detection.py
    ├── test_metadata_cache.py
    ├── test_metrics_collector.py
    ├── test_mlflow_service.py
    ├── test_prediction_service.py
    ├── test_training_lock.py
    └── test_training_service.py
```

## How to run

Tests run inside Docker — no local `uv` installation required.

```bash
# Unit + integration tests only (fast, no external services)
make test-unit

# End-to-end tests (spins up full Docker stack: Postgres, MinIO, MLflow, Redis, API)
make test-e2e

# Both
make test

# With coverage report (requires uv locally)
uv run pytest tests/ --ignore=tests/e2e --cov=src --cov-report=term-missing
```

---

## `tests/core/test_database.py`

Validates the `get_db` async generator that provides SQLAlchemy sessions to route handlers. The `AsyncSessionFactory` is replaced by a mock — no real database required.

| Test                                     | Description                                                                                          |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `test_get_db_yields_session_and_commits` | On successful exit, `get_db` must yield the session and call `session.commit()`                      |
| `test_get_db_rolls_back_on_exception`    | When an exception is thrown inside the context, `get_db` must call `session.rollback()` and re-raise |

---

## `tests/core/test_structlog_config.py`

Validates `configure_logging()`, `get_logger()`, and `_ensure_log_dir()`. Uses `tmp_path` for file-based assertions; root logger state is restored after each test.

| Test                                                      | Description                                                           |
| --------------------------------------------------------- | --------------------------------------------------------------------- |
| `test_configure_logging_adds_exactly_two_handlers`        | Must install exactly two handlers (console + file) on the root logger |
| `test_configure_logging_installs_file_handler`            | Must include a `TimedRotatingFileHandler`                             |
| `test_configure_logging_installs_console_handler`         | Must include a `StreamHandler` for console output                     |
| `test_configure_logging_creates_nested_log_directory`     | Must create all missing parent directories for the log file path      |
| `test_configure_logging_accepts_string_log_levels`        | Must accept string levels such as `'INFO'` and `'DEBUG'`              |
| `test_configure_logging_json_console_mode_does_not_raise` | `json_console=True` must succeed without errors                       |
| `test_get_logger_returns_a_logger`                        | `get_logger()` must return a non-None logger                          |
| `test_get_logger_uses_provided_name`                      | Different name arguments must each produce a usable logger            |
| `test_ensure_log_dir_creates_parent_directories`          | Must create all nested parent directories for the given path          |
| `test_ensure_log_dir_is_idempotent`                       | Calling it twice must not raise even if the directory already exists  |

---

## `tests/api/test_dependencies.py`

Validates the FastAPI dependency factories in `src.api.dependencies`. No I/O is involved — the singleton getters are checked by identity and the factory functions are called with mock arguments.

| Test                                                              | Description                                                                          |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `test_get_metadata_cache_returns_module_singleton`                | `get_metadata_cache()` must return the module-level `_metadata_cache` instance       |
| `test_get_mlflow_service_returns_module_singleton`                | `get_mlflow_service()` must return the module-level `_mlflow_service` instance       |
| `test_get_metrics_collector_returns_module_singleton`             | `get_metrics_collector()` must return the module-level `_metrics_collector` instance |
| `test_get_training_service_returns_training_service_instance`     | `get_training_service()` must construct and return a `TrainingService`               |
| `test_get_prediction_service_returns_prediction_service_instance` | `get_prediction_service()` must construct and return a `PredictionService`           |

---

## `tests/schemas/test_train_request.py`

Validates the business rules of the `TrainRequest` Pydantic schema before any service logic is involved.

| Test                                                     | Description                                                              |
| -------------------------------------------------------- | ------------------------------------------------------------------------ |
| `test_valid_request_passes_validation`                   | A payload with exactly 10 valid points must be accepted without errors   |
| `test_valid_request_with_more_than_minimum_points`       | A payload with more than 10 points must also be accepted                 |
| `test_raises_when_fewer_than_10_timestamps`              | Fewer than 10 timestamps must fail field-level validation (`min_length`) |
| `test_raises_when_fewer_than_10_values`                  | Fewer than 10 values must fail field-level validation (`min_length`)     |
| `test_raises_when_timestamps_and_values_length_mismatch` | Lists of different lengths must be rejected by the `model_validator`     |
| `test_raises_when_timestamps_not_unique`                 | Duplicate timestamps must be rejected                                    |
| `test_raises_when_timestamps_not_sorted`                 | Out-of-order timestamps must be rejected                                 |
| `test_raises_when_timestamp_is_negative`                 | Negative timestamps must be rejected                                     |
| `test_raises_when_values_are_constant`                   | Constant values (zero standard deviation) must be rejected               |

---

## `tests/services/test_anomaly_detection.py`

Validates the Z-score anomaly detection algorithm (`AnomalyDetectionModel`).

| Test                                          | Description                                                                      |
| --------------------------------------------- | -------------------------------------------------------------------------------- |
| `test_fit_computes_correct_mean`              | After `fit()`, `model.mean` must equal the numpy mean of the provided values     |
| `test_fit_computes_correct_std`               | After `fit()`, `model.std` must equal the numpy standard deviation of the values |
| `test_fit_returns_self`                       | `fit()` must return the model instance itself to allow fluent chaining           |
| `test_predict_returns_false_for_normal_value` | A value near the mean must not be flagged as an anomaly                          |
| `test_predict_returns_true_for_outlier`       | A value clearly above `mean + 3σ` must be flagged as an anomaly                  |

---

## `tests/services/test_training_service.py`

Validates the orchestration logic of `TrainingService.fit()`. All external collaborators (lock, MLflow, repository) are mocked.

| Test                                                                | Description                                                                                                                          |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `test_fit_returns_correct_train_response`                           | The return value of `fit()` must contain the correct `series_id`, `version`, and `points_used`                                       |
| `test_fit_calls_mlflow_save_model_with_correct_args`                | `MLflowService.save_model` must be called with the correct `series_id` and point count                                               |
| `test_fit_calls_repo_save_with_correct_args`                        | `ModelMetadataRepository.save` must be called with all correct fields, including the `run_id` and `model_version` returned by MLflow |
| `test_fit_acquires_lock_for_series_id`                              | The training lock must be acquired using the correct `series_id`                                                                     |
| `test_fit_raises_when_mlflow_save_fails`                            | If MLflow raises an exception, `fit()` must propagate it                                                                             |
| `test_fit_raises_when_repo_save_fails`                              | If the repository raises an exception, `fit()` must propagate it                                                                     |
| `test_fit_does_not_call_repo_when_mlflow_fails`                     | If MLflow fails, the repository must not be called                                                                                   |
| `test_fit_returns_existing_version_when_data_is_identical`          | Idempotency: identical data must return the existing version without re-training                                                     |
| `test_fit_skips_mlflow_and_repo_when_data_is_identical`             | Idempotency: no MLflow or DB write must occur when data is identical                                                                 |
| `test_fit_trains_when_values_differ`                                | Different values must trigger a new training + version increment                                                                     |
| `test_fit_trains_when_no_existing_model`                            | A brand-new series must train from scratch and return version `1`                                                                    |
| `test_fit_returns_successfully_when_post_training_cache_warm_fails` | If `load_model` raises after training, `fit()` must log a warning and still return a valid response                                  |
| `test_fit_does_not_propagate_load_model_exception_after_save`       | The post-training cache-warming exception must not bubble out of `fit()`                                                             |

---

## `tests/services/test_training_lock.py`

Validates the concurrency behaviour of `RedisTrainingLock` and `LocalTrainingLock`. `RedisTrainingLock` tests use an in-memory `fakeredis` instance.

| Test                                                    | Description                                                                                        |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `test_same_series_id_executes_sequentially`             | Two concurrent acquires for the same `series_id` must not overlap (mutex) — `RedisTrainingLock`    |
| `test_different_series_ids_run_concurrently`            | Acquires for different `series_id` values must not block each other — `RedisTrainingLock`          |
| `test_lock_released_after_exception`                    | The lock must be released even if the protected block raises — `RedisTrainingLock`                 |
| `test_independent_locks_per_series_id`                  | Each `series_id` has its own independent lock — acquiring two different ones must not deadlock     |
| `test_local_lock_same_series_id_executes_sequentially`  | Same mutex guarantee for `LocalTrainingLock`                                                       |
| `test_local_lock_different_series_ids_run_concurrently` | Different series must run concurrently under `LocalTrainingLock`                                   |
| `test_local_lock_released_after_exception`              | `LocalTrainingLock` must release the lock after an exception in the protected block                |
| `test_redis_lock_raises_timeout_when_blocked_too_long`  | `RedisTrainingLock` must raise `TimeoutError` when the lock cannot be acquired within the deadline |

---

## `tests/api/test_fit_route.py`

Tests the `POST /fit/{series_id}` HTTP route via FastAPI's `TestClient`. The `TrainingService` is replaced by a mock through `dependency_overrides`, with no real I/O. The real application `lifespan` (DB cache-warming) is bypassed via patch.

| Test                                                              | Description                                                                          |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `test_fit_returns_200_with_valid_body`                            | A valid request must return HTTP 200                                                 |
| `test_fit_response_body_matches_schema`                           | The response body must contain the correct `series_id`, `version`, and `points_used` |
| `test_fit_returns_422_when_body_missing_fields`                   | An empty body must return HTTP 422 (Unprocessable Entity)                            |
| `test_fit_returns_422_when_fewer_than_10_timestamps`              | A payload with fewer than 10 points must return HTTP 422                             |
| `test_fit_returns_422_when_timestamps_and_values_length_mismatch` | Lists of different lengths must return HTTP 422                                      |
| `test_fit_returns_422_when_timestamps_not_sorted`                 | Out-of-order timestamps must return HTTP 422                                         |
| `test_fit_returns_422_when_timestamps_not_unique`                 | Duplicate timestamps must return HTTP 422                                            |
| `test_fit_returns_500_when_service_raises`                        | An exception in the service must be caught by the route and return HTTP 500          |
| `test_fit_500_response_contains_error_detail`                     | The `detail` field of the 500 response must contain the original exception message   |

---

## `tests/api/test_healthcheck_route.py`

Tests the `GET /healthcheck` HTTP route in isolation. External dependencies (database, MLflow) are fully mocked.

| Test                                            | Description                                                                     |
| ----------------------------------------------- | ------------------------------------------------------------------------------- |
| `test_healthcheck_returns_200_when_healthy`     | All dependencies up → HTTP 200 with metrics and latency statistics              |
| `test_healthcheck_returns_populated_metrics`    | Metrics collector with latencies → correct `avg` and `p95` values in response   |
| `test_healthcheck_returns_503_when_db_down`     | Database unreachable → HTTP 503 with partial metrics (series count + latencies) |
| `test_healthcheck_returns_503_when_mlflow_down` | MLflow unreachable → HTTP 503 with partial metrics                              |

---

## `tests/services/test_prediction_service.py`

Validates the orchestration logic of `PredictionService.predict()`. All external collaborators (MLflow, repository) are mocked.

| Test                                                              | Description                                                                                           |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `test_predict_returns_correct_response_for_normal_value`          | A normal value must return `anomaly=False` and the correct `model_version`                            |
| `test_predict_returns_correct_response_for_anomalous_value`       | An anomalous value must return `anomaly=True`                                                         |
| `test_predict_uses_latest_model_when_version_not_provided`        | Without a `version` param, `get_latest_by_series_id` must be called and `get_by_version` must not     |
| `test_predict_uses_specific_version_when_provided`                | With a `version` param, `get_by_version` must be called and `get_latest_by_series_id` must not        |
| `test_predict_loads_model_using_run_id_from_metadata`             | `MLflowService.load_model` must be called with the `run_id` stored in the metadata record             |
| `test_predict_calls_model_predict_with_correct_data_point`        | `model.predict()` must receive a `DataPoint` with the correct `timestamp` and `value`                 |
| `test_predict_raises_model_not_found_when_series_does_not_exist`  | When no metadata is found for the series, `ModelNotFoundError` must be raised                         |
| `test_predict_raises_model_not_found_when_version_does_not_exist` | When the requested version is not found, `ModelNotFoundError` must be raised                          |
| `test_predict_populates_cache_after_db_fallback`                  | When cache misses but DB has the model, `predict()` must call `cache.set()` with the fetched metadata |
| `test_predict_returns_correct_response_after_db_fallback`         | When metadata is fetched from DB, `predict()` must return the correct version and anomaly result      |

---

## `tests/api/test_predict_route.py`

Tests the `POST /predict/{series_id}` HTTP route via FastAPI's `TestClient`. The `PredictionService` is replaced by a mock through `dependency_overrides`, with no real I/O. The real application `lifespan` is bypassed via patch.

| Test                                                     | Description                                                                        |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `test_predict_returns_200_with_valid_body`               | A valid request must return HTTP 200                                               |
| `test_predict_response_body_matches_schema`              | The response body must contain the correct `anomaly` and `model_version` fields    |
| `test_predict_returns_anomaly_true_when_service_says_so` | When the service returns `anomaly=True`, the route must reflect that               |
| `test_predict_passes_version_query_param_to_service`     | A `?version=` query param must be accepted without error                           |
| `test_predict_returns_422_when_body_is_empty`            | An empty body must return HTTP 422                                                 |
| `test_predict_returns_422_when_value_is_missing`         | A body with only `timestamp` must return HTTP 422                                  |
| `test_predict_returns_422_when_timestamp_is_missing`     | A body with only `value` must return HTTP 422                                      |
| `test_predict_returns_200_when_timestamp_is_string`      | A string `timestamp` must be accepted (coerced by Pydantic)                        |
| `test_predict_returns_404_when_series_does_not_exist`    | A `ModelNotFoundError` from the service must be caught and returned as HTTP 404    |
| `test_predict_404_response_contains_error_detail`        | The `detail` field of the 404 response must contain the `series_id`                |
| `test_predict_returns_500_when_service_raises`           | An unexpected exception in the service must be caught and returned as HTTP 500     |
| `test_predict_500_response_contains_error_detail`        | The `detail` field of the 500 response must contain the original exception message |

---

## `tests/repositories/test_model_metadata_repository.py`

Validates the behaviour of `ModelMetadataRepository`. The SQLAlchemy `AsyncSession` is mocked throughout — no real database is required.

| Test                                                       | Description                                                                                          |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `test_save_adds_record_to_session`                         | `save()` must call `session.add()` exactly once with the created record                              |
| `test_save_calls_commit`                                   | `save()` must call `session.commit()` to persist the transaction                                     |
| `test_save_returns_model_metadata_with_correct_fields`     | The returned object must have the correct `series_id`, `mlflow_run_id`, `version`, and `points_used` |
| `test_save_preserves_add_then_commit_order`                | `add()` must be called before `commit()` — the reverse order would lose data                         |
| `test_get_latest_by_series_id_returns_record_when_found`   | Returns the record when one exists for the given `series_id`                                         |
| `test_get_latest_by_series_id_returns_none_when_not_found` | Returns `None` when no record exists                                                                 |
| `test_get_by_version_returns_record_when_found`            | Returns the record matching the given `series_id` and `version`                                      |
| `test_get_by_version_returns_none_when_not_found`          | Returns `None` when the requested version does not exist                                             |
| `test_get_all_latest_returns_list_of_records`              | Returns a list with the latest record per series                                                     |
| `test_get_all_latest_returns_empty_list_when_no_records`   | Returns an empty list when no records exist                                                          |
| `test_count_distinct_series_returns_correct_count`         | Returns the count of distinct series from the database                                               |
| `test_count_distinct_series_returns_zero_when_empty`       | Returns `0` when the table is empty                                                                  |

---

## `tests/services/test_metrics_collector.py`

Validates the in-process metrics collection logic of `MetricsCollector`. No Prometheus server is required — metrics are recorded to an in-memory store.

| Test                                                    | Description                                                                                                |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `test_record_predict_appends_total_ms_to_latencies`     | `record_predict()` must store `total_ms` in the rolling predict latency buffer                             |
| `test_record_predict_multiple_calls_accumulate`         | Multiple `record_predict()` calls must accumulate in the buffer                                            |
| `test_record_predict_updates_inference_stats`           | After recording, `get_inference_stats()` must return the correct `avg` and `p95`                           |
| `test_record_training_appends_ms_to_latencies`          | `record_training()` must store the value in the training latency buffer                                    |
| `test_record_training_multiple_calls_accumulate`        | Multiple `record_training()` calls must accumulate in the buffer                                           |
| `test_record_training_updates_training_stats`           | After recording, `get_training_stats()` must return the correct `avg` and `p95`                            |
| `test_inc_series_trained_increments_by_one`             | `inc_series_trained()` must increase the series count by 1                                                 |
| `test_inc_series_trained_is_cumulative`                 | Multiple `inc_series_trained()` calls must accumulate                                                      |
| `test_latency_window_caps_buffer_size`                  | The latency buffer must not exceed the configured `latency_window` — oldest entries are dropped            |
| `test_update_cache_metrics_completes_without_error`     | `update_cache_metrics()` must succeed when Redis and cache objects are provided                            |
| `test_update_cache_metrics_counts_redis_keys_by_prefix` | Must count `model:*` and `metadata:*` Redis keys separately and update the corresponding Prometheus gauges |
| `test_update_cache_metrics_sums_l1_cache_lengths`       | Must set `L1_CACHE_ITEMS` to the combined length of both L1 caches (`metadata._local` + `mlflow._local`)   |

---

## `tests/services/test_metadata_cache.py`

Validates `MetadataCache` against an in-memory `fakeredis.FakeRedis` instance injected via `monkeypatch`. No live Redis is required.

| Test                                                      | Description                                                                                   |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `test_get_latest_returns_none_when_cache_is_empty`        | `get_latest()` must return `None` before any `set()` call                                     |
| `test_get_latest_returns_metadata_after_set`              | `get_latest()` must return the stored metadata after `set()`                                  |
| `test_get_latest_preserves_all_fields`                    | Serialisation round-trip must preserve `version`, `mlflow_run_id`, `points_used`, `data_hash` |
| `test_get_latest_preserves_trained_at`                    | `trained_at` timestamp must survive serialisation to JSON and back                            |
| `test_get_by_version_returns_none_when_cache_is_empty`    | `get_by_version()` must return `None` before any `set()` call                                 |
| `test_get_by_version_returns_metadata_after_set`          | `get_by_version()` must return the stored metadata after `set()`                              |
| `test_get_by_version_preserves_all_fields`                | Serialisation round-trip must preserve all fields                                             |
| `test_get_by_version_returns_none_for_wrong_version`      | Querying a different version must return `None`                                               |
| `test_get_latest_returns_none_for_different_series_id`    | Caching one series must not affect lookups for another                                        |
| `test_set_multiple_series_are_independent`                | Each `series_id` must maintain its own independent cache entry                                |
| `test_set_and_get_latest_with_no_data_hash`               | `set()` and `get_latest()` must work correctly when `data_hash` is `None`                     |
| `test_get_latest_falls_back_to_redis_when_l1_is_cold`     | `get_latest()` must fetch from Redis and repopulate L1 when the L1 TTL has expired            |
| `test_get_latest_repopulates_l1_after_redis_hit`          | After a Redis fallback, the next call must be served from L1 without hitting Redis again      |
| `test_get_by_version_falls_back_to_redis_when_l1_is_cold` | `get_by_version()` must fetch from Redis and repopulate L1 when the L1 TTL has expired        |
| `test_get_by_version_repopulates_l1_after_redis_hit`      | After a Redis fallback, the next `get_by_version()` call must be served from L1               |
| `test_dict_to_metadata_preserves_all_fields`              | All fields serialised to Redis must be faithfully reconstructed by `_dict_to_metadata`        |
| `test_dict_to_metadata_handles_null_trained_at`           | `_dict_to_metadata` must reconstruct metadata correctly when `trained_at` is `None`           |

---

## `tests/services/test_mlflow_service.py`

Validates `MLflowService` with a `fakeredis.FakeRedis` instance for the Redis layer and a `MagicMock` for the MLflow client. No live MLflow or Redis is required.

| Test                                                             | Description                                                                                              |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `test_get_cached_model_returns_none_when_key_not_in_redis`       | `get_cached_model()` must return `None` when nothing is cached for the given `run_id`                    |
| `test_set_then_get_cached_model_returns_correct_model`           | After caching, `get_cached_model()` must return a model with the correct `mean` and `std`                |
| `test_get_cached_model_returns_none_for_different_run_id`        | Caching under one `run_id` must not affect lookups for another                                           |
| `test_set_cached_model_stores_correct_json_in_redis`             | `_set_cached_model()` must serialise `mean` and `std` as valid JSON in Redis                             |
| `test_load_model_returns_cached_model_without_calling_download`  | `load_model()` must serve from cache and skip `download_artifacts` when the model is cached              |
| `test_load_model_downloads_from_mlflow_when_not_cached`          | `load_model()` must call `download_artifacts` and unpickle the model when the cache is cold              |
| `test_load_model_caches_downloaded_model`                        | After downloading, `load_model()` must cache the model so a second call skips the download               |
| `test_save_model_returns_run_id_and_model_version`               | `save_model()` must return the `run_id` and `version` from MLflow                                        |
| `test_save_model_logs_mean_and_std_as_params`                    | `save_model()` must log `mean` and `std` as MLflow params                                                |
| `test_save_model_uses_existing_experiment_when_found`            | `save_model()` must reuse an existing experiment and not call `create_experiment` again                  |
| `test_get_cached_model_returns_model_from_redis_when_l1_is_cold` | `get_cached_model()` must deserialise a model from Redis when the L1 cache is empty                      |
| `test_get_cached_model_refreshes_redis_ttl_on_hit`               | Retrieving a model from Redis must refresh the key's TTL                                                 |
| `test_get_cached_model_repopulates_l1_from_redis`                | After a Redis hit, a subsequent call must be served from L1 without touching Redis                       |
| `test_save_model_logs_training_data_artifact_when_provided`      | `save_model()` must log `training_data.json` as a second artifact when timestamps/values are given       |
| `test_save_model_does_not_log_training_data_when_not_provided`   | `save_model()` must log only `model.pkl` when timestamps/values are omitted                              |
| `test_save_model_succeeds_when_create_registered_model_raises`   | An exception from `create_registered_model` must be silenced — `save_model()` must still return normally |

---

## `tests/e2e/test_fit.py`

End-to-end integration tests for the `POST /fit/{series_id}` endpoint. These tests run against the live API stack (Postgres, MinIO, MLflow, Redis, API) inside Docker.

| Test                                      | Description                                                |
| ----------------------------------------- | ---------------------------------------------------------- |
| `test_fit_returns_200`                    | Valid training data returns HTTP 200                       |
| `test_fit_response_contains_series_id`    | Response body contains the requested `series_id`           |
| `test_fit_response_contains_version`      | Response body contains a version string                    |
| `test_fit_response_contains_points_used`  | Response body contains the number of data points consumed  |
| `test_fit_retrain_increments_version`     | Retraining with different data increments the version      |
| `test_fit_same_data_returns_same_version` | Idempotency: identical data returns the same version       |
| `test_fit_rejects_too_few_points`         | Payload with fewer than 10 points returns HTTP 422         |
| `test_fit_rejects_mismatched_lengths`     | Timestamps and values of different lengths return HTTP 422 |
| `test_fit_rejects_duplicate_timestamps`   | Duplicate timestamps return HTTP 422                       |
| `test_fit_rejects_unsorted_timestamps`    | Unsorted timestamps return HTTP 422                        |
| `test_fit_rejects_negative_timestamps`    | Negative timestamps return HTTP 422                        |
| `test_fit_rejects_empty_body`             | Empty body returns HTTP 422                                |

---

## `tests/e2e/test_predict.py`

End-to-end integration tests for the `POST /predict/{series_id}` endpoint. These tests run against the live API stack (Postgres, MinIO, MLflow, Redis, API) inside Docker. A shared `trained_series` fixture is used to guarantee a known model state.

| Test                                              | Description                                               |
| ------------------------------------------------- | --------------------------------------------------------- |
| `test_predict_returns_200`                        | Valid prediction returns HTTP 200                         |
| `test_predict_response_has_anomaly_field`         | Response body contains the `anomaly` boolean              |
| `test_predict_response_has_model_version`         | Response body contains the `model_version` string         |
| `test_predict_model_version_matches_trained`      | Returned version matches the model trained in the fixture |
| `test_predict_normal_point_is_not_anomaly`        | A value near the mean returns `anomaly=false`             |
| `test_predict_point_within_3sigma_is_not_anomaly` | A value within `mean ± 3σ` returns `anomaly=false`        |
| `test_predict_point_above_3sigma_is_anomaly`      | A value above `mean + 3σ` returns `anomaly=true`          |
| `test_predict_with_explicit_version`              | Querying a specific `?version=` succeeds                  |
| `test_predict_with_unknown_version_returns_404`   | Querying a non-existent version returns HTTP 404          |
| `test_predict_unknown_series_returns_404`         | Predicting on an untrained series returns HTTP 404        |
| `test_predict_returns_422_without_body`           | Empty body returns HTTP 422                               |
| `test_predict_returns_422_without_value`          | Missing `value` field returns HTTP 422                    |
| `test_predict_returns_422_without_timestamp`      | Missing `timestamp` field returns HTTP 422                |

---

## Test dependencies

| Package          | Purpose                                      |
| ---------------- | -------------------------------------------- |
| `pytest`         | Main test framework                          |
| `pytest-asyncio` | Support for `async/await` tests (auto mode)  |
| `pytest-cov`     | Coverage reporting (`--cov=src`)             |
| `httpx`          | Required by FastAPI/Starlette's `TestClient` |
| `fakeredis`      | In-memory Redis mock for unit tests          |
