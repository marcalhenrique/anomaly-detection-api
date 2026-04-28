# Tests

Test coverage for the training flow (`POST /fit/{series_id}`), prediction flow (`POST /predict/{series_id}`), healthcheck, and end-to-end integration.

## Structure

```
tests/
├── api/
│   ├── test_fit_route.py
│   ├── test_healthcheck_route.py
│   └── test_predict_route.py
├── e2e/
│   ├── test_fit.py
│   └── test_predict.py
├── repositories/
│   └── test_model_metadata_repository.py
├── schemas/
│   └── test_train_request.py
└── services/
    ├── test_anomaly_detection.py
    ├── test_prediction_service.py
    ├── test_training_lock.py
    └── test_training_service.py
```

## How to run

Tests run inside Docker — no local `uv` installation required.

```bash
# Unit + integration tests only (fast, no external services)
make test-unit

# End-to-end tests (spins up full Docker stack: Postgres, MinIO, MLflow, API)
make test-e2e

# Both
make test

# With coverage report (requires uv locally)
uv run pytest tests/ -v --cov=src --cov-report=term-missing
```

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

| Test                                                 | Description                                                                                                                          |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `test_fit_returns_correct_train_response`            | The return value of `fit()` must contain the correct `series_id`, `version`, and `points_used`                                       |
| `test_fit_calls_mlflow_save_model_with_correct_args` | `MLflowService.save_model` must be called with the correct `series_id` and point count                                               |
| `test_fit_calls_repo_save_with_correct_args`         | `ModelMetadataRepository.save` must be called with all correct fields, including the `run_id` and `model_version` returned by MLflow |
| `test_fit_acquires_lock_for_series_id`               | The training lock must be acquired using the correct `series_id`                                                                     |
| `test_fit_raises_when_mlflow_save_fails`             | If MLflow raises an exception, `fit()` must propagate it                                                                             |
| `test_fit_raises_when_repo_save_fails`               | If the repository raises an exception, `fit()` must propagate it                                                                     |
| `test_fit_does_not_call_repo_when_mlflow_fails`      | If MLflow fails, the repository must not be called                                                                                   |
| `test_fit_returns_existing_version_when_data_is_identical` | Idempotency: identical data must return the existing version without re-training                                               |
| `test_fit_skips_mlflow_and_repo_when_data_is_identical`    | Idempotency: no MLflow or DB write must occur when data is identical                                                           |
| `test_fit_trains_when_values_differ`                 | Different values must trigger a new training + version increment                                                                     |
| `test_fit_trains_when_no_existing_model`             | A brand-new series must train from scratch and return version `1`                                                                    |

---

## `tests/services/test_training_lock.py`

Validates the concurrency behaviour of `LocalTrainingLock`.

| Test                                         | Description                                                                                    |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `test_same_series_id_executes_sequentially`  | Two concurrent acquires for the same `series_id` must not overlap (mutex)                      |
| `test_different_series_ids_run_concurrently` | Acquires for different `series_id` values must not block each other (true concurrency)         |
| `test_lock_released_after_exception`         | The lock must be released even if the protected block raises an exception                      |
| `test_independent_locks_per_series_id`       | Each `series_id` has its own independent lock — acquiring two different ones must not deadlock |

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

| Test                                                 | Description                                                                           |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `test_healthcheck_returns_200_when_healthy`          | All dependencies up → HTTP 200 with metrics and latency statistics                    |
| `test_healthcheck_returns_populated_metrics`         | Metrics collector with latencies → correct `avg` and `p95` values in response         |
| `test_healthcheck_returns_503_when_db_down`          | Database unreachable → HTTP 503 with partial metrics (series count + latencies)       |
| `test_healthcheck_returns_503_when_mlflow_down`      | MLflow unreachable → HTTP 503 with partial metrics                                    |

---

## `tests/services/test_prediction_service.py`

Validates the orchestration logic of `PredictionService.predict()`. All external collaborators (MLflow, repository) are mocked.

| Test                                                              | Description                                                                                       |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `test_predict_returns_correct_response_for_normal_value`          | A normal value must return `anomaly=False` and the correct `model_version`                        |
| `test_predict_returns_correct_response_for_anomalous_value`       | An anomalous value must return `anomaly=True`                                                     |
| `test_predict_uses_latest_model_when_version_not_provided`        | Without a `version` param, `get_latest_by_series_id` must be called and `get_by_version` must not |
| `test_predict_uses_specific_version_when_provided`                | With a `version` param, `get_by_version` must be called and `get_latest_by_series_id` must not    |
| `test_predict_loads_model_using_run_id_from_metadata`             | `MLflowService.load_model` must be called with the `run_id` stored in the metadata record         |
| `test_predict_calls_model_predict_with_correct_data_point`        | `model.predict()` must receive a `DataPoint` with the correct `timestamp` and `value`             |
| `test_predict_raises_model_not_found_when_series_does_not_exist`  | When no metadata is found for the series, `ModelNotFoundError` must be raised                     |
| `test_predict_raises_model_not_found_when_version_does_not_exist` | When the requested version is not found, `ModelNotFoundError` must be raised                      |

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

Validates the behaviour of `ModelMetadataRepository.save()`. The SQLAlchemy `AsyncSession` is mocked.

| Test                                                   | Description                                                                                          |
| ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `test_save_adds_record_to_session`                     | `save()` must call `session.add()` exactly once with the created record                              |
| `test_save_calls_commit`                               | `save()` must call `session.commit()` to persist the transaction                                     |
| `test_save_returns_model_metadata_with_correct_fields` | The returned object must have the correct `series_id`, `mlflow_run_id`, `version`, and `points_used` |
| `test_save_preserves_add_then_commit_order`            | `add()` must be called before `commit()` — the reverse order would lose data                         |

---

## `tests/e2e/test_fit.py`

End-to-end integration tests for the `POST /fit/{series_id}` endpoint. These tests run against the live API stack (Postgres, MinIO, MLflow, API) inside Docker.

| Test                                              | Description                                                                    |
| ------------------------------------------------- | ------------------------------------------------------------------------------ |
| `test_fit_returns_200`                            | Valid training data returns HTTP 200                                           |
| `test_fit_response_contains_series_id`            | Response body contains the requested `series_id`                               |
| `test_fit_response_contains_version`              | Response body contains a version string                                        |
| `test_fit_response_contains_points_used`          | Response body contains the number of data points consumed                      |
| `test_fit_retrain_increments_version`             | Retraining with different data increments the version                          |
| `test_fit_same_data_returns_same_version`         | Idempotency: identical data returns the same version                           |
| `test_fit_rejects_too_few_points`                 | Payload with fewer than 10 points returns HTTP 422                             |
| `test_fit_rejects_mismatched_lengths`             | Timestamps and values of different lengths return HTTP 422                     |
| `test_fit_rejects_duplicate_timestamps`           | Duplicate timestamps return HTTP 422                                           |
| `test_fit_rejects_unsorted_timestamps`            | Unsorted timestamps return HTTP 422                                            |
| `test_fit_rejects_negative_timestamps`            | Negative timestamps return HTTP 422                                            |
| `test_fit_rejects_empty_body`                     | Empty body returns HTTP 422                                                    |

---

## `tests/e2e/test_predict.py`

End-to-end integration tests for the `POST /predict/{series_id}` endpoint. These tests run against the live API stack inside Docker. A shared `trained_series` fixture is used to guarantee a known model state.

| Test                                               | Description                                                                |
| -------------------------------------------------- | -------------------------------------------------------------------------- |
| `test_predict_returns_200`                         | Valid prediction returns HTTP 200                                          |
| `test_predict_response_has_anomaly_field`          | Response body contains the `anomaly` boolean                               |
| `test_predict_response_has_model_version`          | Response body contains the `model_version` string                          |
| `test_predict_model_version_matches_trained`       | Returned version matches the model trained in the fixture                  |
| `test_predict_normal_point_is_not_anomaly`         | A value near the mean returns `anomaly=false`                              |
| `test_predict_point_within_3sigma_is_not_anomaly`  | A value within `mean ± 3σ` returns `anomaly=false`                         |
| `test_predict_point_above_3sigma_is_anomaly`       | A value above `mean + 3σ` returns `anomaly=true`                           |
| `test_predict_with_explicit_version`               | Querying a specific `?version=` succeeds                                   |
| `test_predict_with_unknown_version_returns_404`    | Querying a non-existent version returns HTTP 404                           |
| `test_predict_unknown_series_returns_404`          | Predicting on an untrained series returns HTTP 404                         |
| `test_predict_returns_422_without_body`            | Empty body returns HTTP 422                                                |
| `test_predict_returns_422_without_value`           | Missing `value` field returns HTTP 422                                     |
| `test_predict_returns_422_without_timestamp`       | Missing `timestamp` field returns HTTP 422                                 |

---

## Test dependencies

| Package          | Purpose                                      |
| ---------------- | -------------------------------------------- |
| `pytest`         | Main test framework                          |
| `pytest-asyncio` | Support for `async/await` tests (auto mode)  |
| `httpx`          | Required by FastAPI/Starlette's `TestClient` |
