# Unit Tests — `/fit` Route

Test coverage for the training flow (`POST /fit/{series_id}`).

## Structure

```
tests/
├── api/
│   └── test_fit_route.py
├── repositories/
│   └── test_model_metadata_repository.py
├── schemas/
│   └── test_train_request.py
└── services/
    ├── test_training_lock.py      # pre-existing
    ├── test_training_service.py
    └── test_anomaly_detection.py
```

## How to run

```bash
# All tests
uv run pytest tests/ -v

# With coverage report
uv run pytest tests/ -v --cov=src --cov-report=term-missing

# Single module
uv run pytest tests/services/test_training_service.py -v
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

Tests the `POST /fit/{series_id}` HTTP route via FastAPI's `TestClient`. The `TrainingService` is replaced by a mock through `dependency_overrides`, with no real I/O.

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

## `tests/repositories/test_model_metadata_repository.py`

Validates the behaviour of `ModelMetadataRepository.save()`. The SQLAlchemy `AsyncSession` is mocked.

| Test                                                   | Description                                                                                          |
| ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `test_save_adds_record_to_session`                     | `save()` must call `session.add()` exactly once with the created record                              |
| `test_save_calls_commit`                               | `save()` must call `session.commit()` to persist the transaction                                     |
| `test_save_returns_model_metadata_with_correct_fields` | The returned object must have the correct `series_id`, `mlflow_run_id`, `version`, and `points_used` |
| `test_save_preserves_add_then_commit_order`            | `add()` must be called before `commit()` — the reverse order would lose data                         |

---

## Test dependencies

| Package          | Purpose                                      |
| ---------------- | -------------------------------------------- |
| `pytest`         | Main test framework                          |
| `pytest-asyncio` | Support for `async/await` tests (auto mode)  |
| `httpx`          | Required by FastAPI/Starlette's `TestClient` |
