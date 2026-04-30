"""Unit tests for MLflowService.

The MLflow client and Redis are both mocked/faked so no live services
are required.
"""

import json
import pickle
from unittest.mock import MagicMock, patch

import fakeredis
import pytest

from src.services.anomaly_detection import AnomalyDetectionModel
from src.services.mlflow_service import MLflowService

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

RUN_ID = "run-abc123"
SERIES_ID = "sensor-01"
ARTIFACT_BUCKET = "test-bucket"
TRACKING_URI = "http://localhost:5001"


def _make_model(mean: float = 3.0, std: float = 1.5) -> AnomalyDetectionModel:
    model = AnomalyDetectionModel()
    model.mean = mean
    model.std = std
    return model


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def mock_mlflow_client():
    return MagicMock()


@pytest.fixture
def svc(fake_redis, mock_mlflow_client):
    """MLflowService with a fakeredis backend and a mocked MlflowClient."""
    with (
        patch("src.services.mlflow_service.get_redis_client", return_value=fake_redis),
        patch("src.services.mlflow_service.mlflow.set_tracking_uri"),
        patch(
            "src.services.mlflow_service.mlflow.tracking.MlflowClient",
            return_value=mock_mlflow_client,
        ),
    ):
        return MLflowService(tracking_uri=TRACKING_URI, artifact_bucket=ARTIFACT_BUCKET)


# ---------------------------------------------------------------------------
# _get_cached_model / _set_cached_model
# ---------------------------------------------------------------------------


def test_get_cached_model_returns_none_when_key_not_in_redis(svc):
    """get_cached_model() must return None when nothing is cached for the run_id."""
    assert svc.get_cached_model(RUN_ID) is None


def test_set_then_get_cached_model_returns_correct_model(svc):
    """After caching a model, get_cached_model() must return it with correct stats."""
    model = _make_model(mean=7.5, std=2.0)
    svc._set_cached_model(RUN_ID, model)
    result = svc.get_cached_model(RUN_ID)
    assert result is not None
    assert result.mean == pytest.approx(7.5)
    assert result.std == pytest.approx(2.0)


def test_get_cached_model_returns_none_for_different_run_id(svc):
    """Caching a model for one run_id must not affect another."""
    svc._set_cached_model("run-X", _make_model())
    assert svc.get_cached_model("run-Y") is None


def test_set_cached_model_stores_correct_json_in_redis(svc, fake_redis):
    """_set_cached_model() must serialise mean/std to Redis as valid JSON."""
    model = _make_model(mean=4.0, std=0.5)
    svc._set_cached_model(RUN_ID, model)
    raw = fake_redis.get(f"model:{RUN_ID}")
    assert raw is not None
    data = json.loads(raw)
    assert data["mean"] == pytest.approx(4.0)
    assert data["std"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# load_model — cached path
# ---------------------------------------------------------------------------


def test_load_model_returns_cached_model_without_calling_download(
    svc, mock_mlflow_client
):
    """load_model() must return the cached model and skip download_artifacts."""
    model = _make_model(mean=9.0, std=0.1)
    svc._set_cached_model(RUN_ID, model)

    result = svc.load_model(RUN_ID)

    assert result.mean == pytest.approx(9.0)
    mock_mlflow_client.download_artifacts.assert_not_called()


# ---------------------------------------------------------------------------
# load_model — download path
# ---------------------------------------------------------------------------


def test_load_model_downloads_from_mlflow_when_not_cached(
    svc, mock_mlflow_client, tmp_path
):
    """load_model() must download from MLflow and return a valid model when cache is cold."""
    expected = _make_model(mean=5.0, std=1.0)
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(expected, f)

    mock_mlflow_client.download_artifacts.return_value = str(tmp_path)

    result = svc.load_model(RUN_ID)

    assert result.mean == pytest.approx(5.0)
    assert result.std == pytest.approx(1.0)
    mock_mlflow_client.download_artifacts.assert_called_once()
    call_args = mock_mlflow_client.download_artifacts.call_args
    assert call_args.args[0] == RUN_ID
    assert call_args.args[1] == "model"


def test_load_model_caches_downloaded_model(svc, mock_mlflow_client, tmp_path):
    """After downloading a model, load_model() must cache it so a second call skips download."""
    model = _make_model(mean=3.0, std=0.5)
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    mock_mlflow_client.download_artifacts.return_value = str(tmp_path)

    svc.load_model(RUN_ID)  # first call — downloads
    svc.load_model(RUN_ID)  # second call — should use cache

    assert mock_mlflow_client.download_artifacts.call_count == 1


# ---------------------------------------------------------------------------
# save_model
# ---------------------------------------------------------------------------


def test_save_model_returns_run_id_and_model_version(svc, mock_mlflow_client, tmp_path):
    """save_model() must return the MLflow run_id and registered model version."""
    mock_mlflow_client.get_experiment_by_name.return_value = None
    mock_mlflow_client.create_experiment.return_value = "exp-1"

    run_mock = MagicMock()
    run_mock.info.run_id = "run-xyz"
    mock_mlflow_client.create_run.return_value = run_mock

    run_info = MagicMock()
    run_info.info.artifact_uri = f"s3://{ARTIFACT_BUCKET}/{SERIES_ID}/mlruns/1"
    mock_mlflow_client.get_run.return_value = run_info

    version_mock = MagicMock()
    version_mock.version = "1"
    mock_mlflow_client.create_model_version.return_value = version_mock

    model = _make_model()
    run_id, version = svc.save_model(SERIES_ID, model, 100)

    assert run_id == "run-xyz"
    assert version == "1"


def test_save_model_logs_mean_and_std_as_params(svc, mock_mlflow_client):
    """save_model() must log model.mean and model.std as MLflow params."""
    mock_mlflow_client.get_experiment_by_name.return_value = None
    mock_mlflow_client.create_experiment.return_value = "exp-1"

    run_mock = MagicMock()
    run_mock.info.run_id = "run-xyz"
    mock_mlflow_client.create_run.return_value = run_mock

    run_info = MagicMock()
    run_info.info.artifact_uri = "s3://bucket/series/mlruns"
    mock_mlflow_client.get_run.return_value = run_info
    mock_mlflow_client.create_model_version.return_value = MagicMock(version="1")

    model = _make_model(mean=7.0, std=2.5)
    svc.save_model(SERIES_ID, model, 50)

    logged_params = {
        call.args[1]: call.args[2]
        for call in mock_mlflow_client.log_param.call_args_list
    }
    assert logged_params["mean"] == pytest.approx(7.0)
    assert logged_params["std"] == pytest.approx(2.5)


def test_save_model_uses_existing_experiment_when_found(svc, mock_mlflow_client):
    """save_model() must reuse an existing experiment rather than creating a duplicate."""
    existing_exp = MagicMock()
    existing_exp.experiment_id = "existing-exp-id"
    mock_mlflow_client.get_experiment_by_name.return_value = existing_exp

    run_mock = MagicMock()
    run_mock.info.run_id = "run-xyz"
    mock_mlflow_client.create_run.return_value = run_mock

    run_info = MagicMock()
    run_info.info.artifact_uri = "s3://bucket/series/mlruns"
    mock_mlflow_client.get_run.return_value = run_info
    mock_mlflow_client.create_model_version.return_value = MagicMock(version="2")

    svc.save_model(SERIES_ID, _make_model(), 10)

    mock_mlflow_client.create_experiment.assert_not_called()
    mock_mlflow_client.create_run.assert_called_once_with(
        experiment_id="existing-exp-id", tags={"series_id": SERIES_ID}
    )


# ---------------------------------------------------------------------------
# _get_cached_model — Redis fallback (L1 cold, Redis warm)
# ---------------------------------------------------------------------------


def test_get_cached_model_returns_model_from_redis_when_l1_is_cold(svc, fake_redis):
    """_get_cached_model must deserialise a model from Redis when L1 cache is empty."""
    import json

    payload = json.dumps({"mean": 5.5, "std": 2.0})
    fake_redis.set(f"model:{RUN_ID}", payload)

    result = svc.get_cached_model(RUN_ID)

    assert result is not None
    assert result.mean == pytest.approx(5.5)
    assert result.std == pytest.approx(2.0)


def test_get_cached_model_refreshes_redis_ttl_on_hit(svc, fake_redis):
    """Retrieving a model from Redis must refresh the key's TTL."""
    import json

    payload = json.dumps({"mean": 1.0, "std": 0.5})
    fake_redis.set(f"model:{RUN_ID}", payload, ex=10)

    svc.get_cached_model(RUN_ID)

    # TTL must still be alive (was refreshed, not deleted)
    ttl = fake_redis.ttl(f"model:{RUN_ID}")
    assert ttl > 0


def test_get_cached_model_repopulates_l1_from_redis(svc, fake_redis):
    """After a Redis hit, a second call must be served from L1 (no Redis access)."""
    import json

    payload = json.dumps({"mean": 3.0, "std": 1.0})
    fake_redis.set(f"model:{RUN_ID}", payload)

    svc.get_cached_model(RUN_ID)  # Redis hit — populates L1
    fake_redis.flushall()  # Clear Redis

    result = svc.get_cached_model(RUN_ID)  # Must come from L1
    assert result is not None
    assert result.mean == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# save_model — training data artifact branch
# ---------------------------------------------------------------------------


def _setup_save_model_mocks(mock_mlflow_client):
    mock_mlflow_client.get_experiment_by_name.return_value = None
    mock_mlflow_client.create_experiment.return_value = "exp-1"
    run_mock = MagicMock()
    run_mock.info.run_id = "run-xyz"
    mock_mlflow_client.create_run.return_value = run_mock
    run_info = MagicMock()
    run_info.info.artifact_uri = f"s3://{ARTIFACT_BUCKET}/{SERIES_ID}/mlruns/1"
    mock_mlflow_client.get_run.return_value = run_info
    mock_mlflow_client.create_model_version.return_value = MagicMock(version="1")


def test_save_model_logs_training_data_artifact_when_provided(svc, mock_mlflow_client):
    """save_model() must log training_data.json when timestamps and values are given."""
    _setup_save_model_mocks(mock_mlflow_client)

    svc.save_model(
        SERIES_ID,
        _make_model(),
        3,
        timestamps=[1, 2, 3],
        values=[1.0, 2.0, 3.0],
    )

    # Two log_artifact calls: model.pkl + training_data.json
    assert mock_mlflow_client.log_artifact.call_count == 2
    artifact_paths = [
        call.kwargs.get("artifact_path") or call.args[2]
        if len(call.args) > 2
        else call.kwargs.get("artifact_path")
        for call in mock_mlflow_client.log_artifact.call_args_list
    ]
    assert artifact_paths == ["model", "model"]


def test_save_model_does_not_log_training_data_when_not_provided(
    svc, mock_mlflow_client
):
    """save_model() must log only model.pkl when timestamps/values are omitted."""
    _setup_save_model_mocks(mock_mlflow_client)

    svc.save_model(SERIES_ID, _make_model(), 3)

    assert mock_mlflow_client.log_artifact.call_count == 1


# ---------------------------------------------------------------------------
# save_model — create_registered_model exception is silenced
# ---------------------------------------------------------------------------


def test_save_model_succeeds_when_create_registered_model_raises(
    svc, mock_mlflow_client
):
    """save_model() must not propagate an exception from create_registered_model."""
    _setup_save_model_mocks(mock_mlflow_client)
    mock_mlflow_client.create_registered_model.side_effect = Exception(
        "Model already registered"
    )

    run_id, version = svc.save_model(SERIES_ID, _make_model(), 10)

    assert run_id == "run-xyz"
    assert version == "1"
