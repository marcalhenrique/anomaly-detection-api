import json
import os
import tempfile

import mlflow
import mlflow.tracking
from cachetools import TTLCache

from src.services.anomaly_detection import AnomalyDetectionModel
from src.core.config import get_settings
from src.core.redis_client import get_redis_client

settings = get_settings()

class MLflowService:
    def __init__(self, tracking_uri: str, artifact_bucket: str) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self._client = mlflow.tracking.MlflowClient()
        self._artifact_bucket = artifact_bucket
        self._redis = get_redis_client()
        self._local = TTLCache(
            maxsize=settings.local_cache_maxsize,
            ttl=settings.local_cache_ttl_seconds,
        )

    def _model_key(self, run_id: str) -> str:
        return f"model:{run_id}"

    def _get_cached_model(self, run_id: str) -> AnomalyDetectionModel | None:

        cached = self._local.get(run_id)
        if cached is not None:
            return cached

        key = self._model_key(run_id)
        payload = self._redis.get(key)
        if payload is None:
            return None

        data = json.loads(payload)
        model = AnomalyDetectionModel()
        model.mean = data["mean"]
        model.std = data["std"]
        self._redis.expire(key, settings.redis_model_ttl_seconds)
        self._local[run_id] = model
        return model

    def _set_cached_model(self, run_id: str, model: AnomalyDetectionModel) -> None:
        payload = json.dumps({"mean": model.mean, "std": model.std})
        self._redis.set(
            self._model_key(run_id), payload, ex=settings.redis_model_ttl_seconds
        )
        self._local[run_id] = model

    def _get_or_create_experiment(self, series_id: str) -> str:
        experiment = self._client.get_experiment_by_name(series_id)
        if experiment:
            return experiment.experiment_id
        return self._client.create_experiment(
            name=series_id,
            artifact_location=f"s3://{self._artifact_bucket}/{series_id}",
        )

    def save_model(
        self,
        series_id: str,
        model: AnomalyDetectionModel,
        points_used: int,
        timestamps: list[int] | None = None,
        values: list[float] | None = None,
    ) -> tuple[str, str]:
        experiment_id = self._get_or_create_experiment(series_id)
        run = self._client.create_run(
            experiment_id=experiment_id,
            tags={"series_id": series_id},
        )
        run_id = run.info.run_id

        self._client.log_param(run_id, "mean", model.mean)
        self._client.log_param(run_id, "std", model.std)
        self._client.log_param(run_id, "points_used", points_used)

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.pkl")
            with open(model_path, "wb") as f:
                import pickle

                pickle.dump(model, f)
            self._client.log_artifact(run_id, model_path, artifact_path="model")

            if timestamps is not None and values is not None:
                data_path = os.path.join(tmp, "training_data.json")
                payload = [
                    {"timestamp": t, "value": v} for t, v in zip(timestamps, values)
                ]
                with open(data_path, "w") as f:
                    json.dump(payload, f)
                self._client.log_artifact(run_id, data_path, artifact_path="model")

        self._client.set_terminated(run_id, status="FINISHED")

        try:
            self._client.create_registered_model(series_id)
        except Exception:
            pass

        artifact_uri = self._client.get_run(run_id).info.artifact_uri
        model_version = self._client.create_model_version(
            name=series_id,
            source=f"{artifact_uri}/model",
            run_id=run_id,
        )

        self._set_cached_model(run_id, model)
        return run_id, model_version.version

    def load_model(self, run_id: str) -> AnomalyDetectionModel:
        cached = self._get_cached_model(run_id)
        if cached is not None:
            return cached

        with tempfile.TemporaryDirectory() as tmp:
            local_dir = self._client.download_artifacts(run_id, "model", dst_path=tmp)
            model_path = os.path.join(local_dir, "model.pkl")
            with open(model_path, "rb") as f:
                import pickle

                model = pickle.load(f)

        self._set_cached_model(run_id, model)
        return model

    def get_cached_model(self, run_id: str) -> AnomalyDetectionModel | None:
        return self._get_cached_model(run_id)

