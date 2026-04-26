import pickle
import tempfile
import os

import mlflow
import mlflow.tracking

from cachetools import LRUCache

from src.services.anomaly_detection import AnomalyDetectionModel
from src.core.config import get_settings

settings = get_settings()


class MLflowService:
    def __init__(self, tracking_uri: str, artifact_bucket: str) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self._client = mlflow.tracking.MlflowClient()
        self._artifact_bucket = artifact_bucket
        self._model_cache = LRUCache(settings.lru_cache_size)

    def _get_or_create_experiment(self, series_id: str) -> str:
        experiment = self._client.get_experiment_by_name(series_id)
        if experiment:
            return experiment.experiment_id
        return self._client.create_experiment(
            name=series_id,
            artifact_location=f"s3://{self._artifact_bucket}/{series_id}",
        )

    def save_model(
        self, series_id: str, model: AnomalyDetectionModel, points_used: int
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
                pickle.dump(model, f)
            self._client.log_artifact(run_id, model_path, artifact_path="model")

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

        return run_id, model_version.version

    def load_model(self, run_id: str) -> AnomalyDetectionModel:
        if run_id in self._model_cache:
            return self._model_cache[run_id]

        with tempfile.TemporaryDirectory() as tmp:
            local_dir = self._client.download_artifacts(run_id, "model", dst_path=tmp)
            model_path = os.path.join(local_dir, "model.pkl")
            with open(model_path, "rb") as f:
                model = pickle.load(f)

        self._model_cache[run_id] = model
        return model
