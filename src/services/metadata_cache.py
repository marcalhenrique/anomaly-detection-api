import json

from cachetools import TTLCache

from src.models.model_metadata import ModelMetadata
from src.core.config import get_settings
from src.core.redis_client import get_redis_client

_settings = get_settings()

def _metadata_to_dict(metadata: ModelMetadata) -> dict:
    return {
        "id": metadata.id,
        "series_id": metadata.series_id,
        "version": metadata.version,
        "mlflow_run_id": metadata.mlflow_run_id,
        "points_used": metadata.points_used,
        "data_hash": metadata.data_hash,
        "trained_at": metadata.trained_at.isoformat() if metadata.trained_at else None,
    }

def _dict_to_metadata(data: dict) -> ModelMetadata:
    from datetime import datetime

    obj = ModelMetadata(
        series_id=data["series_id"],
        version=data["version"],
        mlflow_run_id=data["mlflow_run_id"],
        points_used=data["points_used"],
        data_hash=data.get("data_hash"),
    )
    obj.id = data.get("id")
    trained_at = data.get("trained_at")
    if trained_at:
        obj.trained_at = datetime.fromisoformat(trained_at)
    return obj

class MetadataCache:
    def __init__(self) -> None:
        self._redis = get_redis_client()
        self._local = TTLCache(
            maxsize=_settings.local_cache_maxsize,
            ttl=_settings.local_cache_ttl_seconds,
        )

    def _latest_key(self, series_id: str) -> str:
        return f"metadata:latest:{series_id}"

    def _version_key(self, series_id: str, version: str) -> str:
        return f"metadata:version:{series_id}:{version}"

    def _local_key(self, series_id: str, version: str | None = None) -> str:
        if version is None:
            return f"latest:{series_id}"
        return f"version:{series_id}:{version}"

    def set(self, metadata: ModelMetadata) -> None:
        payload = json.dumps(_metadata_to_dict(metadata))
        ttl = _settings.redis_metadata_ttl_seconds

        self._redis.set(self._latest_key(metadata.series_id), payload, ex=ttl)
        self._redis.set(
            self._version_key(metadata.series_id, metadata.version), payload, ex=ttl
        )

        self._local[self._local_key(metadata.series_id)] = metadata
        self._local[
            self._local_key(metadata.series_id, metadata.version)
        ] = metadata

    def get_latest(self, series_id: str) -> ModelMetadata | None:

        local_key = self._local_key(series_id)
        cached = self._local.get(local_key)
        if cached is not None:
            return cached

        payload = self._redis.get(self._latest_key(series_id))
        if payload is None:
            return None

        metadata = _dict_to_metadata(json.loads(payload))
        self._local[local_key] = metadata
        return metadata

    def get_by_version(self, series_id: str, version: str) -> ModelMetadata | None:

        local_key = self._local_key(series_id, version)
        cached = self._local.get(local_key)
        if cached is not None:
            return cached

        payload = self._redis.get(self._version_key(series_id, version))
        if payload is None:
            return None

        metadata = _dict_to_metadata(json.loads(payload))
        self._local[local_key] = metadata
        return metadata

