from src.models.model_metadata import ModelMetadata


class MetadataCache:
    def __init__(self) -> None:
        self._latest: dict[str, ModelMetadata] = {}
        self._by_version: dict[tuple[str, str], ModelMetadata] = {}

    def set(self, metadata: ModelMetadata) -> None:
        current = self._latest.get(metadata.series_id)
        if current is None or metadata.trained_at >= current.trained_at:
            self._latest[metadata.series_id] = metadata
        self._by_version[(metadata.series_id, metadata.version)] = metadata

    def get_latest(self, series_id: str) -> ModelMetadata | None:
        return self._latest.get(series_id)

    def get_by_version(self, series_id: str, version: str) -> ModelMetadata | None:
        return self._by_version.get((series_id, version))
