from sqlalchemy.ext.asyncio import AsyncSession

from src.models.model_metadata import ModelMetadata


class ModelMetadataRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save(
        self,
        series_id: str,
        run_id: str,
        model_version: str,
        points_used: int,
    ) -> ModelMetadata:
        record = ModelMetadata(
            series_id=series_id,
            mlflow_run_id=run_id,
            version=model_version,
            points_used=points_used,
        )
        self.session.add(record)
        await self.session.commit()
        return record
