from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

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

    async def get_latest_by_series_id(self, series_id: str) -> ModelMetadata | None:
        result = await self.session.execute(
            select(ModelMetadata)
            .where(ModelMetadata.series_id == series_id)
            .order_by(ModelMetadata.trained_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_by_version(
        self, series_id: str, version: str
    ) -> ModelMetadata | None:
        result = await self.session.execute(
            select(ModelMetadata).where(
                ModelMetadata.series_id == series_id, ModelMetadata.version == version
            )
        )
        return result.scalar_one_or_none()

    async def get_all_latest(self) -> list[ModelMetadata]:
        subquery = (
            select(
                ModelMetadata.series_id,
                func.max(ModelMetadata.trained_at).label("max_trained_at"),
            )
            .group_by(ModelMetadata.series_id)
            .subquery()
        )
        result = await self.session.execute(
            select(ModelMetadata).join(
                subquery,
                (ModelMetadata.series_id == subquery.c.series_id)
                & (ModelMetadata.trained_at == subquery.c.max_trained_at),
            )
        )
        return list(result.scalars().all())
