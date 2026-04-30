from fastapi import APIRouter, Depends, HTTPException

from src.api.schemas import TrainRequest, TrainResponse
from src.api.dependencies import get_training_service
from src.services.training_service import TrainingService

router = APIRouter()


@router.post("/fit/{series_id}", response_model=TrainResponse, tags=["Training"])
async def fit(
    series_id: str,
    body: TrainRequest,
    svc: TrainingService = Depends(get_training_service),
) -> TrainResponse:
    try:
        return await svc.fit(series_id=series_id, body=body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
