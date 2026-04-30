import time

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.schemas import PredictResponse, PredictRequest
from src.api.dependencies import get_prediction_service, get_metrics_collector
from src.services.prediction_service import PredictionService, ModelNotFoundError
from src.services.metrics_collector import MetricsCollector

router = APIRouter()

@router.post(
    "/predict/{series_id}", response_model=PredictResponse, tags=["Prediction"]
)
async def predict(
    series_id: str,
    body: PredictRequest,
    version: str | None = Query(None),
    svc: PredictionService = Depends(get_prediction_service),
    metrics: MetricsCollector = Depends(get_metrics_collector),
) -> PredictResponse:
    start = time.perf_counter()
    try:
        response, timings = await svc.predict(
            series_id=series_id,
            timestamp=body.timestamp,
            value=body.value,
            version=version,
        )
        timings["total_ms"] = round((time.perf_counter() - start) * 1000, 3)
        metrics.record_predict(timings)
        return response
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

