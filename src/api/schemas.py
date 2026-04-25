from pydantic import BaseModel, Field, model_validator

MIN_TRAINING_POINTS = 10


class TrainRequest(BaseModel):
    timestamps: list[int] = Field(
        ...,
        description="List of unix timestamps for the training data",
        min_length=MIN_TRAINING_POINTS,
    )
    values: list[float] = Field(
        ...,
        description="List of corresponding values for the training data",
        min_length=MIN_TRAINING_POINTS,
    )

    @model_validator(mode="after")
    def validate_series(self):
        if len(self.timestamps) != len(self.values):
            raise ValueError("timestamps and values must have the same length")
        if any(t < 0 for t in self.timestamps):
            raise ValueError("timestamps must be non-negative unix timestamps")
        if len(self.timestamps) != len(set(self.timestamps)):
            raise ValueError("timestamps must be unique")
        if self.timestamps != sorted(self.timestamps):
            raise ValueError("timestamps must be in ascending order")
        return self


class TrainResponse(BaseModel):
    series_id: str = Field(..., description="Unique identifier for the trained series")
    version: str = Field(..., description="Version of the trained model")
    points_used: int = Field(..., description="Number of data points used for training")


class PredictRequest(BaseModel):
    timestamp: str = Field(..., description="Timestamp for the prediction")
    value: float = Field(..., description="Value to be evaluated for anomaly detection")


class PredictResponse(BaseModel):
    anomaly: bool = Field(
        ..., description="Indicates whether the input value is an anomaly"
    )
    model_version: str = Field(
        ..., description="Version of the model used for prediction"
    )


class HealthCheckMetrics(BaseModel):
    avg: float = Field(None, description="Average value of the series")
    p95: float = Field(None, description="95th percentile of the series")


class HealthCheckResponse(BaseModel):
    series_trained: int = Field(..., description="Number of series trained")
    inference_latency_ms: HealthCheckMetrics = Field(
        ..., description="Inference latency metrics"
    )
    training_latency_ms: HealthCheckMetrics = Field(
        ..., description="Training latency metrics"
    )
