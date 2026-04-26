from pydantic import BaseModel, Field, model_validator

MIN_TRAINING_POINTS = 10

_EXAMPLE_TIMESTAMPS = [
    1745000000,
    1745000060,
    1745000120,
    1745000180,
    1745000240,
    1745000300,
    1745000360,
    1745000420,
    1745000480,
    1745000540,
]
_EXAMPLE_VALUES = [10.1, 10.3, 9.8, 10.0, 10.2, 9.9, 10.1, 10.4, 9.7, 10.0]


class TrainRequest(BaseModel):
    timestamps: list[int] = Field(
        ...,
        description="List of unix timestamps for the training data",
        min_length=MIN_TRAINING_POINTS,
        examples=[_EXAMPLE_TIMESTAMPS],
    )
    values: list[float] = Field(
        ...,
        description="List of corresponding values for the training data",
        min_length=MIN_TRAINING_POINTS,
        examples=[_EXAMPLE_VALUES],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"timestamps": _EXAMPLE_TIMESTAMPS, "values": _EXAMPLE_VALUES}]
        }
    }

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
    series_id: str = Field(
        ...,
        description="Unique identifier for the trained series",
        examples=["sensor-vibration-01"],
    )
    version: str = Field(
        ...,
        description="Version of the trained model",
        examples=["3"],
    )
    points_used: int = Field(
        ...,
        description="Number of data points used for training",
        examples=[10],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"series_id": "sensor-vibration-01", "version": "3", "points_used": 10}
            ]
        }
    }


class PredictRequest(BaseModel):
    timestamp: int = Field(
        ...,
        description="Timestamp for the prediction",
        examples=[1745000600],
    )
    value: float = Field(
        ...,
        description="Value to be evaluated for anomaly detection",
        examples=[42.7],
    )

    model_config = {
        "json_schema_extra": {"examples": [{"timestamp": 1745000600, "value": 42.7}]}
    }


class PredictResponse(BaseModel):
    anomaly: bool = Field(
        ...,
        description="Indicates whether the input value is an anomaly",
        examples=[True],
    )
    model_version: str = Field(
        ...,
        description="Version of the model used for prediction",
        examples=["3"],
    )

    model_config = {
        "json_schema_extra": {"examples": [{"anomaly": True, "model_version": "3"}]}
    }


class HealthCheckMetrics(BaseModel):
    avg: float = Field(None, description="Average value of the series", examples=[12.4])
    p95: float = Field(
        None, description="95th percentile of the series", examples=[28.1]
    )


class HealthCheckResponse(BaseModel):
    series_trained: int = Field(
        ...,
        description="Number of series trained",
        examples=[5],
    )
    inference_latency_ms: HealthCheckMetrics = Field(
        ..., description="Inference latency metrics"
    )
    training_latency_ms: HealthCheckMetrics = Field(
        ..., description="Training latency metrics"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "series_trained": 5,
                    "inference_latency_ms": {"avg": 12.4, "p95": 28.1},
                    "training_latency_ms": {"avg": 340.2, "p95": 512.0},
                }
            ]
        }
    }
