from typing import Sequence

from pydantic import BaseModel, Field

class DataPoint(BaseModel):
    timestamp: int = Field(
        ..., description="Unix timestamp of the time the data point was collected"
    )
    value: float = Field(
        ..., description="Value of the time series measured at time `timestamp`"
    )

class TimeSeries(BaseModel):
    data: Sequence[DataPoint] = Field(
        ...,
        description="List of datapoints, ordered in time, of subsequent measurements of some quantity",
    )

