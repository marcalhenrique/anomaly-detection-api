import numpy as np

from src.core.time_series import DataPoint, TimeSeries

class AnomalyDetectionModel:
    def fit(self, data: TimeSeries) -> "AnomalyDetectionModel":
        values = [d.value for d in data.data]
        self.mean = np.mean(values)
        self.std = np.std(values)
        return self

    def predict(self, data_point: DataPoint) -> bool:
        return data_point.value > self.mean + 3 * self.std

