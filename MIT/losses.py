import numpy as np


class Loss:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = y_true
        self.y_pred = y_pred


class BinaryCrossEntropy(Loss):
    def call(self):
        return -np.mean(
            self.y_true * np.log(self.y_pred)
            + (1 - self.y_true) * np.log(1 - self.y_pred)
        )


class MeanSquaredError(Loss):
    def call(self):
        return np.mean(np.power((self.y_true - self.y_pred), 2))
