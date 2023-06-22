from ..losses import Loss, BinaryCrossEntropy, MeanSquaredError
import numpy as np


def test_loss_initialization():
    y_true = np.array([1.0, 1.0, 1.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    loss = Loss(y_pred=y_pred, y_true=y_true)
    assert loss.y_pred is y_pred
    assert loss.y_true is y_true


def test_binary_cross_entropy():
    y_true = np.array([1.0, 1.0, 1.0])
    y_pred = np.array([0.9, 0.9, 0.9])
    binary_cross = BinaryCrossEntropy(y_true=y_true, y_pred=y_pred)
    res = binary_cross.call()
    assert round(res, 2) == 0.11


def test_mean_squared_error():
    y_true = np.array([1.0, 1.0, 1.0])
    y_pred = np.array([0.9, 0.9, 0.9])
    mse = MeanSquaredError(y_pred=y_pred, y_true=y_true)
    res = mse.call()
    assert round(res, 2) == 0.01
