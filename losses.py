import numpy as np


def mse(y_pred: np.ndarray, y_true: np.ndarray):
    return ( y_pred - y_true ) ** 2


def mse_prime(y_pred: np.ndarray, y_true: np.ndarray):
    return 2 * ( y_pred - y_true )

def cross_entropy(y_pred: np.ndarray, y_true: np.ndarray):
    return (-1) * np.sum( y_true * np.log(y_pred) )

def cross_entropy_prime(y_pred: np.ndarray, y_true: np.ndarray):
    return (-1) * (y_true / y_pred)