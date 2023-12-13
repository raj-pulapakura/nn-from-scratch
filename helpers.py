import pandas as pd
import numpy as np

def load_mnist():
    train_X = pd.read_csv("data/mnist_train.csv")
    train_y = train_X.pop("label")

    val_X = pd.read_csv("data/mnist_test.csv")
    val_y = val_X.pop("label")

    return (
        train_X.to_numpy().astype(np.float32), 
        train_y.to_numpy(),
        val_X.to_numpy().astype(np.float32),
        val_y.to_numpy(),
    )

def one_hot(y):
    return np.eye(len(np.unique(y)))[y]

def softmax(x: np.ndarray):
    x = x - np.max(x, axis=0) # this is to avoid high values (it doesn't change the outputs)
    e = np.exp(x)
    s = e / np.sum(e, axis=0)
    return s

def cross_entropy(y_pred: np.ndarray, y_true: np.ndarray):
    m = y_pred.shape[-1]
    log_probs =  (-1) * np.sum( y_true * np.log(y_pred+1e-7) )
    loss = log_probs / m
    return float(loss)