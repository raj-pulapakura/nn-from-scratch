from units.layers import Dense
from units.activations import Sigmoid
from units.losses import mse, mse_prime

X = None
Y = None

network = [
    Dense(784, 16),
]