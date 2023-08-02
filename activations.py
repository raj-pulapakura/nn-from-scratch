import numpy as np
from layers import Layer


class Activation(Layer):
    def __init___(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.activation_prime(self.input)
    
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.e ** (-x))
        def sigmoid_prime(x):
            return sigmoid(x) * (1-sigmoid(x))
        super().__init__(sigmoid, sigmoid_prime)