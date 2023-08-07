import numpy as np
from helpers import load_mnist_train
from losses import cross_entropy, cross_entropy_prime


class Layer:
    def __init__(self, name):
        self.name = name

    def forward(self, input: np.ndarray):
        pass

    def backward(self):
        pass


class Dense(Layer):
    """
    Fully Connected Layer of Neurons
    """
    def __init__(self, input_size, output_size, name="Dense"):
        self.weights = np.random.rand(output_size, input_size)
        self.bias = np.random.rand(output_size, 1)
        super().__init__(name)

    def forward(self, input: np.ndarray):
        return np.dot(self.weights, input) + self.bias
    
    def backward(self):
        pass


class Activation(Layer):
    """
    Activation function
    """
    def __init__(self, activation, activation_prime, name):
        self.activation = activation
        self.activation_prime = activation_prime
        self.name = name

    def forward(self, input: np.ndarray):
        self.input = input
        return self.activation(input)

    def backward(self):
        pass


class ReLU(Activation):
    """
    ReLU activation.
    Defined as max(x, 0).
    """
    def __init__(self, name):
        relu = lambda x: np.maximum(x, 0)
        relu_prime = lambda x: x > 0
        super().__init__(relu, relu_prime, name)


class Softmax(Activation):
    """
    Softmax activation.
    Converts outputs to probability distribution.
    """
    def __init__(self, name):
        def softmax(x: np.ndarray):
            e = np.exp(x)
            return e / np.sum(e)

        def softmax_prime(x: np.ndarray):
            s = softmax(x)
            prime = []
            for j, xj in enumerate(x):
                p = 0
                for i, si in enumerate(s):
                    deriv = 0
                    if i == j:
                        deriv = si * (1 - si)
                    else:
                        deriv = -si * s[j]
                    p += deriv
                prime.append(p)
            return np.array(prime)

        super().__init__(softmax, softmax_prime, name)

    def backward(self):
        return self.activation_prime(self.input)


if __name__ == "__main__":
    layers = [
        Dense(784, 16, "Input Dense"), # input = (784, 1) | output = (16, 1)
        ReLU("Relu"), # input = (16, 1) | output = (16, 1)
        Dense(16, 10, "Hidden Dense"), # input = (16, 1) | output = (10, 1)
        Softmax("Softmax"), # input = (10, 1) | output = (10, 1)
    ]

    # load data
    labels, mnist = load_mnist_train()
    
    # transform data
    mnist = mnist.T
    labels = np.eye(len(np.unique(labels)))[labels].T

    # get sample
    X = mnist[:, 0:1]
    y = labels[:, 0:1]

    # scale sample
    X /= 255

    # forward prop
    output = X
    for layer in layers:
        output = layer.forward(output)
        print(f"Layer {layer.name} output:  {output.shape}")

    # print("Output")
    # print(output)

    # print("Ground truth")
    # print(y)

    # calculate loss
    loss = cross_entropy(output, y)
    print(f"Cross Entropy Loss: {loss}")

    # back prop
    loss_derivative = cross_entropy_prime(output, y)

    softmax_derivative = layers[3].backward()

    print(loss_derivative * softmax_derivative)
    print(output - y)