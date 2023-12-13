import numpy as np


class Layer:
    """
    Abstract class for a neural network layer.
    """

    def __init__(self, name):
        """
        Parameters
        ----------

        name : str
            Name of the layer.
        """
        self.name = name

    def forward(self, input: np.ndarray):
        """
        Forward propagation.

        Parameters
        ----------

        input : np.ndarray
            Input to the layer. Shape = (input_dim, batch)

        Returns
        -------

        Tensor through forward propagation of input.
        """
        self.input = input

    def backward(self, output_gradient: np.ndarray, learning_rate: float):
        """
        Backward propagation.

        Parameters
        ----------

        output_gradient : np.ndarray
            Derivative of loss with respect to this layer's outputs.
        
        learning_rate : float
            Learning rate.

        Returns
        -------

        Derivative of the loss with respect to this layer's inputs. 
        """
        pass


class Dense(Layer):
    """
    Fully Connected Layer of Neurons.
    """
    
    def __init__(self, input_size: int, output_size: int, name:str="Dense"):
        """
        input_size : int
            Number of input dimensions to the layer.
        
        output_size: int
            Number of hidden neurons.

        name : int
            Name of the layer, defaults to "Dense".
        """
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2/input_size)
        self.bias = np.zeros((output_size, 1))
        super().__init__(name)

    def forward(self, input: np.ndarray):
        self.input = input
        return np.dot(self.weights, input) + self.bias
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float):
        m = self.input.shape[-1] # get batch size from input
        grad_weights = (1/m) * output_gradient @ self.input.T
        grad_bias = (1/m) * np.sum(output_gradient, axis=-1, keepdims=True)
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        return self.weights.T @ output_gradient # return the derivative of the loss with respect to inputs


class Activation(Layer):
    """
    Activation function.
    """

    def __init__(self, activation, activation_prime, name):
        self.activation = activation
        self.activation_prime = activation_prime
        self.name = name

    def forward(self, input: np.ndarray):
        self.input = input
        return self.activation(input)


class ReLU(Activation):
    """
    ReLU activation.
    Defined as max(x, 0).
    """

    def __init__(self, name):
        relu = lambda x: np.maximum(x, 0)
        relu_prime = lambda x: x > 0
        super().__init__(relu, relu_prime, name)

    def backward(self, output_gradient: np.ndarray, learning_rate: float):
        return output_gradient  * self.activation_prime(self.input)