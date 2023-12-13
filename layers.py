import numpy as np
from helpers import load_mnist, one_hot, softmax, cross_entropy

class Layer:
    """
    Abstract class for a neura network layer.
    """
    def __init__(self, name):
        """
        name : str
            Name of the layer, defaults to "Dense".
        """
        self.name = name

    def forward(self, input: np.ndarray):
        """
        input : np.ndarray
            Input to the layer. Shape = (input_dim, batch)
        """
        self.input = input

    def backward(self, output_gradient: np.ndarray, learning_rate: float):
        """
        output_gradient : np.ndarray
            Derivative of the loss with respect to this layer's outputs.
        
        learning_rate : float
            Learning rate.
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


if __name__ == "__main__":

    # DEFINE LAYERS
    layers = [
        Dense(784, 16, "Input Dense"), # input = (784, 1) | output = (16, 1)
        ReLU("Relu"), # input = (16, 1) | output = (16, 1)
        Dense(16, 10, "Hidden Dense"), # input = (16, 1) | output = (10, 1)
    ]

    # LOAD DATA
    train_X, train_y, val_X, val_y = load_mnist()

    # PREPARE DATA
    train_X = train_X.T / 255 # (784, trainN)
    train_y = one_hot(train_y).T # (10, trainN)
    val_X = val_X.T / 255 # (784, valN)
    val_y = one_hot(val_y).T # (10, valN)

    print(f"Training X:  {train_X.shape}")
    print(f"Training Y:  {train_y.shape}")
    print(f"Val X:  {val_X.shape}")
    print(f"Val Y:  {val_y.shape}")

    # PARAMETERS
    trainN = train_X.shape[-1]
    valN = val_X.shape[-1]
    BATCH_SIZE = 100
    EPOCHS = 100

    for e in range(1, EPOCHS+1):

        # TRAINING LOOP (FORWARD PROP, BACK PROP)

        print(f"Epoch {e}... ", end="")
        train_loss = 0
        train_acc = 0
        for i in range(0, trainN, BATCH_SIZE):
            # get batch
            X = train_X[:, i:i+BATCH_SIZE] # (784, BATCH)
            y = train_y[:, i:i+BATCH_SIZE] # (10, BATCH)
            # forward prop
            output = X
            for layer in layers:
                output = layer.forward(output)
            # softmax
            y_pred = softmax(output)
            # calculate loss
            loss = cross_entropy(y_pred, y)
            train_loss += loss
            # calculate accuracy
            acc = np.sum(np.equal(np.argmax(y_pred, axis=0), np.argmax(y, axis=0)))
            acc /= BATCH_SIZE
            train_acc += acc
            # back prop
            grad = y_pred - y # derivative of the loss with respect to outputs of last dense layer
            learning_rate = 0.01
            for layer in layers[::-1]:
                grad = layer.backward(grad, learning_rate)
        # divide metrics by number of batches
        train_loss = train_loss / ( trainN / BATCH_SIZE ) 
        train_acc = train_acc / ( trainN / BATCH_SIZE ) 
        print(f"Training Loss: {round(train_loss, 5)} | Training Accuracy: {round(train_acc, 3)} | ", end="")

        # VALIDATION LOOP (FORWARD PROP, NO BACK PROP)

        val_loss = 0
        val_acc = 0
        for i in range(0, valN, BATCH_SIZE):
            # get batch
            X = val_X[:, i:i+BATCH_SIZE] # (784, BATCH)
            y = val_y[:, i:i+BATCH_SIZE] # (10, BATCH)
            # forward prop
            output = X
            for layer in layers:
                output = layer.forward(output)
            # softmax
            y_pred = softmax(output)
            # calculate loss
            loss = cross_entropy(y_pred, y)
            val_loss += loss
            # calculate accuracy
            acc = np.sum(np.equal(np.argmax(y_pred, axis=0), np.argmax(y, axis=0)))
            acc /= BATCH_SIZE
            val_acc += acc
        # divide metrics by number of batches
        val_loss = val_loss / ( valN / BATCH_SIZE )
        val_acc = val_acc / ( valN / BATCH_SIZE ) 
        print(f"Validation Loss: {round(val_loss, 5)} | Validation Accuracy: {round(val_acc, 3)}")