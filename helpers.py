import numpy as np
import mnist

def load_mnist()->tuple[np.ndarray]:
    """
    Loads MNIST dataset (train and validation sets) using the `mnist` package (https://github.com/datapythonista/mnist).

    Parameters
    ----------

    None

    Returns
    -------

    Tuple of (training_images, training_labels, validation_images, validation_labels)

    Shapes:
    - training_images: (trainN, 784)
    - training_labels: (trainN)
    - validation_labels: (valN, 784)
    - validation_labels: (valN)
    """

    train_X = mnist.train_images()
    train_y = mnist.train_labels()

    val_X = mnist.test_images()
    val_y = mnist.test_labels()

    return (
        train_X.reshape(train_X.shape[0], -1), # reshape from (N, 28, 28) to (N, 784)
        train_y, 
        val_X.reshape(val_X.shape[0], -1), # reshape from (N, 28, 28) to (N, 784)
        val_y
    )

def one_hot(y: np.ndarray)->np.ndarray:
    """
    One hot encodes target variable.

    Parameters
    ----------

    y : np.ndarray
        Target variable.

    Returns
    -------

    One hot encoded target variable.
    """
    return np.eye(len(np.unique(y)))[y]

def softmax(logits: np.ndarray)->np.ndarray:
    """
    Computes stable softmax; computes probabilites from logits.

    Parameters
    ----------

    logits : np.ndarray
        Output of last dense layer. 
        
        Shape = (n_classes, m)

    Returns
    -------

    Probabilities for each target class.
    
    Shape = (n_classes, m)
    """
    logits = logits - np.max(logits, axis=0) # this is to avoid high values (it doesn't change the outputs)
    e = np.exp(logits)
    s = e / np.sum(e, axis=0)
    return s

def cross_entropy(y_pred: np.ndarray, y_true: np.ndarray)->int:
    """
    Computes cross entropy loss between predictions and ground truth.

    Parameters
    ----------

    y_pred : np.ndarray
        Probabilities for predictions.

        Shape = (n_classes, m)

    y_true : np.ndarray
        Ground truth.

        Shape = (n_classes, m)

    Returns
    -------

    Cross entropy loss.
    """
    m = y_pred.shape[-1]
    log_probs =  (-1) * np.sum( y_true * np.log(y_pred+1e-7) )
    loss = log_probs / m
    return float(loss)