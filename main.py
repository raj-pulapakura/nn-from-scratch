import numpy as np
from time import sleep
from layers import Dense, ReLU
from helpers import load_mnist, one_hot, softmax, cross_entropy, print_architecture
from tabulate import tabulate

if __name__ == "__main__":

    delay = 1
    print()
    print("❤️  Neural Network from Scratch with NumPy. Enjoy 🤗")
    print("-----------------------------------------------------")
    sleep(delay)

    # DEFINE LAYERS
    layers = [
        Dense(784, 16, "Hidden Neurons"), # input = (784, BATCH) | output = (16, BATCH)
        ReLU("ReLU"), # input = (16, BATCH) | output = (16, BATCH)
        Dense(16, 10, "Output"), # input = (16, BATCH) | output = (10, BATCH)
    ]

    print("\n🏫  Architecture")
    print("----------------\n")
    sleep(delay)
    print_architecture(layers)
    print()
    sleep(delay)

    # LOAD DATA
    train_X, train_y, val_X, val_y = load_mnist()

    # PREPARE DATA
    train_X = train_X.T / 255 # (784, trainN)
    train_y = one_hot(train_y).T # (10, trainN)
    val_X = val_X.T / 255 # (784, valN)
    val_y = one_hot(val_y).T # (10, valN)

    trainN = train_X.shape[-1]
    valN = val_X.shape[-1]

    print("🔢  MNIST Data")
    print("--------------\n")
    sleep(delay)
    print(f"Training images:  {train_X.shape}")
    sleep(delay)
    print(f"Training labels:  {train_y.shape}")
    sleep(delay)
    print(f"Validation images:  {val_X.shape}")
    sleep(delay)
    print(f"Validation labels:  {val_y.shape}")
    print()
    sleep(delay)

    # PARAMETERS
    BATCH_SIZE = 100
    EPOCHS = 10
    LR = 0.01

    print("⏱️  Hyperparameters")
    print("-------------------\n")
    sleep(delay)
    print(f"Batch Size: {BATCH_SIZE}")
    sleep(delay)
    print(f"Epochs: {EPOCHS}")
    sleep(delay)
    print(f"Learning rate: {LR}")
    print()
    print()
    sleep(delay)

    print("✋  ON YOUR MARKS!")
    sleep(delay+1)
    print("✌️  GET SET!")
    sleep(delay+1)
    print("🏃  TRAIN!")
    print()
    sleep(delay+1)

    for e in range(1, EPOCHS+1):
        
        print(f"Epoch {e}... ", end="")

        # TRAINING LOOP (FORWARD PROP, BACK PROP)

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
            for layer in layers[::-1]:
                grad = layer.backward(grad, LR)
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

    print("\n🏁  Training complete")
    print()
    sleep(delay)

    # CALCULATE FINAL LOSS AND ACCURACY

    print("🏆  Final results")
    print("-----------------\n")
    sleep(delay)

    output = train_X
    for layer in layers:
        output = layer.forward(output)
    # softmax
    y_pred = softmax(output)
    # calculate loss
    train_loss = cross_entropy(y_pred, train_y)
    # calculate accuracy
    train_acc = np.sum(np.equal(np.argmax(y_pred, axis=0), np.argmax(train_y, axis=0)))
    train_acc /= trainN

    output = val_X
    for layer in layers:
        output = layer.forward(output)
    # softmax
    y_pred = softmax(output)
    # calculate loss
    val_loss = cross_entropy(y_pred, val_y)
    # calculate accuracy
    val_acc = np.sum(np.equal(np.argmax(y_pred, axis=0), np.argmax(val_y, axis=0)))
    val_acc /= valN
    
    table = [["Training loss", round(train_loss, 5)],
             ["Validation loss", round(val_loss, 5)],
             ["Training accuracy", f"{round(train_acc, 4)*100}%"],
             ["Validation accuracy", f"{round(val_acc, 4)*100}%"]]
    
    print(tabulate(table, tablefmt='fancy_grid'))
    sleep(delay+1)

    print("\n😍  Done. Hope you enjoyed 😊")
    sleep(delay)