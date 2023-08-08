import pandas as pd
import numpy as np

def load_mnist_train():
    df = pd.read_csv("data/mnist_train.csv")
    labels = df.pop("label")
    return labels.to_numpy(), df.to_numpy().astype(np.float32)




if __name__ == "__main__":

    # x = np.array([5, 7, 1])

    # s = softmax(x)
    # print(s)

    # s_prime = softmax_prime(x)
    # print(s_prime)

    x = [1, 2, 3, 4, 5]
    print(x[::-1])