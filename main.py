import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal


def load_random_image(dataset_path):
    """
    Returns a random 28x28 MNIST image (hand-written digit)
    """

    # read dataset as pandas dataframe
    df = pd.read_csv(dataset_path)
    # get random index
    index = np.random.randint(0, len(df))
    # get the image vector, we start from the 1st column as the 0th column is the label
    image_vector = np.array(df.iloc[index, 1:])
    # reshape the vector into a matrix (image)
    image_matrix = image_vector.reshape((28, 28))
    return image_matrix


def cross_correlation(img, filter):
    return scipy.signal.correlate(img, filter, mode="valid")


def maxpool(img, pool_size=2, stride=2):
    m, n = img.shape

    if m != n:
        raise Exception(f"Input dimensions must be equal. Received image of shape: {img.shape}")

    # check if the image has odd dimensions
    if m % 2 == 1:
        # add extra dimensions of zeroes on all edges
        img = np.pad(img, 1, mode="constant", constant_values=[0])
        # crop so that the extra dimension is only on the right and bottom edges
        img = img[1:, 1:]

    pooled_img = []
    p_i = 0

    for i in range(0, img.shape[0], stride):
        pooled_img.append([])
        for j in range(0, img.shape[1], stride):
            patch = img[i:i+pool_size, j:j+pool_size]
            pooled_img[p_i].append(patch.max())
        p_i +=1        

    pooled_img = np.array(pooled_img)
    return pooled_img

if __name__ == "__main__":

    img = load_random_image("mnist_train.csv")
    print(f"Image shape: {img.shape}")
   
    maxpool(img)
   
    filter = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])

    convolved1 = cross_correlation(img, filter)
    print(f"Convolved 1: {convolved1.shape}")
    pooled1 = maxpool(convolved1)
    print(f"Pooled 1: {pooled1.shape}")

    convolved2 = cross_correlation(pooled1, filter)
    print(f"Convolved 2: {convolved2.shape}")
    pooled2 = maxpool(convolved2)
    print(f"Pooled 2: {pooled2.shape}")

    convolved3 = cross_correlation(pooled2, filter)
    print(f"Convolved 3: {convolved3.shape}")
    pooled3 = maxpool(convolved3)
    print(f"Pooled 3: {pooled3.shape}")
