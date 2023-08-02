import numpy as np
from main import load_random_image
import cv2
import scipy.signal

class ConvLayer:

    def __init__(self, input_shape, n_kernels=10, kernel_size=3):
        # for e.g. input_shape = (3, 28, 28,) for 3 channels and 28x28 pixels per channel
        input_depth, input_height, input_width = input_shape
        self.input_depth = input_depth
        self.input_height = input_height
        self.input_width = input_width

        self.kernel_size = kernel_size
        self.n_kernels = n_kernels

        self.output_shape = (self.n_kernels, self.input_height-self.kernel_size+1, self.input_width-self.kernel_size+1)

        self.kernels = np.random.uniform(-1, 1, (self.n_kernels, self.input_depth, self.kernel_size, self.kernel_size))
        self.biases = np.random.uniform(-1, 1, self.output_shape)

    def forward(self, input):
        output = np.zeros(self.output_shape)

        for i in range(self.n_kernels):
            kernels = self.kernels[i]
            convolved = scipy.signal.correlate(kernels, input, mode="valid")
            output[i] = convolved

        output += self.biases

        return output

    def backward(self):
        pass

class MaxPoolLayer:

    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

if __name__ == "__main__":

    img = cv2.imread("flower.jpg", cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))
    img = np.array(img).reshape((3, 200, 200))

    layer1 = ConvLayer(img.shape, n_kernels=10, kernel_size=3)
    layer2 = ConvLayer(layer1.output_shape, n_kernels=20, kernel_size=3)

    output = layer2.forward(layer1.forward(img))
    print(output.shape)