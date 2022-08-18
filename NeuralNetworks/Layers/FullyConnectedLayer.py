import numpy as np
import Layers
class FCLayer(Layers.Layer):
    def __init__(self, input_shape, output_shape, init_option, activation=None):
        self.weight = np.ones((input_shape, output_shape))
        self.bias = np.ones((1, output_shape))
        self.activation = activation

    def forward(self, input_x):
        if self.activation is not None:
            return self.activation.forward((input_x @ self.weight) + self.bias)
        else:
            return (input_x @ self.weight) + self.bias

    def backward(self, loss):
        return loss
