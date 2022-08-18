import numpy as np
from Layers import Layer


class Softmax(Layer):
    def forward(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1,1)

    def backward(self, loss):
        return loss