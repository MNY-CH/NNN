import numpy as np
from Layers import Layer

class Sigmoid(Layer):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, loss):
        return self.forward(loss) * (1 - self.forward(loss))