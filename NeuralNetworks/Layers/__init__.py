from abc import *


class Layer:
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass
