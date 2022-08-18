import Layers


class BasicModel:
    def __init__(self):
        self.forward_layers = []
        self.backward_layers = []

    def addLayer(self, layer):

        if issubclass(layer, Layers.Layer):
            print("HHHI")
        self.forward_layers.append(layer)
        self.backward_layers = self.forward_layers.reverse()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss):
        pass
