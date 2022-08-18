import numpy as np
from Utils import log, create_random_sample
from Layers import FullyConnectedLayer
from Activations.Sigmoid import Sigmoid
from Activations.Softmax import Softmax
from Loss.BinaryCrossEntropyLoss import BCELoss
from Models import BasicModel

log_flag = True

X, y = create_random_sample(10, 4)
y = y.reshape(-1,1)
log("Data samples : " + str(X), log_flag)
log("Data samples shape : " + str(X.shape), log_flag)

FC_layer1 = FullyConnectedLayer.FCLayer(4, 3, 0, Sigmoid())
FC_layer2 = FullyConnectedLayer.FCLayer(3, 1, 0, Sigmoid())
loss = BCELoss()
model = BasicModel.BasicModel()
model.addLayer(FC_layer1)
log(loss.caculate_loss(FC_layer2.forward(FC_layer1.forward(X)), y), log_flag)


