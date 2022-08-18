import numpy as np


class BCELoss:
    def __init__(self):
        pass
    def caculate_loss(self, predict, lable):
        return -1 * (lable * np.log(predict) + (1 - lable) * np.log(1 - predict))