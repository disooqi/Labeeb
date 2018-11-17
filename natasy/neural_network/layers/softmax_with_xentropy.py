import numpy as np
from natasy.neural_network.layers import OutputLayer


class SoftmaxWithXEntropy(OutputLayer):
    def __init__(self, n_units, n_in, *args, **kwargs):
        super(SoftmaxWithXEntropy, self).__init__(n_units, n_in, *args, **kwargs)

    def calculate_layer_gradients(self, *args, **kwargs):
        y = kwargs.get('y', None)
        dLdZ = self.A-y

        dZdW = self.A_l_1

        dJdW = np.dot(dLdZ, dZdW.T) / dLdZ.shape[1]
        dJdb = np.sum(dLdZ, axis=1, keepdims=True) / dLdZ.shape[1]

        dZdA_1 = self.W
        dLdA_1 = np.dot(dZdA_1.T, dLdZ)  # computing dLd(A-1)

        return dLdA_1, dJdW, dJdb


if __name__ == '__main__':
    SoftmaxWithXEntropy(25, 10)