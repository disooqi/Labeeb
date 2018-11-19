import numpy as np
from natasy.neural_network.layers import OutputLayer


class SoftmaxWithXEntropy(OutputLayer):
    """

    Calculating exponentials and logarithmics are computationally expensive. As we can see from the previous 2 parts,
    the softmax layer is raising the logit scores to exponential in order to get probability vectors, and then the loss
    function is doing the log to calculate the entropy of the loss.

    If we combine these 2 stages in one layer, logarithmics and exponentials kind of cancel out each others, and we can
    get the same final result with much less computational resources. That’s why in many neural network frameworks and
    libraries there is a “softmax-log-loss” function, which is much more optimal than having the 2 functions separated
    """
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