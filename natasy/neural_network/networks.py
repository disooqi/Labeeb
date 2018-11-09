import numpy as np
from .layers import FullyConnectedLayer


class NeuralNetwork:
    def __init__(self, n_features, n_classes):
        self.n = n_features
        self.n_classes = n_classes
        self.layers = list()

    def add_layer(self, n_units, activation='sigmoid', dropout_keep_prob=1):
        if self.layers:
            n_units_previous_layer = self.layers[-1].n_units
        else:
            n_units_previous_layer = self.n

        layer = FullyConnectedLayer(n_units, n_units_previous_layer, activation=activation, keep_prob=dropout_keep_prob)

        self.layers.append(layer)

    def add_output_layer(self, activation='sigmoid'):
        if self.n_classes == 2:
            n_units = 1
        elif self.n_classes < 2:
            pass # TODO: This should be an error on the data
        else:
            n_units = self.n_classes

        if not self.layers:
            self.add_layer(n_units=n_units, activation=activation)
            self.layers[-1].output_layer = True
        elif not self.layers[-1].output_layer:
            self.add_layer(n_units=n_units, activation=activation)
            self.layers[-1].output_layer = True
        else:
            # TODO: you should raise an error and message that says you need to delete existing output_layer
            pass

    @staticmethod
    def calculate_single_layer_gradients(dLdA, layer_cache, compute_dLdA_1=True):
        # todo (4): this function works in the training phase and I guess it should be in a different class than here
        '''
        :param dLdA:
        :return: dJdA_1, dJdW, dJdb
        '''
        # For the first iteration where loss is cross entropy and activation func of output layer
        # is sigmoid, that could be shorten to,
        # dZ[L] = A[L]-Y
        # In general, you can compute dZ as follows
        # dZ = dA * g'(Z) TODO: currently we pass A instead of Z, I guess it is much better to follow "A. Ng" and pass Z

        # During forward propagation, you had divided A1 by keep_prob. In backpropagation, you'll therefore have to
        # divide dA1 by keep_prob again (the calculus interpretation is that if  A[1]A[1]  is scaled by keep_prob, then
        # its derivative  dA[1]dA[1]  is also scaled by the same keep_prob).
        dLdA = np.multiply(dLdA, layer_cache.D) / layer_cache.keep_prob
        dAdZ = layer_cache.dAdZ(layer_cache.A)

        if len(dAdZ.shape) == 3:
            dLdZ = np.einsum('ik,ijk->jk',dLdA, dAdZ) # dot product (element-wise and then sum over columns)
        else:
            dLdZ = dLdA * dAdZ  # Element-wise product

        # dw = dz . a[l-1]
        dZdW = layer_cache.A_l_1

        # this is two steps in one line; getting dLdw and then dJdW
        # if you want to elaborate on that,
        # then dLdW = dLdZ * dZdW
        # followed by dJdW = np.sum(dLdW, axis=1, keepdims=True) / dLdA.shape[1]
        # see https://www.coursera.org/learn/neural-networks-deep-learning/lecture/udiAq/gradient-descent-on-m-examples
        # dLdA.shape[1] is m and dLdZ, dZdW is n*m dimension
        dJdW = np.dot(dLdZ, dZdW.T) / dLdA.shape[1]
        dJdb = np.sum(dLdZ, axis=1, keepdims=True) / dLdA.shape[1]
        dLdA_1 = None
        if compute_dLdA_1:
            # da[l-1] = w[l].T . dz[l]
            dZdA_1 = layer_cache.W
            dLdA_1 = np.dot(dZdA_1.T, dLdZ)  # computing dLd(A-1)
        return dLdA_1, dJdW, dJdb




if __name__ == '__main__':
    pass