import numpy as np
from natasy.neural_network.layers.neuralnetworklayer import NeuralNetworkLayer


class FullyConnectedLayer(NeuralNetworkLayer):
    # def __init__(self, n_units, n_in, initialization, activation=Activation.sigmoid, output_layer=False, keep_prob=1):
    def __init__(self, n_units, n_in, *args, **kwargs):
        super(FullyConnectedLayer, self).__init__(n_units, n_in, *args, **kwargs)
        self.n_units = n_units
        #  It means at every iteration you shut down each neuron of the layer with "1-keep_prob" probability.
        self.keep_prob = kwargs.get('keep_prob', 1)
        # TODO (3): weight initialization happens in the training phase and I guess it should be in a different class than here
        self.activation = kwargs.get('activation')
        self.dAdZ = self.activation.prime
        # print(activation.__name__)

        if not kwargs.get('initialization'):
            self.W, self.b = self.activation.recommended_initialization(n_units, n_in)
        else:
            # TODO: raise error no Weight Intialization avaialable
            raise Exception

        self.output_layer = kwargs.get('output_layer', False)
        self.D = None
        self.A = None
        self.A_l_1 = None

    def calculate_layer_feedforward(self, A_1):
        self.A_l_1 = A_1  # this is A-1 from last loop step
        Z = np.dot(self.W, A_1) + self.b  # (called "logits" in ML folklore)
        A = self.activation(Z)

        # NB! we don't not apply dropout to the input layer or output layer.
        D = np.random.rand(*A.shape) <= self.keep_prob  # dropout
        A = np.multiply(A, D) / self.keep_prob  # inverted dropout

        self.D = D
        self.A = A

        return A

    def calculate_layer_gradients(self, dLdA, *args, compute_dLdA_1=True, **kwargs):
        """
                :param dLdA:
                :return: dJdA_1, dJdW, dJdb
        """
        # For the first iteration where loss is cross entropy and activation func of output layer
        # is sigmoid, that could be shorten to,
        # dZ[L] = A[L]-Y
        # In general, you can compute dZ as follows
        # dZ = dA * g'(Z) TODO: currently we pass A instead of Z, I guess it is much better to follow "A. Ng" and pass Z

        # During forward propagation, you had divided A1 by keep_prob. In backpropagation, you'll therefore have to
        # divide dA1 by keep_prob again (the calculus interpretation is that if  A[1]A[1]  is scaled by keep_prob, then
        # its derivative  dA[1]dA[1]  is also scaled by the same keep_prob).

        dLdA = np.multiply(dLdA, self.D) / self.keep_prob
        dAdZ = self.dAdZ(self.A)

        if len(dAdZ.shape) == 3:
            dLdZ = np.einsum('ik,ijk->jk',dLdA, dAdZ) # dot product (element-wise and then sum over columns)
        else:
            dLdZ = dLdA * dAdZ  # Element-wise product

        # dw = dz . a[l-1]
        dZdW = self.A_l_1
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
            dZdA_1 = self.W
            dLdA_1 = np.dot(dZdA_1.T, dLdZ)  # computing dLd(A-1)
        return dLdA_1, dJdW, dJdb

    def __repr__(self):
        return 'FullyConnectedLayer(n_units={0.n_units}, activation={0.activation.__name__}, output_layer=' \
               '{0.output_layer}, keep_prob={0.keep_prob})'.format(self)


if __name__ == '__main__':
    pass