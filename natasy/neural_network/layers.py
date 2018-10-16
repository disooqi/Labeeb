import numpy as np
from scipy.special import expit, logit


class HiddenLayer:
    def __init__(self):
        pass


class ConvLayer(HiddenLayer):
    def __init__(self):
        super().__init__()


class FullyConnectedLayer(HiddenLayer):
    def __init__(self, n_units, n_in, activation='sigmoid', output_layer=False, keep_prob=1):
        super().__init__()
        self.n_units = n_units
        #  It means at every iteration you shut down each neuron of the layer with "1-keep_prob" probability.
        self.keep_prob = keep_prob
        # todo (3): weight initialization happens in the training phase and I guess it should be in a different class than here
        if activation == 'sigmoid':
            self.activation = self.sigmoid
            self.dAdZ = self.sigmoid_prime
            # self.dAdZ = elementwise_grad(self.sigmoid)
            self._weights_initialization(n_in)
        elif activation == 'relu':
            self.activation = self.relu
            self.dAdZ = self.relu_prime
            self._He_initialization(n_in)  # this an Andrew Ng recommendation to use He for relu
        elif activation == 'tanh':
            self.activation = self.tanh
            self.dAdZ = self.tanh_prime
            self._Xavier_initialization(n_in)  # this an Andrew Ng recommendation to use He for leaky_relu
        elif activation == 'leaky_relu':
            self.activation = self.leaky_relu
            self.dAdZ = self.leaky_relu_prime
            self._He_initialization(n_in)
        elif activation == 'softmax':
            self.activation = self.stable_softmax
            self.dAdZ = self.softmax_prime
            # self.dAdZ = elementwise_grad(self.stable_softmax)
            self._weights_initialization(n_in)

        self.activation_type = activation
        self.output_layer = output_layer

    def _weights_initialization(self, n_in):
        # multiplying W by a small number makes the learning fast
        # however from a practical point of view when multiplied by 0.01 using l>2 the NN does not converge
        # that is beacuse it runs into gradients vanishing problem
        self.W = np.random.randn(self.n_units, n_in) * 0.01
        self.b = np.zeros((self.n_units, 1))

    def _He_initialization(self, n_in):
        self.W = np.random.randn(self.n_units, n_in) * np.sqrt(2 / n_in)
        self.b = np.zeros((self.n_units, 1))

    def _Xavier_initialization(self, n_in):
        """Initialize weight W using Xavier Initialization

        So if the input features of activations are roughly mean 0 and standard variance and variance 1 then this would
        cause z to also take on a similar scale and this doesn't solve, but it definitely helps reduce the vanishing,
        exploding gradients problem because it's trying to set each of the weight matrices W so that it's not
        too much bigger than 1 and not too much less than 1 so it doesn't explode or vanish too quickly.
        """
        self.W = np.random.randn(self.n_units, n_in) * np.sqrt(1 / n_in)
        self.b = np.zeros((self.n_units, 1))

    def _Benjio_initialization(self, n_in):
        self.W = np.random.randn(self.n_units, n_in) * np.sqrt(2 / (n_in + self.n_units))
        self.b = np.zeros((self.n_units, 1))

    @staticmethod
    def softmax(Z):
        """Compute softmax of Matrix Z

        :param Z: is in the shape of (n * m), where n is the number of classes and m is the number of examples
        :return:
        """
        Z_exp = np.exp(Z)
        return Z_exp / np.sum(Z_exp, axis=0)

    @staticmethod
    def stable_softmax(Z):
        """Compute the softmax of vector Z in a numerically stable way."""

        shift_Z = Z - np.max(Z, axis=0)
        Z_exp = np.exp(shift_Z)
        return Z_exp / np.sum(Z_exp, axis=0)

    @staticmethod
    def softmax_prime(A):
        """N/A

        https://stackoverflow.com/questions/40575841/numpy-calculate-the-derivative-of-the-softmax-function
        https://stackoverflow.com/questions/26511401/numpy-fastest-way-of-computing-diagonal-for-each-row-of-a-2d-array
        https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        # Kronecker delta function
        :param A:
        :return:
        """
        b = np.zeros((A.shape[0], A.shape[0], A.shape[1]))
        diag = np.arange(A.shape[0])
        b[diag, diag, :] = A

        SM = A.reshape((-1, 1))
        jac = np.diag(A) - np.dot(SM, SM.T)
        return jac

    @staticmethod
    def sigmoid(Z):
        # https://docs.scipy.org/doc/scipy/reference/generated /scipy.special.expit.html
        # return 1 / (1 + np.exp(-Z))
        return expit(np.clip(Z, -709, 36.73))

    @classmethod
    def sigmoid_prime(cls, A):
        """ calculate dAdZ

        :param A:
        :return: dAdZ
        """
        return A * (1 - A)

    @staticmethod
    def tanh(Z):
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

    @classmethod
    def tanh_prime(cls, A):
        return 1 - A ** 2

    @staticmethod
    def relu(Z):
        # a[a<0] = 0
        # return np.clip(Z, 0, Z)
        return np.maximum(Z, 0)

    @staticmethod
    def relu_prime(A):
        A[A > 0] = 1
        return A

    @staticmethod
    def leaky_relu(Z, alpha=0.01):
        """
        :param Z:
        :param alpha: Slope of the activation function at x < 0.
        :return:

        """
        # return np.clip(Z, alpha * Z, Z)
        return np.where(Z < 0, alpha * Z, Z)

    @staticmethod
    def leaky_relu_prime(A, alpha=0.01):
        return np.where(A > 0, 1, alpha)

    def __repr__(self):
        return 'FullyConnectedLayer(n_units={0.n_units}, activation={0.activation_type}, output_layer=' \
               '{0.output_layer}, keep_prob={0.keep_prob})'.format(self)


if __name__ == '__main__':
    pass