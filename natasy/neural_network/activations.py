import numpy as np
from scipy.special import expit, logit
from .initializations import _zeros_initialization, _weights_initialization, _Xavier_initialization, \
    _He_initialization, _Benjio_initialization


class Activation:
    sigmoid = None
    sigmoid_prime = None
    tanh = None
    tanh_prime = None
    relu = None
    relu_prime = None
    leaky_relu = None
    leaky_relu_prime = None
    softmax = None
    softmax_prime = None
    softmax_stable = None

    # sigmoid = lambda Z: expit(np.clip(Z, -709, 36.73))
    # sigmoid_prime = lambda A: A * (1 - A)
    # tanh = lambda Z: (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    # tanh_prime = lambda A: 1 - A ** 2
    # relu = lambda Z: np.maximum(Z, 0)
    # relu_prime = lambda A: Activation.__relu_prime(A)
    # leaky_relu = lambda Z, alpha=0.01: np.where(Z < 0, alpha * Z, Z)
    # leaky_relu_prime = lambda A, alpha=0.01: np.where(A > 0, 1, alpha)


def _sigmoid(Z):
    # https://docs.scipy.org/doc/scipy/reference/generated /scipy.special.expit.html
    # return 1 / (1 + np.exp(-Z))
    return expit(np.clip(Z, -709, 36.73))


def _sigmoid_prime(A):
    """ calculate dAdZ

    :param A:
    :return: dAdZ
    """
    return A * (1 - A)


def _tanh(Z):
    a = np.exp(Z)
    b = np.exp(-Z)
    return (a - b) / (a + b)


def _tanh_prime(A):
    return 1 - A ** 2


def _relu(Z):
    # a[a<0] = 0
    # return np.clip(Z, 0, Z)
    return np.maximum(Z, 0)


def _relu_prime(A):
    A[A > 0] = 1
    return A


def _leaky_relu(Z, alpha=0.01):
    """
    :param Z:
    :param alpha: Slope of the activation function at x < 0.
    :return:

    """
    # return np.clip(Z, alpha * Z, Z)
    return np.where(Z < 0, alpha * Z, Z)


def _leaky_relu_prime(A, alpha=0.01):
    return np.where(A > 0, 1, alpha)


def _softmax(Z):
    """Compute softmax of Matrix Z

    :param Z: is in the shape of (n * m), where n is the number of classes and m is the number of examples
    :return:
    """
    Z_exp = np.exp(Z)
    return Z_exp / np.sum(Z_exp, axis=0)


def _stable_softmax(Z):
    """Compute the softmax of vector Z in a numerically stable way."""

    shift_Z = Z - np.max(Z, axis=0)
    Z_exp = np.exp(shift_Z)
    return Z_exp / np.sum(Z_exp, axis=0)


def _softmax_prime(S):
    """Computes the gradient of the softmax function.

    https://stackoverflow.com/questions/40575841/numpy-calculate-the-derivative-of-the-softmax-function
    https://stackoverflow.com/questions/26511401/numpy-fastest-way-of-computing-diagonal-for-each-row-of-a-2d-array
    https://stackoverflow.com/questions/41469647/outer-product-of-each-column-of-a-2d-array-to-form-a-3d-array-numpy
    https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    # Kronecker delta function
    :param S: (T, 1) array of input values where the gradient is computed. T is the
       number of output classes.
    :return: dAdZ (T, T) the Jacobian matrix of softmax(Z) at the given Z. D[i, j]
    is DjSi - the partial derivative of Si w.r.t. input j.
    """
    # -SjSi can be computed using an outer product between Sz and itself. Then
    # we add back Si for the i=j cases by adding a diagonal matrix with the
    # values of Si on its diagonal.
    dAdZ_struct = np.zeros((S.shape[0], S.shape[0], S.shape[1]))
    diag_indecies = np.arange(S.shape[0])
    dAdZ_struct[diag_indecies, diag_indecies, :] = S

    # D = -np.outer(S, S) + np.diag(S.flatten())
    dAdZ = dAdZ_struct - S[:, None, :] * S
    return dAdZ


Activation.sigmoid = _sigmoid
Activation.sigmoid.prime = _sigmoid_prime
# self.dAdZ = elementwise_grad(self.sigmoid)
Activation.sigmoid.recommended_initialization = _Xavier_initialization

Activation.tanh = _tanh
Activation.tanh.prime = _tanh_prime
Activation.tanh.recommended_initialization = _Xavier_initialization

Activation.relu = _relu
Activation.relu.prime = _relu_prime
Activation.relu.recommended_initialization = _He_initialization # this an Andrew Ng recommendation to use He for relu

Activation.leaky_relu = _leaky_relu
Activation.leaky_relu.prime = _leaky_relu_prime
Activation.leaky_relu.recommended_initialization = _He_initialization # A. Ng recommendation to use He for leaky_relu

Activation.softmax = _softmax
Activation.softmax.prime = _softmax_prime
Activation.softmax.recommended_initialization = _Xavier_initialization

Activation.softmax_stable = _stable_softmax
Activation.softmax_stable.prime = _softmax_prime
Activation.softmax_stable.recommended_initialization = _Xavier_initialization


if __name__ == '__main__':
    print(dir(Activation.sigmoid))