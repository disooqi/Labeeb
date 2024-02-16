import numpy as np
np.random.seed(0)


class Initialization:
    zeros_initialization = None


def _zeros_initialization(n_units: int, n_in: int):
    W = np.zeros((n_units, n_in))
    b = np.zeros((n_units, 1))
    return W, b


def _weights_initialization(n_units, n_in):
    # multiplying W by a small number makes the learning fast
    # however from a practical point of view when multiplied by 0.01 using l>2 the NN does not converge
    # that is because it runs into gradients vanishing problem
    W = np.random.randn(n_units, n_in) * 0.01
    b = np.zeros((n_units, 1))
    return W, b


def _He_initialization(n_units, n_in):
    """ Goes better with ReLU (a generalization this initializer is called variance_scaling_initializer)

    :param n_units:
    :param n_in:
    :return:
    """
    W = np.random.randn(n_units, n_in) * np.sqrt(2 / n_in)
    b = np.zeros((n_units, 1))
    return W, b


def _Xavier_initialization(n_units, n_in):
    """Initialize weight W using Xavier Initialization (also known as Glorot Initialization)

    So if the input features of activations are roughly mean 0 and standard variance and variance 1 then this would
    cause z to also take on a similar scale and this doesn't solve, but it definitely helps reduce the vanishing,
    exploding gradients problem because it's trying to set each of the weight matrices W so that it's not
    too much bigger than 1 and not too much less than 1 so it doesn't explode or vanish too quickly.
    P.S. Goes better with Sigmoid and Softmax and tanh
    """
    W = np.random.randn(n_units, n_in) * np.sqrt(1 / n_in)
    b = np.zeros((n_units, 1))
    return W, b


def _Benjio_initialization(n_units, n_in):
    W = np.random.randn(n_units, n_in) * np.sqrt(2 / (n_in + n_units))
    b = np.zeros((n_units, 1))
    return W, b


if __name__ == '__main__':
    pass
