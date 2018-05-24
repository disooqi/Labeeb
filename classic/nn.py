import numpy as np
from scipy.special import expit, logit
import time
import logging

np.random.seed(4)  # 4
logger = logging.getLogger(__name__)
fr = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
sh = logging.StreamHandler()
# sh.setFormatter(fr)
logger.addHandler(sh)
logger.setLevel(logging.DEBUG)

logger2 = logging.getLogger('other')
file_handler = logging.FileHandler('run.log')
file_handler.setFormatter(fr)
file_handler.setLevel(logging.INFO)
logger2.addHandler(file_handler)

logger2.setLevel(logging.INFO)


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
        #  It means at every iteration you shut down each neurons of the layer with "1-keep_prob" probability.
        self.keep_prob = keep_prob
        # todo (3): weight initialization should be in the Network class
        if activation == 'sigmoid':
            self.activation = self.sigmoid
            self.dAdZ = self.sigmoid_prime
            self._weights_initialization(n_in)
        elif activation == 'relu':
            self.activation = self.relu
            self.dAdZ = self.relu_prime
            self._He_initialization(n_in)
        elif activation == 'tanh':
            self.activation = self.tanh
            self.dAdZ = self.tanh_prime
            self._Xavier_initialization(n_in)
        elif activation == 'leaky_relu':
            self.activation = self.leaky_relu
            self.dAdZ = self.leaky_relu_prime
            self._He_initialization(n_in)

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
        return Z_exp/np.sum(Z_exp, axis=0)

    @staticmethod
    def stable_softmax(Z):
        """Compute the softmax of vector Z in a numerically stable way."""

        shift_Z = Z - np.max(Z, axis=0)
        Z_exp = np.exp(shift_Z)
        return Z_exp / np.sum(Z_exp, axis=0)

    @staticmethod
    def softmax_prime(A):
        """N/A

        https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        # Kronecker delta function
        :param A:
        :return:
        """
        pass

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
        '''
        :param Z:
        :param alpha: Slope of the activation function at x < 0.
        :return:

        '''
        # return np.clip(Z, alpha * Z, Z)
        return np.where(Z < 0, alpha * Z, Z)

    @staticmethod
    def leaky_relu_prime(A, alpha=0.01):
        return np.where(A > 0, 1, alpha)

    def __repr__(self):
        return 'FullyConnectedLayer(n_units={}, activation={}, output_layer={}, keep_prob={})'.format(
            self.n_units, self.activation_type, self.output_layer, self.keep_prob)


class NN:
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

    def add_output_layer(self):
        if not self.layers:
            self.add_layer(self.n_classes, activation='sigmoid')
            self.layers[-1].output_layer = True
        if not self.layers[-1].output_layer:
            self.add_layer(self.n_classes, activation='sigmoid')
            self.layers[-1].output_layer = True
        else:
            # TODO: you should raise an error and message that says you need to delete existing output_layer
            pass

    @staticmethod
    def _calculate_single_layer_gradients(dLdA, layer_cache, compute_dLdA_1=True):
        '''
        :param dJdA:
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
        dLdZ = dLdA * dAdZ  # Element-wise product

        # dw = dz . a[l-1]
        dZdW = layer_cache.A_l_1
        dJdW = np.dot(dLdZ, dZdW.T) / dLdA.shape[1]  # this is two steps in one line; getting dLdw and then dJdW
        dJdb = np.sum(dLdZ, axis=1, keepdims=True) / dLdA.shape[1]
        dLdA_1 = None
        if compute_dLdA_1:
            # da[l-1] = w[l].T . dz[l]
            dZdA_1 = layer_cache.W
            dLdA_1 = np.dot(dZdA_1.T, dLdZ)  # computing dLd(A-1)
        return dLdA_1, dJdW, dJdb

    def accuracy(self, X, y):
        # You only use dropout during training. Don't use dropout (randomly eliminate nodes) during test time.
        A = X
        for layer in self.layers:
            Z = np.dot(layer.W, A) + layer.b
            A = layer.activation(Z)
        else:
            y = y.argmax(axis=0) + 1
            prediction = A.argmax(axis=0) + 1
            res = np.equal(prediction, y)
            return 100 * np.sum(res) / y.size


class Optimization:
    def __init__(self, loss='cross_entropy', method='gradient-descent'):
        self.method = method
        self.VsnSs = list()
        if loss == 'cross_entropy':
            self.loss = self.cross_entropy_loss
            self.activation_prime = self.cross_entropy_loss_prime

        if method == 'gradient-descent':
            self.optimizer = self.gradient_descent
        elif method == 'gd-with-momentum':
            self.optimizer = self.gradient_descent_with_momentum
        elif method == 'rmsprop':
            self.optimizer = self.RMSprop
        elif method == 'adam':
            self.optimizer = self.adam

    @staticmethod
    def weight_decay(m, alpha, lmbda):
        # L2 Regularization
        return 1 - ((alpha * lmbda) / m)

    @staticmethod
    def learning_rate_decay(decay_rate, epoch_num):
        return 1/(1+decay_rate*epoch_num)

    @staticmethod
    def o_learning_rate_decay(k, epoch_num):
        return k/np.sqrt(epoch_num)

    @staticmethod
    def exponential_learning_rate_decay(decay_rate, epoch_num):
        return 0.95**epoch_num

    def discrete_staircase_learning_rate_decay(self):
        pass

    @classmethod
    def learning_rate(cls, alpha, decay_rate, epoch_num):
        return cls.learning_rate_decay(decay_rate, epoch_num) * alpha
    @classmethod
    def gradient_descent(cls, dJdW, dJdb, W, b, m, **kwargs):
        alpha0 = kwargs['alpha']
        lmbda = kwargs['lmbda']
        epoch_num = kwargs['epoch']
        decay_rate = kwargs['decay_rate']
        alpha = cls.learning_rate(alpha0, decay_rate, epoch_num)
        W = cls.weight_decay(m, alpha, lmbda) * W - alpha * dJdW
        b -= alpha * dJdb

        return W, b

    @classmethod
    def gradient_descent_with_momentum(cls, dJdW, dJdb, W, b, m, **kwargs):
        beta1 = kwargs['beta1']
        Vs = kwargs['VS']
        alpha0 = kwargs['alpha']
        lmbda = kwargs['lmbda']
        epoch_num = kwargs['epoch']
        decay_rate = kwargs['decay_rate']
        alpha = cls.learning_rate(alpha0, decay_rate, epoch_num)
        Vs['Vdw'] = beta1*Vs['Vdw'] + (1-beta1)*dJdW
        Vs['Vdb'] = beta1*Vs['Vdb'] + (1-beta1)*dJdb

        W = cls.weight_decay(m, alpha, lmbda) * W - alpha * Vs['Vdw']
        b = b - alpha * Vs['Vdb']

        return W, b

    @classmethod
    def RMSprop(cls, dJdW, dJdb, W, b, m, **kwargs):
        beta2 = kwargs['beta2']
        Ss = kwargs['VS']
        alpha0 = kwargs['alpha']
        lmbda = kwargs['lmbda']
        epoch_num = kwargs['epoch']
        decay_rate = kwargs['decay_rate']
        alpha = cls.learning_rate(alpha0, decay_rate, epoch_num)
        epsilon = np.finfo(np.float32).eps
        Ss['Sdw'] = beta2 * Ss['Sdw'] + (1 - beta2) * np.square(dJdW)
        Ss['Sdb'] = beta2 * Ss['Sdb'] + (1 - beta2) * np.square(dJdb)

        W = cls.weight_decay(m, alpha, lmbda)*W - alpha * (dJdW/(np.sqrt(Ss['Sdw'])+epsilon))
        b = b - alpha * (dJdb/(np.sqrt(Ss['Sdb'])+epsilon))

        return W, b

    @classmethod
    def adam(cls, dJdW, dJdb, W, b, m, **kwargs):
        beta1 = kwargs['beta1']
        beta2 = kwargs['beta2']
        VsSs = kwargs['VS']
        t = kwargs['t']
        alpha0 = kwargs['alpha']
        lmbda = kwargs['lmbda']
        epoch_num = kwargs['epoch']
        decay_rate = kwargs['decay_rate']
        alpha = cls.learning_rate(alpha0, decay_rate, epoch_num)
        epsilon = np.finfo(np.float32).eps

        VsSs['Vdw'] = beta1 * VsSs['Vdw'] + (1 - beta1) * dJdW
        VsSs['Vdb'] = beta1 * VsSs['Vdb'] + (1 - beta1) * dJdb
        VsSs['Sdw'] = beta2 * VsSs['Sdw'] + (1 - beta2) * np.square(dJdW)
        VsSs['Sdb'] = beta2 * VsSs['Sdb'] + (1 - beta2) * np.square(dJdb)

        Vdw_corrected = VsSs['Vdw']/(1-beta1**t)
        Vdb_corrected = VsSs['Vdb']/(1-beta1**t)
        Sdw_corrected = VsSs['Sdw']/(1-beta2**t)
        Sdb_corrected = VsSs['Sdb']/(1-beta2**t)

        W = cls.weight_decay(m, alpha, lmbda) * W - alpha * (Vdw_corrected / (np.sqrt(Sdw_corrected) + epsilon))
        b = b - alpha * (Vdb_corrected / (np.sqrt(Sdb_corrected) + epsilon))

        return W, b

    @staticmethod
    def cross_entropy_loss(y, a):
        # http://christopher5106.github.io/deep/learning/2016/09/16/about-loss-functions-multinomial-logistic-logarithm-cross-entropy-square-errors-euclidian-absolute-frobenius-hinge.html
        # https://stats.stackexchange.com/questions/260505/machine-learning-should-i-use-a-categorical-cross-entropy-or-binary-cross-entro
        # here we penalize every class even the zero ones
        # the classes here are independent i.e you can reduce the error of one without affecting the other
        return -(y * np.log(a) + (1 - y) * np.log(1 - a))

    @staticmethod
    def softmax_loss(y, a):
        # here we penalize only the targeted class and this is intuitive because they are all dependent i.e. if targeted
        # error is reduced the rest will give less probability because of the softmax relation
        return - np.sum(y * np.log(a), axis=0, keepdims=True)

    @staticmethod
    def cross_entropy_loss_prime(y, a):
        return -y / a + (1 - y) / (1 - a)

    @staticmethod
    def softmax_loss_prime(y, a):
        return -np.sum(y/a)

    @staticmethod
    def regularization_term(network, m, lmbda):
        agg = 0
        for layer in network.layers:
            agg = np.sum(np.square(layer.W))
        else:
            return (lmbda / (2 * m)) * agg

    def cost(self, network, X, y, lmbda=0):
        A = X
        for layer in network.layers:
            Z = np.dot(layer.W, A) + layer.b
            A = layer.activation(Z)
        else:
            loss_matrix = self.loss(y, A)
            sum_over_all_examples = np.sum(loss_matrix, axis=1) / loss_matrix.shape[1]
            return (np.sum(sum_over_all_examples) / sum_over_all_examples.size) + self.regularization_term(network,
                                                                                                           X.shape[1],
                                                                                                           lmbda=lmbda)

    def _update_weights(self, X, y, network, alpha, lmbda, t, beta1, beta2, decay_rate, epoch_num):
        A = X
        for layer in network.layers:
            layer.A_l_1 = A  # this is A-1 from last loop step
            Z = np.dot(layer.W, A) + layer.b  # (called "logits" in ML folklore)
            A = layer.activation(Z)

            # NB! we don't not apply dropout to the input layer or output layer.
            D = np.random.rand(*A.shape) <= layer.keep_prob  # dropout
            A = np.multiply(A, D) / layer.keep_prob  # inverted dropout

            layer.D = D
            layer.A = A

        with np.errstate(invalid='raise'):
            try:
                dLdA = self.activation_prime(y, A)
            except FloatingPointError:
                raise
        # To avoid the confusion: reversed() doesn't modify the list. reversed() doesn't make a copy of the list
        # (otherwise it would require O(N) additional memory). If you need to modify the list use alist.reverse(); if
        # you need a copy of the list in reversed order use alist[::-1]
        for l, layer, VsnSs in zip(range(len(network.layers), 0, -1), reversed(network.layers), reversed(self.VsnSs)):
            dLdA, dJdW, dJdb = network._calculate_single_layer_gradients(dLdA, layer, compute_dLdA_1=(l > 1))

            layer.W, layer.b = self.optimizer(dJdW, dJdb, layer.W, layer.b, X.shape[1], alpha=alpha, lmbda=lmbda,
                                              VS=VsnSs, beta1=beta1, beta2=beta2, t=t, decay_rate=decay_rate, epoch=epoch_num)

    def minimize(self, network, epochs=1, mini_batch_size=0, learning_rate=0.1, regularization_parameter=0,
                 momentum=0.9, beta2=0.999, learning_rate_decay=0, dataset=None):
        bef = time.time()
        for layer in network.layers:
            self.VsnSs.append({"Vdw": np.zeros_like(layer.W), "Vdb": np.zeros_like(layer.b),
                               "Sdw": np.zeros_like(layer.W), "Sdb": np.zeros_like(layer.b)})

        for epoch in range(1, epochs+1):
            for t, mini_batch in enumerate(dataset.next_mini_batch(size=mini_batch_size), start=1):
                self._update_weights(mini_batch.X, mini_batch.y, network, learning_rate,
                                     regularization_parameter, t, beta1=momentum, beta2=beta2, decay_rate=learning_rate_decay, epoch_num=epoch)
            else:
                if epoch % 10 == 0:
                    cost = self.cost(network, dataset.X_train, dataset.y_train, lmbda=regularization_parameter)
                    logger.info('epoch {} (error: {:.5f})'.format(epoch, cost))
        else:
            aft = time.time()
            cost = self.cost(network, dataset.X_train, dataset.y_train, lmbda=regularization_parameter)
            logger.debug('-' * 80)
            logger.debug('| Summary')
            logger.debug('-' * 80)
            logger.debug('training time: {:.2f} SECs'.format(aft - bef))
            logger.debug('-' * 80)
            logger.debug('Finish error: {:.5f}'.format(
                self.cost(network, dataset.X_train, dataset.y_train, lmbda=regularization_parameter)))

            ss = ''
            for i, layer in enumerate(network.layers):
                ss += '\n layer# ' + str(i + 1) + ' - ' + repr(layer)

            logger2.info('train error: {:.2f}, '
                         'time: {:.2f}SECs, '
                         '#layers {}, '
                         '#epochs: {}, '
                         'learning rate: {},\n'
                         'regularization parameter: {}, '
                         'mini-batch size: {}, '
                         'optimizer: [{}], '
                         'dataset: [{}, dev_size:{}, shuffle:{}], {}'.format(cost, aft - bef, len(network.layers),
                                                                             epochs, learning_rate,
                                                                             regularization_parameter, mini_batch_size,
                                                                             self.method, dataset.name, dataset.dev_size,
                                                                             dataset.shuffle, ss))


if __name__ == '__main__':
    pass
