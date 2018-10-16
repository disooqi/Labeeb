import numpy as np
import time
import logging


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


class Optimizer:
    def __init__(self, loss='binary_cross_entropy', method='gradient-descent'):
        self.method = method
        self.VsnSs = list()
        if loss == 'binary_cross_entropy':
            self.loss = self.binary_cross_entropy_loss
            self.loss_prime = self.binary_cross_entropy_loss_prime
        elif loss == 'multinomial_cross_entropy':
            self.loss = self.multinomial_cross_entropy_loss
            self.loss_prime = self.multinomial_cross_entropy_loss_prime

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
    def binary_cross_entropy_loss(y, a):
        # http://christopher5106.github.io/deep/learning/2016/09/16/about-loss-functions-multinomial-logistic-logarithm-cross-entropy-square-errors-euclidian-absolute-frobenius-hinge.html
        # https://stats.stackexchange.com/questions/260505/machine-learning-should-i-use-a-categorical-cross-entropy-or-binary-cross-entro
        # here we penalize every class even the zero ones
        # the classes here are independent i.e you can reduce the error of one without affecting the other
        return -(y * np.log(a) + (1 - y) * np.log(1 - a))

    @staticmethod
    def multinomial_cross_entropy_loss(y, a):
        """ Multinomial Cross Entropy Loss function

        Calculate the error in multiclass application where the output classes are dependent on each others and the
        instance should belong to one class only, this loss function is a perfect match for SOFTMAX func where it
        calculate the discrete probability distribution of the output classes
        :param y: is the target probabilities of the class, y.shape = (n, m) where (n) is classes count and (m) is the
        instance count
        :param a: is the probabilities of the classes as predicted by the model a.shape is also (n, m)
        :return: a vector of instances' loss values
        """
        # here we penalize only the targeted class and this is intuitive because they are all dependent i.e. if targeted
        # error is reduced the rest will give less probability because of the softmax relation
        return - np.sum(y * np.log(a), axis=0, keepdims=True)

    @staticmethod
    def binary_cross_entropy_loss_prime(y, a):
        return -y / a + (1 - y) / (1 - a)

    @staticmethod
    def multinomial_cross_entropy_loss_prime(y, a):
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
                dLdA = self.loss_prime(y, A)
            except FloatingPointError:
                raise
        # To avoid the confusion: reversed() doesn't modify the list. reversed() doesn't make a copy of the list
        # (otherwise it would require O(N) additional memory). If you need to modify the list use alist.reverse(); if
        # you need a copy of the list in reversed order use alist[::-1]
        for l, layer, VsnSs in zip(range(len(network.layers), 0, -1), reversed(network.layers), reversed(self.VsnSs)):
            dLdA, dJdW, dJdb = network.calculate_single_layer_gradients(dLdA, layer, compute_dLdA_1=(l > 1))

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
            logger.debug('| Summary: Training time: {:.2f} SECs, Finish error: {:.5f}'.format(aft - bef,
                                 self.cost(network, dataset.X_train, dataset.y_train, lmbda=regularization_parameter)))
            logger.debug('-' * 80)

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

