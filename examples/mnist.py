import numpy as np
import scipy.io
from natasy.neural_network.network import NeuralNetwork
from natasy.optimization import Optimizer
from natasy.data_preparation.dataset import Dataset
from natasy.neural_network import Activation


class MNIST_dataset(Dataset):
    def __init__(self, X, y, dev_size=0.20):
        """

        :param X: examples (excpected to be in the shape of n*m)
        :param y:
        :param dev_size:
        """
        super(MNIST_dataset, self).__init__(X, y, dev_size, name='MNIST')

    @staticmethod
    def prepare_target(y):
        classes = np.unique(y)
        incidence_y = np.zeros((classes.size, y.size))
        incidence_y[y.ravel() - 1, np.arange(y.size)] = 1  # (5000, 10)
        return incidence_y, classes

    def accuracy(self, network, training_accuracy=False):
        # You only use dropout during training. Don't use dropout (randomly eliminate nodes) during test time.
        if training_accuracy:
            X, y = self.X_train, self.y_train
        else:
            X, y = self.X_dev, self.y_dev

        A = X
        for layer in network.layers:
            Z = np.dot(layer.W, A) + layer.b
            A = layer.activation(Z)
        else:
            y = y.argmax(axis=0) + 1
            prediction = A.argmax(axis=0) + 1
            res = np.equal(prediction, y)
            return 100 * np.sum(res) / y.size


if __name__ == '__main__':
    handwritten_digits = scipy.io.loadmat("data/ex3data1.mat")
    mnist = MNIST_dataset(handwritten_digits['X'], handwritten_digits['y'], dev_size=0.2111)

    nn01 = NeuralNetwork(n_features=400, n_classes=10)
    nn01.add_layer(100, activation=Activation.leaky_relu, dropout_keep_prob=1)
    nn01.add_layer(12, activation=Activation.softmax_stable, output_layer=True)

    # gd_optimizer = Optimizer(loss='multinomial_cross_entropy', method='gradient-descent') # gd-with-momentum gradient-descent rmsprop adam
    gd_optimizer = Optimizer(loss='multinomial_cross_entropy', method='adam') # gd-with-momentum gradient-descent rmsprop adam
    gd_optimizer.minimize(nn01, epochs=100, mini_batch_size=5000, learning_rate=.1, regularization_parameter=0, dataset=mnist)

    train_acc = mnist.accuracy(nn01, training_accuracy=True)
    dev_acc = mnist.accuracy(nn01)
    print('train acc: {:.2f}%, Dev acc: {:.2f}%'.format(train_acc, dev_acc))
