"""
This example has been created to mimic the keras example in https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py
In Keras example, they claim that they achieved about 98.40% accuracy. Using almost the same setup Natasy achieve about
98.58%
"""
import numpy as np
# from keras.datasets import mnist
from labeeb.data_preparation import Dataset
from labeeb.neural_network.network import NeuralNetwork
from labeeb.neural_network import Activation
from labeeb.optimization import Optimizer

class MNISTDataset(Dataset):
    def __init__(self, *args, **kwargs):
        X_train, y_train = kwargs.get('train')
        X_dev, y_dev = kwargs.get('dev')

        self.X_train = (np.reshape(X_train, (X_train.shape[0], -1)).T/255).astype('float32')
        self.y_train = np.reshape(y_train, (y_train.shape[0], -1)).T
        self.X_dev = (np.reshape(X_dev, (X_dev.shape[0], -1)).T/255).astype('float32')
        self.y_dev = np.reshape(y_dev, (y_dev.shape[0], -1)).T

        self.y_train, self.classes = self.prepare_target(self.y_train)
        self.y_dev, _ = self.prepare_target(self.y_dev)

        self.n, self.m = self.X_train.shape

        self.name = 'mnist-original'
        self.dev_size = None
        self.shuffle = None

    @staticmethod
    def prepare_target(y):
        classes = np.unique(y)
        incidence_y = np.zeros((classes.size, y.size))
        incidence_y[y.ravel(), np.arange(y.size)] = 1  # (5000, 10)
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
            y = y.argmax(axis=0)
            prediction = A.argmax(axis=0)
            res = np.equal(prediction, y)
            return 100 * np.sum(res) / y.size


if __name__ == '__main__':
    with np.load("./data/mnist.npz", allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    # return (x_train, y_train), (x_test, y_test)
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    mnist = MNISTDataset(train=(x_train, y_train), dev=(x_test, y_test))

    nn01 = NeuralNetwork(n_features=784, n_classes=10)
    nn01.add_layer(512, activation=Activation.relu, dropout_keep_prob=0.8)
    nn01.add_layer(512, activation=Activation.relu, dropout_keep_prob=0.8)

    nn01.add_layer(12, activation=Activation.softmax_stable, output_layer=True)

    gd_optimizer = Optimizer(loss='multinomial_cross_entropy',
                             method='rmsprop')  # gd-with-momentum gradient-descent rmsprop adam
    # keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    gd_optimizer.minimize(nn01, epochs=20, mini_batch_size=128, learning_rate=.0005, regularization_parameter=0,
                          dataset=mnist)

    train_acc = mnist.accuracy(nn01, training_accuracy=True)
    dev_acc = mnist.accuracy(nn01)
    print('train acc: {:.2f}%, Dev acc: {:.2f}%'.format(train_acc, dev_acc))
