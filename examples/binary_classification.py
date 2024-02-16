import numpy as np
from labeeb.data_preparation.dataset import Dataset
from labeeb.neural_network.network import NeuralNetwork
from labeeb.optimization import Optimizer


class MyDataset(Dataset):
    def __init__(self, dev_size=0.2, shuffle=False, normalize_input_features=False, name='ex2data1'):
        data = np.loadtxt('data/ex2data1.txt', delimiter=',')
        features = data[:, :2]
        y = data[:, 2:]
        super().__init__(features.T, y.T, dev_size=dev_size, shuffle=shuffle, normalize_input_features=normalize_input_features, name=name)

    @staticmethod
    def prepare_target(y):
        classes = np.unique(y)
        incidence_y = np.zeros(y.shape)
        incidence_y[y==classes[0]] = 0
        incidence_y[y==classes[1]] = 1

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
            # y = y.argmax(axis=0) + 1
            # prediction = A.argmax(axis=0) + 1
            A[A>0.5] = 1.0
            A[A<=0.5] = 0.0
            a, b = np.squeeze(A), np.squeeze(y)
            res = np.equal(a, b)
            return 100 * np.sum(res) / y.size


if __name__ == '__main__':
    # import matplotlib.pyplot as plt

    bdata = MyDataset(dev_size=0)

    myNN = NeuralNetwork(n_features=bdata.n, n_classes=bdata.classes.size)
    # myNN.add_layer(4, activation='relu', dropout_keep_prob=1)
    myNN.add_output_layer()

    # gd-with-momentum gradient-descent rmsprop adam
    gd_optimizer = Optimizer(loss='binary_cross_entropy', method='gradient-descent')
    gd_optimizer.minimize(myNN, epochs=40000, mini_batch_size=1000, learning_rate=.00109, regularization_parameter=0,dataset=bdata)

    train_acc = bdata.accuracy(myNN, training_accuracy=True)
    # dev_acc = bdata.accuracy(myNN)

    print('train acc: {:.2f}%, Dev acc: {:.2f}%'.format(train_acc, 0))

