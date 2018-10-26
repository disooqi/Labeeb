import numpy as np
from natasy.data_preparation.dataset import Dataset
from natasy.neural_network.networks import FullyConnectedNetwork
from natasy.optimization import Optimizer


class MyDataset(Dataset):
    def __init__(self):
        data = np.loadtxt('../data/ex2data1.txt', delimiter=',')
        features = data[:, :2]
        y = data[:, 2:]

        super().__init__(features.T, y.T, dev_size=.99, shuffle=True, normalize_input_features=False, name='ex2data1')

    @staticmethod
    def prepare_target(y):
        classes = np.unique(y)
        incidence_y = np.zeros(y.shape)
        incidence_y[y==classes[0]] = 0
        incidence_y[y==classes[1]] = 1

        return incidence_y, classes


if __name__ == '__main__':
    bdata = MyDataset()
    myNN = FullyConnectedNetwork(n_features=bdata.n, n_classes=bdata.classes.size)
    myNN.add_output_layer()

    gd_optimizer = Optimizer(loss='binary_cross_entropy',
                             method='gradient-descent')  # gd-with-momentum gradient-descent rmsprop adam
    gd_optimizer.minimize(myNN, epochs=1, mini_batch_size=1, learning_rate=.1, regularization_parameter=0,
                          dataset=bdata)

    train_acc = myNN.accuracy(bdata.X_train, bdata.y_train)
    dev_acc = myNN.accuracy(bdata.X_dev, bdata.y_dev)
    print('train acc: {:.2f}%, Dev acc: {:.2f}%'.format(train_acc, dev_acc))

