import scipy.io
from natasy.neural_network.networks import FullyConnectedNetwork
from natasy.optimization import Optimizer
from natasy.data_preparation.dataset import MNIST_dataset


if __name__ == '__main__':
    handwritten_digits = scipy.io.loadmat("natasy/data/ex3data1.mat")
    mnist = MNIST_dataset(handwritten_digits['X'].T, handwritten_digits['y'].T, dev_size=0.2)



    nn01 = FullyConnectedNetwork(n_features=400, n_classes=10)
    nn01.add_layer(33, activation='leaky_relu', dropout_keep_prob=1)
    nn01.add_output_layer(activation='sigmoid')



    gd_optimizer = Optimizer(loss='binary_cross_entropy', method='gradient-descent') # gd-with-momentum gradient-descent rmsprop adam
    gd_optimizer.minimize(nn01, epochs=100, mini_batch_size=512, learning_rate=.1, regularization_parameter=0, dataset=mnist)

    train_acc = nn01.accuracy(mnist.X_train, mnist.y_train)
    dev_acc = nn01.accuracy(mnist.X_dev, mnist.y_dev)
    print('train acc: {:.2f}%, Dev acc: {:.2f}%'.format(train_acc, dev_acc))
