import math
import numpy as np
from collections import namedtuple


class Dataset:
    def __init__(self, X, y, dev_size, name=None, shuffle=True, normalize_input_features=True):
        self.name = name
        self.dev_size = dev_size
        self.shuffle = shuffle
        y, self.classes = self.prepare_target(y)

        if shuffle:
            X, y = self.unison_shuffled_copies(X, y.T)

        n_dev_sample = int(dev_size*len(X))
        self.X_train, self.X_dev = X[n_dev_sample:, :].T, X[:n_dev_sample, :].T
        self.y_train, self.y_dev = y[n_dev_sample:, :].T, y[:n_dev_sample, :].T

        self.n, self.m = self.X_train.shape

        if normalize_input_features:
            self.X_train, self.mu, self.sigma = self._normalize_input_features(self.X_train)
            self.X_dev = self._normalize_testset(self.X_dev, self.mu, self.sigma)

    @staticmethod
    def prepare_target(y):
        raise NotImplementedError

    def next_mini_batch(self, size=0):
        mini_batch_size = self.m if size <= 0 else size
        n_mini_batch = math.ceil(self.m/mini_batch_size)
        mini_batch = namedtuple('mini_batch', ['X', 'y'])
        for i in range(n_mini_batch):
            start = i*mini_batch_size
            end = start+mini_batch_size
            yield mini_batch(self.X_train[:, start:end], self.y_train[:, start:end])

    @staticmethod
    def _normalize_input_features(X):
        mu = np.mean(X, axis=1, keepdims=True)
        centered_X = X - mu
        # sigma_squared = np.sum(np.square(centered_X), axis=1, keepdims=True)/m
        sigma = np.std(centered_X, axis=1,keepdims=True, ddof=1) # you need to square it
        standard_normalized_X = np.divide(centered_X, sigma, where=sigma!=0)
        # andrew_normalized_X = np.divide(centered_X, sigma_squared, where=sigma_squared!=0)
        return standard_normalized_X, mu, sigma

    @staticmethod
    def _normalize_testset(X, mu, sigma):
        centered_X = X - mu
        standard_normalized_X = np.divide(centered_X, sigma, where=sigma != 0)
        return standard_normalized_X

    def accuracy(self, network, training_accuracy):
        raise NotImplementedError

    @staticmethod
    def unison_shuffled_copies(a, b):
        print(len(a), len(b))
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]








