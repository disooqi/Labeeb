import numpy as np
from .layers import FullyConnectedLayer


class NeuralNetwork:
    def __init__(self, n_features, n_classes):
        self.n = n_features
        self.n_classes = n_classes
        self.layers = list()

    def add_layer(self, n_units, *args, **kwargs):
        n_units_previous_layer = self.layers[-1].n_units if self.layers else self.n
        layer = FullyConnectedLayer(n_units=n_units, n_in=n_units_previous_layer, *args, **kwargs)
        self.layers.append(layer)

    def add_output_layer(self, *args, **kwargs):
        if self.n_classes == 2:
            n_units = 1
        elif self.n_classes < 2:
            pass # TODO: This should be an error on the data
        else:
            n_units = self.n_classes

        if not self.layers:
            self.add_layer(n_units=n_units, *args, **kwargs)
            self.layers[-1].output_layer = True
        elif not self.layers[-1].output_layer:
            self.add_layer(n_units=n_units, *args, **kwargs)
            self.layers[-1].output_layer = True
        else:
            # TODO: you should raise an error and message that says you need to delete existing output_layer
            pass

    def feedforward(self, X):
        A = X
        for i, layer in enumerate(self.layers):
            A = layer.calculate_layer_feedforward(A)
        else:
            return A


if __name__ == '__main__':
    pass