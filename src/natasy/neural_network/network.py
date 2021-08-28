from .layers import FullyConnectedLayer, OutputLayer, SoftmaxWithXEntropy


class NeuralNetwork:
    def __init__(self, n_features, n_classes):
        self.n = n_features
        self.n_classes = n_classes
        self.layers = list()
        self.has_output_layer = False
        self.require_dLdA = False

    def add_layer(self, n_units, *args, **kwargs):
        if self.has_output_layer:
            raise Exception # TODO: implement Error "OutputLayerExistError"
        n_units_previous_layer = self.layers[-1].n_units if self.layers else self.n

        if kwargs.get('output_layer', False):
            n_units = self.n_classes if self.n_classes > 2 else 1 # TODO: This should be error in the data (n_classes<2)
            layer = SoftmaxWithXEntropy(n_units=n_units, n_in=n_units_previous_layer, *args, **kwargs)
            layer.output_layer = True
            self.has_output_layer = True
        else:
            layer = FullyConnectedLayer(n_units=n_units, n_in=n_units_previous_layer, *args, **kwargs)
        self.layers.append(layer)

    def feedforward(self, X):
        A = X
        for layer in self.layers:
            A = layer.calculate_layer_feedforward(A)
        else:
            return A


if __name__ == '__main__':
    pass