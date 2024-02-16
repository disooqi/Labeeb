from . import NeuralNetworkLayer


class HiddenLayer(NeuralNetworkLayer):
    def __init__(self, *args, **kwargs):
        super(HiddenLayer, self).__init__(*args, **kwargs)

