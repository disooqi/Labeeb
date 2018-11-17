from . import FullyConnectedLayer


class OutputLayer(FullyConnectedLayer):
    def __init__(self, *args, **kwargs):
        super(OutputLayer, self).__init__(*args, **kwargs)


class SigmoidWithXEntroy(OutputLayer):
    def __init__(self, *args, **kwargs):
        super(SigmoidWithXEntroy, self).__init__(*args, **kwargs)