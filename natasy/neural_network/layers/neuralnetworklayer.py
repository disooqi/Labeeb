class NeuralNetworkLayer:
    def __init__(self, *args, **kwargs):
        pass


class HiddenLayer(NeuralNetworkLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class OutputLayer(NeuralNetworkLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SoftmaxWithXEntropy(OutputLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SigmoidWithXEntroy(OutputLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    pass