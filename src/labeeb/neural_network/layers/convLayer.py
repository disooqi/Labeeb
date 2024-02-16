import numpy as np
from . import NeuralNetworkLayer


class ConvLayer(NeuralNetworkLayer):
    # output (n1+2p-f1)/s1 +1*(n2+2p-f2)/s2+1; where n1 and n2 are image dimensions, and f1 and f2 are filter dimensions
    # valid (no padding) and same (Pad so that output size is the same as the input size) convolutions
    # f is usually square and odd (almost always)
    # convolution operation is called cross-correlation in mathematics
    def __init__(self, height, width, n_channels, convolution='valid'):
        super().__init__()
        self.height, self.width, self.n_channels = height, width, n_channels
        self.filters = list() # sometimes they call it kernel

        self.convolution_schema = convolution
        if convolution == 'valid':
            self.p = 0
        elif convolution == 'same':
            # self.p = (f-s-n)/2 # for all filters and all strides
            pass

    def add_filter(self, height, width):
        self.Filter(height, width)
        # if self.convolution_schema ==
        assert height <= self.height
        assert width <= self.width
        self.initialize_filter(height, width)

    class Filter:
        def __init__(self, height, width):
            pass

        # def initialize_filter(self, height, width):
        #     return np.zeros((height, width, self. .n_channels))


if __name__ == '__main__':
    # http://www.cs.toronto.edu/~kriz/cifar.html
    # http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    pass
