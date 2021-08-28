import numpy as np

from natasy.neural_network import Initialization, initializations


def test_initialization():
    assert Initialization()


def test__zeros_initialization():
    W, b = initializations._zeros_initialization(4, 6)
    assert W.shape == (4, 6)
    assert b.shape == (4, 1)

    assert W.all() == 0 and b.all() == 0


def test__he_initialization():
    W, b = initializations._He_initialization(4, 6)
    assert W.shape == (4, 6)
    assert b.shape == (4, 1)
