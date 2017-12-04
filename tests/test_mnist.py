import numpy as np
import unittest

from sklearn.datasets import fetch_mldata

from slugnet.activation import Sigmoid
from slugnet.layers import Dense
from slugnet.model import Model


def get_mnist():
    ndigits = 10
    mnist_bunch = fetch_mldata('MNIST original')
    mnist_target = mnist_bunch['target']
    y = mnist_target.astype(np.int8)
    y_ohe = np.zeros(shape=(len(y), ndigits))
    y_ohe[np.arange(len(y)), y] = 1

    return mnist_bunch['data'], y_ohe


class TestMNIST(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.X, self.y = get_mnist()
        self.model = Model(lr=0.01, l1=0.0, l2=0.5)

        self.model.add_layer(Dense(784, 30, activation=Sigmoid()))
        self.model.add_layer(Dense(30, 10))

        self.model.fit(self.X, self.y)

    def test_mnist(self):
        pass
