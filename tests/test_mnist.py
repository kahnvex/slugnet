import numpy as np
import unittest

from sklearn.datasets import fetch_mldata

from slugnet.activation import ReLU, Softmax
from slugnet.layers import Dense, Dropout
from slugnet.loss import SoftmaxCategoricalCrossEntropy as SCCE
from slugnet.model import Model
from slugnet.optimizers import RMSProp


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
        self.model = Model(lr=0.01, n_epoch=3, loss=SCCE(),
                           metrics=['loss', 'accuracy'], optimizer=RMSProp())

        self.model.add_layer(Dense(784, 200, activation=ReLU()))
        self.model.add_layer(Dense(200, 10, activation=Softmax()))

        self.fit_metrics = self.model.fit(self.X, self.y)

    def test_training_accuracy_above_ninety(self):
        self.assertGreater(self.fit_metrics['train']['accuracy'], 0.9)

    def test_validation_accuracy_above_ninety(self):
        self.assertGreater(self.fit_metrics['val']['accuracy'], 0.9)


class TestMNISTWithDropout(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.X, self.y = get_mnist()
        self.model = Model(lr=0.01, n_epoch=3, loss=SCCE(),
                           metrics=['loss', 'accuracy'], optimizer=RMSProp())

        self.model.add_layer(Dense(784, 200, activation=ReLU()))
        self.model.add_layer(Dropout(0.5))
        self.model.add_layer(Dense(200, 10, activation=Softmax()))

        self.fit_metrics = self.model.fit(self.X, self.y)

    def test_training_accuracy_above_ninety(self):
        self.assertGreater(self.fit_metrics['train']['accuracy'], 0.8)

    def test_validation_accuracy_above_ninety(self):
        self.assertGreater(self.fit_metrics['val']['accuracy'], 0.8)
