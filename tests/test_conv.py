import numpy as np
import unittest

from sklearn.datasets import fetch_mldata

from slugnet.activation import ReLU, Softmax
from slugnet.layers import Convolution, Dense, MeanPooling, Flatten
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
    X = mnist_bunch['data'].reshape((-1, 1, 28, 28)) / 255.0

    return X, y_ohe


class TestMNISTWithConvnet(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.X, self.y = get_mnist()
        self.model = Model(lr=0.01, n_epoch=3, loss=SCCE(),
                           metrics=['loss', 'accuracy'], optimizer=RMSProp())

        self.model.add_layer(Convolution((None, 1, 28, 28), 1, (3, 3)))
        self.model.add_layer(MeanPooling((26, 26), (2, 2)))
        self.model.add_layer(Convolution((None, 1, 13, 13), 2, (4, 4)))
        self.model.add_layer(MeanPooling((10, 10), (2, 2)))
        self.model.add_layer(Flatten((2, 5, 5)))
        self.model.add_layer(Dense(50, 26, activation=Softmax()))

        self.fit_metrics = self.model.fit(self.X, self.y)

    def test_training_accuracy_above_ninety(self):
        self.assertGreater(self.fit_metrics['train']['accuracy'], 0.8)

    def test_validation_accuracy_above_ninety(self):
        self.assertGreater(self.fit_metrics['val']['accuracy'], 0.8)
